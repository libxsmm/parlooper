/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include <stdio.h>
#include <array>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <functional>
#include <omp.h>
#include "jit_compile.h"
#include "par_loop_cost_estimator.h"
#include "par_loop_generator.h"
#include "threaded_loops.h"
#include <cstring>
#include <unistd.h>
#include <libxsmm.h>
#include <dnn_common.h>

double ifreq;

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

inline double getFreq() {
  long long int s = rdtsc();
  sleep(1);
  long long int e = rdtsc();
  return (e - s) * 1.0;
}

inline double getTime() {
  return rdtsc() * ifreq;
}

void find_prime_factors(long num, std::vector<long>& res) {
  long n = num;
  if (n == 1) {
    res.push_back(n);   
  }
  while (n % 2 == 0) {
    res.push_back(2);
    n = n/2;
  }
  for (int i = 3; i <= sqrt(n); i = i + 2) {
    while (n % i == 0) {
      res.push_back(i);
      n = n/i;
    }
  }
  if (n > 2) {
    res.push_back(n);
  }
  return;
}

template<typename DType>
int gemm_benchmark(int argc, char** argv) {
  // Setup default GEMM sizes
  int check_correctness = 1;
  char loop_specs_str[256] = "aBC";  
  long M = 1024*4, N = 1024*4, K = 1024*4;
  long bm = 32, bn = 32, bk = 32;
  long kbf = 1;
  long n_layers = 1;
  long n_iters = 1;
  long i;
  // Setup model and trace
  int use_model = 0;
  const char* const env_use_model = getenv("USE_MODEL");
  if (0 == env_use_model) {
    use_model = 0;
  } else {
    use_model = atoi(env_use_model);
  }
  ifreq = 1.0 / getFreq();
  std::vector<std::string> inp_trace[128];
  platform_spec_t my_platform;
  tensor_metadata_t tensor_metadata;
  set_platform_specs( CLX, omp_get_max_threads(), &my_platform);

  if (argc > 1) {
    sprintf(loop_specs_str, "%s", argv[1]);
  }
  if (argc > 2) {
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    bm = atoi(argv[5]);
    bn = atoi(argv[6]);
    bk = atoi(argv[7]);
    if (argc > 8) {
      kbf = atoi(argv[8]);
    }
    if (argc > 9) {
      n_layers = atoi(argv[9]);
    }
    if (argc > 10) {
      n_iters = atoi(argv[10]);
    }
  }

  if ((n_layers > 1) && !(M == K && bm == bk && bk == bn) ) {
    printf("MLP support only for M == K and bm == bn == bk\n");
    return 1;
  }

  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  long brcount = Kb/kbf;
  while (Kb % kbf != 0) {
    kbf--;
  }
  brcount = Kb/kbf;

  /* Early exit to avoid testing the same combos since in this case the "a" loop has trip count 1 */
  if (kbf == 1 && loop_specs_str[0] != 'a') {
    return 0;
  }

  // Allocate buffers
  DType **ACT = (DType**) malloc((n_layers+1)*sizeof(DType*));
  DType **WGT = (DType**) malloc(n_layers    *sizeof(DType*));
  for (i = 0; i < (n_layers+1); i++) {
    if (i % 2 == 0) {
      ACT[i] = (DType*) libxsmm_aligned_malloc(N*K*sizeof(DType), 2097152);
    } else {
      ACT[i] = (DType*) libxsmm_aligned_malloc(M*N*sizeof(DType), 2097152);
    }
    if (i < n_layers) {
      WGT[i] = (DType*) libxsmm_aligned_malloc(M*K*sizeof(DType), 2097152);
    }
  }
  float *naive_input  = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), 2097152);
  float *naive_output = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 2097152);
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 2097152);
  float *naive_filter = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 2097152);
  DType *naive_input_bf16  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), 2097152);
  DType *naive_output_bf16 = (DType*)libxsmm_aligned_malloc( M*N*sizeof(DType), 2097152);
  DType *naive_filter_bf16 = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), 2097152);
  libxsmm_matdiff_info norms, diff;
  libxsmm_matdiff_clear(&norms);
  libxsmm_matdiff_clear(&diff);

  // Init buffers
  init_buf( naive_input,     K*N, 0, 0 );
  init_buf( naive_output,    M*N, 0, 0 );
  init_buf( naive_filter,    M*K, 0, 0 );
  if (sizeof(DType) == 2) {
    libxsmm_rne_convert_fp32_bf16( naive_input,     (libxsmm_bfloat16*)naive_input_bf16,     N*K );
    libxsmm_rne_convert_fp32_bf16( naive_output,    (libxsmm_bfloat16*)naive_output_bf16,    N*M );
    libxsmm_rne_convert_fp32_bf16( naive_filter,    (libxsmm_bfloat16*)naive_filter_bf16,    M*K );
    matrix_copy_NC_to_NCNC_bf16(  (libxsmm_bfloat16*)naive_input_bf16,  (libxsmm_bfloat16*)ACT[0],     1, N, K, bn, bk );
    matrix_copy_NC_to_NCNC_bf16(  (libxsmm_bfloat16*)naive_output_bf16, (libxsmm_bfloat16*)ACT[n_layers],     1, N, M, bn, bm );
    for (i = 0; i < n_layers; i++) {
      matrix_copy_KC_to_KCCK_bf16( (libxsmm_bfloat16*)naive_filter_bf16, (libxsmm_bfloat16*)WGT[i]       , K, M, bk, bm );
    }
  } else {
    matrix_copy_NC_to_NCNC( naive_input,     (float*)ACT[0],     1, N, K, bn, bk );
    matrix_copy_NC_to_NCNC( naive_output,    (float*)ACT[n_layers],     1, N, M, bn, bm );
    for (i = 0; i < n_layers; i++) {
      matrix_copy_KC_to_KCCK( naive_filter,    (float*)WGT[i]       , K, M, bk, bm );
    }
  }
   
  // Setup TPP kernels
  auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tc_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tr_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto dtype      = (sizeof(DType) == 2) ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;
  auto l_shape = libxsmm_create_gemm_shape( bm, bn, bk, bm, bk, bm, dtype, dtype, dtype, dtype );
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*bk*sizeof(DType), bk*bn*sizeof(DType), brcount );
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bm*bn, 1, bm*bn, bm*bn, dtype, dtype, dtype);

  auto zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
  auto tileconfig_kernel  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
  auto tilerelease_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
  auto brgemm_kernel      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  
  // Compute reference if requested
  if (check_correctness) {
    naive_fullyconnected_t naive_param;
    naive_param.N = N;
    naive_param.C = K;
    naive_param.K = M;
    naive_param.fuse_type = 0;
    for (i = 0; i < n_layers; i++) {
      if (i % 2 == 0) {
        naive_fullyconnected_fused_fp(&naive_param, naive_input, naive_output, naive_filter, NULL);
      } else {
        naive_fullyconnected_fused_fp(&naive_param, naive_output, naive_input, naive_filter, NULL);
      }
    }
  }

  // JIT requested nested loop specs
  long k_step = brcount;
  long m_step = 1;
  long n_step = 1;
  // Prime factorization of trip-counts to find factors k0,m0 etc
  long k_trips = Kb/k_step;
  long m_trips = Mb/m_step;
  long n_trips = Nb/n_step;
  long m0, m1, n0, n1, k0, k1;

  std::vector<long> k_factors;
  find_prime_factors(k_trips, k_factors);
  std::vector<long> m_factors;
  find_prime_factors(m_trips, m_factors);
  std::vector<long> n_factors;
  find_prime_factors(n_trips, n_factors);
#if 0
  printf("K factors are: ");
  for (auto i = 0; i < k_factors.size(); i++) {
    printf("%d ", k_factors[i]);
  }
  printf("\n");
  printf("M factors are: ");
  for (auto i = 0; i < m_factors.size(); i++) {
    printf("%d ", m_factors[i]);
  }
  printf("\n");
  printf("N factors are: ");
  for (auto i = 0; i < n_factors.size(); i++) {
    printf("%d ", n_factors[i]);
  }
  printf("\n");
#endif
  k0 = k_factors[0];
  k1 = (k_factors.size() > 1) ? k_factors[1] : 1;
  m0 = m_factors[0];
  m1 = (m_factors.size() > 1) ? m_factors[1] : 1;
  n0 = n_factors[0];
  n1 = (n_factors.size() > 1) ? n_factors[1] : 1;
 
  long l0_k_step = k0 * k_step;
  long l0_m_step = m0 * m_step;
  long l0_n_step = n0 * n_step;
  long l1_k_step = k1 * l0_k_step;
  long l1_m_step = m1 * l0_m_step;
  long l1_n_step = n1 * l0_n_step;


  auto t0 = getTime();
  auto gemm_loop = ThreadedLoop<3>({
      LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // Logical K loop specs
      LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // Logical M loop specs
      LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // Logical N loop specs
      loop_specs_str);
  auto t1 = getTime();

  // Warmup iteration for i-caches
  for (i = 0; i < n_layers; i++) {
    gemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;
          gemm_param.op.tertiary = (void*)&brcount;
          gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
          gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
          gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
          if (i_k == 0) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)gemm_param.c.primary;
            zero_kernel( &zero_param );
          }
          brgemm_kernel( &gemm_param );
        },
        [&]() {if (sizeof(DType) == 2) tileconfig_kernel(NULL);},
        [&]() {if (sizeof(DType) == 2) tilerelease_kernel(NULL);});
  }

#if 0
  // Nuke buffers from cache in case we run 1 layer
  if (n_layers == 1) {
    gemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          DType *my_A = (DType*)WGT[0] + i_m * K * bm + i_k * bk * bm;
          DType *my_B = (DType*)ACT[0] + i_n * K * bn + i_k * bk * bn;
          DType *my_C = (DType*)ACT[1] + i_n * M * bn + i_m * bn * bm;
          unsigned long long  __i, bri;
          for (bri = 0 ; bri < brcount; bri++) {
            for (__i = 0; __i < (bk*bm)/(64/sizeof(DType)); __i++) {
              _mm_clflush((DType*)my_A + bri * bk * bm + __i * (64/sizeof(DType)));
            }
            for (__i = 0; __i < (bn*bk)/(64/sizeof(DType)); __i++) {
              _mm_clflush((DType*)my_B + bri * bk * bn + __i * (64/sizeof(DType)));
            }
          }
          for (__i = 0; __i < (bm*bn)/(64/sizeof(DType)); __i++) {
            _mm_clflush((DType*)my_C + __i * (64/sizeof(DType)));
          }
        },
        [&]() {},
        [&]() {});
  }
#endif
  
  // benchmark the GEMM
  auto t_start = getTime();
  for (long it = 0; it < n_iters; it++) {
    for (i = 0; i < n_layers; i++) {
      gemm_loop(
          [&](int* ind) {
            int i_k = ind[0], i_m = ind[1], i_n = ind[2];
            libxsmm_gemm_param gemm_param;
            gemm_param.op.tertiary = (void*)&brcount;
            gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
            gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
            gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
            if (i_k == 0) {
              libxsmm_meltw_unary_param zero_param;
              zero_param.out.primary = (void*)gemm_param.c.primary;
              zero_kernel( &zero_param );
            }
            brgemm_kernel( &gemm_param );
          },
          [&]() {if (sizeof(DType) == 2) tileconfig_kernel(NULL);},
          [&]() {if (sizeof(DType) == 2) tilerelease_kernel(NULL);});
    }
  }
  auto t_end = getTime();
 
  // Check correctness if requested
  if (n_layers == 1) {
    printf("##########################################\n");
    printf("#  GEMM %d x %d x %d  (M x N x K)        \n", M, N, K);
    printf("##########################################\n");
  } else {
    printf("##########################################\n");
    printf("#  %d Layer MLP with sizes  %d x %d x %d  (M x N x K)  \n", n_layers, M, N, K);
    printf("##########################################\n");
  }
  if (check_correctness) {
    if (sizeof(DType) == 2) {
      matrix_copy_NCNC_to_NC_bf16( (libxsmm_bfloat16*)ACT[n_layers], (libxsmm_bfloat16*)naive_output_bf16, 1, N, M, bn, bm );
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_output_opt, N*M );
    } else {
      matrix_copy_NCNC_to_NC( (float*)ACT[n_layers], naive_output_opt, 1, N, M, bn, bm );
    }
    printf("##########################################\n");
    printf("#           Correctness                  #\n");
    printf("##########################################\n");
    if (n_layers % 2 == 1) {
      libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*M, 1, naive_output, naive_output_opt, 0, 0);
    } else {
      libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*M, 1, naive_input, naive_output_opt, 0, 0);
    }
    printf("L1 reference  : %.25g\n", norms.l1_ref);
    printf("L1 test       : %.25g\n", norms.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms.l2_rel);
    printf("Linf abs.error: %.24f\n", norms.linf_abs);
    printf("Linf rel.error: %.24f\n", norms.linf_rel);
    printf("Check-norm    : %.24f\n", norms.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms);
  }

  // Model GEMM
  auto t_trace_start = getTime();
  double modeled_time = 0.0;
  if (use_model > 0) {
    set_tensor_metadata(bm, bn, bk, brcount, sizeof(DType), &tensor_metadata);
    for (i = 0; i < n_layers; i++) {
      gemm_loop(
          [&](int* ind) {
            int nc = ind[0], s1 = ind[1], nk = ind[2];
            char record[256];
            int my_thread_id = omp_get_thread_num();
            sprintf(record, "WGT%d[%d][%d]", i, s1, nc);
            std::string a_access(record);
            inp_trace[my_thread_id].push_back(a_access);
            sprintf(record, "ACT%d[%d][%d]", i,  nk, nc);
            std::string b_access(record);
            inp_trace[my_thread_id].push_back(b_access);
            sprintf(record, "ACT%d[%d][%d]", i+1,  nk, s1);
            std::string c_access(record);
            inp_trace[my_thread_id].push_back(c_access);
          },
          [&]() {},
          [&]() {});
    }
  }
  auto t2 = getTime();
  if (use_model > 0) {
    modeled_time = tensor_contraction_cost_estimator(
        PARALLEL_TRACES, inp_trace, tensor_metadata, my_platform);
  }
  auto t3 = getTime();

  // Print performance/model numbers
  double gflop = (2.0*(double)n_layers*(double)M*(double)N*(double)K) / (1000*1000*1000);
  printf("Time is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  if (use_model > 0) {
    printf("Model time gemm is %.5g ms (%.5g GFLOPS)\n", modeled_time, gflop/(modeled_time/1000.0));
    printf("Tracing takes %.5g ms and modeling takes %.5g ms\n", 1000.0*(t2-t_trace_start), 1000.0*(t3-t2));
    printf("Compilation time is %.5g s\n", t1-t0);
    printf("MODELED %.5g %s_%d_%d_%d_%d_%d_%d_bf%d_threads%d\n", gflop/(modeled_time/1000.0), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads());
  }
  printf("MEASURE %.5g %s_%d_%d_%d_%d_%d_%d_bf%d_threads%d\n", gflop/((t_end-t_start)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads());

  // Free buffers
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_output_opt);
  libxsmm_free(naive_input_bf16);
  libxsmm_free(naive_output_bf16);
  libxsmm_free(naive_filter_bf16);
  for (i = 0; i < (n_layers+1); i++) {
    libxsmm_free(ACT[i]);
    if (i < n_layers) {
      libxsmm_free(WGT[i]);
    }
  }
  free(ACT);
  free(WGT);

  return 0;
}

int main(int argc, char** argv) {
  int use_prec_bf16 = 0;
  const char* const env_prec_str = getenv("USE_BF16");
  if (0 == env_prec_str) {
    use_prec_bf16 = 0;
  } else {
    use_prec_bf16 = atoi(env_prec_str);
  }
  if (use_prec_bf16 == 0) {
    return gemm_benchmark<float>(argc, argv);  
  } else {
    return gemm_benchmark<libxsmm_bfloat16>(argc, argv);  
  }
}

