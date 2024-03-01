/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "common_utils.h"
#include "threaded_loops.h"
#include "gemm_common_utils.h"

#define N_BRGEMMS_PER_GEMM 128
#define N_TASKS_PER_GEMM 8


template<typename DType>
int gemm_benchmark(int argc, char** argv) {
  // Setup default GEMM sizes
  char loop_specs_str[256] = "aBC";
  long M = 1024*4, N = 1024*4, K = 1024*4;
  long bm = 32, bn = 32, bk = 32;
  long kbf = 1;
  long n_layers = 1;
  long n_iters = 1;
  long i;
  long check_correctness = 0;
  long cache_resident_acts = 0;
  long flat_act = 0;

  ifreq = 1.0 / getFreq();
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
    if (argc > 15) {
      cache_resident_acts = atoi(argv[15]);
    } 
    if (argc > 16) {
      check_correctness = atoi(argv[16]);
    } 
    if (argc > 17) {
      flat_act = atoi(argv[17]);
    } 
  }

  
  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  long  brcount = Kb/kbf;
  while (Kb % kbf != 0) {
    kbf--;
  }
  brcount = Kb/kbf;

  // Allocate buffers
  DType **ACT = (DType**) malloc((n_layers+1)*sizeof(DType*));
  check_null_ptr(ACT, "ACT array");
  DType **WGT = (DType**) malloc(n_layers    *sizeof(DType*));
  check_null_ptr(WGT, "WGT array");
  for (i = 0; i < (n_layers+1); i++) {
    ACT[i] = (DType*) libxsmm_aligned_malloc(LIBXSMM_MAX(K,M)*N*sizeof(DType), 64);
    check_null_ptr(ACT[i], "ACT[i] array"); 
    if (i < n_layers) {
      WGT[i] = (DType*) libxsmm_aligned_malloc(M*K*sizeof(DType), 64);
      check_null_ptr(WGT[i], "WGT[i] array"); 
    }
  }
  float *naive_input  = (float*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(float), 64);
  check_null_ptr(naive_input, "naive_input array");
  float *naive_output = (float*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(float), 64);
  check_null_ptr(naive_output, "naive_output array");
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(float), 64);
  check_null_ptr(naive_output_opt, "naive_output_opt array");
  float *naive_filter = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 64);
  check_null_ptr(naive_filter, "naive_filter array");
  DType *naive_input_bf16  = (DType*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(DType), 64);
  check_null_ptr(naive_input_bf16, "naive_input_bf16 array");
  DType *naive_output_bf16 = (DType*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(DType), 64);
  check_null_ptr(naive_output_bf16, "naive_output_bf16 array");
  DType *naive_filter_bf16 = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), 64);
  check_null_ptr(naive_filter_bf16, "naive_filter_bf16 array");
  
  // Init buffers
  init_buf( naive_input,     LIBXSMM_MAX(K,M)*N, 0, 0 );
  init_buf( naive_output,    LIBXSMM_MAX(K,M)*N, 0, 0 );
  init_buf( naive_filter,    M*K, 0, 0 );

  libxsmm_rne_convert_fp32_bf16( naive_input,     (libxsmm_bfloat16*)naive_input_bf16,     N*LIBXSMM_MAX(K,M));
  libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_input_bf16, naive_input, N*LIBXSMM_MAX(K,M));
  libxsmm_rne_convert_fp32_bf16( naive_output,    (libxsmm_bfloat16*)naive_output_bf16,    N*LIBXSMM_MAX(K,M) );
  libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_output, N*M);
  libxsmm_rne_convert_fp32_bf16( naive_filter,    (libxsmm_bfloat16*)naive_filter_bf16,    M*K );
  libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_filter_bf16, naive_filter, M*K);
  for (i = 0; i < n_layers; i++) {
    matrix_copy_KC_to_KCCK_bf16_local( (libxsmm_bfloat16*)naive_filter_bf16, (libxsmm_bfloat16*)WGT[i], K, M, bk, bm );
    if (flat_act > 0) {
      memcpy((libxsmm_bfloat16*)ACT[i],(libxsmm_bfloat16*)naive_input_bf16, (n_layers == 1) ? K*N*sizeof(libxsmm_bfloat16) : LIBXSMM_MAX(K,M)*N*sizeof(libxsmm_bfloat16));
    } else {
      matrix_copy_NC_to_NCNC_bf16_local( (libxsmm_bfloat16*)naive_input_bf16, (libxsmm_bfloat16*)ACT[i] , N, (n_layers == 1) ? K : LIBXSMM_MAX(K,M), bn, bk );
    } 
  }
  if (flat_act > 0) {
    memcpy((libxsmm_bfloat16*)ACT[n_layers],(libxsmm_bfloat16*)naive_output, (n_layers == 1) ? M*N*sizeof(libxsmm_bfloat16) : LIBXSMM_MAX(K,M)*N*sizeof(libxsmm_bfloat16));
  } else {
    matrix_copy_NC_to_NCNC_bf16_local( (libxsmm_bfloat16*)naive_output, (libxsmm_bfloat16*)ACT[n_layers], N, (n_layers == 1) ? M : LIBXSMM_MAX(K,M), bn, bk );
  }
  
  // Setup TPP kernels
  auto l_flags    = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ;
  auto l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  auto l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  
  auto dtype      = LIBXSMM_DATATYPE_BF16;
  auto l_shape = libxsmm_create_gemm_shape( bm, bn, bk, bm, (flat_act > 0) ? K : bk, (flat_act > 0) ? M : bm, dtype, dtype, dtype, LIBXSMM_DATATYPE_F32 );
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*bk*sizeof(DType), (flat_act > 0) ? bk*sizeof(DType) : bk*bn*sizeof(DType), brcount );
  auto l_unary_shape = libxsmm_create_meltw_unary_shape((flat_act > 0) ? bm : bm*bn, (flat_act > 0) ? bn : 1, (flat_act > 0) ? M : bm*bn, (flat_act > 0) ? M : bm*bn, dtype, dtype, dtype);

  if (brcount == Kb) l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

  auto zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  check_null_ptr((void*)zero_kernel, "zero_kernel TPP");
  auto tileconfig_kernel  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
  check_null_ptr((void*)tileconfig_kernel, "tileconfig_kernel TPP");
  auto tilerelease_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
  check_null_ptr((void*)tilerelease_kernel, "tilerelease_kernel TPP");
  auto brgemm_kernel      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  check_null_ptr((void*)brgemm_kernel, "brgemm_kernel TPP");

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
        libxsmm_rne_convert_fp32_bf16( naive_output,     (libxsmm_bfloat16*)naive_output_bf16,     N*M );
        libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_output, N*M);
      } else {
        naive_fullyconnected_fused_fp(&naive_param, naive_output, naive_input, naive_filter, NULL);
        libxsmm_rne_convert_fp32_bf16( naive_input,     (libxsmm_bfloat16*)naive_output_bf16,     N*M );
        libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_input, N*M);
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

  long n_threads = omp_get_max_threads();
  long n_brgemms = 0;

  auto gemm_loop = ThreadedLoop<3>({
      LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // Logical K loop specs
      LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // Logical M loop specs
      LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // Logical N loop specs
      loop_specs_str);

  for (i = 0; i < n_layers; i++) {
    gemm_loop( [&](int* ind) {
      int i_k = ind[0], i_m = ind[1], i_n = ind[2];
      int m_id = gemm_loop.get_tid_in_parallel_dim('b', ind);
      int n_id = gemm_loop.get_tid_in_parallel_dim('c', ind);
      if (m_id == 0 && n_id == 0) {
        //printf("I am thread with global ID %d and m_id %d and n_is %d\n", gemm_loop.get_tid(ind), m_id, n_id);
        n_brgemms++;
      }
    },
    [&]() {},
    [&]() {});
  }

  printf("In total there are %d BRGEMMS per iter\n", n_brgemms);
  double *timelines[n_threads];
  long index_in_timeline[n_threads];
  long n_record_iters = 5;
   for (i = 0; i < n_threads; i++) {
    timelines[i] = (double*) libxsmm_aligned_malloc((n_record_iters*n_brgemms+1)*sizeof(double), 64);
    memset(timelines[i], 0, (n_record_iters*n_brgemms+1)*sizeof(double));
  }
  memset(index_in_timeline, 0, n_threads*sizeof(long));

  long iter_to_record = 300;
  double profile_times[4*n_threads];
  long long local_brgemm_ids[n_threads];
  memset(profile_times, 0, (4*n_threads)*sizeof(double));
  memset(local_brgemm_ids, 0, n_threads*sizeof(long long));
  long long n_prof_0 = 0;
  long long n_prof_1 = 0;
  long long n_prof_2 = 0;
  long long n_prof_3 = 0;

  // Warmup iteration for i-caches
  for (i = 0; i < n_layers; i++) {
    gemm_loop(
      [&](int* ind) {
        int i_k = ind[0], i_m = ind[1], i_n = ind[2];
        libxsmm_gemm_param gemm_param;
        gemm_param.op.tertiary = (void*)&brcount;
        gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
        if (flat_act == 0) {
          if (cache_resident_acts > 0) {
            gemm_param.b.primary = (void*)((DType*)ACT[0] + i_n * K * bn + i_k * bk * bn );
          } else {
            gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
          }
          gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
        } else {
          if (cache_resident_acts > 0) {
            gemm_param.b.primary = (void*)((DType*)ACT[0] + i_n * K * bn + i_k * bk );
          } else {
            gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk );
          }
          gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bm );
        }
     
        if ((i_k == 0) && (brcount != Kb)) {
          libxsmm_meltw_unary_param zero_param;
          zero_param.out.primary = (void*)gemm_param.c.primary;
          zero_kernel( &zero_param );
        }
        brgemm_kernel( &gemm_param );
      },
      [&]() {tileconfig_kernel(NULL);},
      [&]() {tilerelease_kernel(NULL);});
  }


  // Check correctness if requested
  if (n_layers == 1) {
    printf("##########################################\n");
    printf("#  GEMM %ld x %ld x %ld  (M x N x K)        \n", M, N, K);
    printf("##########################################\n");
  } else {
    printf("##############################################################\n");
    printf("    %ld Layer MLP with sizes  %ld x %ld x %ld  (M x N x K)  \n", n_layers, M, N, K);
    printf("##############################################################\n");
  }

  if (check_correctness) {
    libxsmm_matdiff_info norms, diff;
    libxsmm_matdiff_clear(&norms);
    libxsmm_matdiff_clear(&diff);
    if (flat_act > 0) {
      memcpy((libxsmm_bfloat16*)naive_output_bf16, (libxsmm_bfloat16*)ACT[n_layers], M*N*sizeof(libxsmm_bfloat16));
    } else {
      matrix_copy_NCNC_to_NC_bf16( (libxsmm_bfloat16*)ACT[n_layers], (libxsmm_bfloat16*)naive_output_bf16, 1, N, M, bn, bm );
    }
    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_output_opt, N*M );
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

  // benchmark the GEMM
  auto t_start = getTime();
  for (long it = 0; it < n_iters; it++) {
    for (i = 0; i < n_layers; i++) {
      gemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;
          
          long tid = gemm_loop.get_tid(ind);
          long long local_brgemm_id = local_brgemm_ids[tid];               
          double t_before = 0.0, t_after = 0.0;
          long record_iter = ((it >= iter_to_record) && (it <= iter_to_record+n_record_iters-1)) ? 1 : 0;
          long cur_index_in_timeline = 0;

          gemm_param.op.tertiary = (void*)&brcount;
          gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
          if (flat_act == 0) {
            if (cache_resident_acts > 0) {
              gemm_param.b.primary = (void*)((DType*)ACT[0] + i_n * K * bn + i_k * bk * bn );
            } else {
              gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
            }
            gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
          } else {
            if (cache_resident_acts > 0) {
              gemm_param.b.primary = (void*)((DType*)ACT[0] + i_n * K * bn + i_k * bk );
            } else {
              gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk );
            }
            gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bm );
          }
          if ((i_k == 0) && (brcount != Kb)) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)gemm_param.c.primary;
            zero_kernel( &zero_param );
          }
          t_before = getTime();
          brgemm_kernel( &gemm_param );
          t_after = getTime();
          if (record_iter) {
            cur_index_in_timeline = index_in_timeline[tid];
            index_in_timeline[tid]++;
            timelines[tid][cur_index_in_timeline+1] = (t_after-t_before)+timelines[tid][cur_index_in_timeline];
          }
          /* Add time based on profile type */
          if (local_brgemm_id % N_BRGEMMS_PER_GEMM == 0) {
            /* Streaming from mem/llc both A and B */
            if (tid == 0) n_prof_0++;
            profile_times[tid*4 + 0] += (t_after-t_before);
          } else if (local_brgemm_id % N_BRGEMMS_PER_GEMM < N_TASKS_PER_GEMM) {
            /* Streaming from mem/llc B, A in L2 */
            if (tid == 0) n_prof_1++;
            profile_times[tid*4 + 1] += (t_after-t_before);
          } else if (local_brgemm_id % N_TASKS_PER_GEMM == 0) {
            /* Streaming from mem/llc A, B in L2 */
            if (tid == 0) n_prof_2++;
            profile_times[tid*4 + 2] += (t_after-t_before);
          } else {
            /* Streaming from L2 both A and B*/
            if (tid == 0) n_prof_3++;
            profile_times[tid*4 + 3] += (t_after-t_before);
          }
          local_brgemm_ids[tid]++;
        },
        [&]() {tileconfig_kernel(NULL);},
        [&]() {tilerelease_kernel(NULL);});
    }
  }
  auto t_end = getTime();
  
  // Print performance/model numbers
  double gflop = (2.0*(double)n_layers*(double)M*(double)N*(double)K) / (1000*1000*1000);
  printf("Time is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  printf("Effective model sizes: %.5g GB\n", ((double)sizeof(DType)*(double)n_layers*(double)M*(double)K)/(1024.0*1024.0*1024.0));
  printf("Effective A BW is %.5g GB/s\n", (((double)sizeof(DType)*(double)n_layers*(double)M*(double)K) / (1024.0*1024.0*1024.0))/((t_end-t_start)/(1.0*n_iters)));
  printf("MEASURE %.5g %s_%ld_%ld_%ld_%ld_%ld_%ld_bf%ld_threads%d\n", gflop/((t_end-t_start)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads());

  long thread_stats_id = 0;
  printf("\n\nTmeline stats for thread %d\n", thread_stats_id);
  double n_flops_per_brgemm = 2.0*bm*bn*bk*brcount;
  double slab_size = 1.0*bm*bk*brcount*sizeof(DType)/1024.0/1024.0/1024.0;
  double epsilon = 0.0;
  long layer_to_print = 20;

  FILE *fp = fopen("timeline.txt", "w");
  for (i = layer_to_print * (n_brgemms/n_layers) * (n_record_iters-1); i < (n_brgemms/n_layers) + layer_to_print * (n_brgemms/n_layers) * (n_record_iters-1) ; i++) {
    double _t0 = timelines[thread_stats_id][i];
    double _t1 = timelines[thread_stats_id][i+1];
    double effective_gflops = 1.0*n_flops_per_brgemm/(1000000000.0*(_t1-_t0));
    double effective_bw_1_slab = 1.0*slab_size/(1.0*(_t1-_t0));
    double effective_bw_2_slab = 2.0*slab_size/(1.0*(_t1-_t0));
    if (i % n_brgemms == 0) {
      printf("New set of n_brgemms\n");
    }
    //printf("Effective Gflops is %.5g\n", effective_gflops);
    //printf("Effective GB/s 1 slab is %.5g\n", effective_bw_1_slab);
    //printf("Effective GB/s 2 slab is %.5g\n", effective_bw_2_slab);
    /* Print two time points: _t0 and _t1 - epsilon with the effective GFLOPS */
    epsilon = (_t1 -_t0)/1000.0;
    fprintf(fp, "%.10g\t%.5g\n", _t0, effective_gflops);
    fprintf(fp, "%.10g\t%.5g\n", _t1-epsilon, effective_gflops);
    //printf("%.10g\t%.5g\n", _t0, effective_gflops);
    //rintf("%.10g\t%.5g\n", _t1-epsilon, effective_gflops);
  }
  fclose(fp);

  /* Now take averages for the 4 profiles across threads */
  double prof_avg_0 = 0.0, prof_avg_1 = 0.0, prof_avg_2 = 0.0, prof_avg_3 = 0.0;
  for (i = 0; i < n_threads; i++) {
    prof_avg_0 += profile_times[i*4 + 0];
    prof_avg_1 += profile_times[i*4 + 1];
    prof_avg_2 += profile_times[i*4 + 2];
    prof_avg_3 += profile_times[i*4 + 3];
  }
  prof_avg_0 = prof_avg_0/(n_threads*1.0*n_prof_0);
  prof_avg_1 = prof_avg_1/(n_threads*1.0*n_prof_1);
  prof_avg_2 = prof_avg_2/(n_threads*1.0*n_prof_2);
  prof_avg_3 = prof_avg_3/(n_threads*1.0*n_prof_3);
  //printf("Avg 0 / 1 / 2 / 3 are %.5g %.5g %.5g %.5g\n", prof_avg_0, prof_avg_1, prof_avg_2, prof_avg_3);
  //printf("Count 0 / 1 / 2 / 3 are %lld %lld %lld %lld\n", n_prof_0, n_prof_1, n_prof_2, n_prof_3);

  double core_freq = 1.9;
  printf("Avg BW (GB/s) for profile 0 (A & B from mem/llc) is %.5g\n", 2.0*slab_size/(1.0*prof_avg_0));
  printf("Avg BW (Bytes/c) for profile 0 is %.5g\n", (2.0*slab_size/(1.0*prof_avg_0))*1024.0*1024.0*1024.0/(core_freq*1000000000.0));
  printf("Avg BW (GB/s) for profile 1 (B from mem/llc, A in L2) is %.5g\n", 1.0*slab_size/(1.0*prof_avg_1));
  printf("Avg BW (Bytes/c) for profile 1 is %.5g\n", (1.0*slab_size/(1.0*prof_avg_1))*1024.0*1024.0*1024.0/(core_freq*1000000000.0));
  printf("Avg BW (GB/s) for profile 2 (A from mem/llc, B in L2) is %.5g\n", 1.0*slab_size/(1.0*prof_avg_2));
  printf("Avg BW (Bytes/c) for profile 2 is %.5g\n", (1.0*slab_size/(1.0*prof_avg_2))*1024.0*1024.0*1024.0/(core_freq*1000000000.0));
  printf("Avg GFLOPS for profile 3 is %.5g\n", 1.0*n_flops_per_brgemm/(1000000000.0*prof_avg_3));

  // Free buffers
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_opt);
  libxsmm_free(naive_filter);
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
  return gemm_benchmark<libxsmm_bfloat16>(argc, argv);  
}

