/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "common_utils.h"

template<typename DType>
int gemm_benchmark(int argc, char** argv) {
  // Setup default GEMM sizes
  int check_correctness = 1;
  char loop_specs_str[256] = "aBC";  
  long M = 1024*4, N = 1024*4, K = 1024*4;
  long bm = 32, bn = 32, bk = 32;
  long mbf = 1;
  long private_wt_trans = 0;
  long n_iters = 1;
  long i;
  long n_threads = omp_get_max_threads();
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
      mbf = atoi(argv[8]);
    }
    if (argc > 9) {
      private_wt_trans = atoi(argv[9]);
    }
    if (argc > 10) {
      n_iters = atoi(argv[10]);
    }
  }

  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  long brcount = Mb/mbf;
  while (Mb % mbf != 0) {
    mbf--;
  }
  brcount = Mb/mbf;

  // Allocate buffers
  float *naive_input  = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), 2097152);
  float *naive_output = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 2097152);
  float *naive_filter = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 2097152);
  float *naive_input_check = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), 2097152);
  float *naive_input_opt = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), 2097152);
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 2097152);
  float *naive_filter_opt = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 2097152);
  float *tr_filter_ = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 2097152);
  float *tr_filter_prv_ = (float*)libxsmm_aligned_malloc( n_threads*M*K*sizeof(float), 2097152);

  DType *naive_input_bf16  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), 2097152);
  DType *naive_output_bf16 = (DType*)libxsmm_aligned_malloc( M*N*sizeof(DType), 2097152);
  DType *naive_filter_bf16 = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), 2097152);
  DType *naive_input_bf16_opt  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), 2097152);
  DType *naive_output_bf16_opt = (DType*)libxsmm_aligned_malloc( M*N*sizeof(DType), 2097152);
  DType *naive_filter_bf16_opt = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), 2097152);

  long trans_tracker_size = Kb + 64 - 64%Kb;
  char *trans_tracker = (char*)libxsmm_aligned_malloc( n_threads*trans_tracker_size*sizeof(char), 2097152);
  
  libxsmm_matdiff_info norms, diff;
  libxsmm_matdiff_clear(&norms);
  libxsmm_matdiff_clear(&diff);

  // Init buffers
  init_buf( (float*)trans_tracker,   (n_threads*trans_tracker_size)/4, 0, 0 );
  init_buf( naive_input,     K*N, 0, 0 );
  init_buf( naive_output,    M*N, 0, 0 );
  init_buf( naive_filter,    M*K, 0, 0 );
  if (sizeof(DType) == 2) {
    libxsmm_rne_convert_fp32_bf16( naive_input,     (libxsmm_bfloat16*)naive_input_bf16,     N*K );
    libxsmm_rne_convert_fp32_bf16( naive_output,    (libxsmm_bfloat16*)naive_output_bf16,    N*M );
    libxsmm_rne_convert_fp32_bf16( naive_filter,    (libxsmm_bfloat16*)naive_filter_bf16,    M*K );
    matrix_copy_NC_to_NCNC_bf16(  (libxsmm_bfloat16*)naive_input_bf16,  (libxsmm_bfloat16*)naive_input_bf16_opt,     1, N, K, bn, bk );
    matrix_copy_NC_to_NCNC_bf16(  (libxsmm_bfloat16*)naive_output_bf16, (libxsmm_bfloat16*)naive_output_bf16_opt,     1, N, M, bn, bm );
    matrix_copy_KC_to_KCCK_bf16( (libxsmm_bfloat16*)naive_filter_bf16, (libxsmm_bfloat16*)naive_filter_bf16_opt      , K, M, bk, bm );
  } else {
    matrix_copy_NC_to_NCNC( naive_input,     (float*)naive_input_opt,     1, N, K, bn, bk );
    matrix_copy_NC_to_NCNC( naive_output,    (float*)naive_output_opt,     1, N, M, bn, bm );
    matrix_copy_KC_to_KCCK( naive_filter,    (float*)naive_filter_opt       , K, M, bk, bm );
  }
   
  // Setup TPP kernels
  auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tc_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tr_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto dtype      = (sizeof(DType) == 2) ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;
  auto l_shape = libxsmm_create_gemm_shape( bk, bn, bm, bk, bm, bk, dtype, dtype, dtype, dtype );
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*bk*sizeof(DType), bm*bn*sizeof(DType), brcount );
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*bn, 1, bk*bn, bk*bn, dtype, dtype, dtype);
  auto zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
  auto tileconfig_kernel  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
  auto tilerelease_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
  if (brcount == Mb) l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  auto brgemm_kernel      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

  libxsmm_meltwfunction_unary wt_trans_kernel;
  auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bm, bk, bm, bk, dtype, dtype, dtype);
  if (dtype == LIBXSMM_DATATYPE_F32) {
    wt_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  } else {
    wt_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }

  // Compute reference if requested
  if (check_correctness) {
    naive_fullyconnected_t naive_param;
    naive_param.N = N;
    naive_param.C = K;
    naive_param.K = M;
    naive_param.fuse_type = 0;
    naive_fullyconnected_fused_bp(&naive_param, naive_input, naive_output, naive_filter, NULL, NULL);
  }

  // JIT requested nested loop specs
  long k_step = 1;
  long m_step = brcount;
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

  auto t0 = getTime();
  auto wt_trans_loop = ThreadedLoop<2>({
      LoopSpecs{0, Mb, 1, true},
      LoopSpecs{0, Kb, 1, true}},
      "AB");

  auto gemm_loop = ThreadedLoop<3>({
      LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // Logical K loop specs
      LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // Logical M loop specs
      LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // Logical N loop specs
      loop_specs_str);
  auto t1 = getTime();
  
  // benchmark the GEMM
  void *filter = (sizeof(DType) == 2) ? (void*)naive_filter_bf16_opt : (void*)naive_filter_opt;
  void *tr_filter = (void*)tr_filter_;
  void *input = (sizeof(DType) == 2) ? (void*)naive_input_bf16_opt : (void*)naive_input_opt;
  void *output = (sizeof(DType) == 2) ? (void*)naive_output_bf16_opt : (void*)naive_output_opt;

  double t_start, t_end;
  for (long it = 0; it < n_iters + 1; it++) {
    if (it == 1) t_start = getTime();

    if (private_wt_trans == 0) {
      wt_trans_loop(
        [&](int* ind) {
          int i_m = ind[0], i_k = ind[1];
          libxsmm_meltw_unary_param trans_param;
          trans_param.in.primary  = LIBXSMM_ACCESS_RAW(4, sizeof(DType),    filter, i_m, i_k, 0, 0, Kb, bk, bm);
          trans_param.out.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), tr_filter, i_k, i_m, 0, 0, Mb, bm, bk);
          wt_trans_kernel(&trans_param);
        },
        [&]() {},
        [&]() {});
    } 

    gemm_loop(
      [&](int* ind) {
        int i_k = ind[0], i_m = ind[1], i_n = ind[2];
        libxsmm_gemm_param gemm_param;
        gemm_param.op.tertiary = (void*)&brcount;
        gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), tr_filter, i_k, i_m, 0, 0, Mb, bm, bk);
        gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output, i_n, i_m, 0, 0, Mb, bn, bm);
        gemm_param.c.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input, i_n, i_k, 0, 0, Kb, bn, bk);
        
        if (private_wt_trans > 0) {
          int tid = omp_get_thread_num();
          char is_transposed = trans_tracker[tid*trans_tracker_size+i_k];
          if (is_transposed == 0) {
            int _i_m = 0;
            trans_tracker[tid*trans_tracker_size+i_k] = 1;
            for (_i_m = 0; _i_m < Mb; _i_m++) {
              libxsmm_meltw_unary_param trans_param;
              trans_param.in.primary  = LIBXSMM_ACCESS_RAW(4, sizeof(DType),    filter, _i_m, i_k, 0, 0, Kb, bk, bm);
              trans_param.out.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), (char*)tr_filter_prv_ + tid * sizeof(DType) * M * K, i_k, _i_m, 0, 0, Mb, bm, bk);
              wt_trans_kernel(&trans_param);
            }
          }
          gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), (char*)tr_filter_prv_ + tid * sizeof(DType) * M * K, i_k, i_m, 0, 0, Mb, bm, bk);     
        }

        if ((i_m == 0) && (brcount != Mb)) {
          libxsmm_meltw_unary_param zero_param;
          zero_param.out.primary = (void*)gemm_param.c.primary;
          zero_kernel( &zero_param );
        }

        brgemm_kernel( &gemm_param );
      },
      [&]() {
        if (sizeof(DType) == 2) tileconfig_kernel(NULL);
        if (private_wt_trans > 0) {
          int tid = omp_get_thread_num();
          memset((char*)trans_tracker + tid * trans_tracker_size, 0, trans_tracker_size*sizeof(char));
        }
      },
      [&]() {if (sizeof(DType) == 2) tilerelease_kernel(NULL);});

    if (it == n_iters) t_end = getTime();
  }
 
  // Check correctness if requested
  printf("##########################################\n");
  printf("#  GEMM-BWD %d x %d x %d  (M x N x K)     \n", M, N, K);
  printf("##########################################\n");
  if (private_wt_trans > 0) {
    printf("Using private filter transposes...\n");
  } else {
    printf("Using upfront filter transposes...\n");
  }

  if (check_correctness) {
    if (sizeof(DType) == 2) {
      matrix_copy_NCNC_to_NC_bf16( (libxsmm_bfloat16*)input, (libxsmm_bfloat16*)naive_input_bf16, 1, N, K, bn, bk );
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_input_bf16, naive_input_check, N*K );
    } else {
      matrix_copy_NCNC_to_NC( (float*)input, naive_input_check, 1, N, K, bn, bk );
    }
    printf("##########################################\n");
    printf("#           Correctness                  #\n");
    printf("##########################################\n");

    libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*K, 1, naive_input, naive_input_check, 0, 0);
    
    printf("L1 reference  : %.25g\n", norms.l1_ref);
    printf("L1 test       : %.25g\n", norms.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms.l2_rel);
    printf("Linf abs.error: %.24f\n", norms.linf_abs);
    printf("Linf rel.error: %.24f\n", norms.linf_rel);
    printf("Check-norm    : %.24f\n", norms.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms);
  }

  // Print performance numbers
  double gflop = (2.0*(double)M*(double)N*(double)K) / (1000*1000*1000);
  printf("Time is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  printf("MEASURE %.5g %s_%d_%d_%d_%d_%d_%d_bf%d_threads%d_private_wt_trans%d\n", gflop/((t_end-t_start)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, mbf, omp_get_max_threads(), private_wt_trans);

  // Free buffers
  libxsmm_free(tr_filter_);
  libxsmm_free(tr_filter_prv_);
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_check);
  libxsmm_free(naive_output);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_output_opt);
  libxsmm_free(naive_input_opt);
  libxsmm_free(naive_filter_opt);
  libxsmm_free(naive_input_bf16);
  libxsmm_free(naive_output_bf16);
  libxsmm_free(naive_filter_bf16);
  libxsmm_free(naive_input_bf16_opt);
  libxsmm_free(naive_output_bf16_opt);
  libxsmm_free(naive_filter_bf16_opt);
  libxsmm_free(trans_tracker);

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

