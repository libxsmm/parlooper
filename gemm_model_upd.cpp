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
  long nbf = 1;
  long private_trans = 0;
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
      nbf = atoi(argv[8]);
    }
    if (argc > 9) {
      private_trans = atoi(argv[9]);
    }
    if (argc > 10) {
      n_iters = atoi(argv[10]);
    }
  }

  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  long brcount = Nb/nbf;
  while (Nb % nbf != 0) {
    nbf--;
  }
  brcount = Nb/nbf;

  // Allocate buffers
  float *naive_input  = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), 2097152);
  float *naive_output = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 2097152);
  float *naive_filter = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 2097152);
  float *naive_filter_check = (float*)libxsmm_aligned_malloc( K*M*sizeof(float), 2097152);
  float *naive_input_opt = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), 2097152);
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 2097152);
  float *naive_filter_opt = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 2097152);

  DType *naive_input_bf16  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), 2097152);
  DType *naive_output_bf16 = (DType*)libxsmm_aligned_malloc( M*N*sizeof(DType), 2097152);
  DType *naive_filter_bf16 = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), 2097152);
  DType *naive_input_bf16_opt  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), 2097152);
  DType *naive_output_bf16_opt = (DType*)libxsmm_aligned_malloc( M*N*sizeof(DType), 2097152);
  DType *naive_filter_bf16_opt = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), 2097152);

  DType *tr_input = (DType*)libxsmm_aligned_malloc( N*K*sizeof(DType), 2097152);
  DType *tr_input_prv_ = (DType*)libxsmm_aligned_malloc( n_threads*N*K*sizeof(DType), 2097152);
  DType *tr_output = (DType*)libxsmm_aligned_malloc( N*M*sizeof(DType), 2097152);
  DType *tr_output_prv_ = (DType*)libxsmm_aligned_malloc( n_threads*N*M*sizeof(DType), 2097152);
  DType *filter_prv_ = (DType*)libxsmm_aligned_malloc( n_threads*K*M*sizeof(DType), 2097152);

  long inp_trans_tracker_size = Kb + 64 - 64%Kb;
  char *inp_trans_tracker = (char*)libxsmm_aligned_malloc( n_threads*inp_trans_tracker_size*sizeof(char), 2097152);
  long  out_trans_tracker_size = Mb + 64 - 64%Mb;
  char *out_trans_tracker = (char*)libxsmm_aligned_malloc( n_threads*out_trans_tracker_size*sizeof(char), 2097152);

  libxsmm_matdiff_info norms, diff;
  libxsmm_matdiff_clear(&norms);
  libxsmm_matdiff_clear(&diff);

  // Init buffers
  init_buf( (float*)inp_trans_tracker,   (n_threads*inp_trans_tracker_size)/4, 0, 0 );
  init_buf( (float*)out_trans_tracker,   (n_threads*out_trans_tracker_size)/4, 0, 0 );
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
  auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto l_tc_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tr_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto dtype      = (sizeof(DType) == 2) ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;
  auto l_shape = (sizeof(DType) == 2) ? libxsmm_create_gemm_shape( bm, bk, bn, bm, bn, bm, dtype, dtype, dtype, dtype ) 
                                      : libxsmm_create_gemm_shape( bm, bk, bn, bm, bk, bm, dtype, dtype, dtype, dtype );
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  auto l_brconfig = (sizeof(DType) == 2) ? libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*bn*sizeof(DType), bk*bn*sizeof(DType), brcount )
                                         : libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, M*bn*sizeof(DType), K*bn*sizeof(DType), brcount );
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*bm, 1, bk*bm, bk*bm, dtype, dtype, dtype);
  auto zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
  auto tileconfig_kernel  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
  auto tilerelease_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
  if (brcount == Nb) {
    l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
    if (sizeof(DType) == 2) {
      l_flags |= LIBXSMM_GEMM_FLAG_VNNI_C;  
    }
  }
  auto brgemm_kernel      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

  libxsmm_meltwfunction_unary inp_trans_kernel, out_trans_kernel, wt_vnni_kernel, wt_copy_kernel;
  if (dtype == LIBXSMM_DATATYPE_BF16) {
    auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bn, bk, bn, dtype, dtype, dtype);
    inp_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    tr_unary_shape = libxsmm_create_meltw_unary_shape(bm, bn, bm, bm, dtype, dtype, dtype);
    out_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    tr_unary_shape = libxsmm_create_meltw_unary_shape(bm, bk, bm, bm, dtype, dtype, dtype);
    wt_vnni_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    wt_copy_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }

  // Compute reference if requested
  if (check_correctness) {
    naive_fullyconnected_t naive_param;
    naive_param.N = N;
    naive_param.C = K;
    naive_param.K = M;
    naive_param.fuse_type = 0;
    naive_fullyconnected_wu(&naive_param, naive_input, naive_output, naive_filter);
  }

  // JIT requested nested loop specs
  long k_step = 1;
  long m_step = 1;
  long n_step = brcount;
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
  auto inp_trans_loop = ThreadedLoop<2>({
      LoopSpecs{0, Nb, 1, true},
      LoopSpecs{0, Kb, 1, true}},
      "AB");

  auto out_trans_loop = ThreadedLoop<2>({
      LoopSpecs{0, Nb, 1, true},
      LoopSpecs{0, Mb, 1, true}},
      "AB");

  auto gemm_loop = ThreadedLoop<3>({
      LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // Logical K loop specs
      LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // Logical M loop specs
      LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // Logical N loop specs
      loop_specs_str);
  auto t1 = getTime();
  
  // benchmark the GEMM
  void *filter = (sizeof(DType) == 2) ? (void*)naive_filter_bf16_opt : (void*)naive_filter_opt;
  void *input = (sizeof(DType) == 2) ? (void*)naive_input_bf16_opt : (void*)naive_input_opt;
  void *output = (sizeof(DType) == 2) ? (void*)naive_output_bf16_opt : (void*)naive_output_opt;

  double t_start, t_end;
  // FP32 algorithms
  if (sizeof(DType) == 4){
    for (long it = 0; it < n_iters + 1; it++) {
      if (it == 1) t_start = getTime();

      gemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;
          gemm_param.op.tertiary = (void*)&brcount;
          gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output, i_n, i_m, 0, 0, Mb, bn, bm);
          gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input, i_n, i_k, 0, 0, Kb, bn, bk);
          gemm_param.c.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), filter, i_m, i_k, 0, 0, Kb, bk, bm);
          
          if ((i_n == 0) && (brcount != Nb)) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)gemm_param.c.primary;
            zero_kernel( &zero_param );
          }

          brgemm_kernel( &gemm_param );
        },
        [&]() {},
        [&]() {});

      if (it == n_iters) t_end = getTime();
    }
  }

  //BF16 algorithms
  if (sizeof(DType) == 2){
    for (long it = 0; it < n_iters + 1; it++) {
      if (it == 1) t_start = getTime();

      if (private_trans == 0) {
        inp_trans_loop(
          [&](int* ind) {
            int i_n = ind[0], i_k = ind[1];
            libxsmm_meltw_unary_param trans_param;
            trans_param.in.primary  = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input, i_n, i_k, 0, 0, Kb, bn, bk);
            trans_param.out.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), tr_input, i_k, i_n, 0, 0, Nb, bk, bn);
            inp_trans_kernel(&trans_param);
          },
          [&]() {},
          [&]() {});

        out_trans_loop(
          [&](int* ind) {
            int i_n = ind[0], i_m = ind[1];
            libxsmm_meltw_unary_param trans_param;
            trans_param.in.primary  = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output, i_n, i_m, 0, 0, Mb, bn, bm);
            trans_param.out.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), tr_output, i_m, i_n, 0, 0, Nb, bn, bm);
            out_trans_kernel(&trans_param);
          },
          [&]() {},
          [&]() {});
      } 

      gemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;
          gemm_param.op.tertiary = (void*)&brcount;
          gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), tr_output, i_m, i_n, 0, 0, Nb, bn, bm);
          gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), tr_input, i_k, i_n, 0, 0, Nb, bk, bn);
          gemm_param.c.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), filter, i_m, i_k, 0, 0, Kb, bk, bm);
          
          if (private_trans > 0) {
            int tid = omp_get_thread_num();
            char is_inp_transposed = inp_trans_tracker[tid*inp_trans_tracker_size+i_k];
            char is_out_transposed = out_trans_tracker[tid*out_trans_tracker_size+i_m];
            if (is_out_transposed == 0) {
              int _i_n = 0;
              out_trans_tracker[tid*out_trans_tracker_size+i_m] = 1;
              for (_i_n = 0; _i_n < Nb; _i_n++) {
                libxsmm_meltw_unary_param trans_param;
                trans_param.in.primary  = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output, _i_n, i_m, 0, 0, Mb, bn, bm);
                trans_param.out.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), (char*)tr_output_prv_ + tid * sizeof(DType) * N * M, i_m, _i_n, 0, 0, Nb, bn, bm);
                out_trans_kernel(&trans_param);
              }
            }
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), (char*)tr_output_prv_ + tid * sizeof(DType) * N * M, i_m, i_n, 0, 0, Nb, bn, bm);     

            if (is_inp_transposed == 0) {
              int _i_n = 0;
              inp_trans_tracker[tid*inp_trans_tracker_size+i_k] = 1;
              for (_i_n = 0; _i_n < Nb; _i_n++) {
                libxsmm_meltw_unary_param trans_param;
                trans_param.in.primary  = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input, _i_n, i_k, 0, 0, Kb, bn, bk);
                trans_param.out.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), (char*)tr_input_prv_ + tid * sizeof(DType) * N * K, i_k, _i_n, 0, 0, Nb, bk, bn);
                inp_trans_kernel(&trans_param);
              }
            }
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), (char*)tr_input_prv_ + tid * sizeof(DType) * N * K, i_k, i_n, 0, 0, Nb, bk, bn);     
          }
          
          if ((i_n == 0) && (brcount != Nb)) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)gemm_param.c.primary;
            zero_kernel( &zero_param );
          }

          brgemm_kernel( &gemm_param );

          if ((i_n + brcount >= Nb) && (brcount != Nb)) {
            libxsmm_bfloat16 tmp[bk*bm];
            libxsmm_meltw_unary_param trans_param;
            trans_param.in.primary  = LIBXSMM_ACCESS_RAW(4, sizeof(DType), filter, i_m, i_k, 0, 0, Kb, bk, bm);
            trans_param.out.primary = tmp;
            wt_copy_kernel(&trans_param);
            trans_param.in.primary = tmp;
            trans_param.out.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), filter, i_m, i_k, 0, 0, Kb, bk, bm);
            wt_vnni_kernel(&trans_param);
          }
        },
        [&]() {
          if (sizeof(DType) == 2) tileconfig_kernel(NULL);
          if (private_trans > 0) {
            int tid = omp_get_thread_num();
            memset((char*)inp_trans_tracker + tid * inp_trans_tracker_size, 0, inp_trans_tracker_size*sizeof(char));
            memset((char*)out_trans_tracker + tid * out_trans_tracker_size, 0, out_trans_tracker_size*sizeof(char));          
          }
        },
        [&]() {if (sizeof(DType) == 2) tilerelease_kernel(NULL);});

      if (it == n_iters) t_end = getTime();
    }
  }
  // Check correctness if requested
  printf("##########################################\n");
  printf("#  GEMM-UPD %d x %d x %d  (M x N x K)     \n", M, N, K);
  printf("##########################################\n");
  if (sizeof(DType) == 2) {
    if (private_trans > 0) {
      printf("Using private act transposes...\n"); 
    } else { 
      printf("Using upfront act transposes...\n"); 
    } 
  }
  if (check_correctness) {
    if (sizeof(DType) == 2) {
      matrix_copy_KCCK_to_KC_bf16( (libxsmm_bfloat16*)filter, (libxsmm_bfloat16*)naive_filter_bf16, K, M, bk, bm );
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_filter_bf16, naive_filter_check, K*M );
    } else {
      matrix_copy_KCCK_to_KC( (float*)filter, naive_filter_check, K, M, bk, bm );
    }
    printf("##########################################\n");
    printf("#           Correctness                  #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, M*K, 1, naive_filter, naive_filter_check, 0, 0);
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
  printf("MEASURE %.5g %s_%d_%d_%d_%d_%d_%d_bf%d_threads%d_private_act_trans%d\n", gflop/((t_end-t_start)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, nbf, omp_get_max_threads(), private_trans); 
  // Free buffers
  libxsmm_free(tr_input); 
  libxsmm_free(tr_input_prv_);
  libxsmm_free(tr_output);
  libxsmm_free(tr_output_prv_);
  libxsmm_free(naive_input);
  libxsmm_free(naive_filter_check); 
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
  libxsmm_free(inp_trans_tracker);
  libxsmm_free(out_trans_tracker);
  libxsmm_free(filter_prv_); 
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

