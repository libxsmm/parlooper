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
  long flat_weight_layout = 0;
  long trans_a = 0;
  long trans_b = 0;
  long use_sf_curve = 0;
  long unit_step = 1;
  char gemm_config[256] = "VN";
  long upfront_xforms = 0, xform_A_upfront = 0, xform_B_upfront = 0;

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
    if (argc > 11) {
      if (strcmp(argv[11], "FLAT_A") == 0) {
        flat_weight_layout = 1;
      }
    } 
    if (argc > 12) {
      if (strcmp(argv[12], "TRA") == 0) {
        trans_a = 1;
      }
    } 
    if (argc > 13) {
      if (strcmp(argv[13], "TRB") == 0) {
        trans_b = 1;
      }
    } 
    if (argc > 14) {
      check_correctness = atoi(argv[14]);
    }
    if (argc > 15) {
      if (strcmp(argv[15], "UPFRONT_XFORM") == 0) {
        upfront_xforms = 1;
      }    
    }
  }

  if (strcmp(argv[1], "SFC") == 0) {
    use_sf_curve = 1;
  }
  
  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  long  brcount = Kb/kbf;
  while (Kb % kbf != 0) {
    kbf--;
  }
  brcount = Kb/kbf;

  // Allocate buffers
  DType *scratch_A = NULL;
  DType *scratch_B = NULL;
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
    matrix_copy_KC_to_KCCK_bf16_local( (libxsmm_bfloat16*)naive_filter_bf16, (libxsmm_bfloat16*)WGT[i], K, M, bk, bm, flat_weight_layout, trans_a );
    matrix_copy_NC_to_NCNC_bf16_local( (libxsmm_bfloat16*)naive_input_bf16, (libxsmm_bfloat16*)ACT[i] , N, (n_layers == 1) ? K : LIBXSMM_MAX(K,M), bn, bk, trans_b );
  }
  matrix_copy_NC_to_NCNC_bf16_local( (libxsmm_bfloat16*)naive_output, (libxsmm_bfloat16*)ACT[n_layers], N, (n_layers == 1) ? M : LIBXSMM_MAX(K,M), bn, bm, 0 );
  
  // Setup TPP kernels
  auto l_flags    = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ;
  auto l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  auto l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  
  if (upfront_xforms == 0) {
    if (flat_weight_layout > 0 && trans_a == 0 && trans_b == 0) {
      l_flags    = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'N', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ;
      l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'N', 'N');
      l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'N', 'N');
      strcpy(gemm_config, "NN");  
    } else if (flat_weight_layout > 0 && trans_a > 0 && trans_b == 0) {
      l_flags    = LIBXSMM_GEMM_VNNI_FLAGS('T', 'N', 'N', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ;
      l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('T', 'N', 'N', 'N');
      l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('T', 'N', 'N', 'N');
      strcpy(gemm_config, "TN");   
    } else if (flat_weight_layout > 0 && trans_a == 0 && trans_b > 0) {
      l_flags    = LIBXSMM_GEMM_VNNI_FLAGS('N', 'T', 'N', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ;
      l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'T', 'N', 'N');
      l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'T', 'N', 'N');
      strcpy(gemm_config, "NT"); 
    } else if (flat_weight_layout == 0 && trans_a == 0 && trans_b == 0) {
      l_flags    = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ;
      l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
      l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
      strcpy(gemm_config, "VN");
    } else {
      printf("INVALID GEMM CONFIGURATION. EXITING...\n");
      return 0;
    }
  }

  auto dtype = LIBXSMM_DATATYPE_BF16;
  auto a_xform_loop = ThreadedLoop<2>({ LoopSpecs{0, Mb, 1, true}, LoopSpecs{0, Kb, 1, true}}, "AB");
  auto b_xform_loop = ThreadedLoop<2>({ LoopSpecs{0, Nb, 1, true}, LoopSpecs{0, Kb, 1, true}}, "AB");
  libxsmm_meltwfunction_unary a_xform_kernel, b_xform_kernel;
  if (upfront_xforms > 0) {
    if (flat_weight_layout > 0 && trans_a == 0 && trans_b == 0) {
      auto xform_unary_shape = libxsmm_create_meltw_unary_shape(bm, bk, bm, bm, dtype, dtype, dtype);
      a_xform_kernel = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, xform_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ); 
      xform_A_upfront = 1;
      strcpy(gemm_config, "NN");
    } else if (flat_weight_layout > 0 && trans_a > 0 && trans_b == 0) {
      auto xform_unary_shape = libxsmm_create_meltw_unary_shape(bk/2, bm, bk/2, bm, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
      a_xform_kernel = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, xform_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ); 
      xform_A_upfront = 1;
      strcpy(gemm_config, "TN");   
    } else if (flat_weight_layout > 0 && trans_a == 0 && trans_b > 0) {
      auto xform_unary_shape = libxsmm_create_meltw_unary_shape(bm, bk, bm, bm, dtype, dtype, dtype);
      a_xform_kernel = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, xform_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ); 
      xform_A_upfront = 1;
      xform_unary_shape = libxsmm_create_meltw_unary_shape(bn, bk, bn, bk, dtype, dtype, dtype);
      b_xform_kernel = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, xform_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ); 
      xform_B_upfront = 1;
      strcpy(gemm_config, "NT"); 
    } else if (flat_weight_layout == 0 && trans_a == 0 && trans_b == 0) {
      printf("INVALID GEMM CONFIGURATION. EXITING...\n");
      return 0;
    } else {
      printf("INVALID GEMM CONFIGURATION. EXITING...\n");
      return 0;
    }
  }

  if (xform_A_upfront > 0) {
    scratch_A = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), 64);
    check_null_ptr(scratch_A, "scratch A array");
  }
  if (xform_B_upfront > 0) {
    scratch_B  = (DType*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(DType), 64);
    check_null_ptr(scratch_B, "scratch B array");
  }

  auto l_shape = libxsmm_create_gemm_shape( bm, bn, bk, (trans_a > 0 && upfront_xforms == 0) ? bk : bm, (trans_b > 0 && upfront_xforms == 0) ? bn : bk, bm, dtype, dtype, dtype, LIBXSMM_DATATYPE_F32 );
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*bk*sizeof(DType), bk*bn*sizeof(DType), brcount );
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bm*bn, 1, bm*bn, bm*bn, dtype, dtype, dtype);

  if (brcount == Kb) l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

  auto zero_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
  auto tileconfig_kernel  = libxsmm_dispatch_tilecfg_gemm( l_shape, l_tc_flags );
  auto tilerelease_kernel = libxsmm_dispatch_tilecfg_gemm( l_shape, l_tr_flags );
  auto brgemm_kernel      = libxsmm_dispatch_brgemm( l_shape, l_flags, l_prefetch_flags, l_brconfig );

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

  auto gemm_loop = (use_sf_curve == 0) ?
      ThreadedLoop<3>({
      LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // Logical K loop specs
      LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // Logical M loop specs
      LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // Logical N loop specs
      loop_specs_str) :
      ThreadedLoop<3>({
      LoopSpecs{0, Kb, k_step, {}},             // Logical K loop
      LoopSpecs{0, Mb*Nb, unit_step,{}},        // Logical MxN loop over the SF curve index space
      LoopSpecs{0, unit_step, unit_step, {}}},  // Degenerate loop, just to match types with gemm_loop of 3 nested loops
      "aB");

  unsigned char *sf_curve_index_map = NULL;
  unsigned int index_tsize = 4;
  if (use_sf_curve > 0) {
    index_tsize = fill_sf_curve_index_map(&sf_curve_index_map, Mb, Nb);
  }

  // Warmup iteration for i-caches
  for (i = 0; i < n_layers; i++) {
    if (upfront_xforms > 0) {
      if (xform_A_upfront > 0) {
        a_xform_loop(
          [&](int* ind) {
            int i_m = ind[0], i_k = ind[1];
            libxsmm_meltw_unary_param xform_param;
            xform_param.in.primary  = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
            xform_param.out.primary = (void*)((DType*)scratch_A + i_m * K * bm + i_k * bk * bm );
            a_xform_kernel(&xform_param);
          },
          [&]() {},
          [&]() {});   
      }
      if (xform_B_upfront > 0) {
        b_xform_loop(
          [&](int* ind) {
            int i_n = ind[0], i_k = ind[1];
            libxsmm_meltw_unary_param xform_param;
            xform_param.in.primary  = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
            xform_param.out.primary = (void*)((DType*)scratch_B + i_n * K * bn + i_k * bk * bn);
            b_xform_kernel(&xform_param);
          },
          [&]() {},
          [&]() {});   
      }
    }
    gemm_loop(
      [&](int* ind) {
        int i_k = ind[0], i_m, i_n;
        if (use_sf_curve > 0) {
          extract_indices_from_sf_curve(&i_m, &i_n, sf_curve_index_map, ind[1] /* This is the index in the SF curve*/, index_tsize);
        } else {
          i_m = ind[1];
          i_n = ind[2];
        }
        libxsmm_gemm_param gemm_param;
        gemm_param.op.tertiary = (void*)&brcount;
        if (xform_A_upfront > 0) {
          gemm_param.a.primary = (void*)((DType*)scratch_A + i_m * K * bm + i_k * bk * bm );   
        } else {
          gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
        }
        if (xform_B_upfront > 0) {
          gemm_param.b.primary = (void*)((DType*)scratch_B + i_n * K * bn + i_k * bk * bn );      
        } else {
          gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
        }
        gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
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
    matrix_copy_NCNC_to_NC_bf16( (libxsmm_bfloat16*)ACT[n_layers], (libxsmm_bfloat16*)naive_output_bf16, 1, N, M, bn, bm );
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
      if (upfront_xforms > 0) {
        if (xform_A_upfront > 0) {
          a_xform_loop(
            [&](int* ind) {
              int i_m = ind[0], i_k = ind[1];
              libxsmm_meltw_unary_param xform_param;
              xform_param.in.primary  = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
              xform_param.out.primary = (void*)((DType*)scratch_A + i_m * K * bm + i_k * bk * bm );
              a_xform_kernel(&xform_param);
            },
            [&]() {},
            [&]() {});   
        }
        if (xform_B_upfront > 0) {
          b_xform_loop(
            [&](int* ind) {
              int i_n = ind[0], i_k = ind[1];
              libxsmm_meltw_unary_param xform_param;
              xform_param.in.primary  = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
              xform_param.out.primary = (void*)((DType*)scratch_B + i_n * K * bn + i_k * bk * bn);
              b_xform_kernel(&xform_param);
            },
            [&]() {},
            [&]() {});   
        }
      }
      gemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m, i_n;
          if (use_sf_curve > 0) {
            extract_indices_from_sf_curve(&i_m, &i_n, sf_curve_index_map, ind[1] /* This is the index in the SF curve*/, index_tsize);
          } else {
            i_m = ind[1];
            i_n = ind[2];
          }
          libxsmm_gemm_param gemm_param;
          gemm_param.op.tertiary = (void*)&brcount;
          if (xform_A_upfront > 0) {
            gemm_param.a.primary = (void*)((DType*)scratch_A + i_m * K * bm + i_k * bk * bm );   
          } else {
            gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
          }
          if (xform_B_upfront > 0) {
            gemm_param.b.primary = (void*)((DType*)scratch_B + i_n * K * bn + i_k * bk * bn );      
          } else {
            gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
          }
          gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
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
  }
  auto t_end = getTime();
  
  // Print performance/model numbers
  double gflop = (2.0*(double)n_layers*(double)M*(double)N*(double)K) / (1000*1000*1000);
  printf("Time is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  printf("Effective model sizes: %.5g GB\n", ((double)sizeof(DType)*(double)n_layers*(double)M*(double)K)/(1024.0*1024.0*1024.0));
  printf("Effective A BW is %.5g GB/s\n", (((double)sizeof(DType)*(double)n_layers*(double)M*(double)K) / (1024.0*1024.0*1024.0))/((t_end-t_start)/(1.0*n_iters)));
  printf("MEASURE %.5g %s_%ld_%ld_%ld_%ld_%ld_%ld_bf%ld_threads%d_config_%s\n", gflop/((t_end-t_start)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads(),gemm_config);

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
  if (sf_curve_index_map != NULL) {
    libxsmm_free(sf_curve_index_map);
  }
  if (scratch_A != NULL) {
    libxsmm_free(scratch_A);
  }
  if (scratch_B != NULL) {
    libxsmm_free(scratch_B);
  }
  free(ACT);
  free(WGT);
  return 0;
}

int main(int argc, char** argv) {
  return gemm_benchmark<libxsmm_bfloat16>(argc, argv);  
}

