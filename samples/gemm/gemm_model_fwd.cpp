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

#define ALIGNMENT_SIZE 64
//#define USE_EQN_REDUCE
#define BENCH_REDUCE

template<typename DType>
void run_gemm(long n_layers, long M, long N, long K,
              long Mb, long Nb, long Kb, long bm, long bn, long bk, long split_K_factor, long brcount_in, long upfront_xforms,
              DType **WGT, DType **ACT, DType *scratch_A, DType *scratch_B, DType **output_partial,
              ThreadedLoop<3> gemm_loop, libxsmm_gemmfunction brgemm_kernel, libxsmm_meltwfunction_unary zero_kernel, libxsmm_tilecfgfunction tileconfig_kernel, libxsmm_tilecfgfunction tilerelease_kernel, long use_sf_curve, unsigned char *sf_curve_index_map, unsigned int index_tsize,
              ThreadedLoop<2> reduce_output_loop, libxsmm_meltwfunction_binary l_add_kernel, libxsmm_meqn_function reduce_func, libxsmm_meltwfunction_unary l_reduce_kernel, long skip_reduce,
              long xform_A_upfront, ThreadedLoop<2> a_xform_loop, libxsmm_meltwfunction_unary a_xform_kernel,
              long xform_B_upfront, ThreadedLoop<2> b_xform_loop, libxsmm_meltwfunction_unary b_xform_kernel) {
  long brcount = brcount_in;
  for (int i = 0; i < n_layers; i++) {
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
            xform_param.in.primary  = (void*)((DType*)ACT[2*i] + i_n * K * bn + i_k * bk * bn );
            xform_param.out.primary = (void*)((DType*)scratch_B + i_n * K * bn + i_k * bk * bn);
            b_xform_kernel(&xform_param);
          },
          [&]() {},
          [&]() {});   
      }
    }
    gemm_loop(
      [&](int* ind) {
        int i_k = ind[0], i_m, i_n, i_k_split;
        if (use_sf_curve > 0) {
          extract_indices_from_sf_curve(&i_m, &i_n, sf_curve_index_map, ind[1]%(Mb*Nb) /* This is the index in the SF curve*/, index_tsize);
          i_k_split = ind[1]/(Mb*Nb);  
        } else {
          i_m = ind[1];
          i_n = ind[2];
        }
        libxsmm_gemm_param gemm_param;
        gemm_param.op.tertiary = (void*)&brcount;
        if (xform_A_upfront > 0) {
          gemm_param.a.primary = (void*)((DType*)scratch_A + i_m * K * bm + i_k * bk * bm + i_k_split * (K/split_K_factor) * bm );   
        } else {
          gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm + i_k_split * (K/split_K_factor) * bm );
        }
        if (xform_B_upfront > 0) {
          gemm_param.b.primary = (void*)((DType*)scratch_B + i_n * K * bn + i_k * bk * bn + i_k_split * (K/split_K_factor) * bn  );      
        } else {
          gemm_param.b.primary = (void*)((DType*)ACT[2*i] + i_n * K * bn + i_k * bk * bn + i_k_split * (K/split_K_factor) * bn );
        }
        if (i_k_split > 0) {
          gemm_param.c.primary = (void*)((DType*)output_partial[i_k_split-1] + i_n * M * bn + i_m * bn * bm );    
        } else {
          gemm_param.c.primary = (void*)((DType*)ACT[2*i+1] + i_n * M * bn + i_m * bn * bm );
        }
        if ((i_k == 0) && (brcount != (Kb/split_K_factor))) {
          libxsmm_meltw_unary_param zero_param;
          zero_param.out.primary = (void*)gemm_param.c.primary;
          zero_kernel( &zero_param );
        }
        brgemm_kernel( &gemm_param );
      },
      [&]() {tileconfig_kernel(NULL);},
      [&]() {tilerelease_kernel(NULL);});
    
    if ((split_K_factor > 1) && (skip_reduce == 0)) {
      reduce_output_loop(
        [&](int* ind) {
          int i_m = ind[0], i_n = ind[1];
          if (split_K_factor == 2) {
            libxsmm_meltw_binary_param add_param;
            add_param.in0.primary  = (void*)((DType*)output_partial[0] + i_n * M * bn + i_m * bn * bm );
            add_param.in1.primary  = (void*)((DType*)ACT[2*i+1] + i_n * M * bn + i_m * bn * bm );       
            add_param.out.primary = (void*)((DType*)ACT[2*i+1] + i_n * M * bn + i_m * bn * bm );
            l_add_kernel(&add_param);
          } else {
#ifdef USE_EQN_REDUCE
            libxsmm_meqn_param eqn_param;
            libxsmm_matrix_arg  arg_array[2];
            arg_array[0].primary = (void*)((DType*)output_partial[0] + i_n * M * bn + i_m * bn * bm );
            arg_array[1].primary = (void*)((DType*)ACT[2*i+1] + i_n * M * bn + i_m * bn * bm );
            eqn_param.inputs = arg_array;
            eqn_param.output.primary = (void*)((DType*)ACT[2*i+1] + i_n * M * bn + i_m * bn * bm );
            reduce_func(&eqn_param);
#else
            libxsmm_meltw_binary_param add_param;
            libxsmm_meltw_unary_param reduce_param;
            DType reduce_scratch[bm*bn];
            reduce_param.in.primary = (void*)((DType*)output_partial[0] + i_n * M * bn + i_m * bn * bm );
            reduce_param.out.primary  = (void*)reduce_scratch;
            add_param.in0.primary  = (void*)reduce_scratch;
            add_param.in1.primary  = (void*)((DType*)ACT[2*i+1] + i_n * M * bn + i_m * bn * bm );       
            add_param.out.primary = (void*)((DType*)ACT[2*i+1] + i_n * M * bn + i_m * bn * bm );
            l_reduce_kernel(&reduce_param);
            l_add_kernel(&add_param);
#endif
          }
        },
        [&]() {},
        [&]() {});  
    }
  }
  return;
}

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
  long split_K_factor = 1;
  libxsmm_blasint my_eqn0;
  libxsmm_meqn_arg_metadata arg_metadata;
  libxsmm_meqn_op_metadata  op_metadata;
  libxsmm_meqn_arg_shape          arg_shape_in, arg_shape_out;
  libxsmm_matrix_arg_attributes   arg_singular_attr = libxsmm_create_matrix_arg_attributes( LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);
  libxsmm_meqn_function reduce_func;

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
    if (argc > 16) {
      split_K_factor = atoi(argv[16]);
    }
  }

  if (strcmp(argv[1], "SFC") == 0) {
    use_sf_curve = 1;
  }
  
  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  long  brcount = (Kb/split_K_factor)/kbf;
  while ((Kb/split_K_factor) % kbf != 0) {
    kbf--;
  }
  brcount = (Kb/split_K_factor)/kbf;

  // Allocate buffers
  DType *scratch_A = NULL;
  DType *scratch_B = NULL;
  DType **ACT = (DType**) malloc((2*n_layers)*sizeof(DType*));
  check_null_ptr(ACT, "ACT array");
  DType **WGT = (DType**) malloc(n_layers    *sizeof(DType*));
  check_null_ptr(WGT, "WGT array");
  for (i = 0; i < n_layers; i++) {
    WGT[i] = (DType*) libxsmm_aligned_malloc(M*K*sizeof(DType), ALIGNMENT_SIZE);
  }
  for (i = 0; i < 2*n_layers; i++) {
    if (i%2 == 0) {
      ACT[i] = (DType*) libxsmm_aligned_malloc(K*N*sizeof(DType), ALIGNMENT_SIZE);
    } else {
      ACT[i] = (DType*) libxsmm_aligned_malloc(M*N*sizeof(DType), ALIGNMENT_SIZE);
    }
    check_null_ptr(ACT[i], "ACT[i] array"); 
  }

  float *naive_input  = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_input, "naive_input array");
  float *naive_output = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_output, "naive_output array");
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_output_opt, "naive_output_opt array");
  float *naive_filter = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), ALIGNMENT_SIZE);
  check_null_ptr(naive_filter, "naive_filter array");
  DType *naive_input_lp  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), ALIGNMENT_SIZE);
  check_null_ptr(naive_input_lp, "naive_input_lp array");
  DType *naive_output_lp = (DType*)libxsmm_aligned_malloc( M*N*sizeof(DType), ALIGNMENT_SIZE);
  check_null_ptr(naive_output_lp, "naive_output_lp array");
  DType *naive_filter_lp = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), ALIGNMENT_SIZE);
  check_null_ptr(naive_filter_lp, "naive_filter_lp array");
  
  // Init buffers
  init_buf( naive_input,     K*N, 0, 0 );
  init_buf( naive_output,    M*N, 0, 0 );
  init_buf( naive_filter,    M*K, 0, 0 );

  int n_out_copies = LIBXSMM_MAX(1, split_K_factor-1);
  DType *output_partial[n_out_copies];
  DType *global_scratch = NULL;
  for (i = 1; i < split_K_factor; i++) {
    if (i == 1) {
      global_scratch = (DType*)libxsmm_aligned_malloc( M*N*sizeof(DType)*(split_K_factor-1), ALIGNMENT_SIZE);
    }
    output_partial[i-1] = (DType*)global_scratch + (i-1) * M*N;
  }

  parlooper_rne_convert_fp32_lp<DType>( naive_input,     (void*)naive_input_lp,     N*K);
  parlooper_convert_lp_f32<DType>( (void*)naive_input_lp, naive_input, N*K);
  parlooper_rne_convert_fp32_lp<DType>( naive_output,    (void*)naive_output_lp,    N*M);
  parlooper_convert_lp_f32<DType>( (void*)naive_output_lp, naive_output, N*M);
  parlooper_rne_convert_fp32_lp<DType>( naive_filter,    (void*)naive_filter_lp,    M*K);
  parlooper_convert_lp_f32<DType>( (void*)naive_filter_lp, naive_filter, M*K);
  for (i = 0; i < n_layers; i++) {
    parlooper_matrix_copy_KC_to_KCCK<DType>( (void*)naive_filter_lp, (void*)WGT[i], K, M, bk, bm, flat_weight_layout, trans_a );
    parlooper_matrix_copy_NC_to_NCNC<DType>( (void*)naive_input_lp, (void*)ACT[2*i] , N, K, bn, bk, trans_b );
    parlooper_matrix_copy_NC_to_NCNC<DType>( (void*)naive_output_lp, (void*)ACT[2*i+1], N, M, bn, bm, 0 );
  }
  
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

  auto dtype = parlooper_get_lixbxsmm_dtype<DType>();
  auto a_xform_loop = ThreadedLoop<2>({ LoopSpecs{0, Mb, 1, true}, LoopSpecs{0, Kb, 1, true}}, "AB");
  auto b_xform_loop = ThreadedLoop<2>({ LoopSpecs{0, Nb, 1, true}, LoopSpecs{0, Kb, 1, true}}, "AB");
  libxsmm_meltwfunction_unary a_xform_kernel, b_xform_kernel;
  if (upfront_xforms > 0) {
    if (flat_weight_layout > 0 && trans_a == 0 && trans_b == 0) {
      auto xform_unary_shape = libxsmm_create_meltw_unary_shape(bm, bk, bm, bm, dtype, dtype, dtype);
      a_xform_kernel = libxsmm_dispatch_meltw_unary( parlooper_get_vnni_xform<DType>(), xform_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ); 
      xform_A_upfront = 1;
      strcpy(gemm_config, "NN");
    } else if (flat_weight_layout > 0 && trans_a > 0 && trans_b == 0) {
      auto xform_unary_shape = libxsmm_create_meltw_unary_shape(bk/2, bm, bk/2, bm, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
      a_xform_kernel = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, xform_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ); 
      xform_A_upfront = 1;
      strcpy(gemm_config, "TN");   
    } else if (flat_weight_layout > 0 && trans_a == 0 && trans_b > 0) {
      auto xform_unary_shape = libxsmm_create_meltw_unary_shape(bm, bk, bm, bm, dtype, dtype, dtype);
      a_xform_kernel = libxsmm_dispatch_meltw_unary( parlooper_get_vnni_xform<DType>(), xform_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ); 
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
    scratch_A = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), ALIGNMENT_SIZE);
    check_null_ptr(scratch_A, "scratch A array");
  }
  if (xform_B_upfront > 0) {
    scratch_B  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), ALIGNMENT_SIZE);
    check_null_ptr(scratch_B, "scratch B array");
  }

  auto l_shape = libxsmm_create_gemm_shape( bm, bn, bk, (trans_a > 0 && upfront_xforms == 0) ? bk : bm, (trans_b > 0 && upfront_xforms == 0) ? bn : bk, bm, dtype, dtype, dtype, LIBXSMM_DATATYPE_F32 );
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*bk*sizeof(DType), bk*bn*sizeof(DType), brcount );
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bm*bn, 1, bm*bn, bm*bn, dtype, dtype, dtype);

  if (brcount == (Kb/split_K_factor)) l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

  auto zero_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
  auto tileconfig_kernel  = libxsmm_dispatch_tilecfg_gemm( l_shape, l_tc_flags );
  auto tilerelease_kernel = libxsmm_dispatch_tilecfg_gemm( l_shape, l_tr_flags );
  auto brgemm_kernel      = libxsmm_dispatch_brgemm( l_shape, l_flags, l_prefetch_flags, l_brconfig );

  auto l_binary_shape = libxsmm_create_meltw_binary_shape(bm, bn, bm, bm, bm, dtype, dtype, dtype, LIBXSMM_DATATYPE_F32);
  auto l_add_kernel = libxsmm_dispatch_meltw_binary( LIBXSMM_MELTW_TYPE_BINARY_ADD, l_binary_shape, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  auto reduce_output_loop = ThreadedLoop<2>({ LoopSpecs{0, Mb, 1, true}, LoopSpecs{0, Nb, 1, true}}, "AB");

  auto l_reduce_shape = libxsmm_create_meltw_unary_shape(bm*bn, n_out_copies, M*N, bm*bn, dtype, dtype, LIBXSMM_DATATYPE_F32);
  auto l_reduce_kernel = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_reduce_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);

  my_eqn0 = libxsmm_meqn_create();
  op_metadata   = libxsmm_create_meqn_op_metadata(my_eqn0, -1);
  libxsmm_meqn_push_back_binary_op(op_metadata, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  libxsmm_meqn_push_back_unary_op(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
  arg_shape_in  = libxsmm_create_meqn_arg_shape( bm*bn, n_out_copies, M*N, dtype );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 0);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_shape_in  = libxsmm_create_meqn_arg_shape( bm*bn, 1, bm*bn, dtype );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 1);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_shape_out = libxsmm_create_meqn_arg_shape( bm*bn, 1, bm*bn, dtype );
  reduce_func = libxsmm_dispatch_meqn( my_eqn0, arg_shape_out );

  // Compute reference if requested
  if (check_correctness) {
    naive_fullyconnected_t naive_param;
    naive_param.N = N;
    naive_param.C = K;
    naive_param.K = M;
    naive_param.fuse_type = 0;
    naive_fullyconnected_fused_fp(&naive_param, naive_input, naive_output, naive_filter, NULL);
    parlooper_rne_convert_fp32_lp<DType>( naive_output,     (void*)naive_output_lp, (N*M) );
    parlooper_convert_lp_f32<DType>( (void*)naive_output_lp, naive_output, N*M);
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
      LoopSpecs{0, Kb/split_K_factor, k_step, {}},             // Logical K loop
      LoopSpecs{0, Mb*Nb*split_K_factor, unit_step,{}},        // Logical MxN loop over the SF curve index space
      LoopSpecs{0, unit_step, unit_step, {}}},  // Degenerate loop, just to match types with gemm_loop of 3 nested loops
      "aB");

  unsigned char *sf_curve_index_map = NULL;
  unsigned int index_tsize = 4;
  if (use_sf_curve > 0) {
    index_tsize = fill_sf_curve_index_map(&sf_curve_index_map, Mb, Nb);
  }

  // Warmup iteration for i-caches
  run_gemm<DType>(n_layers, M, N, K,
      Mb, Nb, Kb, bm, bn, bk, split_K_factor, brcount, upfront_xforms,
      WGT, ACT, scratch_A, scratch_B, output_partial,
      gemm_loop, brgemm_kernel, zero_kernel, tileconfig_kernel, tilerelease_kernel, use_sf_curve, sf_curve_index_map, index_tsize,
      reduce_output_loop, l_add_kernel, reduce_func, l_reduce_kernel, 0,
      xform_A_upfront, a_xform_loop, a_xform_kernel,
      xform_B_upfront, b_xform_loop, b_xform_kernel);

  // Check correctness if requested
  printf("##############################################################\n");
  printf("    %ld Sets of GEMMS with sizes  %ld x %ld x %ld  (M x N x K)  \n", n_layers, M, N, K);
  printf("##############################################################\n");

  if (check_correctness) {
    libxsmm_matdiff_info norms, diff;
    libxsmm_matdiff_clear(&norms);
    libxsmm_matdiff_clear(&diff);
    parlooper_matrix_copy_NCNC_to_NC<DType>( (void*)ACT[2*n_layers-1], (void*)naive_output_lp, N, M, bn, bm );
    parlooper_convert_lp_f32<DType>( (void*)naive_output_lp, naive_output_opt, N*M );
    printf("##########################################\n");
    printf("#           Correctness                  #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*M, 1, naive_output, naive_output_opt, 0, 0);
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
    run_gemm<DType>(n_layers, M, N, K,
        Mb, Nb, Kb, bm, bn, bk, split_K_factor, brcount, upfront_xforms,
        WGT, ACT, scratch_A, scratch_B, output_partial,
        gemm_loop, brgemm_kernel, zero_kernel, tileconfig_kernel, tilerelease_kernel, use_sf_curve, sf_curve_index_map, index_tsize,
        reduce_output_loop, l_add_kernel, reduce_func, l_reduce_kernel, 0,
        xform_A_upfront, a_xform_loop, a_xform_kernel,
        xform_B_upfront, b_xform_loop, b_xform_kernel);
  }
  auto t_end = getTime();

#ifdef BENCH_REDUCE
 // benchmark the GEMM without REDUCTIONS
  auto t_start_noreduce = getTime();
  for (long it = 0; it < n_iters; it++) {
    run_gemm<DType>(n_layers, M, N, K,
        Mb, Nb, Kb, bm, bn, bk, split_K_factor, brcount, upfront_xforms,
        WGT, ACT, scratch_A, scratch_B, output_partial,
        gemm_loop, brgemm_kernel, zero_kernel, tileconfig_kernel, tilerelease_kernel, use_sf_curve, sf_curve_index_map, index_tsize,
        reduce_output_loop, l_add_kernel, reduce_func, l_reduce_kernel, 1,
        xform_A_upfront, a_xform_loop, a_xform_kernel,
        xform_B_upfront, b_xform_loop, b_xform_kernel);
  }
  auto t_end_noreduce = getTime();
#endif
 
  // Print performance/model numbers
  double gflop = (2.0*(double)n_layers*(double)M*(double)N*(double)K) / (1000*1000*1000);
  printf("Time is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  printf("Effective model sizes: %.5g GB\n", ((double)sizeof(DType)*(double)n_layers*(double)M*(double)K)/(1024.0*1024.0*1024.0));
  printf("Effective total GEMM sizes: %.5g GB\n", ((double)sizeof(DType)*(double)n_layers*((double)M*(double)K + (double)M*(double)N + (double)K*(double)N ))/(1024.0*1024.0*1024.0));
  printf("Effective A BW is %.5g GB/s\n", (((double)sizeof(DType)*(double)n_layers*(double)M*(double)K) / (1024.0*1024.0*1024.0))/((t_end-t_start)/(1.0*n_iters)));
  printf("MEASURE %.5g %s_%ld_%ld_%ld_%ld_%ld_%ld_bf%ld_threads%d_config_%s\n", gflop/((t_end-t_start)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads(),gemm_config);
#ifdef BENCH_REDUCE
  printf("MEASURE2 %.5g %s_%ld_%ld_%ld_%ld_%ld_%ld_bf%ld_threads%d_config_%s\n", gflop/((t_end_noreduce-t_start_noreduce)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads(),gemm_config);
  printf("Time is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  printf("Time2 is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end_noreduce-t_start_noreduce)/(1.0*n_iters), gflop/((t_end_noreduce-t_start_noreduce)/(1.0*n_iters)));
  printf("Reduction diff is %.5g sec\n", ((t_end-t_start)-(t_end_noreduce-t_start_noreduce))/(1.0*n_iters));
  double reduction_vol = (((double)sizeof(DType)*(double)n_layers*(double)M*(double)N*(double)split_K_factor)/(1024.0*1024.0*1024.0));
  printf("Reductions run at %.5g GB/s\n", reduction_vol/(((t_end-t_start)-(t_end_noreduce-t_start_noreduce))/(1.0*n_iters)));
#endif

  // Free buffers
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_opt);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_input_lp);
  libxsmm_free(naive_output_lp);
  libxsmm_free(naive_filter_lp);
  for (i = 0; i < n_layers; i++) {
    libxsmm_free(WGT[i]);
  }
  for (i = 0; i < 2*n_layers; i++) {
    libxsmm_free(ACT[i]);
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
  if (split_K_factor > 1) {
    libxsmm_free(global_scratch);
  } 
  free(ACT);
  free(WGT);
  return 0;
}

int main(int argc, char** argv) {
  int use_dtype = 1;
  if (argc > 17) {
    if (strcmp(argv[17],"BF16") == 0) {
      use_dtype = 1;
    }
    if (strcmp(argv[17],"BF8") == 0) {
      use_dtype = 2;
    }
    if (strcmp(argv[17],"FP32") == 0) {
      use_dtype = 3;
    }
  }
  if (use_dtype == 1) {
    return gemm_benchmark<libxsmm_bfloat16>(argc, argv);  
  } else if (use_dtype == 2) {
    return gemm_benchmark<libxsmm_bfloat8>(argc, argv);
  } else if (use_dtype == 3) {
    return gemm_benchmark<float>(argc, argv);
  } else {
    return 0;
  }
}

