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
int conv_benchmark(int argc, char** argv) {
  int error = 0;
  // Setup default GEMM sizes
  int check_correctness = 1;
  char loop_specs_str[256] = "aBC";  
  long N = 14, H = 28, W = 28, C = 512, K = 1024, R = 1, S = 1, stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0;
  long bc = 32, bk = 32, bn = 32;
  long n_iters = 1;
  long i;
#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif
  /* Some algorithmic knobs  */
  /* Uses parallelism in the MB dimension for f32 precision */
  long use_mb_par_f32 = 1;

  /* Fuse bf16 necessary transposes */ 
  long bf16_use_nchw_format = 1;
  long bf16_use_chwn_format = 1;
  long bf16_fuse_upd_transposes = 1;

  /* Control variants for chwn format */
  long bf16_acc_nw = 1;
  long par_over_h_pixels = 1;
  long use_private_trans = 0;

  /* Control variants for nchw format */
  long pack_input_upfront = 0;
  long compute_pixels = 0;
  long remainder_pixels = 0;
  long upd_remaining_pixels = 0;
  long accum_length_pixels = 0;
  long max_init_offset = 0;
  long input_compute_pad = 0;
  long input_pixels = 0;
  long output_pixels = 0;
  long pixel_blocking = 0;
  long n_used_pixels = 0;
  long use_intermediate_f32_wt_tensor = 0;
  long use_hybrid_imgfm_parallelization = 0;
  long n_img_teams = 7;
  long n_ofm_teams = 4;
  long weight_copies = 0;
  long multiple_target = 2;
  long max_compute_offset_input = 0;
  long use_f32_wt_reduction_and_external_wt_vnni = 0;
  long compute_full_wt_output_block = 0;
  long pixels_blocking_factor = 1;
  long logical_padding = 0;

  // Setup model and trace
  ifreq = 1.0 / getFreq();
  std::vector<std::string> inp_trace[128];
  platform_spec_t my_platform;
  tensor_metadata_t tensor_metadata;
  set_platform_specs( CLX, omp_get_max_threads(), &my_platform);

  if (argc > 1) {
    sprintf(loop_specs_str, "%s", argv[1]);
  }
  if (argc > 2) {
    N = atoi(argv[2]);
    H = atoi(argv[3]);
    W = atoi(argv[4]);
    C = atoi(argv[5]);
    K = atoi(argv[6]);
    R = atoi(argv[7]);
    S = atoi(argv[8]);
    stride_h = atoi(argv[9]);
    stride_w = atoi(argv[10]);
    pad_h  = atoi(argv[11]);
    pad_w = atoi(argv[12]);
    bc  = atoi(argv[13]);
    bk  = atoi(argv[14]);
    if (argc > 15) {
      n_iters = atoi(argv[15]);
    }

    if (sizeof(DType) == 2) {
      bf16_use_nchw_format            = atoi(argv[16]);
      bf16_fuse_upd_transposes        = atoi(argv[17]);
      bf16_acc_nw                     = atoi(argv[18]);
      par_over_h_pixels               = atoi(argv[19]);
      pack_input_upfront              = atoi(argv[20]);
      use_intermediate_f32_wt_tensor  = atoi(argv[21]);
      use_hybrid_imgfm_parallelization = atoi(argv[22]);
      n_img_teams                     = atoi(argv[23]);
      n_ofm_teams                     = atoi(argv[24]);
      use_f32_wt_reduction_and_external_wt_vnni = atoi(argv[25]);
      compute_full_wt_output_block    = atoi(argv[26]);
      if (argc > 27) {
        pixels_blocking_factor = atoi(argv[27]);
      }
      if (argc > 28) {
        logical_padding = atoi(argv[28]);
      }
    } else {
      use_mb_par_f32 = atoi(argv[16]);    
    }
  }

  bf16_use_chwn_format = (bf16_use_nchw_format > 0) ? 0 : 1;
  use_private_trans = bf16_fuse_upd_transposes;

  if (logical_padding && ((sizeof(DType) != 2) || (bf16_use_chwn_format == 0) || (use_private_trans != 0)) ) {
    printf("Error: logical padding is only supported for bf16 and chwn format and use_private_trans == 0 \n");
    exit(-1);
  }

  long Kb = K/bk, Cb = C/bc;
  // For now only physical padding
  long  pad_h_in = pad_h;
  long  pad_w_in = pad_w;
  long  pad_h_out = pad_h;
  long  pad_w_out = pad_w;

  // Deriving some aux values
  long ofh = (H + 2 * pad_h - R) / stride_h + 1;
  long ofw = (W + 2 * pad_w - S) / stride_w + 1;
  long ifhp = H + 2 * pad_h_in;
  long ifwp = W + 2 * pad_w_in;
  long ofhp = ofh + 2 * pad_h_out;
  long ofwp = ofw + 2 * pad_w_out;
  long ifh = H;
  long ifw = W;
  bn = N;

  // Allocate buffers
  float *naive_input  = (float*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(float), 2097152);
  float *naive_input_nchwc  = (float*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(float), 2097152);
  float *naive_input_unpad  = (float*)libxsmm_aligned_malloc( N*ifh*ifw*C*sizeof(float), 2097152);
  float *naive_input_unpad_nchwc  = (float*)libxsmm_aligned_malloc( N*ifh*ifw*C*sizeof(float), 2097152);
  float *naive_output = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_output_nchwc = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_output_unpad = (float*)libxsmm_aligned_malloc( N*ofh*ofw*K*sizeof(float), 2097152);
  float *naive_output_unpad_nchwc = (float*)libxsmm_aligned_malloc( N*ofh*ofw*K*sizeof(float), 2097152);
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_filter = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  float *naive_filter_opt = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  float *naive_filter_kcrsck = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
  DType *input_unpad_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifh*ifw*C*sizeof(DType), 2097152);
  DType *output_libxsmm = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  DType *output_unpad_libxsmm = (DType*)libxsmm_aligned_malloc( N*ofh*ofw*K*sizeof(DType), 2097152);
  DType *tr_input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
  DType *tr_output_libxsmm = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  DType **private_tr_input_libxsmm  = (DType**)libxsmm_aligned_malloc( nThreads*sizeof(DType*), 2097152);
  DType **private_tr_output_libxsmm = (DType**)libxsmm_aligned_malloc( nThreads*sizeof(DType*), 2097152);
  for (int thr = 0; thr < nThreads; thr++) {
    private_tr_input_libxsmm[thr] = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
    private_tr_output_libxsmm[thr] = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  }
  DType *filter_libxsmm = (DType*)libxsmm_aligned_malloc( C*K*R*S*sizeof(DType), 2097152);
  float *scratch_libxsmm = (float*)libxsmm_aligned_malloc( nThreads*C*K*R*S*sizeof(float), 2097152);
  libxsmm_bfloat16 *scratch_libxsmm_bf16_weights = (libxsmm_bfloat16*)libxsmm_aligned_malloc(C*K*R*S*sizeof(libxsmm_bfloat16), 2097152);
  DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
  unsigned long long *A_offsets = (unsigned long long*) libxsmm_aligned_malloc(Cb * R * S * sizeof(unsigned long long), 2097152);
  unsigned long long *B_offsets = (unsigned long long*) libxsmm_aligned_malloc(Cb * R * S * sizeof(unsigned long long), 2097152);
  int trans_tracker_size = Cb + Kb + 64 - 64%(Cb+Kb);
  int *trans_tracker = (int*)libxsmm_aligned_malloc( nThreads*trans_tracker_size*sizeof(int), 2097152);
  DType *input_linearized_pixels;
  DType *output_linearized_pixels;

  libxsmm_matdiff_info norms, diff;
  libxsmm_matdiff_clear(&norms);
  libxsmm_matdiff_clear(&diff);

  // Init buffers
  float *naive_input_tmp = (float*)libxsmm_aligned_malloc( (size_t)N*C*ifhp*ifwp*sizeof(float), 2097152);
  init_buf(naive_input_tmp,          N*C*ifh*ifw, 0, 0);
  copy_internal_nchw( naive_input_unpad, naive_input_tmp, N, C, ifh, ifw, 0, 0);
  copy_internal_nchw( naive_input , naive_input_tmp, N, C, ifh, ifw, pad_h, pad_w);
  libxsmm_free(naive_input_tmp);
  set_zeropad_nchw(naive_input, N, C, ifhp, ifwp, pad_h_in, pad_w_in);

  float *naive_output_tmp = (float*)libxsmm_aligned_malloc( (size_t)N*K*ofhp*ofwp*sizeof(float), 2097152);
  init_buf(naive_output_tmp,          N*K*ofh*ofw, 0, 0);
  copy_internal_nchw( naive_output_unpad, naive_output_tmp, N, K, ofh, ofw, 0, 0);
  copy_internal_nchw( naive_output , naive_output_tmp, N, K, ofh, ofw, pad_h, pad_w);
  libxsmm_free(naive_output_tmp);
  set_zeropad_nchw(naive_output, N, K, ofhp, ofwp, pad_h_out, pad_w_out);

  //init_buf(naive_output,         N*K*ofwp*ofhp, 0, 0);
  //set_zeropad_nchw(naive_output, N, K, ofhp, ofwp, pad_h_out, pad_w_out);

  init_buf(naive_filter,         K*C*R*S, 0, 0);
  
  if (sizeof(DType) == 2) {
    tensor_copy_NCHW_to_NCHWc (naive_input , naive_input_nchwc,  N, C, ifhp, ifwp, bc);
    tensor_copy_NCHW_to_NCHWc (naive_input_unpad, naive_input_unpad_nchwc,  N, C, ifh, ifw, bc);
    tensor_copy_NCHW_to_NCHWc (naive_output, naive_output_nchwc, N, K, ofhp, ofwp, bk);
    tensor_copy_NCHW_to_NCHWc (naive_output_unpad, naive_output_unpad_nchwc,  N, K, ofh, ofw, bk);
    tensor_copy_KCRS_to_KCRSck_bf16(naive_filter, (libxsmm_bfloat16*)filter_libxsmm, K, C, R, S, bc, bk);
    libxsmm_rne_convert_fp32_bf16( naive_input_nchwc,     (libxsmm_bfloat16*)input_libxsmm,     N*C*ifhp*ifwp );
    libxsmm_rne_convert_fp32_bf16( naive_output_nchwc,    (libxsmm_bfloat16*)output_libxsmm,    N*K*ofhp*ofwp );
    libxsmm_rne_convert_fp32_bf16( naive_input_unpad_nchwc,     (libxsmm_bfloat16*)input_unpad_libxsmm,     N*C*ifh*ifw );
    libxsmm_rne_convert_fp32_bf16( naive_output_unpad_nchwc,    (libxsmm_bfloat16*)output_unpad_libxsmm,    N*K*ofh*ofw );
  } else {
    tensor_copy_NCHW_to_NCHWc (naive_input , (float*)input_libxsmm,  N, C, ifhp, ifwp, bc);
    tensor_copy_NCHW_to_NCHWc (naive_output, (float*)output_libxsmm, N, K, ofhp, ofwp, bk);
    tensor_copy_KCRS_to_KCRSck(naive_filter, (float*)filter_libxsmm, K, C, R, S, bc, bk);
  }

  // Compute reference if requested
  if (check_correctness) {
    naive_conv_t naive_param;
    naive_param.nImg = N;
    naive_param.nIfm = C;
    naive_param.nOfm = K;
    naive_param.ifhp = ifhp;
    naive_param.ifwp = ifwp;
    naive_param.ofhp = ofhp;
    naive_param.ofwp = ofwp;
    naive_param.ifh = ifh;
    naive_param.ifw = ifw;
    naive_param.ofh = ofh;
    naive_param.ofw = ofw;
    naive_param.pad_h = pad_h;
    naive_param.pad_w = pad_w;
    naive_param.pad_h_in = pad_h_in;
    naive_param.pad_w_in = pad_w_in;
    naive_param.pad_h_out = pad_h_out;
    naive_param.pad_w_out = pad_w_out;
    naive_param.kh = R;
    naive_param.kw = S;
    naive_param.stride_h = stride_h;
    naive_param.stride_w = stride_w;
    zero_buf(naive_filter,    K*C*R*S);
    naive_conv_wu(&naive_param, naive_input, naive_output, naive_filter); 
  }

  // TPP kernels that may be used
  libxsmm_meltwfunction_unary zero_kernel;
  libxsmm_meltwfunction_unary zero_kernel_chwn;
  libxsmm_meltwfunction_unary zero_kernel_khwn;
  libxsmm_meltwfunction_unary zero_kernel_bf16;
  libxsmm_meltwfunction_unary zero_input_pad_kernel_bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel0_f32;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_f32;
  libxsmm_meltwfunction_unary wt_reduce_kernel0_f32bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_f32bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel0_bf16bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_bf16bf16;
  libxsmm_meltwfunction_unary trans_xform_kernel;
  libxsmm_meltwfunction_unary vnni_xform_kernel;
  libxsmm_meltwfunction_unary fp32bf16_cvt_kernel;
  libxsmm_meltwfunction_unary wt_vnni_kernel;
  libxsmm_meltwfunction_unary vnni_output_compute_pixels_bf16;
  libxsmm_meltwfunction_unary vnni_output_zero_remaining_pixels_bf16;
  libxsmm_meltwfunction_unary transpose_input_pixels_bf16;
  libxsmm_meltwfunction_unary transposeNpack_input_pixels_bf16;

  libxsmm_xmmfunction tileconfig_kernel;
  libxsmm_xmmfunction tilerelease_kernel;
  libxsmm_xmmfunction gemm_kernel;
  libxsmm_xmmfunction brgemm_kernel_acc_pixel;
  libxsmm_xmmfunction brgemm_kernel_acc_pixel_zerobeta_cvnni;
  libxsmm_xmmfunction gemm_kernel_non_hybrid;
  libxsmm_xmmfunction gemm_kernel_non_hybrid_zerobeta_cvnni;
  libxsmm_xmmfunction brgemm_kernel_hybrid;
  libxsmm_xmmfunction brgemm_kernel_hybrid_zerobeta_cvnni;

  
  char bf16_conv_spec_string[256];
  char fp32_conv_spec_string[256];

  if (sizeof(DType) == 4) {
    sprintf(fp32_conv_spec_string, "%s", loop_specs_str);
    sprintf(bf16_conv_spec_string, "Abcdef");
  } else {
    sprintf(fp32_conv_spec_string, "Abcdefg");
    sprintf(bf16_conv_spec_string, "%s", loop_specs_str);
  }

  // Setup basic GEMM flags
  auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto l_tc_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto l_tr_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto dtype      = (sizeof(DType) == 2) ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;
  
  // Configure WT Reduction related TPP kernels
  long fm_blocking = (bk % 16 == 0) ? 16 : bk;
  long reduce_work = Kb * C * R * S * (bk/fm_blocking);
  long reduce_chunk_size = (reduce_work + nThreads - 1)/nThreads;
  long reduce_work_tripcount = (reduce_work + reduce_chunk_size - 1) / reduce_chunk_size;
  long chunk0 = reduce_chunk_size * fm_blocking;
  long chunk1 = K * C * R * S  - (reduce_work_tripcount-1) * chunk0;
  chunk1 = (chunk1 <= 0) ? chunk0 : chunk1; 

  if (use_hybrid_imgfm_parallelization > 0) {
    bf16_fuse_upd_transposes = 0;
    weight_copies = n_img_teams;
  } else {
    weight_copies = nThreads;
  }

  auto l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, weight_copies, K * C *R * S, chunk0, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
  wt_reduce_kernel0_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
  l_unary_shape.m         = chunk1;
  l_unary_shape.ldo       = chunk1;
  wt_reduce_kernel1_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ; 
 
  l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, weight_copies, K * C *R * S, chunk0, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32);
  wt_reduce_kernel0_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
  l_unary_shape.m         = chunk1;
  l_unary_shape.ldo       = chunk1;
  wt_reduce_kernel1_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;  

  l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, weight_copies, K * C *R * S, chunk0, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32);
  wt_reduce_kernel0_bf16bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
  l_unary_shape.m         = chunk1;
  l_unary_shape.ldo       = chunk1;
  wt_reduce_kernel1_bf16bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;

  // Configure zero TPP kernels
  l_unary_shape = libxsmm_create_meltw_unary_shape(bk*bc, 1, bk*bc, bk*bc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
  zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  l_unary_shape = libxsmm_create_meltw_unary_shape(bk*bc, 1, bk*bc, bk*bc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
  zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  // Generate XForm TPP kernels
  auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bk, dtype, dtype, dtype);
  wt_vnni_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  if (logical_padding)
    tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bn, K*ofh*ofw, bk, dtype, dtype, dtype);
  else
    tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bn, K*ofhp*ofwp, bk, dtype, dtype, dtype);
  vnni_xform_kernel =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

  if (logical_padding)
    tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, bn, C*ifh*ifw, bn, dtype, dtype, dtype);
  else
    tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, bn, C*ifhp*ifwp, bn, dtype, dtype, dtype);
  trans_xform_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

  // Generate f32->bf16 cvt TPP kernel
  l_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bk, LIBXSMM_DATATYPE_F32, dtype, LIBXSMM_DATATYPE_F32);
  fp32bf16_cvt_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  if (sizeof(DType) == 4) {
    auto gemm_n = bc;
    auto gemm_m = bk;
    auto gemm_k = ofw;
    auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
  } else {
    if (bf16_use_nchw_format > 0) {
     if (R == 1 && S == 1 && (stride_w != 1 || stride_h != 1)) {
        pack_input_upfront = 1;
      } else {
        pack_input_upfront = 0;
      }
      compute_pixels = ofw * ofh + 2 * pad_w * (ofh-1);
      remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
      accum_length_pixels = compute_pixels + remainder_pixels;
      max_init_offset = 2 * pad_h * ifwp + 2 * pad_w;
      max_compute_offset_input = max_init_offset + accum_length_pixels;
      input_compute_pad = (max_compute_offset_input > ifwp*ifhp) ? max_compute_offset_input - ifwp*ifhp : 0;
      input_pixels = ifwp*ifhp+ input_compute_pad;
      if (pack_input_upfront) {
        input_pixels = accum_length_pixels;
        auto pack_unary_shape = libxsmm_create_meltw_unary_shape(bc, ofw, stride_w * bc, input_pixels, dtype, dtype, dtype);
        transposeNpack_input_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, pack_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      }
      output_pixels = accum_length_pixels;
      n_used_pixels = accum_length_pixels;
      pixel_blocking = accum_length_pixels;
      while (pixel_blocking % pixels_blocking_factor != 0) {
        pixels_blocking_factor--;
      }
      pixel_blocking = accum_length_pixels/pixels_blocking_factor;
      use_intermediate_f32_wt_tensor = (pixel_blocking == n_used_pixels) ? 0 : 1;
      float beta = (use_intermediate_f32_wt_tensor) ? (float)1.0 : (float)0.0;

      if (pack_input_upfront)
        l_unary_shape = libxsmm_create_meltw_unary_shape(remainder_pixels, bc, input_pixels, input_pixels, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
      else
        l_unary_shape = libxsmm_create_meltw_unary_shape(input_compute_pad, bc, input_pixels, input_pixels, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
      zero_input_pad_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

      if (use_hybrid_imgfm_parallelization == 0) {
        auto new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
        auto new_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
        auto new_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
        if (use_intermediate_f32_wt_tensor == 0) {
          new_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
        }
        gemm_kernel_non_hybrid.gemm = libxsmm_dispatch_gemm_v2( new_shape, new_flags, new_prefetch_flags );
        
        new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, dtype, dtype);
        new_flags |=  LIBXSMM_GEMM_FLAG_BETA_0  | LIBXSMM_GEMM_FLAG_VNNI_C;
        gemm_kernel_non_hybrid_zerobeta_cvnni.gemm      = libxsmm_dispatch_gemm_v2( new_shape, new_flags, new_prefetch_flags );
        tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( new_shape, l_tc_flags, new_prefetch_flags );
        tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( new_shape, l_tr_flags, new_prefetch_flags );
      } else {
        long stride_a = K * output_pixels * sizeof(DType);
        long stride_b = C * input_pixels * sizeof(DType);
        auto new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
        auto new_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
        auto new_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
        if (use_intermediate_f32_wt_tensor == 0) {
          new_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
        }
        auto new_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, stride_a, stride_b, 0 );
        brgemm_kernel_hybrid.gemm   = libxsmm_dispatch_brgemm_v2( new_shape, new_flags, new_prefetch_flags, new_brconfig );

        new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, dtype, dtype);
        new_flags |=  LIBXSMM_GEMM_FLAG_BETA_0  | LIBXSMM_GEMM_FLAG_VNNI_C;
        brgemm_kernel_hybrid_zerobeta_cvnni.gemm   = libxsmm_dispatch_brgemm_v2( new_shape, new_flags, new_prefetch_flags, new_brconfig );

        tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( new_shape, l_tc_flags, new_prefetch_flags );
        tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( new_shape, l_tr_flags, new_prefetch_flags );
      }
      input_linearized_pixels  = (DType*)libxsmm_aligned_malloc( N*input_pixels*C*sizeof(DType), 2097152);
      output_linearized_pixels = (DType*)libxsmm_aligned_malloc( N*output_pixels*K*sizeof(DType), 2097152);
      auto new_tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, ifwp, bc, input_pixels, dtype, dtype, dtype);
      transpose_input_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      new_tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, compute_pixels, bk, bk, dtype, dtype, dtype);
      if ((ofhp * ofwp) % 2 == 0) {
        vnni_output_compute_pixels_bf16 =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      } else {
        vnni_output_compute_pixels_bf16 =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      }
      upd_remaining_pixels = output_pixels - ((compute_pixels+1)/2)*2;
      auto zero_unary_shape = libxsmm_create_meltw_unary_shape(bk*upd_remaining_pixels, 1, bk*upd_remaining_pixels, bk*upd_remaining_pixels, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
      vnni_output_zero_remaining_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, zero_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    } else {
      auto gemm_n = bc;
      auto gemm_m = bk;
      auto gemm_k = bn;
      auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bn, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
      auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
      auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bn*bk*sizeof(DType), stride_w*bc*bn*sizeof(DType), 0 );
      tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
      tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
      gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
      brgemm_kernel_acc_pixel.gemm  = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      l_flags  |=  LIBXSMM_GEMM_FLAG_BETA_0  | LIBXSMM_GEMM_FLAG_VNNI_C;
      l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bn, bk, dtype, dtype, dtype, dtype);
      brgemm_kernel_acc_pixel_zerobeta_cvnni.gemm  = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

      l_unary_shape = libxsmm_create_meltw_unary_shape(bc*bn, 1, bc*bn, bc*bn, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
      zero_kernel_chwn = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

      l_unary_shape = libxsmm_create_meltw_unary_shape(bk*bn, 1, bk*bn, bk*bn, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
      zero_kernel_khwn = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    }
  }
  
  // JIT nested loop specs for various algorithms
  long n_step = 1;
  long c_step = 1;
  long k_step = 1;
  long h_step = 1;
  long w_step = ofw;
  long r_step = 1;
  long s_step = 1;
  long tr_step = 1;

  // Aux steps for linearized algo loops
  long _n_step = 1;
  long _k_step = 1;
  long _c_step = 1;
  long _r_step = 1;
  long _s_step = 1;

  printf("Test parameters: N H W C K R S stride_h stride_w pad_h pad_w bc bk logical_padding: %d %d %d %d %d %d %d %d %d %d %d %d %d %d \n", N, H, W, C, K, R, S, stride_h, stride_w, pad_h, pad_w, bc, bk, logical_padding);
  printf("Tuning parameters: bf16_use_nchw_format bf16_fuse_upd_transposes bf16_acc_nw: %d %d %d \n", bf16_use_nchw_format, bf16_fuse_upd_transposes, bf16_acc_nw);
  printf("Tuning parameters: par_over_h_pixels pack_input_upfront use_intermediate_f32_wt_tensor: %d %d %d \n", par_over_h_pixels, pack_input_upfront, use_intermediate_f32_wt_tensor);
  printf("Tuning parameters: use_hybrid n_img_teams n_ofm_teams: %d %d %d \n", use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams);
  printf("Tuning parameters: use_f32_wt_reduction_and_external_wt_vnni compute_full_wt_output_block pblock: %d %d %d \n", use_f32_wt_reduction_and_external_wt_vnni, compute_full_wt_output_block, pixels_blocking_factor);
  printf("Tuning parameters: use_mb_par_f32: %d \n", use_mb_par_f32);


  auto t0 = getTime();

  // Zeros nThreads F32 wt tensors 
  auto zero_wt_loop = ThreadedLoop<5>({
      LoopSpecs{0, nThreads, 1, true},
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      "Abcde");

  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, true},
      LoopSpecs{0, Cb, c_step, true},
      LoopSpecs{0, Kb, k_step, true},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step, true},
      LoopSpecs{0, S, s_step, true}},
      fp32_conv_spec_string);

  // Transposes input to CHWN format
  auto tr_input_loop = ThreadedLoop<3>({
      LoopSpecs{0, Cb, tr_step},
      LoopSpecs{0, ifhp, tr_step},
      LoopSpecs{0, ifwp, tr_step}},
      "ABC");

  // Transposes output to Kb HW bk N format
  auto tr_output_loop  = ThreadedLoop<3>({
      LoopSpecs{0, Kb, tr_step},
      LoopSpecs{0, ofhp, tr_step},
      LoopSpecs{0, ofwp, tr_step}},
      "ABC");

  if (sizeof(DType) == 2) {
    w_step = 1;
  }

  if (bf16_acc_nw == 1) {
    w_step = ofw;
    h_step = 1;
  }

  if (compute_full_wt_output_block > 0 && bf16_use_chwn_format > 0) {
    w_step = ofw;
    h_step = ofh;
  }
  
  auto conv_loop_bf16 = ThreadedLoop<6>({
      LoopSpecs{0, Cb, c_step, true},
      LoopSpecs{0, Kb, k_step, true},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step, true},
      LoopSpecs{0, S, s_step, true}},
      bf16_conv_spec_string);

  // Weight reduction loop
  auto reduce_wt_loop = ThreadedLoop<1>({
      LoopSpecs{0, reduce_work_tripcount, 1, true}},
      "A");

  // Loop to vnni convert bf16 weights
  auto vnni_wt_loop = ThreadedLoop<4>({
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      "ABCD");

  char nchw_format_loop_spec[256];
  auto tr_input_nchw_loop = ThreadedLoop<2>({
      LoopSpecs{0, N, _n_step},
      LoopSpecs{0, Cb, _c_step}},
      "Ab");

  auto tr_output_nchw_loop = ThreadedLoop<2>({
      LoopSpecs{0, N, _n_step},
      LoopSpecs{0, Kb, _k_step}},
      "Ab");

  if (use_hybrid_imgfm_parallelization == 0) {
    //sprintf(nchw_format_loop_spec, "Abcdef");
  } else {
    if (compute_full_wt_output_block > 0) {
      _n_step = N;
    } else {
      _n_step = N/n_img_teams;
    }
  }

  auto conv_loop_bf16_nchw = ThreadedLoop<6>({
      LoopSpecs{0, N, _n_step, true},
      LoopSpecs{0, Cb, _c_step, true},
      LoopSpecs{0, Kb, _k_step, true},
      LoopSpecs{0, n_used_pixels, pixel_blocking},
      LoopSpecs{0, R, _r_step, true},
      LoopSpecs{0, S, _s_step, true}},
      bf16_conv_spec_string);

  auto t1 = getTime();

  double t_start, t_end;
  //Benchmark convolution
  for (i = 0; i < n_iters + 1; i++) {
    if (i == 1) t_start = getTime();
    if (sizeof(DType) == 4) {
      if (use_mb_par_f32 == 0) {
        conv_loop(
          [&](int* ind) {
            int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];
            libxsmm_gemm_param gemm_param;
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);        
            if (i_n == 0 && i_w == 0 && i_h == 0) {
              libxsmm_meltw_unary_param zero_param;
              zero_param.out.primary = (void*)gemm_param.c.primary;
              zero_kernel( &zero_param );
            }
            gemm_kernel.gemm( &gemm_param );
          },
          [&]() {},
          [&]() {});
      } else {
        zero_wt_loop(
          [&](int* ind) {
            int i_n = ind[0], i_k = ind[1], i_c = ind[2], i_r = ind[3], i_s = ind[4];
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, i_n, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
            zero_kernel( &zero_param );
          },
          [&]() {},
          [&]() {});

        conv_loop(
          [&](int* ind) {
            int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];
            int tid = omp_get_thread_num();
            libxsmm_gemm_param gemm_param;
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);        
            gemm_kernel.gemm( &gemm_param );
          },
          [&]() {},
          [&]() {});

        reduce_wt_loop(
          [&](int* ind) {
            int i_n = ind[0];
            libxsmm_meltw_unary_param reduce_param;
            reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm, i_n, 0, chunk0);
            reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)filter_libxsmm,  i_n, 0, chunk0);
            if (i_n < reduce_work_tripcount - 1) {
              wt_reduce_kernel0_f32( &reduce_param );
            } else {
              wt_reduce_kernel1_f32( &reduce_param );
            }
          },
          [&]() {},
          [&]() {});
      }
    }

    if ( (sizeof(DType) == 2) && (bf16_use_nchw_format > 0)) {
      if (bf16_fuse_upd_transposes == 0) {
        if (pack_input_upfront > 0) {
          tr_input_nchw_loop(
            [&](int* ind) {
              int i_n = ind[0], i_c = ind[1];
              libxsmm_meltw_unary_param unary_param;
              for (int ij = 0; ij < ofh; ij++) {
                unary_param.in.primary = (void*) LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm,           i_n, i_c, ij*stride_h, 0, 0, Cb, ifhp, ifwp, bc);
                unary_param.out.primary= (void*) LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, ij*(ifwp/stride_w), Cb, bc, input_pixels);
                transposeNpack_input_pixels_bf16( &unary_param );
              }
              if (remainder_pixels > 0) {
                libxsmm_meltw_unary_param zero_param;
                zero_param.out.primary = (void*) LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, ofh*(ifwp/stride_w), Cb, bc, input_pixels);
                zero_input_pad_kernel_bf16( &zero_param );
              }
            },
            [&]() {},
            [&]() {});
        } else {
          tr_input_nchw_loop(
            [&](int* ind) {
              int i_n = ind[0], i_c = ind[1];
              libxsmm_meltw_unary_param unary_param;
              for (int ij = 0; ij < ifhp; ij++) {
                unary_param.in.primary = (void*) LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm,           i_n, i_c, ij, 0, 0, Cb, ifhp, ifwp, bc);
                unary_param.out.primary= (void*) LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, ij*ifwp, Cb, bc, input_pixels);
                transpose_input_pixels_bf16( &unary_param );
              }
              if (input_compute_pad > 0) {
                libxsmm_meltw_unary_param zero_param;
                zero_param.out.primary = (void*) LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, ifhp*ifwp, Cb, bc, input_pixels);
                zero_input_pad_kernel_bf16( &zero_param );
              }
            },
            [&]() {},
            [&]() {});
        }

        tr_output_nchw_loop(
          [&](int* ind) {
            int i_n = ind[0], i_k = ind[1];
            libxsmm_meltw_unary_param unary_param;
            unary_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm,           i_n, i_k, pad_h, pad_w, 0, Kb, ofhp, ofwp, bk);
            unary_param.out.primary= LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, 0, 0, Kb, output_pixels, bk);
            vnni_output_compute_pixels_bf16( &unary_param );
            if (upd_remaining_pixels > 0) {
              unary_param.out.primary= LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, ((compute_pixels+1)/2)*2, 0, Kb, output_pixels, bk);
              vnni_output_zero_remaining_pixels_bf16( &unary_param );
            }
          },
          [&]() {},
          [&]() {});
      }
      if (use_hybrid_imgfm_parallelization == 0) {
        conv_loop_bf16_nchw(
          [&](int* ind) {
            int i_n = ind[0], i_c = ind[1], i_k = ind[2], pix = ind[3], i_r = ind[4], i_s = ind[5];
            libxsmm_gemm_param gemm_param;
            libxsmm_meltw_unary_param unary_param;
            int tid = omp_get_thread_num(); 

            if (bf16_fuse_upd_transposes == 1 && pix == 0 && i_c == 0 && i_r == 0 && i_s == 0) {
              unary_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm,           i_n, i_k, pad_h, pad_w, 0, Kb, ofhp, ofwp, bk);
              unary_param.out.primary= LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, 0, 0, Kb, output_pixels, bk);
              vnni_output_compute_pixels_bf16( &unary_param );
              if (upd_remaining_pixels > 0) {
                unary_param.out.primary= LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, ((compute_pixels+1)/2)*2, 0, Kb, output_pixels, bk);
                vnni_output_zero_remaining_pixels_bf16( &unary_param );
              }
            }

            if (bf16_fuse_upd_transposes == 1 && pix == 0 && i_k == 0 && i_r == 0 && i_s == 0) {
              for (int ij = 0; ij < ifhp; ij++) {
                unary_param.in.primary = (void*) LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm,           i_n, i_c, ij, 0, 0, Cb, ifhp, ifwp, bc);
                unary_param.out.primary= (void*) LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, ij*ifwp, Cb, bc, input_pixels);
                transpose_input_pixels_bf16( &unary_param );
              }
              if (input_compute_pad > 0) {
                libxsmm_meltw_unary_param zero_param;
                zero_param.out.primary = (void*) LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, ifhp*ifwp, Cb, bc, input_pixels);
                zero_input_pad_kernel_bf16( &zero_param );
              }
            }
       
            if (use_f32_wt_reduction_and_external_wt_vnni > 0) { 
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);     
              if (pix == 0) {
                libxsmm_meltw_unary_param zero_param;
                zero_param.out.primary = (void*)gemm_param.c.primary;
                zero_kernel( &zero_param );
              }
              gemm_kernel_non_hybrid.gemm( &gemm_param );
            } else {
              /* Use beta = 0 kernel with c_vnni formating */
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);     
              gemm_kernel_non_hybrid_zerobeta_cvnni.gemm( &gemm_param );
            }
          },
          [&]() {if (sizeof(DType) == 2) tileconfig_kernel.gemm(NULL);},
          [&]() {if (sizeof(DType) == 2) tilerelease_kernel.gemm(NULL);});
        
        if (use_f32_wt_reduction_and_external_wt_vnni > 0) { 
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              libxsmm_meltw_unary_param reduce_param;
              reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(float), (float*)scratch_libxsmm, i_n, 0, chunk0);
              reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm_bf16_weights, i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce_kernel0_f32bf16( &reduce_param );
              } else {
                wt_reduce_kernel1_f32bf16( &reduce_param );  
              } 
            },
            [&]() {},
            [&]() {});

          vnni_wt_loop(
            [&](int* ind) {
              int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];
              libxsmm_meltw_unary_param xform_param;
              xform_param.in.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), scratch_libxsmm_bf16_weights, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              wt_vnni_kernel( &xform_param );
            },
            [&]() {},
            [&]() {});
        } else {
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              libxsmm_meltw_unary_param reduce_param;
              reduce_param.in.primary   = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm, i_n, 0, chunk0);
              reduce_param.out.primary  = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)filter_libxsmm, i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce_kernel0_bf16bf16( &reduce_param );
              } else {
                wt_reduce_kernel1_bf16bf16( &reduce_param );  
              } 
            },
            [&]() {},
            [&]() {});
        }
      } else {
        conv_loop_bf16_nchw(
          [&](int* ind) {
            int i_n = ind[0], i_c = ind[1], i_k = ind[2], pix = ind[3], i_r = ind[4], i_s = ind[5];
            int my_col_id;
            unsigned long long brcount = _n_step;
            libxsmm_gemm_param gemm_param;
            libxsmm_meltw_unary_param unary_param;
            
            if (compute_full_wt_output_block == 0) {
              my_col_id = conv_loop_bf16_nchw.get_tid_in_parallel_dim('a', ind);
              if (use_f32_wt_reduction_and_external_wt_vnni > 0) { 
                gemm_param.op.tertiary = (void*)&brcount;        
                gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
                gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, my_col_id, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);     
                if (pix == 0) {
                  libxsmm_meltw_unary_param zero_param;
                  zero_param.out.primary = (void*)gemm_param.c.primary;
                  zero_kernel( &zero_param );
                }
                brgemm_kernel_hybrid.gemm( &gemm_param );
              } else {
                gemm_param.op.tertiary = (void*)&brcount;        
                gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
                gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, my_col_id, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);     
                brgemm_kernel_hybrid_zerobeta_cvnni.gemm( &gemm_param );       
              }
            } else {
              gemm_param.op.tertiary = (void*)&brcount;        
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), (DType*)filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);     
              brgemm_kernel_hybrid_zerobeta_cvnni.gemm( &gemm_param );  
            }
          },
          [&]() {if (sizeof(DType) == 2) tileconfig_kernel.gemm(NULL);},
          [&]() {if (sizeof(DType) == 2) tilerelease_kernel.gemm(NULL);});
        if (use_f32_wt_reduction_and_external_wt_vnni > 0) { 
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              libxsmm_meltw_unary_param reduce_param;
              reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(float), (float*)scratch_libxsmm, i_n, 0, chunk0);
              reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm_bf16_weights, i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce_kernel0_f32bf16( &reduce_param );
              } else {
                wt_reduce_kernel1_f32bf16( &reduce_param );  
              } 
            },
            [&]() {},
            [&]() {});

          vnni_wt_loop(
            [&](int* ind) {
              int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];
              libxsmm_meltw_unary_param xform_param;
              xform_param.in.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), scratch_libxsmm_bf16_weights, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              wt_vnni_kernel( &xform_param );
            },
            [&]() {},
            [&]() {});
        } else if (compute_full_wt_output_block == 0) {
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              libxsmm_meltw_unary_param reduce_param;
              reduce_param.in.primary   = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm, i_n, 0, chunk0);
              reduce_param.out.primary  = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)filter_libxsmm, i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce_kernel0_bf16bf16( &reduce_param );
              } else {
                wt_reduce_kernel1_bf16bf16( &reduce_param );  
              } 
            },
            [&]() {},
            [&]() {});
        }
      }
    }
    
    if ( (sizeof(DType) == 2) && (bf16_use_chwn_format > 0)) {
      if (use_private_trans == 0) {
        if (logical_padding) {
          tr_input_loop(
            [&](int* ind) {
              int i_c = ind[0], i_h = ind[1], i_w = ind[2];
              libxsmm_meltw_unary_param zero_param;
              zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h, i_w, 0, 0, ifhp, ifwp, bc, bn);
              zero_kernel_chwn( &zero_param );

              if (i_h >= pad_h && i_h < ifhp - pad_h && i_w >= pad_w && i_w < ifwp - pad_w) {
                libxsmm_meltw_unary_param trans_param;
                trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_unpad_libxsmm, 0, i_c, i_h - pad_h, i_w - pad_w, 0, Cb, ifh, ifw, bc);
                trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h, i_w, 0, 0, ifhp, ifwp, bc, bn);
                trans_xform_kernel( &trans_param );
              }
            },
            [&]() {},
            [&]() {});
        } else {
          tr_input_loop(
            [&](int* ind) {
              int i_c = ind[0], i_h = ind[1], i_w = ind[2];
              libxsmm_meltw_unary_param trans_param;
              trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, 0, i_c, i_h, i_w, 0, Cb, ifhp, ifwp, bc);
              trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h, i_w, 0, 0, ifhp, ifwp, bc, bn);
              trans_xform_kernel( &trans_param );
            },
            [&]() {},
            [&]() {});
        }


        if (logical_padding) {
          tr_output_loop(
            [&](int* ind) {
              int i_k = ind[0], i_h = ind[1], i_w = ind[2];
              libxsmm_meltw_unary_param zero_param;
              zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h, i_w, 0, 0, ofhp, ofwp, bn, bk);
              zero_kernel_khwn( &zero_param );

              if (i_h >= pad_h && i_h < ofhp - pad_h && i_w >= pad_w && i_w < ofwp - pad_w) {
                libxsmm_meltw_unary_param trans_param;
                trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_unpad_libxsmm, 0, i_k, i_h - pad_h, i_w - pad_w, 0, Kb, ofh, ofw, bk);
                trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h, i_w, 0, 0, ofhp, ofwp, bn, bk);
                vnni_xform_kernel( &trans_param );
              }
            },
            [&]() {},
            [&]() {});
        } else {
          tr_output_loop(
            [&](int* ind) {
              int i_k = ind[0], i_h = ind[1], i_w = ind[2];
              libxsmm_meltw_unary_param trans_param;
              trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, 0, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h, i_w, 0, 0, ofhp, ofwp, bn, bk);
              vnni_xform_kernel( &trans_param );
            },
            [&]() {},
            [&]() {});
        }
      }

      if (par_over_h_pixels > 0) {
        zero_wt_loop(
          [&](int* ind) {
            int i_n = ind[0], i_k = ind[1], i_c = ind[2], i_r = ind[3], i_s = ind[4];
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, i_n, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
            zero_kernel( &zero_param );
          },
          [&]() {},
          [&]() {});
      }

      /* Zero out transpose tracker  */
      //TODO; use zero TPP here instead of memset
      if (use_private_trans > 0) {
        memset(trans_tracker, 0, trans_tracker_size*nThreads*sizeof(int));
      }

      conv_loop_bf16(
        [&](int* ind) {
          int i_c = ind[0], i_k = ind[1], i_h = ind[2], i_w = ind[3], i_r = ind[4], i_s = ind[5];
          int tid = omp_get_thread_num();
          libxsmm_gemm_param gemm_param;
          unsigned long long brcount = w_step*h_step;
          gemm_param.op.tertiary = (void*)&brcount;

          if (i_h == 0 && i_w == 0 && par_over_h_pixels == 0 && compute_full_wt_output_block == 0) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
            zero_kernel( &zero_param );
          }

          if (use_private_trans > 0) {
            int *inp_loc = (int*) trans_tracker + tid * trans_tracker_size + i_c;
            int *out_loc = (int*) trans_tracker + tid * trans_tracker_size + Cb + i_k;
            int is_inp_trans = *inp_loc;
            int is_out_trans = *out_loc;
            
            if (is_inp_trans == 0) {
              for (int _ih = 0; _ih < ifhp; _ih++) {
                for (int _iw = 0; _iw < ifwp; _iw++) {
                  libxsmm_meltw_unary_param trans_param;
                  trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, 0, i_c, _ih, _iw, 0, Cb, ifhp, ifwp, bc);
                  trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_input_libxsmm[tid], i_c, _ih, _iw, 0, 0, ifhp, ifwp, bc, bn);
                  trans_xform_kernel( &trans_param );
                }
              }
              *inp_loc = 1;
            }

            if (is_out_trans == 0) {
              for (int _ih = 0; _ih < ofhp; _ih++) {
                for (int _iw = 0; _iw < ofwp; _iw++) {
                  libxsmm_meltw_unary_param trans_param;
                  trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, 0, i_k, _ih, _iw, 0, Kb, ofhp, ofwp, bk);
                  trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_output_libxsmm[tid], i_k, _ih, _iw, 0, 0, ofhp, ofwp, bn, bk);
                  vnni_xform_kernel( &trans_param );
                }
              }     
              *out_loc = 1;
            }

            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_output_libxsmm[tid], i_k, i_h + pad_h_out, i_w + pad_w_out, 0, 0, ofhp, ofwp, bn, bk);
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_input_libxsmm[tid] , i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, 0, ifhp, ifwp, bc, bn);
          } else {            
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h + pad_h_out, i_w + pad_w_out, 0, 0, ofhp, ofwp, bn, bk);
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, 0, ifhp, ifwp, bc, bn);
          }

          if (compute_full_wt_output_block == 0) {
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);        
            brgemm_kernel_acc_pixel.gemm( &gemm_param );

            if ((i_h == ofh - h_step) && (i_w == ofw - w_step) && (par_over_h_pixels == 0)) {
              libxsmm_meltw_unary_param xform_param;
              xform_param.in.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              xform_param.out.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              fp32bf16_cvt_kernel( &xform_param );
              xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              wt_vnni_kernel( &xform_param );
            }
          } else {
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);        
            brgemm_kernel_acc_pixel_zerobeta_cvnni.gemm( &gemm_param );
          }

        },
        [&]() {if (sizeof(DType) == 2) tileconfig_kernel.gemm(NULL);},
        [&]() {if (sizeof(DType) == 2) tilerelease_kernel.gemm(NULL);});
  
      if (par_over_h_pixels > 0) {
        reduce_wt_loop(
          [&](int* ind) {
            int i_n = ind[0];
            libxsmm_meltw_unary_param reduce_param;
            reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(float), (float*)scratch_libxsmm, i_n, 0, chunk0);
            reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm_bf16_weights, i_n, 0, chunk0);
            if (i_n < reduce_work_tripcount - 1) {
              wt_reduce_kernel0_f32bf16( &reduce_param );
            } else {
              wt_reduce_kernel1_f32bf16( &reduce_param );  
            } 
          },
          [&]() {},
          [&]() {});

        vnni_wt_loop(
          [&](int* ind) {
            int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];
            libxsmm_meltw_unary_param xform_param;
            xform_param.in.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), scratch_libxsmm_bf16_weights, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
            xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
            wt_vnni_kernel( &xform_param );
          },
          [&]() {},
          [&]() {});
      }
    }
    if (i == n_iters) t_end = getTime();
  }

  // Check correctness if requested
  if (check_correctness) {
    if (sizeof(DType) == 2) {
      tensor_copy_KCRSck_vnni_to_norm_f32( (libxsmm_bfloat16*)filter_libxsmm, naive_filter_kcrsck, K, C, R, S, bc, bk);
      tensor_copy_KCRSck_to_KCRS( (float*)naive_filter_kcrsck, naive_filter_opt, K, C, R, S, bc, bk);
    } else {
      tensor_copy_KCRSck_to_KCRS( (float*)filter_libxsmm, naive_filter_opt, K, C, R, S, bc, bk);
    }
    printf("##########################################\n");
    printf("#           Correctness - UPD            #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, K*C*R*S, 1, naive_filter, naive_filter_opt, 0, 0);
    printf("L1 reference  : %.25g\n", norms.l1_ref);
    printf("L1 test       : %.25g\n", norms.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms.l2_rel);
    printf("Linf abs.error: %.24f\n", norms.linf_abs);
    printf("Linf rel.error: %.24f\n", norms.linf_rel);
    printf("Check-norm    : %.24f\n", norms.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms);

    /* "Random" tolerance is set */
    double tolerance = (sizeof(DType) == 2 ? 0.05 : 0.0001);

    if(norms.normf_rel > tolerance) {
      printf("Validation FAILED\n");
      error = -1;
    } else
      printf("Validation PASSED\n");
  }

  // Print performance/model numbers
  double gflop = (2.0*(double)n_iters*(double)N*(double)C*(double)K*(double)R*(double)S*(double)ofh*(double)ofw)/(1000*1000*1000);
  //printf("Compilation time is %.5g s\n", t1-t0);
  printf("GFLOPS %.6g %s\n", gflop/(t_end-t_start), loop_specs_str);
  printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, omp_get_max_threads(), N, C, K,
        H, W, R, S, stride_h, pad_h, pad_w, ((double)((t_end - t_start)/n_iters)), (gflop)/(t_end - t_start), norms.l1_ref, norms.l1_tst,
        norms.l2_abs, norms.l2_rel, norms.linf_abs, norms.linf_rel, norms.normf_rel);

  // Free buffers
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_nchwc);
  libxsmm_free(naive_input_unpad);
  libxsmm_free(naive_input_unpad_nchwc);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_nchwc);
  libxsmm_free(naive_output_unpad);
  libxsmm_free(naive_output_unpad_nchwc);
  libxsmm_free(naive_output_opt);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_filter_opt);
  libxsmm_free(naive_filter_kcrsck);
  libxsmm_free(input_libxsmm);
  libxsmm_free(input_unpad_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(output_unpad_libxsmm);
  libxsmm_free(filter_libxsmm);
  libxsmm_free(scratch_libxsmm);
  libxsmm_free(scratch_libxsmm_bf16_weights);
  libxsmm_free(A_offsets);
  libxsmm_free(B_offsets);
  return error;
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
    return conv_benchmark<float>(argc, argv);  
  } else {
    return conv_benchmark<libxsmm_bfloat16>(argc, argv);  
  }
}

