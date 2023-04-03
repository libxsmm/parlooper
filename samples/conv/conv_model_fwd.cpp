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
  long N = 14, H = 28, W = 28, C = 512, K = 1024, R = 1, S = 1, stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0, pad_h_in = 0, pad_w_in = 0, pad_h_out = 0, pad_w_out = 0;
  long bc = 32, bk = 32;
  long n_iters = 400;
  long i;
  long w_block = 1;
  long c_block = 1;
  long k_block = 1;
  long h_block = 1;
  long h_in_gemm = 1;
  long pack_input = 0;
  long logical_padding = 0;
  long input_padding_copy = 0;
  long zero_output_rims = 0;
  long with_bias = 0;
  long with_relu = 0;
  /* if there is parallelization over W, then zeroing rims (if requested) directly after brgemm will lead to multiple threads zeroing the same rim,
     which can be avoided by doing zeroing of the rims in a separate loop after the convolution (with extra perf cost potentially)  */
  long avoid_racey_zeroing_in_rims = 0; /* by default, zeroing of the rims (if requested through zero_output_rims > 0) is done inside the conv loop */
  // Setup model and trace
  ifreq = 1.0 / getFreq();
  std::vector<std::string> inp_trace[128];
  platform_spec_t my_platform;
  tensor_metadata_t tensor_metadata;
  set_platform_specs( CLX, omp_get_max_threads(), &my_platform);

  printf("argc = %d \n", argc);

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
      h_block  = atoi(argv[15]);
      w_block  = atoi(argv[16]);
      c_block  = atoi(argv[17]);
      k_block  = atoi(argv[18]);
      h_in_gemm  = atoi(argv[19]);
      pack_input = atoi(argv[20]);    
      if (argc > 21) {
        n_iters = atoi(argv[21]);
      }
      if (argc > 22) {
        logical_padding = atoi(argv[22]);
        if (argc > 23) {
          input_padding_copy = atoi(argv[23]);
        }
        if (argc > 27) {
          pad_h_in  = atoi(argv[24]);
          pad_w_in  = atoi(argv[25]);
          pad_h_out = atoi(argv[26]);
          pad_w_out = atoi(argv[27]);
        }
        if (argc > 28) {
          zero_output_rims = atoi(argv[28]);
        }
        if (argc > 30) {
          with_bias = atoi(argv[29]);
          with_relu = atoi(argv[30]);
        }
      }
    }
  }

  pad_h_in = (logical_padding == 0 ? pad_h : 0);
  pad_w_in = (logical_padding == 0 ? pad_w : 0);
  pad_h_out = (logical_padding == 0 ? pad_h : 0);
  pad_w_out = (logical_padding == 0 ? pad_w : 0);

  if ( (h_in_gemm > 1) && (w_block != 1) ) {
    printf("Invalid input GEMM config: When multiple H pixels are handled in the gemm, then the full ofw should be also used as gemm_n...\n");
    return -1;
  }

  if (logical_padding && h_in_gemm > 1 ) {
    printf("Error: logical padding is only supported for h_in_gemm = 1\n");
    return -1;
  }

  if (!logical_padding && input_padding_copy != 0) {
    printf("Error: input_padding_copy only makes sense for logical_padding enabled\n");
    return -1;
  }

  if ((pad_h_in != 0 || pad_w_in != 0 || pad_h_out != 0 || pad_w_out != 0) && logical_padding != 0) {
    printf("Error: Passed value of logical_padding = %d does not match the padding parameters: %d %d %d %d %d %d\n", logical_padding, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out);
    return -1;
  }

  long Kb = K/bk, Cb = C/bc;
  // Deriving some aux values
  long ofh = (H + 2 * pad_h - R) / stride_h + 1;
  long ofw = (W + 2 * pad_w - S) / stride_w + 1;
  long ifhp = H + 2 * pad_h_in;
  long ifwp = W + 2 * pad_w_in;
  long ofhp = ofh + 2 * pad_h_out;
  long ofwp = ofw + 2 * pad_w_out;
  long ifh = H;
  long ifw = W;

  /* used for always-physically-padded input copy when input_padding_copy != 0 */
  long ifhp_physically_padded = H + 2 * pad_h;
  long ifwp_physically_padded = W + 2 * pad_w;

  // Allocate buffers
  float *naive_input  = (float*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(float), 2097152);
  float *naive_input_nchwc  = (float*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(float), 2097152);
  float *naive_output = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_output_nchwc = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_filter = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  float *naive_filter_kcrsck = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  float *naive_bias = (float*)libxsmm_aligned_malloc( K*sizeof(float), 2097152);
  DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
  DType *scratch_input_libxsmm = (DType*)libxsmm_aligned_malloc( N*ifhp_physically_padded*ifwp_physically_padded*C*sizeof(DType), 2097152);
  DType *packed_input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ofh*ofw*C*sizeof(DType), 2097152);
  DType *output_libxsmm = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  DType *filter_libxsmm = (DType*)libxsmm_aligned_malloc( C*K*R*S*sizeof(DType), 2097152);
  DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
  DType *bias_libxsmm = (DType*)libxsmm_aligned_malloc( K*sizeof(DType), 2097152);
  unsigned long long *A_offsets = (unsigned long long*) libxsmm_aligned_malloc(Cb * R * S * sizeof(unsigned long long), 2097152);
  unsigned long long *B_offsets = (unsigned long long*) libxsmm_aligned_malloc(Cb * R * S * sizeof(unsigned long long), 2097152);

  libxsmm_matdiff_info norms, diff;
  libxsmm_matdiff_clear(&norms);
  libxsmm_matdiff_clear(&diff);

  // Init buffers
  float *naive_input_tmp = (float*)libxsmm_aligned_malloc( (size_t)N*C*ifhp*ifwp*sizeof(float), 2097152);
  init_buf(naive_input_tmp,          N*C*ifh*ifw, 0, 0);
  copy_internal_nchw( naive_input , naive_input_tmp, N, C, ifh, ifw, pad_h_in, pad_w_in);
  libxsmm_free(naive_input_tmp);
  set_zeropad_nchw(naive_input, N, C, ifhp, ifwp, pad_h_in, pad_w_in);

  init_buf(naive_output, N*K*ofhp*ofwp, 0, 0);

  init_buf(naive_filter,         K*C*R*S, 0, 0);

  init_buf(naive_bias,           K, 0, 0);
  
  if (sizeof(DType) == 2) {
    tensor_copy_NCHW_to_NCHWc (naive_input , naive_input_nchwc,  N, C, ifhp, ifwp, bc);
    tensor_copy_NCHW_to_NCHWc (naive_output, naive_output_nchwc, N, K, ofhp, ofwp, bk);
    tensor_cvt_copy_KCRS_to_KCRSck_bf16(naive_filter, (libxsmm_bfloat16*)filter_libxsmm, K, C, R, S, bc, bk);
    libxsmm_rne_convert_fp32_bf16( naive_input_nchwc,     (libxsmm_bfloat16*)input_libxsmm,     N*C*ifhp*ifwp );
    libxsmm_rne_convert_fp32_bf16( naive_output_nchwc,    (libxsmm_bfloat16*)output_libxsmm,    N*K*ofhp*ofwp );
    //libxsmm_rne_convert_fp32_bf16( naive_filter_kcrsck,   (libxsmm_bfloat16*)filter_libxsmm,    K*C*R*S );
    libxsmm_rne_convert_fp32_bf16( naive_bias,     (libxsmm_bfloat16*)bias_libxsmm,     K );
  } else {
    tensor_copy_NCHW_to_NCHWc (naive_input , (float*)input_libxsmm,  N, C, ifhp, ifwp, bc);
    tensor_copy_NCHW_to_NCHWc (naive_output, (float*)output_libxsmm, N, K, ofhp, ofwp, bk);
    tensor_copy_KCRS_to_KCRSck(naive_filter, (float*)filter_libxsmm, K, C, R, S, bc, bk);
    memcpy(bias_libxsmm, naive_bias, K * sizeof(float));
  }
  /* Note: have to zeropad reference output after copying (not after init) to keep random non-zero values in the rims of output_libxsmm */
  set_zeropad_nchw(naive_output, N, K, ofhp, ofwp, pad_h_out, pad_w_out);
  
  long avoid_rim_fmas = 0;
  if (ofh <= 7 && ofw <=7 && R == 3 && S == 3 && stride_w == 1 && stride_h == 1 && h_in_gemm == 1) {
    avoid_rim_fmas = 1;
  }

  if (logical_padding && !input_padding_copy)
    avoid_rim_fmas = 1;

  if (avoid_rim_fmas == 1 && (R == 1 || S == 1)) {
    printf("Error: avoid_rim_fmas does not work (and does not make sense) for 1x1 filters\n");
    return -1;
  }

  if (avoid_rim_fmas == 1 && ((R%2) == 0 || (S%2) == 0)) {
    printf("Error: avoid_rim_fmas does not work for even-sized filters\n");
    return -1;
  }

  if (avoid_rim_fmas == 1 && w_block != 1) {
    printf("Warning: w_block != 1 is not thoroughly tested with avoid_rim_fmas\n");
    //return -1;
  }

  if (R != 1 || S != 1) {
    pack_input = 0;
  }

  if (pack_input != 0 && input_padding_copy != 0) {
    printf("Error: input_padding_copy does not work with pack_input enabled\n");
    return -1;
  }

  if (pack_input == 0 && h_in_gemm > 1 && (pad_h_in != pad_h_out || pad_w_in != pad_w_out)) {
    printf("Error: h_in_gemm = %d > 1 does not work with different inout/output paddings when input is not packed (%d != %d or %d != %d)\n", h_in_gemm, pad_h_in, pad_h_out, pad_w_in, pad_w_out);
    return -1;
  }

  if ((pack_input == 1 && h_in_gemm > 1 && ((pad_h_in == 0 && pad_h_in != 0) || (pad_w_in == 0 && pad_w_out != 0)))) {
    printf("Error: h_in_gemm = %d > 1 does not work with zero input and non-zero output paddings when input is packed (%d != %d or %d != %d)\n", h_in_gemm, pad_h_in, pad_h_out, pad_w_in, pad_w_out);
    return -1;
  }

  long Cb_step = Cb/c_block;
  long n_step = 1;
  long c_step = Cb_step;
  long k_step = 1;
  long h_step = h_in_gemm;
  long w_step = ofw/w_block;
  long r_step = R;
  long s_step = S;

  if (avoid_rim_fmas == 1) {
    r_step = 1;
    s_step = 1;
  }

  printf("Test parameters: N H W C K R S stride_h stride_w  pad_h pad_w pad_h_in pad_w_in pad_h_out pad_w_out  bc bk zero_output_rims with_bias with_relu: %d  %d %d %d %d  %d %d %d %d  %d %d %d %d %d %d  %d %d  %d  %d %d\n",
          N, H, W, C, K,
          R, S, stride_h, stride_w,
          pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out,
          bc, bk,
          zero_output_rims,
          with_bias, with_relu);
  printf("Tuning parameters: h_block w_block c_block k_block h_in_gemm pack_input logical_padding input_padding_copy: %d %d %d %d %d %d %d %d\n",
          h_block, w_block, c_block, k_block, h_in_gemm, pack_input, logical_padding, input_padding_copy);
  printf("Tuning parameters: avoid_rim_fmas: %d\n", avoid_rim_fmas);
  printf("Tuning string: %s\n", loop_specs_str);
  printf("Extra parameters: avoid_racey_zeroing_in_rims: %d\n", avoid_racey_zeroing_in_rims);

  int has_W_parallelization = 0;
  for (size_t i = 0; i < strlen(loop_specs_str); i++) {
    if (loop_specs_str[i] == 'E') {
      has_W_parallelization++;
      break;
    }
  }

  if (has_W_parallelization && zero_output_rims && !avoid_racey_zeroing_in_rims) {
    printf("Warning: potentially racey zeroing of the rims will happen as zero_output_rims = %d,"
           " has_W_parallelization = %d and avoid_racey_zeroing_in_rims = %d (0 is default)\n",
           zero_output_rims, has_W_parallelization, has_W_parallelization, avoid_racey_zeroing_in_rims);
  }

  char zero_output_rims_loop_specs_str[256] = "abcd"; /* same parallelization over output as in the conv loop string except non-parallel W dimension*/
  if (has_W_parallelization) {
    std::string tmp_nkhw_string;
    for (size_t i = 0; i < strlen(loop_specs_str); i++) {
      if (loop_specs_str[i] == 'A' || loop_specs_str[i] == 'a')
        tmp_nkhw_string += loop_specs_str[i];
      if (loop_specs_str[i] == 'C' || loop_specs_str[i] == 'c')
        tmp_nkhw_string += loop_specs_str[i] - 1;
      if (loop_specs_str[i] == 'D' || loop_specs_str[i] == 'd')
        tmp_nkhw_string += 'c';
      if (loop_specs_str[i] == 'E' || loop_specs_str[i] == 'e')
        tmp_nkhw_string += loop_specs_str[i] - 1;
    }
    strcpy(zero_output_rims_loop_specs_str, tmp_nkhw_string.c_str());
  }

  
  // Setup TPP kernels
  auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  if (Cb_step == Cb && r_step == R && s_step == S)
    l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  if (sizeof(DType) == 2 && avoid_rim_fmas != 0) { /* enabling tile config and tile release within brgemm as this code path requires calling different brgemms */
    l_flags &= ~LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG;
    l_flags &= ~LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  }
  auto l_tc_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tr_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto dtype      = (sizeof(DType) == 2) ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;

  libxsmm_xmmfunction tileconfig_kernel;
  libxsmm_xmmfunction tilerelease_kernel;
  libxsmm_xmmfunction brgemm_kernel;
  libxsmm_xmmfunction brgemm_kernel_1less;
  libxsmm_xmmfunction brgemm_kernel_2less;
  libxsmm_xmmfunction brgemm_ext_kernel;
  libxsmm_meltwfunction_unary zero_kernel;
  libxsmm_meltwfunction_unary zero_padded_hwbc_kernel;
  libxsmm_meltwfunction_unary zero_hwpad_kernel;
  libxsmm_meltwfunction_unary zero_wpad_kernel;
  libxsmm_meltwfunction_unary copy_wbc_kernel;
  libxsmm_meltwfunction_unary input_pack_kernel;

  libxsmm_meltwfunction_binary colbias_add_kernel;
  libxsmm_meltwfunction_unary  relu_kernel;

  auto w_gemm_pixels = ofw/w_block;
  auto gemm_n = (w_gemm_pixels +  2 * pad_w) * (h_in_gemm - 2) + 2 * (w_gemm_pixels + pad_w);
  auto gemm_m = bk;
  auto gemm_k = bc;

  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, dtype, dtype, dtype);
  zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  l_unary_shape = libxsmm_create_meltw_unary_shape(ifhp_physically_padded*ifwp_physically_padded*bc, 1, ifhp_physically_padded*ifwp_physically_padded*bc, ifhp_physically_padded*ifwp_physically_padded*bc, dtype, dtype, dtype);
  zero_padded_hwbc_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  l_unary_shape = libxsmm_create_meltw_unary_shape(pad_h_out*ofwp*bk, 1, pad_h_out*ofwp*bk, pad_h_out*ofwp*bk, dtype, dtype, dtype);
  zero_hwpad_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  l_unary_shape = libxsmm_create_meltw_unary_shape(pad_w_out*bk, 1, pad_w_out*bk, pad_w_out*bk, dtype, dtype, dtype);
  zero_wpad_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  l_unary_shape = libxsmm_create_meltw_unary_shape(ifw*bc, 1, ifwp*bc, ifwp_physically_padded*bc, dtype, dtype, dtype);
  copy_wbc_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  if (avoid_rim_fmas == 1) {
    auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, R*S*bc*bk*sizeof(DType), bc*ifhp*ifwp*sizeof(DType), Cb_step );
    auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

    l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n-1, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    brgemm_kernel_1less.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n-2, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    brgemm_kernel_2less.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

    if (with_bias) {
      auto l_binary_shape = libxsmm_create_meltw_binary_shape(bk, w_gemm_pixels, bk, bk, bk, dtype, dtype, dtype, LIBXSMM_DATATYPE_F32);
      colbias_add_kernel = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_ADD, l_binary_shape, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0);
    }
    if (with_relu) {
#ifdef __x86_64__
      auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk, w_gemm_pixels, bk, bk, dtype, dtype, dtype);
#else
      auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk, w_gemm_pixels, bk, bk, dtype, dtype, LIBXSMM_DATATYPE_F32); /* there is no bf16 compute relu on non-x86 */
#endif
      relu_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    }
  } else if (R == 1 && S == 1) {
    auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;

    libxsmm_gemm_batch_reduce_config l_brconfig;
    memset( &l_brconfig, 0, sizeof(libxsmm_gemm_batch_reduce_config) );
    libxsmm_gemm_shape l_shape;
    memset( &l_shape, 0, sizeof(libxsmm_gemm_shape) );

    if (pack_input == 0) {
      l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
      l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, R*S*bc*bk*sizeof(DType), bc*ifhp*ifwp*sizeof(DType), Cb_step );
    } else {
      auto l_pack_shape = libxsmm_create_meltw_unary_shape(bc, w_gemm_pixels, bc*stride_w, bc, dtype, dtype, dtype);
      input_pack_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_pack_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
      l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc, bk, dtype, dtype, dtype, dtype );
      l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, R*S*bc*bk*sizeof(DType), bc*ofh*ofw*sizeof(DType), Cb_step );   
    }

    tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );

    brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

    if (with_bias || with_relu) {
      libxsmm_gemm_ext_unary_argops l_argops;
      memset( &l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops) );

      libxsmm_gemm_ext_binary_postops l_postops;
      memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

      if (with_bias)
        l_postops = libxsmm_create_gemm_ext_binary_postops(bk, dtype, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0);
      if (with_relu) {
        l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
        l_argops.ldcp           = l_shape.ldc;
      }
      brgemm_ext_kernel.gemm_ext  = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops );
    }

  } else {
    auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_OFFSET, 0, 0, 0 );
    tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );

    brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

    if (with_bias || with_relu) {
      libxsmm_gemm_ext_unary_argops l_argops;
      memset( &l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops) );

      libxsmm_gemm_ext_binary_postops l_postops;
      memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

      if (with_bias)
        l_postops = libxsmm_create_gemm_ext_binary_postops(bk, dtype, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0);
      if (with_relu) {
        l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
        l_argops.ldcp           = l_shape.ldc;
      }
      brgemm_ext_kernel.gemm_ext  = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops );
    }

    // Prepare offset array
    i = 0;
    for (long ifm = 0; ifm < Cb_step; ifm++) {
      for (long kj = 0; kj < R; kj++) {
        for (long ki = 0; ki < S; ki++) {
          A_offsets[i] = (ifm * R * S * bc * bk +
              kj * S * bc * bk +
              ki * bc * bk) * sizeof(DType);
          if (input_padding_copy)
            B_offsets[i] = (ifm * ifhp_physically_padded * ifwp_physically_padded * bc +
                kj * ifwp_physically_padded * bc +
                ki * bc) * sizeof(DType);
          else
            B_offsets[i] = (ifm * ifhp * ifwp * bc +
                kj * ifwp * bc +
                ki * bc) * sizeof(DType);
          i++;
        }
      }
    }
  }
  
  // Compute reference if requested
  if (check_correctness) {
    naive_conv_t naive_param;
    libxsmm_blasint naive_fuse_type = 0;
    if (with_bias) {
      if (with_relu)
        naive_fuse_type = 3;
      else
        naive_fuse_type = 1;
    }
    else if (with_relu)
      naive_fuse_type = 2;
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
    zero_buf(naive_output,    N*K*ofhp*ofwp);
    naive_fused_conv_fp(&naive_param, naive_input, naive_output, naive_filter, naive_bias, naive_fuse_type);
  }

  // JIT requested nested loop specs

  auto t0 = getTime();
  auto input_pad_loop = ThreadedLoop<2>({
      LoopSpecs{0, N, n_step, true},
      LoopSpecs{0, Cb, c_step}},
      "Ab");
  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, true},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, Kb, k_step, {k_block}},
      LoopSpecs{0, ofh, h_step, {h_block}},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      loop_specs_str);
  auto zero_output_rims_loop = ThreadedLoop<4>({
      LoopSpecs{0, N, n_step, true},
      LoopSpecs{0, Kb, k_step, {k_block}},
      LoopSpecs{0, ofh, h_step, {h_block}},
      LoopSpecs{0, ofw, w_step}},
      zero_output_rims_loop_specs_str);

  auto t1 = getTime();

  // benchmark the convolution
  double t_start, t_end;
  for (i = 0; i < n_iters+1; i++) {
    if (i == 1) t_start = getTime();
    if (input_padding_copy) {
      input_pad_loop(
        [&](int* ind) {
          int i_n = ind[0], i_c = ind[1];

          libxsmm_meltw_unary_param unary_param;

          unary_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), scratch_input_libxsmm, i_n, i_c, 0, 0, 0, Cb, ifhp_physically_padded, ifwp_physically_padded, bc);
          zero_padded_hwbc_kernel( &unary_param );

          for (int _i_h = pad_h; _i_h < ifhp_physically_padded - pad_h; _i_h++) {
            unary_param.in.primary  = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, _i_h - pad_h, 0, 0, Cb, ifhp, ifwp, bc);
            unary_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), scratch_input_libxsmm, i_n, i_c, _i_h, pad_w, 0, Cb, ifhp_physically_padded, ifwp_physically_padded, bc);
            copy_wbc_kernel( &unary_param );
          }
        },
        [&]() {if (sizeof(DType) == 2) {};},
        [&]() {if (sizeof(DType) == 2) {};});
    }
    conv_loop(
      [&](int* ind) {
        int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];
        if (avoid_rim_fmas == 0) {
          unsigned long long brcount = Cb_step * r_step * s_step;
          libxsmm_gemm_param gemm_param;
          libxsmm_gemm_ext_param  gemm_param_ext;
          gemm_param.op.tertiary = (void*)&brcount;
          gemm_param.a.secondary = (void*)A_offsets;
          gemm_param.b.secondary = (void*)B_offsets;
          gemm_param.a.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
          if (pack_input ==  0) {
            if (input_padding_copy)
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), scratch_input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp_physically_padded, ifwp_physically_padded, bc);
            else
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, pad_h_in - pad_h + i_h * stride_h + i_r, pad_w_in - pad_w + i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
          } else {
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), packed_input_libxsmm, i_n, i_c, i_h, i_w, 0, Cb, ofh, ofw, bc);     
          }
          gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
          
          if (pack_input > 0 && i_r == 0 && i_s == 0 && i_k == 0 && i_c == 0) {
            libxsmm_blasint _br, _h;
            for (_br = 0; _br < Cb; _br++) {
              for (_h = 0; _h < h_step; _h++) {
                libxsmm_meltw_unary_param pack_param;
                pack_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, _br, pad_h_in - pad_h + (i_h+_h) * stride_h, pad_w_in - pad_w + i_w * stride_w, 0, Cb, ifhp, ifwp, bc);
                pack_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), packed_input_libxsmm, i_n, _br, i_h+_h, i_w, 0, Cb, ofh, ofw, bc);
                input_pack_kernel( &pack_param );
              }
            }
          } 

          if (Cb_step != Cb || r_step != R || s_step != S) {
            if (i_c == 0 && i_r == 0 && i_s == 0) {
              libxsmm_meltw_unary_param zero_param;
              zero_param.out.primary = (void*)gemm_param.c.primary;
              zero_kernel( &zero_param );
            }
          }

          if ((with_bias || with_relu) && i_c == Cb - c_step) {
            gemm_param_ext.op.tertiary = gemm_param.op.tertiary;
            gemm_param_ext.a.secondary = gemm_param.a.secondary;
            gemm_param_ext.b.secondary = gemm_param.b.secondary;
            gemm_param_ext.a.primary = gemm_param.a.primary;
            gemm_param_ext.b.primary = gemm_param.b.primary;
            gemm_param_ext.c.primary = gemm_param.c.primary;
            if (with_bias)
              gemm_param_ext.d.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), bias_libxsmm, i_k, 0, bk);

            brgemm_ext_kernel.gemm_ext( &gemm_param_ext );
          } else {
            brgemm_kernel.gemm( &gemm_param );
          }
        } else { /* for avoid_rim_fmas dispatch */
          unsigned long long brcount = Cb_step;
          libxsmm_gemm_param gemm_param;
          gemm_param.op.tertiary = (void*)&brcount;
          gemm_param.a.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
          if (i_c == 0 && i_r == 0 && i_s == 0) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
            zero_kernel( &zero_param );
          }
          if (R == 7 && S == 7) {
            if (i_h * stride_h + i_r - R/2 < 0) {
              /* Do no FLOPS  */
            } else if (i_h *stride_h + i_r - R/2 >= ifh ) {
              /* Do no FLOPS  */
            } else if ( i_s < S/2 && i_w * stride_w + (i_s - S/2) < 0 && (i_w + 1) * stride_w + (i_s - S/2) >= 0  ) {
              // the case when left i_s is out of input image for the first pitch only
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, pad_h_in + i_h * stride_h + (i_r - R/2) , pad_w_in + (i_w + 1) * stride_w + (i_s - S/2) , 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w + 1, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel_1less.gemm( &gemm_param );
            } else if ( i_s < S/2 && i_w * stride_w + (i_s - S/2) < 0 && (i_w + 1) * stride_w + (i_s - S/2) < 0 && (i_w + 2) * stride_w + (i_s - S/2) >= 0  ) {
              // the case when left i_s is out of input image for the first two pitches
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, pad_h_in + i_h * stride_h + (i_r - R/2) , pad_w_in + (i_w + 2) * stride_w + (i_s - S/2) , 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w + 2, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel_2less.gemm( &gemm_param );
            } else if ( i_s > S/2 && (i_w + w_step - 1)*stride_w + (i_s - S/2) >= ifw && (i_w + w_step - 2)*stride_w + (i_s - S/2) < ifw ) {
              // the case when right i_s is out of input image for the last pitch only
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, pad_h_in + i_h * stride_h + (i_r - R/2) , pad_w_in + i_w * stride_w + (i_s - S/2) , 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel_1less.gemm( &gemm_param );
            } else if ( i_s > S/2 && (i_w + w_step - 1)*stride_w + (i_s - S/2) >= ifw && (i_w + w_step - 2)*stride_w + (i_s - S/2) >= ifw && (i_w + w_step - 3)*stride_w + (i_s - S/2) < ifw ) {
              // for the case when right i_s is out of input image for the last 2 pitches
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, pad_h_in + i_h * stride_h + (i_r - R/2) , pad_w_in + i_w * stride_w + (i_s - S/2) , 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel_2less.gemm( &gemm_param );
            } else {
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, pad_h_in + i_h * stride_h + (i_r - R/2) , pad_w_in + i_w * stride_w + (i_s - S/2) , 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel.gemm( &gemm_param );
            }
          } else if (R == 3 && S == 3) { /* works for 3x3 stride-1 and stride-2 convolutions */
            if (i_r == 0 && i_h == 0) {
              /* Do no FLOPS  */
            } else if (i_r == R-1 && (i_h + h_step - 1)*stride_h + i_r == ifh + 1 ) {
              /* Do no FLOPS  */
            } else if ( i_w == 0 && i_s == 0 ) {
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, pad_h_in + i_h * stride_h + (i_r - R/2), pad_w_in + (i_w + 1) * stride_w + (i_s - S/2), 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w + 1, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel_1less.gemm( &gemm_param );
            //} else if ( i_w + w_step == ofw  && i_s == S-1) {
            } else if ( (i_w + w_step - 1)*stride_w + i_s == ifw + 1 && i_s == S-1) {
              //exit(-1);
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, pad_h_in + i_h * stride_h + (i_r - R/2), pad_w_in + i_w * stride_w + (i_s - S/2), 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel_1less.gemm( &gemm_param );

            } else {
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, pad_h_in + i_h * stride_h + (i_r - R/2), pad_w_in + i_w * stride_w + (i_s - S/2), 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel.gemm( &gemm_param );
            }
          }

          if ((with_bias || with_relu) && i_r == R - r_step && i_s == S - s_step && i_c == Cb - c_step) {
            if (with_bias) {
              libxsmm_meltw_binary_param binary_param;
              binary_param.in0.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), bias_libxsmm, i_k, 0, bk);
              binary_param.in1.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              binary_param.out.primary = binary_param.in1.primary;
              colbias_add_kernel( &binary_param );
            }
            if (with_relu) {
              libxsmm_meltw_unary_param unary_param;
              unary_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              unary_param.out.primary = unary_param.in.primary;
              relu_kernel( &unary_param );
            }
          } /* handling the fusion of bias/relu */
        } /* end of if-else around avoid_rim_fmas */

        if ((!has_W_parallelization || !avoid_racey_zeroing_in_rims) && zero_output_rims) {
          libxsmm_meltw_unary_param zero_param;
          if (i_c == Cb - c_step && i_r == R - r_step && i_s == S - s_step) {
            if (i_h == 0) {
              zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, 0, 0, 0, Kb, ofhp, ofwp, bk);
              zero_hwpad_kernel( &zero_param );
            }
            if ( i_h == ofh - h_step) {
              zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, pad_h_out + ofh, 0, 0, Kb, ofhp, ofwp, bk);
              zero_hwpad_kernel( &zero_param );
            }
            if (i_w == 0) {
              for (int _h = 0; _h < h_step; _h++) {
                zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, pad_h_out + i_h + _h, 0, 0, Kb, ofhp, ofwp, bk);
                zero_wpad_kernel( &zero_param );
              }
            }
            if (i_w == ofw - w_step) {
              for (int _h = 0; _h < h_step; _h++) {
                zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, pad_h_out + i_h + _h, pad_w_out + ofw, 0, Kb, ofhp, ofwp, bk);
                zero_wpad_kernel( &zero_param );
              }
            }
          }
        } /* for if (zero_output_rims) */

      },
      [&]() {if (sizeof(DType) == 2 && avoid_rim_fmas == 0) tileconfig_kernel.gemm(NULL);},
      [&]() {if (sizeof(DType) == 2 && avoid_rim_fmas == 0) tilerelease_kernel.gemm(NULL);});

      if (avoid_racey_zeroing_in_rims && has_W_parallelization && zero_output_rims) {
        zero_output_rims_loop(
          [&] (int * ind) {
            int i_n = ind[0], i_k = ind[1], i_h = ind[2], i_w = ind[3];

            libxsmm_meltw_unary_param zero_param;

            if (i_h == 0) {
              zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, 0, 0, 0, Kb, ofhp, ofwp, bk);
              zero_hwpad_kernel( &zero_param );
            }
            if ( i_h == ofh - h_step) {
              zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, pad_h_out + ofh, 0, 0, Kb, ofhp, ofwp, bk);
              zero_hwpad_kernel( &zero_param );
            }
            if (i_w == 0) {
              for (int _h = 0; _h < h_step; _h++) {
                zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, pad_h_out + i_h + _h, 0, 0, Kb, ofhp, ofwp, bk);
                zero_wpad_kernel( &zero_param );
              }
            }
            if (i_w == ofw - w_step) {
              for (int _h = 0; _h < h_step; _h++) {
                zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, pad_h_out + i_h + _h, pad_w_out + ofw, 0, Kb, ofhp, ofwp, bk);
                zero_wpad_kernel( &zero_param );
              }
            }

          },
          [&]() {},
          [&]() {});
      }

    if (i == n_iters) t_end = getTime();
  }
  
  // Check correctness if requested
  if (check_correctness) {
    if (sizeof(DType) == 2) {
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)output_libxsmm, naive_output_nchwc, N*K*ofhp*ofwp );
      tensor_copy_NCHWc_to_NCHW (naive_output_nchwc, naive_output_opt, N, K, ofhp, ofwp, bk);
    } else {
      tensor_copy_NCHWc_to_NCHW ((float*)output_libxsmm, naive_output_opt, N, K, ofhp, ofwp, bk);
    }
    /* If output was physically padded and we don't request to zero the output rims within the convolution code, then we zero out the rims here */
    if (!zero_output_rims && (pad_h_out > 0 || pad_w_out > 0)) {
      set_zeropad_nchw(naive_output_opt, N, K, ofhp, ofwp, pad_h_out, pad_w_out);
    }
    printf("##########################################\n");
    printf("#           Correctness - FWD            #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*K*ofhp*ofwp, 1, naive_output, naive_output_opt, 0, 0);
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
  printf("GFLOPS %.6g %s_hb=%d_wb=%d_cb=%d_kb=%d_hg=%d_pu=%d\n", gflop/((double)(t_end-t_start)), loop_specs_str, h_block, w_block, c_block, k_block, h_in_gemm, pack_input);
  printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, omp_get_max_threads(), N, C, K,
        H, W, R, S, stride_h, pad_h, pad_w, ((double)((t_end - t_start)/n_iters)), (gflop)/(t_end - t_start), norms.l1_ref, norms.l1_tst,
        norms.l2_abs, norms.l2_rel, norms.linf_abs, norms.linf_rel, norms.normf_rel);

  // Free buffers
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_nchwc);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_nchwc);
  libxsmm_free(naive_output_opt);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_filter_kcrsck);
  libxsmm_free(naive_bias);
  libxsmm_free(input_libxsmm);
  libxsmm_free(scratch_input_libxsmm);
  libxsmm_free(packed_input_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(filter_libxsmm);
  libxsmm_free(bias_libxsmm);
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

