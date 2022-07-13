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
int conv_benchmark(int argc, char** argv) {
  int error = 0;
  // Setup default GEMM sizes
  int check_correctness = 1;
  char loop_specs_str[256] = "aBC";  
  long N = 14, H = 28, W = 28, C = 512, K = 1024, R = 1, S = 1, stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0;
  long bc = 32, bk = 32;
  long w_block = 1;
  long c_block = 1;
  long k_block = 1;
  long h_block = 1;
  long h_in_gemm = 1;
  long n_iters = 1;
  long i;
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
      h_block  = atoi(argv[15]);
      w_block  = atoi(argv[16]);
      c_block  = atoi(argv[17]);
      k_block  = atoi(argv[18]);
      h_in_gemm  = atoi(argv[19]);
      if (argc > 20) {
        n_iters = atoi(argv[20]);
      }
    }
  }

  printf("Test parameters: N H W C K R S stride_h stride_w pad_h pad_w bc bk: %d %d %d %d %d %d %d %d %d %d %d %d %d \n", N, H, W, C, K, R, S, stride_h, stride_w, pad_h, pad_w, bc, bk);
  printf("Tuning parameters: h_block w_block c_block k_block h_in_gemm: %d %d %d %d %d \n", h_block, w_block, c_block, k_block, h_in_gemm);

  if ( (h_in_gemm > 1) && (w_block != 1) ) {
    printf("Invalid input GEMM config: When multiple H pixels are handled in the gemm, then the full ofw should be also used as gemm_n...\n");
    return 0;
  }

  if ( (h_in_gemm > 1) && (stride_h != 1) ) {
    printf("Invalid input GEMM config: When multiple H pixels are handled in the gemm, stride must be one...\n");
    return 0;
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

  // Allocate buffers
  float *naive_input  = (float*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(float), 2097152);
  float *naive_input_nchwc  = (float*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(float), 2097152);
  float *naive_output = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_output_nchwc = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_input_opt = (float*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(float), 2097152);
  float *naive_filter = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  float *naive_filter_kcrsck = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
  DType *output_libxsmm = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  DType *filter_libxsmm = (DType*)libxsmm_aligned_malloc( C*K*R*S*sizeof(DType), 2097152);
  DType *tr_filter_libxsmm = (DType*)libxsmm_aligned_malloc( C*K*R*S*sizeof(DType), 2097152);
  DType *input_libxsmm_off= (DType*)input_libxsmm + (size_t) (pad_h_in * ifwp * bc + pad_w_in * bc);
  DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
  unsigned long long *A_offsets = (unsigned long long*) libxsmm_aligned_malloc(Kb * R * S * sizeof(unsigned long long), 2097152);
  unsigned long long *B_offsets = (unsigned long long*) libxsmm_aligned_malloc(Kb * R * S * sizeof(unsigned long long), 2097152);

  libxsmm_matdiff_info norms, diff;
  libxsmm_matdiff_clear(&norms);
  libxsmm_matdiff_clear(&diff);

  // Init buffers
  float *naive_input_tmp = (float*)libxsmm_aligned_malloc( (size_t)N*C*ifhp*ifwp*sizeof(float), 2097152);
  init_buf(naive_input_tmp,          N*C*ifh*ifw, 0, 0);
  copy_internal_nchw( naive_input , naive_input_tmp, N, C, ifh, ifw, pad_h, pad_w);
  libxsmm_free(naive_input_tmp);
  set_zeropad_nchw(naive_input, N, C, ifhp, ifwp, pad_h_in, pad_w_in);
  init_buf(naive_output,         N*K*ofwp*ofhp, 0, 0);
  set_zeropad_nchw(naive_output, N, K, ofhp, ofwp, pad_h_out, pad_w_out);
  init_buf(naive_filter,         K*C*R*S, 0, 0);
  
  if (sizeof(DType) == 2) {
    tensor_copy_NCHW_to_NCHWc (naive_input , naive_input_nchwc,  N, C, ifhp, ifwp, bc);
    tensor_copy_NCHW_to_NCHWc (naive_output, naive_output_nchwc, N, K, ofhp, ofwp, bk);
    tensor_copy_KCRS_to_KCRSck_bf16(naive_filter, (libxsmm_bfloat16*)filter_libxsmm, K, C, R, S, bc, bk);
    libxsmm_rne_convert_fp32_bf16( naive_input_nchwc,     (libxsmm_bfloat16*)input_libxsmm,     N*C*ifhp*ifwp );
    libxsmm_rne_convert_fp32_bf16( naive_output_nchwc,    (libxsmm_bfloat16*)output_libxsmm,    N*K*ofhp*ofwp );
    //libxsmm_rne_convert_fp32_bf16( naive_filter_kcrsck,   (libxsmm_bfloat16*)filter_libxsmm,    K*C*R*S );
  } else {
    tensor_copy_NCHW_to_NCHWc (naive_input , (float*)input_libxsmm,  N, C, ifhp, ifwp, bc);
    tensor_copy_NCHW_to_NCHWc (naive_output, (float*)output_libxsmm, N, K, ofhp, ofwp, bk);
    tensor_copy_KCRS_to_KCRSck(naive_filter, (float*)filter_libxsmm, K, C, R, S, bc, bk);
  }
  
  // Setup TPP kernels
  auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tc_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tr_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto dtype      = (sizeof(DType) == 2) ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;

  libxsmm_xmmfunction tileconfig_kernel;
  libxsmm_xmmfunction tilerelease_kernel;
  libxsmm_xmmfunction brgemm_kernel;
  libxsmm_xmmfunction brgemm_kernel2;
  libxsmm_meltwfunction_unary zero_kernel;
  libxsmm_meltwfunction_unary zero_kernel_bc;
  libxsmm_meltwfunction_unary zero_kernel_all_pixels;
  libxsmm_meltwfunction_unary wt_trans_kernel;
  
  auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bc, dtype, dtype, dtype);
  if (dtype == LIBXSMM_DATATYPE_F32) {
    wt_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  } else {
    wt_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  }

  long Kb_step = Kb/k_block;
  long avoid_rim_fmas = 0;
  long non_1x1_with_strides = 0;
  if (ofh <= 7 && ofw <=7 && R == 3 && S == 3 && stride_w == 1 && stride_h == 1) {
    avoid_rim_fmas = 1;
  }

  if ((R != 1 && stride_h != 1) ||
      (S != 1 && stride_w != 1)) {
    non_1x1_with_strides = 1;
  }

  printf("Algorithm decisions: avoid_rim_fmas non_1x1_with-strides: %d %d \n", avoid_rim_fmas, non_1x1_with_strides);

  if ((R == 1 && S == 1) ||
      (avoid_rim_fmas == 1) ||
      (non_1x1_with_strides == 1)) {
    auto w_gemm_pixels = ofw/w_block;
    auto gemm_n = (w_gemm_pixels +  2 * pad_w) * (h_in_gemm - 2) + 2 * (w_gemm_pixels + pad_w);
    auto w_zero_pixels = ifw/w_block;
    auto zero_n = (w_zero_pixels +  2 * pad_w) * (h_in_gemm - 2) + 2 * (w_zero_pixels + pad_w);
    //auto gemm_n = ofw;
    auto gemm_m = bc;
    auto gemm_k = bk;
    auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bc, bk, stride_w*bc, dtype, dtype, dtype, dtype );
    auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, R*S*bc*bk*sizeof(DType), bk*ofhp*ofwp*sizeof(DType), Kb_step );
    auto l_unary_shape = libxsmm_create_meltw_unary_shape(bc*zero_n, 1, bc*zero_n, bc*zero_n, dtype, dtype, dtype);
    zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    l_unary_shape = libxsmm_create_meltw_unary_shape(bc*ifwp*ifhp, 1, bc*ifwp*ifhp, bc*ifwp*ifhp, dtype, dtype, dtype);
    zero_kernel_all_pixels = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
    l_unary_shape = libxsmm_create_meltw_unary_shape(bc, 1, bc, bc, dtype, dtype, dtype);
    zero_kernel_bc = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
    tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
    brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n-1, gemm_k, bc, bk, stride_w*bc, dtype, dtype, dtype, dtype );
    brgemm_kernel2.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  } else {
    auto w_gemm_pixels = ofw/w_block;
    auto gemm_n = (w_gemm_pixels +  2 * pad_w) * (h_in_gemm - 2) + 2 * (w_gemm_pixels + pad_w);
    //auto gemm_n = ofw;
    auto w_zero_pixels = ifw/w_block;
    auto zero_n = (w_zero_pixels +  2 * pad_w) * (h_in_gemm - 2) + 2 * (w_zero_pixels + pad_w);
    auto gemm_m = bc;
    auto gemm_k = bk;
    auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bc, bk, stride_w*bc, dtype, dtype, dtype, dtype );
    auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_OFFSET, 0, 0, 0 );
    auto l_unary_shape = libxsmm_create_meltw_unary_shape(bc*zero_n, 1, bc*zero_n, bc*zero_n, dtype, dtype, dtype);
    zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    l_unary_shape = libxsmm_create_meltw_unary_shape(bc*ifwp*ifhp, 1, bc*ifwp*ifhp, bc*ifwp*ifhp, dtype, dtype, dtype);
    zero_kernel_all_pixels = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);   
    tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
    brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    // Prepare offset array
    i = 0;
    for (long ifm = 0; ifm < Kb_step; ifm++) {
      for (long kj = 0; kj < R; kj++) {
        for (long ki = 0; ki < S; ki++) {
          A_offsets[i] = (ifm * R * S * bc * bk +
              kj * S * bc * bk +
              ki * bc * bk) * sizeof(DType);
          B_offsets[i] = (ifm * ofhp * ofwp * bk +
              kj * ofwp * bk +
              ki * bk) * sizeof(DType);
          i++;
        }
      }
    }
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
    zero_buf(naive_input,    N*C*ifhp*ifwp);
    naive_conv_bp(&naive_param, naive_input, naive_output, naive_filter, NULL); 
  }

  // JIT requested nested loop specs
  long n_step = 1;
  long c_step = 1;
  long k_step = Kb_step;
  long h_step = h_in_gemm;
  long w_step = ofw/w_block;
  long r_step = R;
  long s_step = S;

  if ((avoid_rim_fmas == 1) || (non_1x1_with_strides == 1)) {
    r_step = 1;
    s_step = 1;
  }

  auto t0 = getTime();
  auto wt_trans_loop = ThreadedLoop<4>({
      LoopSpecs{0, Kb, 1, true},
      LoopSpecs{0, Cb, 1, true},
      LoopSpecs{0, R, 1, true},
      LoopSpecs{0, S, 1, true}},
      "ABCD");

  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, true},
      LoopSpecs{0, Cb, c_step, {c_block}},
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, ofh, h_step, {h_block}},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      loop_specs_str);
  auto t1 = getTime();

  // benchmark the convolution
  double t_start, t_end;
  for (i = 0; i < n_iters + 1; i++) {
    if (i == 1) t_start = getTime(); 
    wt_trans_loop(
      [&](int* ind) {
        int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];
        libxsmm_meltw_unary_param trans_param;
        trans_param.in.primary  = LIBXSMM_ACCESS_RAW(6, sizeof(DType),    filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
        trans_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), tr_filter_libxsmm, i_c, i_k, R-1-i_r, S-1-i_s, 0, 0, Kb, R, S, bk, bc);
        wt_trans_kernel(&trans_param);
      },
      [&]() {},
      [&]() {});

    conv_loop(
      [&](int* ind) {
        int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];
        if (avoid_rim_fmas == 0) {
          if (non_1x1_with_strides == 0) {
            unsigned long long brcount = Kb_step * r_step * s_step;
            libxsmm_gemm_param gemm_param;
            gemm_param.op.tertiary = (void*)&brcount;
            gemm_param.a.secondary = (void*)A_offsets;
            gemm_param.b.secondary = (void*)B_offsets;      
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), tr_filter_libxsmm, i_c, i_k, i_r, i_s, 0, 0, Kb, R, S, bk, bc);
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);  
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc); 
            if (i_k == 0 && i_r == 0 && i_s == 0) {
              if (stride_h != 1) {
                if (i_w == 0 && i_h == 0) {
                  libxsmm_meltw_unary_param zero_param;
                  zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, 0, 0, 0, Cb, ifhp, ifwp, bc);
                  zero_kernel_all_pixels( &zero_param );
                }
              } else {
                libxsmm_meltw_unary_param zero_param;
                zero_param.out.primary = (void*)gemm_param.c.primary;
                zero_kernel( &zero_param );
              }
            }
            brgemm_kernel.gemm( &gemm_param );
          } else {
            unsigned long long brcount = Kb_step * r_step * s_step;
            libxsmm_gemm_param gemm_param;
            gemm_param.op.tertiary = (void*)&brcount;
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), tr_filter_libxsmm, i_c, i_k, R-1-i_r, S-1-i_s, 0, 0, Kb, R, S, bk, bc);
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);  
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc); 
            if (i_k == 0 && i_r == 0 && i_s == 0) {
              if (stride_h != 1) {
                if (i_w == 0 && i_h == 0) {
                  libxsmm_meltw_unary_param zero_param;
                  zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, 0, 0, 0, Cb, ifhp, ifwp, bc);
                  zero_kernel_all_pixels( &zero_param );
                }
              } else {
                libxsmm_meltw_unary_param zero_param;
                zero_param.out.primary = (void*)gemm_param.c.primary;
                zero_kernel( &zero_param );
              }
            }
            brgemm_kernel.gemm( &gemm_param );

            /* Zero Rim..  */
            if (i_r == R-1 && i_s == S-1 && i_h == ofh-h_step && i_w == ofw-w_step && i_k == Kb - Kb_step) {
              for (int ij = 0; ij < ifhp; ij++) {
                for (int ii = 0; ii < ifwp; ii++) {
                  if ((ij < pad_h_in || ij >= ifh + pad_h_in) || 
                      (ii < pad_w_in || ii >= ifw + pad_w_in)) {
                    libxsmm_meltw_unary_param zero_param;
                    zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, ij, ii, 0, Cb, ifhp, ifwp, bc);
                    zero_kernel_bc( &zero_param );
                  }
                }
              } 
            } 
          }
        } else {
          unsigned long long brcount = Kb_step;
          libxsmm_gemm_param gemm_param;
          gemm_param.op.tertiary = (void*)&brcount;
          gemm_param.a.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), tr_filter_libxsmm, i_c, i_k, i_r, i_s, 0, 0, Kb, R, S, bk, bc);      
          if (i_k == 0 && i_r == 0 && i_s == 0) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h * stride_h, i_w * stride_w , 0, Cb, ifhp, ifwp, bc);
            zero_kernel( &zero_param );
          }
          if (i_r == 0 && i_h == 0) {
            /* Do no FLOPS  */
          } else if (i_r == R-1 && i_h == ofh-1 ) {
            /* Do no FLOPS  */
          } else if ( i_w == 0 && i_s == 0 ) {
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, i_h + i_r, i_w + i_s + 1, 0, Kb, ofhp, ofwp, bk);
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h, i_w + 1, 0, Cb, ifhp, ifwp, bc);       
            brgemm_kernel2.gemm( &gemm_param );
          } else if ( i_w + w_step == ofw  && i_s == S-1) {
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, i_h + i_r, i_w + i_s, 0, Kb, ofhp, ofwp, bk);
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h, i_w, 0, Cb, ifhp, ifwp, bk);
            brgemm_kernel2.gemm( &gemm_param );
          } else {
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, i_h + i_r, i_w + i_s, 0, Kb, ofhp, ofwp, bk);
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h, i_w, 0, Cb, ifhp, ifwp, bc);    
            brgemm_kernel.gemm( &gemm_param );
          }
        }
      },
      [&]() {if (sizeof(DType) == 2) tileconfig_kernel.gemm(NULL);},
      [&]() {if (sizeof(DType) == 2) tilerelease_kernel.gemm(NULL);});
    
    if (i == n_iters) t_end = getTime(); 
  }
  
  // Check correctness if requested
  if (check_correctness) {
    if (sizeof(DType) == 2) {
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)input_libxsmm, naive_input_nchwc, N*C*ifhp*ifwp );
      tensor_copy_NCHWc_to_NCHW (naive_input_nchwc, naive_input_opt, N, C, ifhp, ifwp, bc);
    } else {
      tensor_copy_NCHWc_to_NCHW ((float*)input_libxsmm, naive_input_opt, N, C, ifhp, ifwp, bc);
    }
    printf("##########################################\n");
    printf("#           Correctness - BWD            #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*C*ifhp*ifwp, 1, naive_input, naive_input_opt, 0, 0);
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
  printf("GFLOPS %.6g %s_hb=%d_wb=%d_cb=%d_kb=%d\n", gflop/(t_end-t_start), loop_specs_str, h_block, w_block, c_block, k_block);

  // Free buffers
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_nchwc);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_nchwc);
  libxsmm_free(naive_input_opt);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_filter_kcrsck);
  libxsmm_free(input_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(filter_libxsmm);
  libxsmm_free(tr_filter_libxsmm);
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

