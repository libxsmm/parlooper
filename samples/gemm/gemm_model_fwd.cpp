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

/*
 *
 *   A: [brcount][bk/2][bm][2]
 * Scf: [brcount][bk/32][bm]
 * Out: [brcount][bk/2][bm][2]
 *
 */
void cvt_kernel_mxfp4(unsigned char *in_ptr, unsigned char *scf_ptr, libxsmm_bfloat16 *out_ptr, long brcount, long bm, long bk) {
  long i_br, i_k, i_bk, i_m;
  __m512i zero_vreg = _mm512_setzero_si512();
  __m512i ones_vreg = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  __m512i lut_vreg = _mm512_set_epi16(33216, 33152, 33088, 33024, 32960, 32896, 32768, 1, 448, 384, 320, 256, 192, 128, 0, 1, 33216, 33152, 33088, 33024, 32960, 32896, 32768, 1, 448, 384, 320, 256, 192, 128, 0, 1);
  __m512i vnni_perm_reg_lo = _mm512_set_epi16(47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41, 9, 40, 8, 39, 7, 38, 6, 37, 5, 36, 4, 35, 3, 34, 2, 33, 1, 32, 0);
  __m512i vnni_perm_reg_hi = _mm512_set_epi16(63, 31, 62, 30, 61, 29, 60, 28, 59, 27, 58, 26, 57, 25, 56, 24, 55, 23, 54, 22, 53, 21, 52, 20, 51, 19, 50, 18, 49, 17, 48, 16);
  /* batch reduce iteration */
  for (i_br = 0; i_br < brcount; i_br++) {
    for (i_bk = 0; i_bk < bk/32; i_bk++) {
      for (i_m = 0; i_m < bm/32; i_m++) {
        __m512i scale_vreg = _mm512_cvtepu8_epi16(_mm256_loadu_epi16((unsigned char*)scf_ptr + (i_br * (bk/32) * bm + i_bk * bm + i_m * 32)));
        __mmask32 mask_scf_zero =  _mm512_cmpeq_epi16_mask(scale_vreg, zero_vreg);    
        _mm_prefetch ((unsigned char*)scf_ptr + ((i_br+1) * (bk/32) * bm + i_bk * bm + i_m * 32), _MM_HINT_T0);     
        scale_vreg  = _mm512_sub_epi16(scale_vreg, ones_vreg);
        scale_vreg  = _mm512_slli_epi16(scale_vreg, 7);
        for (i_k = 0; i_k < 16; i_k++) {
          __mmask32 mask_vreg_zero_k0, mask_vreg_zero_k1;
          __m512i out_vreg_16m0_2k, out_vreg_16m1_2k;     
          __m512i input_vreg_k0 = _mm512_cvtepu8_epi16(_mm256_loadu_epi16((unsigned char*)in_ptr + (i_br * (bk/2) * bm + (i_bk * 16 + i_k) * bm + i_m * 32)));
          __m512i input_vreg_k1 = _mm512_srli_epi16(input_vreg_k0, 4);
          _mm_prefetch ((unsigned char*)in_ptr + ((i_br+1) * (bk/2) * bm + (i_bk * 16 + i_k) * bm + i_m * 32), _MM_HINT_T0);   
          input_vreg_k0  = _mm512_permutexvar_epi16(input_vreg_k0, lut_vreg);
          mask_vreg_zero_k0 =  _mm512_cmpeq_epi16_mask(input_vreg_k0, ones_vreg);
          mask_vreg_zero_k0 = _kor_mask32(mask_vreg_zero_k0, mask_scf_zero);
          input_vreg_k0 = _mm512_add_epi16(input_vreg_k0, scale_vreg);
          input_vreg_k0 = _mm512_mask_blend_epi16(mask_vreg_zero_k0, input_vreg_k0, zero_vreg);
          input_vreg_k1  = _mm512_permutexvar_epi16(input_vreg_k1, lut_vreg);
          mask_vreg_zero_k1 =  _mm512_cmpeq_epi16_mask(input_vreg_k1, ones_vreg);
          mask_vreg_zero_k1 = _kor_mask32(mask_vreg_zero_k1, mask_scf_zero);
          input_vreg_k1 = _mm512_add_epi16(input_vreg_k1, scale_vreg);
          input_vreg_k1 = _mm512_mask_blend_epi16(mask_vreg_zero_k1, input_vreg_k1, zero_vreg);
          out_vreg_16m0_2k  = _mm512_permutex2var_epi16(input_vreg_k0, vnni_perm_reg_lo, input_vreg_k1);
          out_vreg_16m1_2k  = _mm512_permutex2var_epi16(input_vreg_k0, vnni_perm_reg_hi, input_vreg_k1);
          _mm512_storeu_epi16((libxsmm_bfloat16*) out_ptr + (i_br * (bk/2) * bm * 2 + (i_bk * 16 + i_k) * bm * 2 + (2 * i_m + 0) * 32), out_vreg_16m0_2k);
          _mm512_storeu_epi16((libxsmm_bfloat16*) out_ptr + (i_br * (bk/2) * bm * 2 + (i_bk * 16 + i_k) * bm * 2 + (2 * i_m + 1) * 32), out_vreg_16m1_2k);
        }
      }
    }
  }
}

LIBXSMM_INLINE
float convert_mxfp4_to_float(unsigned char x) {
  float fp4_e1m2_lut[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};
  float result = fp4_e1m2_lut[x];
  return result;
}
template<typename DType, typename DTypeLP>

int gemm_benchmark(int argc, char** argv) {
  // Setup default GEMM sizes
  char loop_specs_str[256] = "aBC";
  long M = 1024*4, N = 1024*4, K = 1024*4;
  long bm = 32, bn = 32, bk = 32;
  long kbf = 1;
  long n_layers = 1;
  long n_iters = 1;
  long i, j;
  long check_correctness = 1;
  long fuse_cvts = 1;
  long cache_resident_acts = 0;
  long is_A_bf8 = 0;
  long is_A_bf16 = 0;
  long is_A_mxfp4 = 0;
  long nThreads = omp_get_max_threads();

  if (strcmp(argv[13], "BF16") == 0) {
    is_A_bf16 = 1;
  }
  if (strcmp(argv[13], "BF8BF16") == 0) {
    is_A_bf8 = 1;
  }
  if (strcmp(argv[13], "MXFP4BF16") == 0) {
    is_A_mxfp4 = 1;
  }

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
      fuse_cvts = atoi(argv[16]);
    }     
  }
  
  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  long brcount = Kb/kbf;
  long a_dtype_scale = (is_A_mxfp4 > 0) ? 2 : 1;
  while (Kb % kbf != 0) {
    kbf--;
  }
  brcount = Kb/kbf;

  // Allocate buffers
  DType **ACT = (DType**) malloc((n_layers+1)*sizeof(DType*));
  DTypeLP **WGT = (DTypeLP**) malloc(n_layers    *sizeof(DTypeLP*));
  unsigned char **MXFP_SCALES = (unsigned char**) malloc(n_layers    *sizeof(unsigned char*));

  for (i = 0; i < (n_layers+1); i++) {
    ACT[i] = (DType*) libxsmm_aligned_malloc(LIBXSMM_MAX(K,M)*N*sizeof(DType), 64);
    if (i < n_layers) {
      WGT[i] = (DTypeLP*) libxsmm_aligned_malloc(M*(K/a_dtype_scale)*sizeof(DTypeLP), 64);
      if (is_A_mxfp4 > 0) {
        MXFP_SCALES[i] = (unsigned char*) libxsmm_aligned_malloc(M*(K/32)*sizeof(unsigned char), 64);
      }
    }
  }
  float *naive_input  = (float*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(float), 64);
  float *naive_output = (float*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(float), 64);
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(float), 64);
  float *naive_filter = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 64);
  DType *naive_input_bf16  = (DType*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(DType), 64);
  DType *naive_output_bf16 = (DType*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(DType), 64);
  DTypeLP *naive_filter_lp = (DTypeLP*)libxsmm_aligned_malloc( M*(K/a_dtype_scale)*sizeof(DTypeLP), 64);
  unsigned char *mxfp4_scf = (unsigned char*)libxsmm_aligned_malloc( M*(K/32)*sizeof(unsigned char), 64);

  DType *weight_scratch  = (DType*)libxsmm_aligned_malloc( bk*brcount*bm*sizeof(DType)*nThreads, 64);
  long long *last_slice_cvted = (long long*)libxsmm_aligned_malloc( 64*sizeof(long long)*nThreads, 64);
  for (i = 0; i < 64*nThreads; i++) {
    last_slice_cvted[i] = -1;
  }

  auto dtype_up      = LIBXSMM_DATATYPE_BF16;
  auto dtype_down    = LIBXSMM_DATATYPE_BF8;
  auto l_cvt_shape = libxsmm_create_meltw_unary_shape(bm*2, (bk/2)*brcount, bm*2, bm*2, dtype_down, dtype_up, dtype_up);
  auto cvt_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_cvt_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  

  // Init buffers
  init_buf( naive_input,     LIBXSMM_MAX(K,M)*N, 0, 0 );
  init_buf( naive_output,    LIBXSMM_MAX(K,M)*N, 0, 0 );
  if (is_A_mxfp4 > 0) {
    unsigned char scale_exp = 0;
    unsigned int scale_exp_u32 = 0;
    float *scalef_ptr = (float*)&scale_exp_u32;
    DTypeLP *weight_mxfp = (DTypeLP*)WGT[0];  
    float scale_expf = 0.0;
//#pragma omp parallel for private(j)
    for (i = 0; i < M; i++) {
      for (j = 0; j < K/32; j++) {
        mxfp4_scf[(i/bm)*(K/32)*bm+j*bm+(i%bm)] = (unsigned char) (125+rand()%10);
      }
    }
    unsigned int __i, _i;
//#pragma omp parallel for private(_i, j)
    for (__i = 0; __i < Mb; __i++) {
    for (_i = 0; _i < bm; _i++) {
      i = __i*bm+_i;
      for (j = 0; j < K/2; j++) {
        unsigned char even = (unsigned char)(rand()%16);
        unsigned char odd  = (unsigned char)(rand()%16);
        unsigned char result;
        float evenf;
        float oddf;
        libxsmm_bfloat16 evenbf16;
        libxsmm_bfloat16 oddbf16;
        if (j % 16 == 0) {
          scale_exp = mxfp4_scf[(i/bm)*(K/32)*bm+(j/16)*bm+(i%bm)];
          scale_exp_u32 = (unsigned int) scale_exp;
          scale_exp_u32 = scale_exp_u32 << 23;
          scale_expf = *scalef_ptr;
        } 
        evenf = convert_mxfp4_to_float(even) * scale_expf;
        oddf = convert_mxfp4_to_float(odd) * scale_expf;
        libxsmm_rne_convert_fp32_bf16( &evenf, &evenbf16, 1 );
        libxsmm_rne_convert_fp32_bf16( &oddf, &oddbf16, 1 );
        libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)&evenbf16, &evenf, 1);
        libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)&oddbf16, &oddf, 1);
        even = even & 0x0f;
        odd = (odd & 0x0f) << 4;
        result = even | odd;
        naive_filter[i*K+2*j+0] = evenf;
        naive_filter[i*K+2*j+1] = oddf;
        weight_mxfp[(i/bm)*(K/2)*bm+j*bm+(i%bm)] = result;
      }
    }
    }
  } else {
    init_buf( naive_filter,    M*K, 0, 0 );
  }

  libxsmm_rne_convert_fp32_bf16( naive_input,     (libxsmm_bfloat16*)naive_input_bf16,     N*LIBXSMM_MAX(K,M));
  libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_input_bf16, naive_input, N*LIBXSMM_MAX(K,M));
  libxsmm_rne_convert_fp32_bf16( naive_output,    (libxsmm_bfloat16*)naive_output_bf16,    N*LIBXSMM_MAX(K,M) );
  libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_output, N*M);
  
  if (is_A_bf8 > 0) {
    libxsmm_rne_convert_fp32_bf8( naive_filter,    (libxsmm_bfloat8*)naive_filter_lp,    M*K );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)naive_filter_lp, naive_filter, M*K);
  }
  if (is_A_bf16 > 0) {
    libxsmm_rne_convert_fp32_bf16( naive_filter,    (libxsmm_bfloat16*)naive_filter_lp,    M*K );
    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_filter_lp, naive_filter, M*K);
  }

  for (i = 0; i < n_layers; i++) {
    unsigned int __i, _i;
    if (is_A_bf8 > 0) {  
      matrix_copy_KC_to_KCCK_bf8_local( (libxsmm_bfloat8*)naive_filter_lp, (libxsmm_bfloat8*)WGT[i], K, M, bk, bm );
    }
    if (is_A_bf16 > 0) {  
      matrix_copy_KC_to_KCCK_bf16_local( (libxsmm_bfloat16*)naive_filter_lp, (libxsmm_bfloat16*)WGT[i], K, M, bk, bm );
    }
    if ((is_A_mxfp4 > 0) && (i > 0)) {
#if 0
#pragma omp parallel for private(_i,j)
    for (__i = 0; __i < Mb; __i++) {
    for (_i = 0; _i < bm; _i++) {
      for (j = 0; j < (K/2); j++) {
        unsigned char *fake_ptr_in = (unsigned char*)WGT[0];
        unsigned char *fake_ptr_out = (unsigned char*)WGT[i];    
        fake_ptr_out[(__i*bm+_i)*K/2+j] = fake_ptr_in[ (__i*bm+_i)*K/2+j];
      }
    }
    }
#else
      memcpy(WGT[i], WGT[0], M * (K/2));
#endif
    }
    if (is_A_mxfp4 > 0) {
#if 0 
#pragma omp parallel for private(_i,j)
      for (__i = 0; __i < Mb; __i++) {
    for (_i = 0; _i < bm; _i++) {
      for (j = 0; j < (K/32); j++) {
        unsigned char *fake_ptr_in = (unsigned char*)mxfp4_scf;
        unsigned char *fake_ptr_out = (unsigned char*)MXFP_SCALES[i];    
        fake_ptr_out[(__i*bm+_i)*K/32+j] = fake_ptr_in[(__i*bm+_i)*K/32+j];
      }
    }
    }
#else
      memcpy(MXFP_SCALES[i], mxfp4_scf, M * (K/32));
#endif
    } 
    matrix_copy_NC_to_NCNC_bf16_local( (libxsmm_bfloat16*)naive_input_bf16, (libxsmm_bfloat16*)ACT[i] , N, LIBXSMM_MAX(K,M), bn, bk );
  }
  matrix_copy_NC_to_NCNC_bf16_local( (libxsmm_bfloat16*)naive_output, (libxsmm_bfloat16*)ACT[n_layers], N, LIBXSMM_MAX(K,M), bn, bk );

  // Setup TPP kernels
  auto l_flags    = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ;
  auto l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  auto l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');

  if (is_A_mxfp4 > 0) {
    l_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2;
    l_tc_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2;
    l_tr_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2;
  }
  
  auto dtype      = LIBXSMM_DATATYPE_BF16;
  auto l_shape = libxsmm_create_gemm_shape( bm, bn, bk, bm, bk, bm, (is_A_bf16 > 0) ? dtype : ( (is_A_mxfp4 > 0) ? LIBXSMM_DATATYPE_I8 : LIBXSMM_DATATYPE_BF8), dtype, dtype, LIBXSMM_DATATYPE_F32 );
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*(bk/a_dtype_scale)*sizeof(DTypeLP), bk*bn*sizeof(DType), 0 );
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bm*bn, 1, bm*bn, bm*bn, dtype, dtype, dtype);

  if (brcount == Kb) l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

  auto zero_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
  auto tileconfig_kernel  = libxsmm_dispatch_tilecfg_gemm( l_shape, l_tc_flags );
  auto tilerelease_kernel = libxsmm_dispatch_tilecfg_gemm( l_shape, l_tr_flags );
  auto brgemm_kernel      = libxsmm_dispatch_brgemm( l_shape, l_flags, l_prefetch_flags, l_brconfig );

  auto l_shape_full = libxsmm_create_gemm_shape( bm, bn, bk, bm, bk, bm, dtype, dtype, dtype, LIBXSMM_DATATYPE_F32 );
  auto l_brconfig_full = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*bk*sizeof(DType), bk*bn*sizeof(DType), 0 );
  auto brgemm_kernel_full = libxsmm_dispatch_brgemm( l_shape_full, l_flags, l_prefetch_flags, l_brconfig_full );

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

  auto gemm_loop = ThreadedLoop<3>({
      LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // Logical K loop specs
      LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // Logical M loop specs
      LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // Logical N loop specs
      loop_specs_str);

  // Warmup iteration for i-caches
  for (i = 0; i < n_layers; i++) {
    if (fuse_cvts > 0) {
      gemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;
          gemm_param.op.tertiary = (void*)&brcount;
          gemm_param.a.primary = (void*)((DTypeLP*)WGT[i] + (i_m * K * bm + i_k * bk * bm)/a_dtype_scale);
          if (is_A_mxfp4 > 0) {
            gemm_param.a.tertiary = (void*)((unsigned char*)MXFP_SCALES[i] + (i_m * (K/32) * bm + i_k * (bk/32) * bm));
          }
          if (cache_resident_acts > 0) {
            gemm_param.b.primary = (void*)((DType*)ACT[0] + i_n * K * bn + i_k * bk * bn );
          } else {
            gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
          }
          gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
          if (brcount != Kb && i_k == 0) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)gemm_param.c.primary;
            zero_kernel( &zero_param );
          }
          brgemm_kernel( &gemm_param );
        },
        [&]() {tileconfig_kernel(NULL);},
        [&]() {tilerelease_kernel(NULL);});
    } else {
      gemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;
          libxsmm_meltw_unary_param cvt_param;
          //int tid = gemm_loop.get_tid(ind);
          int tid = omp_get_thread_num();
          long long slice_id = (i_k/brcount)*Mb+i_m;
          gemm_param.op.tertiary = (void*)&brcount;

          cvt_param.in.primary = (void*)((DTypeLP*)WGT[i] + (i_m * K * bm + i_k * bk * bm)/a_dtype_scale);
          cvt_param.out.primary = (void*)((DType*)weight_scratch + tid * bk * brcount * bm);

          if (last_slice_cvted[64*tid] != slice_id) {
            if (is_A_mxfp4 > 0) {
              cvt_kernel_mxfp4((unsigned char*)cvt_param.in.primary, (unsigned char*)MXFP_SCALES[i] + (i_m * (K/32) * bm + i_k * (bk/32) * bm),
                               (libxsmm_bfloat16*)cvt_param.out.primary, brcount, bm, bk);
            } else {
              cvt_kernel( &cvt_param );
            }
            last_slice_cvted[64*tid] = slice_id;        
          }
          gemm_param.a.primary = (void*)cvt_param.out.primary;

          if (cache_resident_acts > 0) {
            gemm_param.b.primary = (void*)((DType*)ACT[0] + i_n * K * bn + i_k * bk * bn );
          } else {
            gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
          }
          gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
          if (brcount != Kb && i_k == 0) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)gemm_param.c.primary;
            zero_kernel( &zero_param );
          }
          brgemm_kernel_full( &gemm_param );
        },
        [&]() {int _tid = omp_get_thread_num(); tileconfig_kernel(NULL); last_slice_cvted[64*_tid] = -1;},
        [&]() {tilerelease_kernel(NULL);});
    }
  }

  // Check correctness if requested
  if (n_layers == 1) {
    printf("##########################################\n");
    printf("#  GEMM %d x %d x %d  (M x N x K)        \n", M, N, K);
    printf("##########################################\n");
  } else {
    printf("##############################################################\n");
    printf("    %d Layer MLP with sizes  %d x %d x %d  (M x N x K)  \n", n_layers, M, N, K);
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
      if (fuse_cvts > 0) {
        gemm_loop(
          [&](int* ind) {
            int i_k = ind[0], i_m = ind[1], i_n = ind[2];
            libxsmm_gemm_param gemm_param;
            gemm_param.op.tertiary = (void*)&brcount;
            gemm_param.a.primary = (void*)((DTypeLP*)WGT[i] + (i_m * K * bm + i_k * bk * bm)/a_dtype_scale);
            if (is_A_mxfp4 > 0) {
              gemm_param.a.tertiary = (void*)((unsigned char*)MXFP_SCALES[i] + (i_m * (K/32) * bm + i_k * (bk/32) * bm));
            }
            if (cache_resident_acts > 0) {
              gemm_param.b.primary = (void*)((DType*)ACT[0] + i_n * K * bn + i_k * bk * bn );
            } else {
              gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
            }
            gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
            brgemm_kernel( &gemm_param );
          },
          [&]() {tileconfig_kernel(NULL);},
          [&]() {tilerelease_kernel(NULL);});
      } else {
        gemm_loop(
          [&](int* ind) {
            int i_k = ind[0], i_m = ind[1], i_n = ind[2];
            libxsmm_gemm_param gemm_param;
            libxsmm_meltw_unary_param cvt_param;
            //int tid = gemm_loop.get_tid(ind);
            int tid = omp_get_thread_num();
            long long slice_id = (i_k/brcount)*Mb+i_m;
            gemm_param.op.tertiary = (void*)&brcount;

            cvt_param.in.primary = (void*)((DTypeLP*)WGT[i] + (i_m * K * bm + i_k * bk * bm)/a_dtype_scale);
            cvt_param.out.primary = (void*)((DType*)weight_scratch + tid * bk * brcount * bm);

            if (last_slice_cvted[64*tid] != slice_id) {
              if (is_A_mxfp4 > 0) {
                cvt_kernel_mxfp4((unsigned char*)cvt_param.in.primary, (unsigned char*)MXFP_SCALES[i] + (i_m * (K/32) * bm + i_k * (bk/32) * bm),
                                 (libxsmm_bfloat16*)cvt_param.out.primary, brcount, bm, bk);
              } else {
                cvt_kernel( &cvt_param );
              }
              last_slice_cvted[64*tid] = slice_id;        
            }
            gemm_param.a.primary = (void*)cvt_param.out.primary;

            if (cache_resident_acts > 0) {
              gemm_param.b.primary = (void*)((DType*)ACT[0] + i_n * K * bn + i_k * bk * bn );
            } else {
              gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
            }
            gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
            if (brcount != Kb && i_k == 0) {
              libxsmm_meltw_unary_param zero_param;
              zero_param.out.primary = (void*)gemm_param.c.primary;
              zero_kernel( &zero_param );
            }
            brgemm_kernel_full( &gemm_param );
          },
          [&]() {int _tid = omp_get_thread_num(); tileconfig_kernel(NULL); last_slice_cvted[64*_tid] = -1;},
          [&]() {tilerelease_kernel(NULL);});
      }
    }
  }
  auto t_end = getTime();
  
  // Print performance/model numbers
  double gflop = (2.0*(double)n_layers*(double)M*(double)N*(double)K) / (1000*1000*1000);
  printf("Time is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  printf("Effective model sizes: %.5g GB\n", ((double)sizeof(DType)*(double)n_layers*(double)M*(double)(K))/(1024.0*1024.0*1024.0));
  printf("Effective A BW is %.5g GB/s\n", (((double)sizeof(DType)*(double)n_layers*(double)M*(double)(K)) / (1024.0*1024.0*1024.0))/((t_end-t_start)/(1.0*n_iters)));
  printf("MEASURE %.5g %s_%d_%d_%d_%d_%d_%d_bf%d_threads%d\n", gflop/((t_end-t_start)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads());

  // Free buffers
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_input_bf16);
  libxsmm_free(naive_output_bf16);
  libxsmm_free(naive_filter_lp);
  libxsmm_free(mxfp4_scf);
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
  char cl_str_precision[256];
  sprintf(cl_str_precision, "%s", argv[13]);
  if (strcmp(cl_str_precision, "BF16") == 0) {
    return gemm_benchmark<libxsmm_bfloat16, libxsmm_bfloat16>(argc, argv);
  } else if (strcmp(cl_str_precision, "BF8BF16") == 0) { 
    return gemm_benchmark<libxsmm_bfloat16, libxsmm_bfloat8>(argc, argv);
  } else if (strcmp(cl_str_precision, "MXFP4BF16") == 0) { 
    return gemm_benchmark<libxsmm_bfloat16, unsigned char>(argc, argv);
  } else {
    return 0;
  }
}

