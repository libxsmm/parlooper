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

/* Activations are in flat format:
   [N][K]
*/

libxsmm_meltwfunction_unary unary_kernel_quant;
libxsmm_meltwfunction_unary unary_kernel_absmax;
libxsmm_meqn_function dequant_func;
int use_tpp_for_quant = 0;

void quantize_K_dim(float *in_ptr, unsigned char *out_ptr, float *out_scales, int group_size_k, int i_n, long K, float scale) {
  int k_groups = K/group_size_k;
  int g = 0;
  int ik = 0;
  float d = 0.0f;
  float id = 0.0f;
  for (g = 0; g < k_groups; g++) {
    /* Find max of current group */
    __m512 max_acc = _mm512_setzero_ps();
    __m512 v_id;  
    float max_val = 0.0;
    float cur_scale = 0.0;;
    for (ik = 0; ik < group_size_k; ik += 16) {
      max_acc = _mm512_max_ps(max_acc, _mm512_abs_ps(_mm512_load_ps((float*)in_ptr + i_n * K + g * group_size_k + ik)));  
    }
    max_val = _mm512_reduce_max_ps(max_acc);
    d = max_val / 127;
    id = (d != 0) ? (1.0f / d) : 0;
    cur_scale = d * scale;
    out_scales[i_n * k_groups + g] = cur_scale;
    v_id = _mm512_set1_ps(id);
    for (ik = 0; ik < group_size_k; ik += 16) {
      __m512 v = _mm512_load_ps((float*)in_ptr + i_n * K + g * group_size_k + ik);
      v = _mm512_mul_round_ps(v, v_id, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
      __m512i v_i32 = _mm512_cvt_roundps_epi32(v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
      __m128i res = _mm512_cvtepi32_epi8(v_i32);
      _mm_storeu_epi8((unsigned char*)out_ptr + i_n * K + g * group_size_k + ik, res);
    }
  }
}

void quantize_64_x_K_gs_64(float *in_ptr, unsigned char *out_ptr, float *out_scales, int i_n, long K, float scale) {
  int in = 0;
  int g = 0;
  int ik = 0;
  float d = 0.0f;
  float id = 0.0f;
  const int group_size_k = 64;
  const int bn = 64;
  int k_groups = K/group_size_k;
  for (in = 0; in < bn; in++) {
    for (g = 0; g < k_groups; g++) {
      /* Find max of current group */
      __m512 max_acc = _mm512_setzero_ps();
      __m512 v_id;  
      float max_val = 0.0;
      float cur_scale = 0.0;;
      for (ik = 0; ik < group_size_k; ik += 16) {
        max_acc = _mm512_max_ps(max_acc, _mm512_abs_ps(_mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik)));  
      }
      max_val = _mm512_reduce_max_ps(max_acc);
      d = max_val / 127;
      id = (d != 0) ? (1.0f / d) : 0;
      cur_scale = d * scale;
      out_scales[(i_n * bn + in) * k_groups + g] = cur_scale;
      v_id = _mm512_set1_ps(id);
      for (ik = 0; ik < group_size_k; ik += 16) {
        __m512 v = _mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik);
        v = _mm512_mul_round_ps(v, v_id, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m512i v_i32 = _mm512_cvt_roundps_epi32(v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m128i res = _mm512_cvtepi32_epi8(v_i32);
        _mm_storeu_epi8((unsigned char*)out_ptr + (i_n * bn + in) * K + g * group_size_k + ik, res);
      }
    }
  }
}

void quantize_64_x_K_gs_128(float *in_ptr, unsigned char *out_ptr, float *out_scales, int i_n, long K, float scale) {
  int in = 0;
  int g = 0;
  int ik = 0;
  float d = 0.0f;
  float id = 0.0f;
  const int group_size_k = 128;
  const int bn = 64;
  int k_groups = K/group_size_k;
  for (in = 0; in < bn; in++) {
    for (g = 0; g < k_groups; g++) {
      /* Find max of current group */
      __m512 max_acc = _mm512_setzero_ps();
      __m512 v_id;  
      float max_val = 0.0;
      float cur_scale = 0.0;;
      for (ik = 0; ik < group_size_k; ik += 16) {
        max_acc = _mm512_max_ps(max_acc, _mm512_abs_ps(_mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik)));  
      }
      max_val = _mm512_reduce_max_ps(max_acc);
      d = max_val / 127;
      id = (d != 0) ? (1.0f / d) : 0;
      cur_scale = d * scale;
      out_scales[(i_n * bn + in) * k_groups + g] = cur_scale;
      v_id = _mm512_set1_ps(id);
      for (ik = 0; ik < group_size_k; ik += 16) {
        __m512 v = _mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik);
        v = _mm512_mul_round_ps(v, v_id, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m512i v_i32 = _mm512_cvt_roundps_epi32(v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m128i res = _mm512_cvtepi32_epi8(v_i32);
        _mm_storeu_epi8((unsigned char*)out_ptr + (i_n * bn + in) * K + g * group_size_k + ik, res);
      }
    }
  }
}

void quantize_64_x_K_gs_256(float *in_ptr, unsigned char *out_ptr, float *out_scales, int i_n, long K, float scale) {
  int in = 0;
  int g = 0;
  int ik = 0;
  float d = 0.0f;
  float id = 0.0f;
  const int group_size_k = 256;
  const int bn = 64;
  int k_groups = K/group_size_k;
  for (in = 0; in < bn; in++) {
    for (g = 0; g < k_groups; g++) {
      /* Find max of current group */
      __m512 max_acc = _mm512_setzero_ps();
      __m512 v_id;  
      float max_val = 0.0;
      float cur_scale = 0.0;;
      for (ik = 0; ik < group_size_k; ik += 16) {
        max_acc = _mm512_max_ps(max_acc, _mm512_abs_ps(_mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik)));  
      }
      max_val = _mm512_reduce_max_ps(max_acc);
      d = max_val / 127;
      id = (d != 0) ? (1.0f / d) : 0;
      cur_scale = d * scale;
      out_scales[(i_n * bn + in) * k_groups + g] = cur_scale;
      v_id = _mm512_set1_ps(id);
      for (ik = 0; ik < group_size_k; ik += 16) {
        __m512 v = _mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik);
        v = _mm512_mul_round_ps(v, v_id, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m512i v_i32 = _mm512_cvt_roundps_epi32(v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m128i res = _mm512_cvtepi32_epi8(v_i32);
        _mm_storeu_epi8((unsigned char*)out_ptr + (i_n * bn + in) * K + g * group_size_k + ik, res);
      }
    }
  }
}

void quantize_64_x_K_gs_512(float *in_ptr, unsigned char *out_ptr, float *out_scales, int i_n, long K, float scale) {
  int in = 0;
  int g = 0;
  int ik = 0;
  float d = 0.0f;
  float id = 0.0f;
  const int group_size_k = 512;
  const int bn = 64;
  int k_groups = K/group_size_k;
  for (in = 0; in < bn; in++) {
    for (g = 0; g < k_groups; g++) {
      /* Find max of current group */
      __m512 max_acc = _mm512_setzero_ps();
      __m512 v_id;  
      float max_val = 0.0;
      float cur_scale = 0.0;;
      for (ik = 0; ik < group_size_k; ik += 16) {
        max_acc = _mm512_max_ps(max_acc, _mm512_abs_ps(_mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik)));  
      }
      max_val = _mm512_reduce_max_ps(max_acc);
      d = max_val / 127;
      id = (d != 0) ? (1.0f / d) : 0;
      cur_scale = d * scale;
      out_scales[(i_n * bn + in) * k_groups + g] = cur_scale;
      v_id = _mm512_set1_ps(id);
      for (ik = 0; ik < group_size_k; ik += 16) {
        __m512 v = _mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik);
        v = _mm512_mul_round_ps(v, v_id, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m512i v_i32 = _mm512_cvt_roundps_epi32(v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m128i res = _mm512_cvtepi32_epi8(v_i32);
        _mm_storeu_epi8((unsigned char*)out_ptr + (i_n * bn + in) * K + g * group_size_k + ik, res);
      }
    }
  }
}

void quantize_bn_x_K_generic(float *in_ptr, unsigned char *out_ptr, float *out_scales, int group_size_k, int i_n, long bn, long K, float scale) {
  int in = 0;
  int k_groups = K/group_size_k;
  int g = 0;
  int ik = 0;
  float d = 0.0f;
  float id = 0.0f;
  for (in = 0; in < bn; in++) {
    for (g = 0; g < k_groups; g++) {
      /* Find max of current group */
      __m512 max_acc = _mm512_setzero_ps();
      __m512 v_id;  
      float max_val = 0.0;
      float cur_scale = 0.0;;
      for (ik = 0; ik < group_size_k; ik += 16) {
        max_acc = _mm512_max_ps(max_acc, _mm512_abs_ps(_mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik)));  
      }
      max_val = _mm512_reduce_max_ps(max_acc);
      d = max_val / 127;
      id = (d != 0) ? (1.0f / d) : 0;
      cur_scale = d * scale;
      out_scales[(i_n * bn + in) * k_groups + g] = cur_scale;
      v_id = _mm512_set1_ps(id);
      for (ik = 0; ik < group_size_k; ik += 16) {
        __m512 v = _mm512_load_ps((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + ik);
        //v = _mm512_mul_round_ps(v, v_id, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        v = _mm512_mul_ps(v, v_id);
        __m512i v_i32 = _mm512_cvt_roundps_epi32(v, (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        __m128i res = _mm512_cvtepi32_epi8(v_i32);
        _mm_storeu_epi8((unsigned char*)out_ptr + (i_n * bn + in) * K + g * group_size_k + ik, res);
      }
    }
  }
}

void quantize_bn_x_K(float *in_ptr, unsigned char *out_ptr, float *out_scales, int group_size_k, int i_n, long bn, long bk, long K, float scale) {
  if (use_tpp_for_quant > 0) {
    libxsmm_meltw_unary_param unary_param;
    int in = 0;
    int k_groups = K/group_size_k;
    int g = 0;
    int ik = 0;
    float d = 0.0f;
    float id = 0.0f;
    int k_subgroup = 0;
    for (in = 0; in < bn; in++) {
      for (g = 0; g < k_groups; g++) {
        float max_val = 0.0;
        float cur_scale = 0.0;
        unary_param.in.primary  = (void*)((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k);
        unary_param.out.primary = (void*)&max_val;
        unary_kernel_absmax( &unary_param );
        d = max_val / 127;
        id = (d != 0) ? (1.0f / d) : 0;
        cur_scale = d * scale;
        out_scales[(i_n * bn + in) * k_groups + g] = cur_scale;
        for (k_subgroup = 0; k_subgroup < group_size_k/bk; k_subgroup++) {
          unary_param.in.primary  = (void*)((float*)in_ptr + (i_n * bn + in) * K + g * group_size_k + k_subgroup * bk);
          unary_param.in.secondary  = (void*)&id;
          unary_param.out.primary = (void*)((unsigned char*)out_ptr + i_n * bn * K + in * bk + g * group_size_k * bn + k_subgroup * bk * bn);
          unary_kernel_quant( &unary_param );
        }
      }
    }
  } else {
    if (bn == 64) {
      if (group_size_k == 64) {
        quantize_64_x_K_gs_64(in_ptr, out_ptr, out_scales, i_n, K, scale);
      } else if (group_size_k == 128) {
        quantize_64_x_K_gs_128(in_ptr, out_ptr, out_scales, i_n, K, scale);  
      } else if (group_size_k == 256) {
        quantize_64_x_K_gs_256(in_ptr, out_ptr, out_scales, i_n, K, scale);
      } else if (group_size_k == 512) {
        quantize_64_x_K_gs_512(in_ptr, out_ptr, out_scales, i_n, K, scale);  
      } else {
        quantize_bn_x_K_generic(in_ptr, out_ptr, out_scales, group_size_k, i_n, bn, K, scale);  
      }
    } else {
      quantize_bn_x_K_generic(in_ptr, out_ptr, out_scales, group_size_k, i_n, bn, K, scale); 
    }
  }
}

void dequantize_64_x_64(unsigned int *int32_acc_ptr, float *f32_acc_ptr,  unsigned short *wei_scales_ptr, float *inp_scales, long k_groups, long gid, long M) {
  const int bm = 64;
  const int bn =64;
  int i_n = 0, i_m = 0;
  for (i_n = 0; i_n < bn; i_n++) {
    __m512 dx = _mm512_set1_ps(inp_scales[i_n * k_groups + gid]);
    for (i_m = 0; i_m < bm/16; i_m++) {
      __m512 dw = _mm512_cvtph_ps (_mm256_loadu_epi16((unsigned short*)wei_scales_ptr + i_m * 16));
      __m512 scale = _mm512_mul_ps(dx, dw);
      __m512 i32_vec_acc = _mm512_cvtepi32_ps(_mm512_loadu_epi32((unsigned int*)int32_acc_ptr + i_n * bm + i_m * 16));
      __m512 f32_vec_acc = _mm512_load_ps((float*)f32_acc_ptr + i_n * M + i_m * 16);
      f32_vec_acc = _mm512_fmadd_ps(scale, i32_vec_acc, f32_vec_acc);
      _mm512_store_ps ((float*)f32_acc_ptr + i_n * M + i_m * 16, f32_vec_acc);
    }
  }
}

void dequantize_bm_x_bn(unsigned int *int32_acc_ptr, float *f32_acc_ptr,  unsigned short *wei_scales_ptr, float *inp_scales, long k_groups, long gid, long M, long bm, long bn) {
  if (use_tpp_for_quant > 0) {
    libxsmm_meqn_param eqn_param;
    libxsmm_matrix_arg  arg_array[4];
    arg_array[0].primary = wei_scales_ptr;
    arg_array[1].primary = &inp_scales[gid];
    arg_array[2].primary = int32_acc_ptr;
    arg_array[3].primary = f32_acc_ptr;
    eqn_param.inputs = arg_array;
    eqn_param.output.primary = f32_acc_ptr;
    dequant_func(&eqn_param);
  } else {
    if (bm == 64 && bn == 64) {
      dequantize_64_x_64(int32_acc_ptr, f32_acc_ptr, wei_scales_ptr, inp_scales, k_groups, gid, M);
    } else {
      int i_n = 0, i_m = 0;
      for (i_n = 0; i_n < bn; i_n++) {
        __m512 dx = _mm512_set1_ps(inp_scales[i_n * k_groups + gid]);
        for (i_m = 0; i_m < bm/16; i_m++) {
          __m512 dw = _mm512_cvtph_ps (_mm256_loadu_epi16((unsigned short*)wei_scales_ptr + i_m * 16));
          __m512 scale = _mm512_mul_ps(dx, dw);
          __m512 i32_vec_acc = _mm512_cvtepi32_ps(_mm512_loadu_epi32((unsigned int*)int32_acc_ptr + i_n * bm + i_m * 16));
          __m512 f32_vec_acc = _mm512_load_ps((float*)f32_acc_ptr + i_n * M + i_m * 16);
          f32_vec_acc = _mm512_fmadd_ps(scale, i32_vec_acc, f32_vec_acc);
          _mm512_store_ps ((float*)f32_acc_ptr + i_n * M + i_m * 16, f32_vec_acc);
        }
      }
    }
  }
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
  long i;
  long check_correctness = 0;
  long group_size_k = 256;
  long nThreads = omp_get_max_threads();
  float scale = 1.0f;
  
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
    kbf = atoi(argv[8]);
    group_size_k = atoi(argv[9]);
    n_layers = atoi(argv[10]);
    n_iters = atoi(argv[11]);
    check_correctness = atoi(argv[12]);
    if (argc > 13) {
      use_tpp_for_quant = atoi(argv[13]);
    }
  }
  
  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  long brcount = Kb/kbf;
  while (Kb % kbf != 0) {
    kbf--;
  }
  brcount = Kb/kbf;

  // Create some auxiliary TPPs
  if (use_tpp_for_quant > 0) {
    libxsmm_blasint my_eqn0;
    libxsmm_meqn_arg_metadata arg_metadata;
    libxsmm_meqn_op_metadata  op_metadata;
    libxsmm_meqn_arg_shape          arg_shape_in, arg_shape_out;
    libxsmm_matrix_arg_attributes   arg_singular_attr = libxsmm_create_matrix_arg_attributes( LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);
    int k_groups = K/group_size_k;

    libxsmm_meltw_unary_shape unary_shape;
    libxsmm_meltw_unary_flags unary_flags;
    libxsmm_meltw_unary_type  unary_type;

    /* Abs max reduce TPP */
    unary_shape.m = group_size_k;
    unary_shape.n = 1;
    unary_shape.ldi = group_size_k;
    unary_shape.ldo = 1;
    unary_shape.in0_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type = LIBXSMM_DATATYPE_F32;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX;
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;  
    unary_kernel_absmax = libxsmm_dispatch_meltw_unary( unary_type, unary_shape, unary_flags );
    
    /* Quant TPP */
    unary_shape.m = bk;
    unary_shape.n = 1;
    unary_shape.ldi = bk;
    unary_shape.ldo = bk;
    unary_shape.in0_type = LIBXSMM_DATATYPE_F32;
    unary_shape.out_type = LIBXSMM_DATATYPE_I8;
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_QUANT;
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;  
    unary_kernel_quant = libxsmm_dispatch_meltw_unary( unary_type, unary_shape, unary_flags );

    /* Create dequant equation */
    my_eqn0 = libxsmm_meqn_create();
    op_metadata   = libxsmm_create_meqn_op_metadata(my_eqn0, -1);
    libxsmm_meqn_push_back_ternary_op( op_metadata, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT );
    libxsmm_meqn_push_back_binary_op( op_metadata, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0 |  LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( bm, 1, bm, LIBXSMM_DATATYPE_F16 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 0);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, bn, k_groups, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 1);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( bm, bn, bm, LIBXSMM_DATATYPE_I32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 2);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( bm, bn, M, LIBXSMM_DATATYPE_F32 );
    arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 3);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);    
    arg_shape_out = libxsmm_create_meqn_arg_shape( bm, bn, M, LIBXSMM_DATATYPE_F32 );
    dequant_func = libxsmm_dispatch_meqn( my_eqn0, arg_shape_out );
  }

  // Allocate buffers
  DType **ACT = (DType**) malloc((n_layers+1)*sizeof(DType*));
  DTypeLP **WGT = (DTypeLP**) malloc(n_layers*sizeof(DTypeLP*));
  libxsmm_float16 **WGT_SCALES = (libxsmm_float16**) malloc(n_layers*sizeof(libxsmm_float16*));
  for (i = 0; i < (n_layers+1); i++) {
    ACT[i] = (DType*) libxsmm_aligned_malloc(LIBXSMM_MAX(K,M)*N*sizeof(DType), 64);
    if (i < n_layers) {
      WGT[i] = (DTypeLP*) libxsmm_aligned_malloc(M*K*sizeof(DTypeLP), 64);
      WGT_SCALES[i] = (libxsmm_float16*) libxsmm_aligned_malloc(M*(K/group_size_k)*sizeof(libxsmm_float16), 64);   
    }
  }
  float *naive_input  = (float*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(float), 64);;
  float *naive_output = (float*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(float), 64);
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( LIBXSMM_MAX(K,M)*N*sizeof(float), 64);
  float *naive_filter = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 64);
  libxsmm_float16 *naive_filter_scales = (libxsmm_float16*)libxsmm_aligned_malloc(M*(K/group_size_k)*sizeof(libxsmm_float16), 64);
  DTypeLP *naive_filter_lp = (DTypeLP*)libxsmm_aligned_malloc( M*K*sizeof(DTypeLP), 64);
  
  // Allocate buffers to convert inputs and input scales
  DTypeLP *int8_acts = (DTypeLP*) libxsmm_aligned_malloc(LIBXSMM_MAX(K,M)*N*sizeof(DTypeLP), 64);
  float *inp_scales = (float*) libxsmm_aligned_malloc((LIBXSMM_MAX(K,M)/group_size_k)*N*sizeof(float), 64);

  // Allocate private scratches
  unsigned int *int32_scratch = (unsigned int*) libxsmm_aligned_malloc(nThreads*bm*bn*sizeof(unsigned int), 64);

  // Init buffers
  for (i = 0; i < LIBXSMM_MAX(K,M)*N; i++) {
//    naive_input[i] = get_random_posneg_p5_num();
    naive_input[i] = get_random_pos_p5_num();
  }
  for (i = 0; i < LIBXSMM_MAX(K,M)*N; i++) {
    naive_output[i] = 0.0;
  }

  // Weight scales init with positive values
  // [Mb][K/group_size_k][bm]
  for (int imb = 0; imb < Mb; imb++) {
    for (int igk = 0; igk < K/group_size_k; igk++) {
      for (int ibm = 0; ibm < bm; ibm++) {
        float tmp = get_random_pos_p5_num();  
        libxsmm_float16 tmpf16 = 0;  
        libxsmm_rne_convert_fp32_f16(&tmp, &tmpf16, 1);
        naive_filter_scales[imb * (K/group_size_k) * bm + igk * bm + ibm] = tmpf16;       
      }
    }
  }

  // Weights init with positive values
  // [Mb][Kb][bk/4][bm][4]
  for (int imb = 0; imb < Mb; imb++) {
    for (int ikb = 0; ikb < Kb; ikb++) {
      for (int ibk = 0; ibk < bk/4; ibk++) {
        for (int ibm = 0; ibm < bm; ibm++) {
          for (int ibkk = 0; ibkk < 4; ibkk++) {
            int logical_k = ikb * bk + ibk * 4 + ibkk;
            int logical_m = imb * bm + ibm;
            int igk = logical_k/group_size_k;
            libxsmm_float16 cur_scale = naive_filter_scales[imb * (K/group_size_k) * bm + igk * bm + ibm];
            float f32_scale = 0.0;
            char tmp = (char) (get_random_pos_p5_num() * 10.0);
            naive_filter_lp[imb * Kb * bk * bm + ikb * bk * bm + ibk * bm * 4 + ibm * 4  + ibkk] = tmp;
            libxsmm_convert_f16_f32( &cur_scale, &f32_scale, 1 );
            naive_filter[logical_m * K + logical_k] = f32_scale * ((float)tmp);
          } 
        }     
      }
    }
  }

  for (i = 0; i < n_layers; i++) {
    memcpy(WGT[i], naive_filter_lp, M * K * sizeof(unsigned char));
    memcpy((libxsmm_float16*)WGT_SCALES[i], naive_filter_scales, M * (K/group_size_k) * sizeof(libxsmm_float16));
    memcpy((float*)ACT[i], naive_input, LIBXSMM_MAX(K,M)*N * sizeof(float));
  }
  memcpy((float*)ACT[n_layers], naive_output, LIBXSMM_MAX(K,M)*N * sizeof(float));


  // Setup TPP kernels
  auto l_flags    = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_BETA_0 | LIBXSMM_GEMM_FLAG_A_UNSIGNED;
  auto l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  auto l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  
  auto dtype      = LIBXSMM_DATATYPE_BF16;
  auto l_shape = libxsmm_create_gemm_shape( bm, bn, bk, bm, bk, bm, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I32, LIBXSMM_DATATYPE_I32 );
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*bk*sizeof(char), bk*bn*sizeof(char), group_size_k/bk );
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bm, bn, M, M, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);

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
    /* Here quantize the input activations from fp32 to int8 */
#pragma omp parallel for
    for (int in = 0; in < Nb; in++) {
      /* Quantize current input block and calculate also scale factor */
      quantize_bn_x_K((float*)ACT[i], (unsigned char*)int8_acts, (float*)inp_scales, group_size_k, in, bn, bk, K, scale);   
    }
    gemm_loop(
      [&](int* ind) {
        int i_k = ind[0], i_m = ind[1], i_n = ind[2];
        int i_k_group;
        unsigned int int32_scratch_ptr[bm*bn];     

        if (i_k == 0) {
          /* Initialize bm x bn f32 accumulator to zero */
          libxsmm_meltw_unary_param zero_param;
          zero_param.out.primary = (void*)((float*)ACT[i+1] + i_n * M * bn + i_m * bm);
          zero_kernel( &zero_param );
        }

        for (i_k_group = 0; i_k_group < ((brcount * bk) / group_size_k); i_k_group++) {
          /* Run beta = 0 brgemm with bm x bn x K' with K' = group_size_k (i.e. br = group_size_k/bk) */
          libxsmm_gemm_param gemm_param;
          long long br_int8 = group_size_k/bk;

          gemm_param.op.tertiary = (void*)&br_int8;
          gemm_param.a.primary = (void*)((unsigned char*)WGT[i] + i_m * K * bm + i_k * bk * bm + i_k_group * group_size_k * bm );
          gemm_param.b.primary = (void*)((unsigned char*)int8_acts + i_n * K * bn + i_k * bk * bn + i_k_group * group_size_k * bn);
          gemm_param.c.primary = (void*)int32_scratch_ptr;
          brgemm_kernel( &gemm_param );

          /* Update running f32 accumulator  */
          dequantize_bm_x_bn(int32_scratch_ptr, (float*)ACT[i+1] + i_n * M * bn + i_m * bm,
                                       (unsigned short*)WGT_SCALES[i] + i_m * (K/group_size_k) * bm + ((i_k * bk)/group_size_k) * bm + i_k_group * bm,
                                                (float*)inp_scales + i_n * (K/group_size_k) * bn, 
                                                K/group_size_k, ((i_k * bk)/group_size_k) + i_k_group, M, bm, bn);    
        }
      },
      [&]() {tileconfig_kernel(NULL);},
      [&]() {tilerelease_kernel(NULL);});
  }

#if 1
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
    memcpy((float*)naive_output_opt, (float*)ACT[n_layers], M*N*sizeof(float));
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
#endif

  // benchmark the GEMM
  auto t_start = getTime();
  for (long it = 0; it < n_iters; it++) {
    for (i = 0; i < n_layers; i++) {
      /* Here quantize the input activations from fp32 to int8 */
#pragma omp parallel for
      for (int in = 0; in < Nb; in++) {
        /* Quantize current input block and calculate also scale factor */
        quantize_bn_x_K((float*)ACT[i], (unsigned char*)int8_acts, (float*)inp_scales, group_size_k, in, bn, bk, K, scale);   
      }
      gemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          int i_k_group;
          unsigned int int32_scratch_ptr[bm*bn];     

          if (i_k == 0) {
            /* Initialize bm x bn f32 accumulator to zero */
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)((float*)ACT[i+1] + i_n * M * bn + i_m * bm);
            zero_kernel( &zero_param );
          }

          for (i_k_group = 0; i_k_group < ((brcount * bk) / group_size_k); i_k_group++) {
            /* Run beta = 0 brgemm with bm x bn x K' with K' = group_size_k (i.e. br = group_size_k/bk) */
            libxsmm_gemm_param gemm_param;
            long long br_int8 = group_size_k/bk;

            gemm_param.op.tertiary = (void*)&br_int8;
            gemm_param.a.primary = (void*)((unsigned char*)WGT[i] + i_m * K * bm + i_k * bk * bm + i_k_group * group_size_k * bm );
            gemm_param.b.primary = (void*)((unsigned char*)int8_acts + i_n * K * bn + i_k * bk * bn + i_k_group * group_size_k * bn);
            gemm_param.c.primary = (void*)int32_scratch_ptr;
            brgemm_kernel( &gemm_param );

            /* Update running f32 accumulator  */
            dequantize_bm_x_bn(int32_scratch_ptr, (float*)ACT[i+1] + i_n * M * bn + i_m * bm,
                                         (unsigned short*)WGT_SCALES[i] + i_m * (K/group_size_k) * bm + ((i_k * bk)/group_size_k) * bm + i_k_group * bm,
                                                  (float*)inp_scales + i_n * (K/group_size_k) * bn, 
                                                  K/group_size_k, ((i_k * bk)/group_size_k) + i_k_group, M, bm, bn);    
          }

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
  printf("MEASURE %.5g %s_%d_%d_%d_%d_%d_%d_bf%d_threads%d\n", gflop/((t_end-t_start)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads());

  // Free buffers
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_output_opt);
  libxsmm_free(naive_filter_scales);
  libxsmm_free(naive_filter_lp);
  libxsmm_free(int8_acts);
  libxsmm_free(inp_scales);
  libxsmm_free(int32_scratch);
  for (i = 0; i < (n_layers+1); i++) {
    libxsmm_free(ACT[i]);
    if (i < n_layers) {
      libxsmm_free(WGT[i]);
      libxsmm_free(WGT_SCALES[i]);   
    }
  }
  free(ACT);
  free(WGT);
  free(WGT_SCALES);
  return 0;
}

int main(int argc, char** argv) {
  return gemm_benchmark<float, unsigned char>(argc, argv);
}

