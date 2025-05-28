/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#include "sf_curves_utils.h"

void check_null_ptr(void* ptr, const char* ptr_name) {
  if (ptr == NULL) {
    printf("Pointer for %s is NULL. Exiting...\n", ptr_name);
    exit(0);
  } 
}

template<typename DType> libxsmm_meltw_unary_type parlooper_get_vnni_xform() {
  return LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
}

template<> libxsmm_meltw_unary_type parlooper_get_vnni_xform<libxsmm_bfloat16>() {
  return LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
}

template<> libxsmm_meltw_unary_type parlooper_get_vnni_xform<libxsmm_bfloat8>() {
  return LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4;
}

template<typename DType> libxsmm_datatype parlooper_get_lixbxsmm_dtype() {
  return LIBXSMM_DATATYPE_BF16;
}

template<> libxsmm_datatype parlooper_get_lixbxsmm_dtype<float>() {
  return LIBXSMM_DATATYPE_F32;
}

template<> libxsmm_datatype parlooper_get_lixbxsmm_dtype<libxsmm_bfloat16>() {
  return LIBXSMM_DATATYPE_BF16;
}

template<> libxsmm_datatype parlooper_get_lixbxsmm_dtype<libxsmm_bfloat8>() {
  return LIBXSMM_DATATYPE_BF8;
}

template<typename DType> void parlooper_matrix_copy_NCNC_to_NC(void *in, void *out, long N, long M, long bn, long bm) {
  matrix_copy_NCNC_to_NC_bf16( (libxsmm_bfloat16*)in, (libxsmm_bfloat16*)out, 1, N, M, bn, bm );
  return;
}

template<> void parlooper_matrix_copy_NCNC_to_NC<float>(void *in, void *out, long N, long M, long bn, long bm) {
  matrix_copy_NCNC_to_NC( (float*)in, (float*)out, 1, N, M, bn, bm );
  return;
}

template<> void parlooper_matrix_copy_NCNC_to_NC<libxsmm_bfloat16>(void *in, void *out, long N, long M, long bn, long bm) {
  matrix_copy_NCNC_to_NC_bf16( (libxsmm_bfloat16*)in, (libxsmm_bfloat16*)out, 1, N, M, bn, bm );
  return;
}

template<> void parlooper_matrix_copy_NCNC_to_NC<libxsmm_bfloat8>(void *in, void *out, long N, long M, long bn, long bm) {
  matrix_copy_NCNC_to_NC_bf8( (libxsmm_bfloat8*)in, (libxsmm_bfloat8*)out, 1, N, M, bn, bm );
  return;
}

template<typename DType> void parlooper_rne_convert_fp32_lp(float *in, void *out, long size) {
  libxsmm_rne_convert_fp32_bf16(in, (libxsmm_bfloat16*)out, size);
  return;
}

template<> void parlooper_rne_convert_fp32_lp<float>(float *in, void *out, long size) {
  memcpy(out, in, size * sizeof(float));
  return;
}

template<> void parlooper_rne_convert_fp32_lp<libxsmm_bfloat16>(float *in, void *out, long size) {
  libxsmm_rne_convert_fp32_bf16(in, (libxsmm_bfloat16*)out, size);
  return;
}

template<> void parlooper_rne_convert_fp32_lp<libxsmm_bfloat8>(float *in, void *out, long size) {
  libxsmm_rne_convert_fp32_bf8(in, (libxsmm_bfloat8*)out, size);
  return;
}

template<typename DType> void parlooper_convert_lp_f32(void *in, float *out, long size) {
  libxsmm_convert_bf16_f32((libxsmm_bfloat16*)in, out, size);
  return;
}

template<> void parlooper_convert_lp_f32<float>(void *in, float *out, long size) {
  memcpy(out, in, size * sizeof(float));
  return;
}

template<> void parlooper_convert_lp_f32<libxsmm_bfloat16>(void *in, float *out, long size) {
  libxsmm_convert_bf16_f32((libxsmm_bfloat16*)in, out, size);
  return;
}

template<> void parlooper_convert_lp_f32<libxsmm_bfloat8>(void *in, float *out, long size) {
  libxsmm_convert_bf8_f32((libxsmm_bfloat8*)in, out, size);
  return;
}

template<typename DType> void parlooper_matrix_copy_KC_to_KCCK(void *src, void *dst, int C, int K, int bc, int bk, int flat_layout, int trans_a )
{
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  int vnni_block = libxsmm_cpuid_dot_pack_factor(parlooper_get_lixbxsmm_dtype<DType>());
  if (flat_layout > 0) {
    vnni_block = 1;
  }
  if (trans_a > 0) {
    vnni_block = 1;
    LIBXSMM_VLA_DECL(2, DType, real_src, (DType*)src, C);
    LIBXSMM_VLA_DECL(5, DType, real_dst, (DType*)dst, cBlocks, bk/vnni_block, bc, vnni_block);
    # pragma omp parallel for
    for (int k1 = 0; k1 < kBlocks; k1++) {
      for (int c1 = 0; c1 < cBlocks; c1++) {
        for (int c2 = 0; c2 < bc; c2++) {
          for (int k2 = 0; k2 < bk; k2++) {
            vnni_block = 1;
            LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, k2, c2/vnni_block, c2%vnni_block, cBlocks, bk, bc/vnni_block, vnni_block) =
              LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);      
          }
        }
      }
    }
  } else {
    LIBXSMM_VLA_DECL(2, DType, real_src, (DType*)src, C);
    LIBXSMM_VLA_DECL(5, DType, real_dst, (DType*)dst, cBlocks, bc/vnni_block, bk, vnni_block);
    # pragma omp parallel for
    for (int k1 = 0; k1 < kBlocks; k1++) {
      for (int c1 = 0; c1 < cBlocks; c1++) {
        for (int c2 = 0; c2 < bc; c2++) {
          for (int k2 = 0; k2 < bk; k2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, c2/vnni_block, k2, c2%vnni_block, cBlocks, bc/vnni_block, bk, vnni_block) =
              LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);
          }
        }
      }
    }
  }
}

template<typename DType> void parlooper_matrix_copy_NC_to_NCNC(void *src, void *dst, int N, int C, int bn, int bc, int trans_b)
{
  int nBlocks = N/bn;
  int cBlocks = C/bc;

  if (trans_b > 0) {
    LIBXSMM_VLA_DECL(3, DType, real_src, (DType*)src, N, C);
    LIBXSMM_VLA_DECL(5, DType, real_dst, (DType*)dst, nBlocks, cBlocks, bc, bn);
    # pragma omp parallel for 
    for (int n1 = 0; n1 < nBlocks; n1++) {
      for (int c1 = 0; c1 < cBlocks; c1++) {
        for (int n2 = 0; n2 < bn; n2++) {
          for (int c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, 0, n1, c1, c2, n2, nBlocks, cBlocks, bc, bn) =
              LIBXSMM_VLA_ACCESS(3, real_src, 0, n1*bn+n2, c1*bc+c2, N, C);
          }
        }
      }
    }
  } else {
    LIBXSMM_VLA_DECL(3, DType, real_src, (DType*)src, N, C);
    LIBXSMM_VLA_DECL(5, DType, real_dst, (DType*)dst, nBlocks, cBlocks, bn, bc);
    # pragma omp parallel for 
    for (int n1 = 0; n1 < nBlocks; n1++) {
      for (int c1 = 0; c1 < cBlocks; c1++) {
        for (int n2 = 0; n2 < bn; n2++) {
          for (int c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, 0, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc) =
              LIBXSMM_VLA_ACCESS(3, real_src, 0, n1*bn+n2, c1*bc+c2, N, C);
          }
        }
      }
    }
  }
}

unsigned int fill_sf_curve_index_map(unsigned char **sf_curve_index_map, unsigned int Mb, unsigned int Nb) {
  long long i, n_tasks = Mb*Nb;
  int m_id, n_id;
  if (Mb < 256 && Nb < 256) {
    unsigned char *map;
    *sf_curve_index_map = (unsigned char*) libxsmm_aligned_malloc( 2*Mb*Nb*sizeof(unsigned char), 2097152);
    map = (unsigned char*) *sf_curve_index_map;
    for (i = 0; i < n_tasks; i++) {
      gilbert_d2xy( &m_id, &n_id, i, Mb, Nb );
      map[2*i+0] = (unsigned char)m_id;
      map[2*i+1] = (unsigned char)n_id;
    }
    return 1;
  } else if (Mb < 65536 && Nb < 65536) {
    unsigned short *map;
    *sf_curve_index_map = (unsigned char*) libxsmm_aligned_malloc( 2*Mb*Nb*sizeof(unsigned short), 2097152);
    map = (unsigned short*) *sf_curve_index_map;
    for (i = 0; i < n_tasks; i++) {
      gilbert_d2xy( &m_id, &n_id, i, Mb, Nb );
      map[2*i+0] = (unsigned short)m_id;
      map[2*i+1] = (unsigned short)n_id;
    }
    return 2;
  } else {
    unsigned int *map;
    *sf_curve_index_map = (unsigned char*) libxsmm_aligned_malloc( 2*Mb*Nb*sizeof(unsigned int), 2097152);
    map = (unsigned int*) *sf_curve_index_map;
    for (i = 0; i < n_tasks; i++) {
      gilbert_d2xy( &m_id, &n_id, i, Mb, Nb );
      map[2*i+0] = (unsigned int)m_id;
      map[2*i+1] = (unsigned int)n_id;
    }
    return 4;
  }
}

void extract_indices_from_sf_curve(int *i_m, int *i_n, unsigned char *sf_curve_index_map, int sf_curve_index, unsigned int index_tsize) {
  if (index_tsize == 1) {
    unsigned char *map;
    map = (unsigned char*) sf_curve_index_map;
    *i_m = (int) map[2*sf_curve_index + 0];
    *i_n = (int) map[2*sf_curve_index + 1];
  } else if (index_tsize == 2) {
    unsigned short *map;
    map = (unsigned short*) sf_curve_index_map;
    *i_m = (int) map[2*sf_curve_index + 0];
    *i_n = (int) map[2*sf_curve_index + 1];
  } else {
    unsigned int *map;
    map = (unsigned int*) sf_curve_index_map;
    *i_m = (int) map[2*sf_curve_index + 0];
    *i_n = (int) map[2*sf_curve_index + 1];
  }
  return;
}

LIBXSMM_INLINE void matrix_copy_NC_to_NCNC_bf16_local(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int N, int C, int bn, int bc, int trans_b)
{
  int nBlocks = N/bn;
  int cBlocks = C/bc;

  if (trans_b > 0) {
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, real_src, src, N, C);
    LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, nBlocks, cBlocks, bc, bn);
    # pragma omp parallel for 
    for (int n1 = 0; n1 < nBlocks; n1++) {
      for (int c1 = 0; c1 < cBlocks; c1++) {
        for (int n2 = 0; n2 < bn; n2++) {
          for (int c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, 0, n1, c1, c2, n2, nBlocks, cBlocks, bc, bn) =
              LIBXSMM_VLA_ACCESS(3, real_src, 0, n1*bn+n2, c1*bc+c2, N, C);
          }
        }
      }
    }
  } else {
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, real_src, src, N, C);
    LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, nBlocks, cBlocks, bn, bc);
    # pragma omp parallel for 
    for (int n1 = 0; n1 < nBlocks; n1++) {
      for (int c1 = 0; c1 < cBlocks; c1++) {
        for (int n2 = 0; n2 < bn; n2++) {
          for (int c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, 0, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc) =
              LIBXSMM_VLA_ACCESS(3, real_src, 0, n1*bn+n2, c1*bc+c2, N, C);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KC_to_KCCK_bf16_local(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int C, int K, int bc, int bk, int flat_layout, int trans_a )
{
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  int vnni_block = libxsmm_cpuid_dot_pack_factor(LIBXSMM_DATATYPE_BF16);
  if (flat_layout > 0) {
    vnni_block = 1;
  }
  if (trans_a > 0) {
    vnni_block = 1;
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_src, src, C);
    LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, cBlocks, bk/vnni_block, bc, vnni_block);
    # pragma omp parallel for
    for (int k1 = 0; k1 < kBlocks; k1++) {
      for (int c1 = 0; c1 < cBlocks; c1++) {
        for (int c2 = 0; c2 < bc; c2++) {
          for (int k2 = 0; k2 < bk; k2++) {
            vnni_block = 1;
            LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, k2, c2/vnni_block, c2%vnni_block, cBlocks, bk, bc/vnni_block, vnni_block) =
              LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);      
          }
        }
      }
    }
  } else {
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_src, src, C);
    LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, cBlocks, bc/vnni_block, bk, vnni_block);
    # pragma omp parallel for
    for (int k1 = 0; k1 < kBlocks; k1++) {
      for (int c1 = 0; c1 < cBlocks; c1++) {
        for (int c2 = 0; c2 < bc; c2++) {
          for (int k2 = 0; k2 < bk; k2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, c2/vnni_block, k2, c2%vnni_block, cBlocks, bc/vnni_block, bk, vnni_block) =
              LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);
          }
        }
      }
    }
  }
}

double get_random_posneg_p5_num(void) {
  double tmp = libxsmm_rng_f64()-0.5;
  if ( tmp < -0.4 ) {
    tmp = -0.4;
  } else if ( tmp < -0.3 ) {
    tmp = -0.3;
  } else if ( tmp < -0.2 ) {
    tmp = -0.2;
  } else if ( tmp < -0.1 ) {
    tmp = -0.1;
  } else if ( tmp < 0 ) {
    tmp = 0;
  } else if ( tmp < 0.1 ) {
    tmp = 0.1;
  } else if ( tmp < 0.2 ) {
    tmp = 0.2;
  } else if ( tmp < 0.3 ) {
    tmp = 0.3;
  } else if ( tmp < 0.4 ) {
    tmp = 0.4;
  } else if ( tmp < 0.5 ) {
    tmp = 0.5;
  } else {
    tmp = 0.5;
  }

  return tmp;
}

double get_random_pos_p5_num(void) {
  double tmp = libxsmm_rng_f64();
  if ( tmp < 0.1 ) {
    tmp = 0.1;
  } else if ( tmp < 0.2 ) {
    tmp = 0.2;
  } else if ( tmp < 0.3 ) {
    tmp = 0.3;
  } else if ( tmp < 0.4 ) {
    tmp = 0.4;
  } else if ( tmp < 0.5 ) {
    tmp = 0.5;
  } else if ( tmp < 0.6 ) {
    tmp = 0.6;
  } else if ( tmp < 0.7 ) {
    tmp = 0.7;
  } else if ( tmp < 0.8 ) {
    tmp = 0.8;
  } else if ( tmp < 0.9 ) {
    tmp = 0.9;
  } else if ( tmp < 1.0 ) {
    tmp = 1.0;
  } else {
    tmp = 1.0;
  }

  return tmp;
}

void naive_fullyconnected_fused_int8( naive_fullyconnected_t* param, const unsigned char* input_ptr, unsigned char* output_ptr, const char* filter_ptr, const float* bias_ptr, float* scf_quant, float* tmp_f32_output_ptr ) {
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;
  int img, ifm, ofm;
  float max_vals[256];
  float max_value = FLT_MIN;
  int maxexp = 0;
  float l_scf_quant = 0.0f;
  LIBXSMM_VLA_DECL(2, const unsigned char, input,  input_ptr,  nIFm);
  LIBXSMM_VLA_DECL(2, const char, filter, filter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, unsigned char, output, output_ptr, nOFm);
  LIBXSMM_VLA_DECL(2,      float, output_f32, tmp_f32_output_ptr, nOFm);
  auto l_quant_unary_shape = libxsmm_create_meltw_unary_shape(1, 1, 1, 1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_F32);
  auto quant_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_QUANT, l_quant_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  for (img = 0; img < 256; img++) {
    max_vals[img] = FLT_MIN;
  }

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ofm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ofm = 0; ofm < nOFm; ++ofm) {
    for(img = 0; img < nImg; ++img) {
      int tid = omp_get_thread_num();
      int accum = 0;
      float f32_res = 0.0f;
      for (ifm = 0; ifm < nIFm; ++ifm) {
        accum += LIBXSMM_VLA_ACCESS(2, filter, ofm, ifm, nIFm) * LIBXSMM_VLA_ACCESS(2, input, img, ifm, nIFm);
      }
      f32_res = (float)accum * 1.0f;
      if ( param->fuse_type == 1 ) {
        f32_res += bias_ptr[ofm];
      } else if ( param->fuse_type == 2 ) {
        f32_res = ( f32_res >= 0.0f ) ? f32_res : 0.0f;
      } else if ( param->fuse_type == 3 ) {
        f32_res += bias_ptr[ofm];
        f32_res = ( f32_res >= 0.0 ) ? f32_res : 0.0f;
      }
      LIBXSMM_VLA_ACCESS(2, output_f32, img, ofm, nOFm) = f32_res;
      if (f32_res > max_vals[tid]) {
        max_vals[tid] = f32_res;
      }
    }
  }

  /* Find max value */
  for (img = 0; img < 256; img++) {    
    if (max_vals[img] > max_value) {
      max_value = max_vals[img];
    }
  }

  LIBXSMM_ELIDE_RESULT(float, LIBXSMM_FREXPF(max_value, &maxexp));
  maxexp -= 6;
  l_scf_quant = libxsmm_sexp2_i8i(-maxexp);
  *scf_quant = l_scf_quant;

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm);
# pragma omp parallel for private(img, ofm)
#endif
  /* Quantize the tensor */
  for (ofm = 0; ofm < nOFm; ++ofm) {
    for(img = 0; img < nImg; ++img) {
#if 1
      libxsmm_meltw_unary_param quant_param;
      quant_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(2, output_f32, img, ofm, nOFm);
      quant_param.in.secondary= (void*)&l_scf_quant;
      quant_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm);
      quant_kernel( &quant_param );
#else
      LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) = (unsigned char) LIBXSMM_NEARBYINTF(LIBXSMM_VLA_ACCESS(2, output_f32, img, ofm, nOFm) * l_scf_quant);
#endif
    }
  }
}

