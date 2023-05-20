/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
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

LIBXSMM_INLINE void matrix_copy_KC_to_KCCK_i8flat(char *src, char *dst, int C, int K, int bc, int bk) {
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, char, real_src, src, C);
  LIBXSMM_VLA_DECL(4, char, real_dst, dst, cBlocks, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(4, real_dst, k1, c1, c2, k2, cBlocks, bc, bk) =
            LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KC_to_KCCK_f16flat(libxsmm_float16 *src, libxsmm_float16 *dst, int C, int K, int bc, int bk) {
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, libxsmm_float16, real_src, src, C);
  LIBXSMM_VLA_DECL(4, libxsmm_float16, real_dst, dst, cBlocks, bc, bk);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(c1); LIBXSMM_OMP_VAR(c2); LIBXSMM_OMP_VAR(k2);
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(4, real_dst, k1, c1, c2, k2, cBlocks, bc, bk) =
            LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);
        }
      }
    }
  }
}

void naive_fullyconnected_fused_int8f16( naive_fullyconnected_t* param, const libxsmm_float16* input_ptr, libxsmm_float16* output_ptr, const char* filter_ptr) {
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;
  int img, ifm, ofm;
  LIBXSMM_VLA_DECL(2, const libxsmm_float16, input,  input_ptr,  nIFm);
  LIBXSMM_VLA_DECL(2, const char, filter, filter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, libxsmm_float16, output, output_ptr, nOFm);
  float l_float_one = 0.0001f;
  libxsmm_float16 l_f16_one;
  libxsmm_rne_convert_fp32_f16(&l_float_one,    (libxsmm_float16*)&l_f16_one,    1 );
  libxsmm_convert_f16_f32( &l_f16_one, &l_float_one, 1 );
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ofm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ofm = 0; ofm < nOFm; ++ofm) {
    for(img = 0; img < nImg; ++img) {
      float f32_res = 0.0f, a_use, b_use;
      libxsmm_float16 c_tmp;
      libxsmm_float16 cur_a, cur_b;
      for (ifm = 0; ifm < nIFm; ++ifm) {
        char char_a = LIBXSMM_VLA_ACCESS(2, filter, ofm, ifm, nIFm);
        short short_a = (short) char_a;
        /* Convert a to float16 and scale  */
        a_use = (float) short_a;
        libxsmm_rne_convert_fp32_f16(&a_use, &cur_a, 1);
        libxsmm_convert_f16_f32( &cur_a, &a_use, 1 );
        a_use = a_use * l_float_one;
        libxsmm_rne_convert_fp32_f16(&a_use, &cur_a, 1);
        libxsmm_convert_f16_f32( &cur_a, &a_use, 1 );
        cur_b = LIBXSMM_VLA_ACCESS(2, input, img, ifm, nIFm);
        libxsmm_convert_f16_f32( &cur_b, &b_use, 1 );
        f32_res += a_use * b_use;
        libxsmm_rne_convert_fp32_f16(&f32_res, &c_tmp, 1);
        libxsmm_convert_f16_f32( &c_tmp, &f32_res, 1 );
      }
      libxsmm_rne_convert_fp32_f16(&f32_res, &c_tmp, 1);
      LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) = c_tmp;
    }
  }
}

void naive_fullyconnected_fused_f16( naive_fullyconnected_t* param, const libxsmm_float16* input_ptr, libxsmm_float16* output_ptr, libxsmm_float16* filter_ptr) {
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;
  int img, ifm, ofm;
  LIBXSMM_VLA_DECL(2, const libxsmm_float16, input,  input_ptr,  nIFm);
  LIBXSMM_VLA_DECL(2, const libxsmm_float16, filter, filter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, libxsmm_float16, output, output_ptr, nOFm);
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ofm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ofm = 0; ofm < nOFm; ++ofm) {
    for(img = 0; img < nImg; ++img) {
      float f32_res = 0.0f, a_use, b_use;
      libxsmm_float16 c_tmp;
      libxsmm_float16 cur_a, cur_b;
      for (ifm = 0; ifm < nIFm; ++ifm) {
        cur_a = LIBXSMM_VLA_ACCESS(2, filter, ofm, ifm, nIFm);
        libxsmm_convert_f16_f32( &cur_a, &a_use, 1 );
        cur_b = LIBXSMM_VLA_ACCESS(2, input, img, ifm, nIFm);
        libxsmm_convert_f16_f32( &cur_b, &b_use, 1 );
        f32_res += a_use * b_use;
        libxsmm_rne_convert_fp32_f16(&f32_res, &c_tmp, 1);
        libxsmm_convert_f16_f32( &c_tmp, &f32_res, 1 );
      }
      libxsmm_rne_convert_fp32_f16(&f32_res, &c_tmp, 1);
      LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) = c_tmp;
    }
  }
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
  auto quant_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_QUANT, l_quant_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

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

