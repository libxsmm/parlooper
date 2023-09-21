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

int ullcompare( const void* a , const void* b ) {
  const unsigned long long aull = *( const unsigned long long* )a;
  const unsigned long long bull = *( const unsigned long long* )b;
  if ( aull < bull ) {
    return -1;
  } else if( aull > bull ) {
    return 1;
  } else {
    return 0;
  }
}

void shuffle_array(unsigned long long *array, int n) {
  if (n > 1)
  {
    int i;
    for (i = 0; i < n - 1; i++)
    {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned long long t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

int is_dense_grid_point(unsigned long long grid_point_id, int n_dense_grid_points, unsigned long long *grid_point_array) {
  unsigned long long key = grid_point_id;
  unsigned long long *found_ptr = (unsigned long long*) bsearch(&key, grid_point_array, n_dense_grid_points, sizeof(unsigned long long), ullcompare);
  return ((found_ptr == NULL) ? 0 : 1);
}

LIBXSMM_INLINE void matrix_copy_KC_to_KCCK_f16(libxsmm_float16 *src, libxsmm_float16 *dst, int C, int K, int bc, int bk)
{
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

template<typename DType>
int gemm_benchmark(int argc, char** argv) {
  // Setup default GEMM sizes
  int check_correctness = 1;
  char loop_specs_str[256] = "aBC";  
  long M = 1024*4, N = 1024*4, K = 1024*4;
  long bm = 32, bn = 32, bk = 32;
  long kbf = 1;
  long n_layers = 1;
  long n_iters = 1;
  long long i, l_i, l_j, l_c = 0;
  long fuse_bias = 0;
  long fuse_relu = 0;
  long int8_gemm = 0;
  long f16_gemm = 0;
  long use_ping_pong_bufs = 0;
  char prec_string[255];
  char fuse_string[255];
  // Setup model and trace
  int use_model = 0;
  const char* const env_use_model = getenv("USE_MODEL");
  double sparse_frac = 0.8;
  unsigned int  use_decompress_kernel = 0;
  if (0 == env_use_model) {
    use_model = 0;
  } else {
    use_model = atoi(env_use_model);
  }
  ifreq = 1.0 / getFreq();
  std::vector<std::string> inp_trace[128];
  platform_spec_t my_platform;
  tensor_metadata_t tensor_metadata;
  set_platform_specs( CLX, omp_get_max_threads(), &my_platform);

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
      fuse_bias = atoi(argv[11]);
    }
    if (argc > 12) {
      fuse_relu = atoi(argv[12]);
    }
    if (argc > 14) {
      use_ping_pong_bufs = atoi(argv[14]);
    }
    if (argc > 15) {
      use_decompress_kernel = 1;
      sparse_frac = atof(argv[15]);
    }
  }

  if (sizeof(DType) == 1) {
    if (strcmp(argv[13], "INT8") == 0) {
      int8_gemm = 1;
    }
  }

  if (sizeof(DType) == 2) {
    if (strcmp(argv[13], "F16") == 0) {
      f16_gemm = 1;
    }
  }

  if (int8_gemm > 0) {
    sprintf(prec_string, "INT8");
  } else if (sizeof(DType) == 1) {
    sprintf(prec_string, "BF8");
  } else if (sizeof(DType) == 2) {
    if (f16_gemm > 0) {
      sprintf(prec_string, "F16"); 
    } else {
      sprintf(prec_string, "BF16");
    }
  } else if (sizeof(DType) == 4) {
    sprintf(prec_string, "FP32");
  }

  if (fuse_bias == 0 && fuse_relu == 0) {
    sprintf(fuse_string, "NONE");
  } else if (fuse_bias == 1 && fuse_relu == 0) {
    sprintf(fuse_string, "BIAS");
  } else if (fuse_bias == 0 && fuse_relu == 1) {
    sprintf(fuse_string, "RELU");
  } else if (fuse_bias == 1 && fuse_relu == 1) {
    sprintf(fuse_string, "BIAS + RELU");
  } else {
    sprintf(fuse_string, "NONE");
  }
  
  if ((n_layers > 1) && !(M == K && bm == bk && bk == bn) ) {
    printf("MLP support only for M == K and bm == bn == bk\n");
    return 1;
  }

  long Mb = M/bm, Nb = N/bn, Kb = K/bk;
  long brcount = Kb/kbf;
  while (Kb % kbf != 0) {
    kbf--;
  }
  brcount = Kb/kbf;

  /* Early exit to avoid testing the same combos since in this case the "a" loop has trip count 1 */
  if (kbf == 1 && loop_specs_str[0] != 'a') {
    return 0;
  }

  // Allocate buffers
  float **naive_bias = (float**) malloc((n_layers+1)*sizeof(float*));
  float  *scf_quant = (float*) malloc(n_layers*sizeof(float));
  DType **ACT = (DType**) malloc((n_layers+1)*sizeof(DType*));
  DType **BIAS = (DType**) malloc((n_layers)*sizeof(DType*));
  DType **WGT = (DType**) malloc(n_layers    *sizeof(DType*));
  for (i = 0; i < (n_layers+1); i++) {
    if (use_ping_pong_bufs == 0) {
      if (i % 2 == 0) {
        ACT[i] = (DType*) libxsmm_aligned_malloc(N*K*sizeof(DType), 2097152);
      } else {
        ACT[i] = (DType*) libxsmm_aligned_malloc(M*N*sizeof(DType), 2097152);
      }
    } else {
      if (i < 2) {
        if (i % 2 == 0) {
          ACT[i] = (DType*) libxsmm_aligned_malloc(N*K*sizeof(DType), 2097152);
        } else {
          ACT[i] = (DType*) libxsmm_aligned_malloc(M*N*sizeof(DType), 2097152);
        }   
      } else {
        ACT[i] = ACT[i-2];
      }
    }
    if (i < n_layers) {
      WGT[i] = (DType*) libxsmm_aligned_malloc(M*K*sizeof(DType), 2097152);
      if (fuse_bias > 0) {
        BIAS[i] = (DType*) libxsmm_aligned_malloc(M*sizeof(DType), 2097152);
        naive_bias[i] = (float*) libxsmm_aligned_malloc(M*sizeof(float), 2097152);
        init_buf( naive_bias[i], M, 0, 0 );
        if (int8_gemm > 0) {
          /* DO nothing, will be using the f32 bias  */
        } else if ((sizeof(DType) == 1) && (int8_gemm == 0)) {
          libxsmm_rne_convert_fp32_bf8(  naive_bias[i],  (libxsmm_bfloat8*)BIAS[i], M );
          libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)BIAS[i], naive_bias[i], M);
        } else if (sizeof(DType) == 2) {
          libxsmm_rne_convert_fp32_bf16( naive_bias[i], (libxsmm_bfloat16*)BIAS[i], M );
          libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)BIAS[i], naive_bias[i], M);
        } else {
          memcpy(BIAS[i], naive_bias[i], M*sizeof(float));
        }
      }
    }
  }
  float *itm_f32_out  = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 2097152);
  float *naive_input  = (float*)libxsmm_aligned_malloc( K*N*sizeof(float), 2097152);
  float *naive_output = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 2097152);
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( M*N*sizeof(float), 2097152);
  float *naive_filter = (float*)libxsmm_aligned_malloc( M*K*sizeof(float), 2097152);
  DType *naive_input_bf16  = (DType*)libxsmm_aligned_malloc( K*N*sizeof(DType), 2097152);
  DType *naive_output_bf16 = (DType*)libxsmm_aligned_malloc( M*N*sizeof(DType), 2097152);
  DType *naive_filter_bf16 = (DType*)libxsmm_aligned_malloc( M*K*sizeof(DType), 2097152);
  DType *compressed_filter = NULL;
  unsigned int *compressed_a_bitmap = NULL;
  long long *compressed_a_offsets = NULL;
  unsigned char *naive_input_i8;
  unsigned char *naive_output_i8;
  unsigned char *naive_output_opt_i8;
  char *naive_filter_i8;

  libxsmm_matdiff_info norms, diff;
  libxsmm_matdiff_clear(&norms);
  libxsmm_matdiff_clear(&diff);

  // Init buffers
  init_buf( naive_input,     K*N, 0, 0 );
  init_buf( naive_output,    M*N, 0, 0 );
  init_buf( naive_filter,    M*K, 0, 0 );

  if (use_decompress_kernel > 0) {
    unsigned int nnz = 0;
    long long n_grid_points = 0, n_dense_grid_points = 0;
    unsigned long long *grid_point_array; 
    n_grid_points = M * K;
    grid_point_array = (unsigned long long *) malloc(n_grid_points * sizeof(unsigned long long));
    n_dense_grid_points = (long long) ((double)(1.0-sparse_frac) * n_grid_points);
    auto t0 = getTime();  
#if defined(_OPENMP)
# pragma omp parallel for
#endif
    for (i = 0; i < n_grid_points; i++) {
      grid_point_array[i] = i;
    }
    auto t1 = getTime();
    printf("Setting up grid point array takes %.5g secs\n", t1-t0);   
    /* Pemute array of n grid points and consider densifying on the ones with id <= n_dense_grid_points */
    shuffle_array(grid_point_array, n_grid_points);
    t0 = getTime();
    printf("Shuffling grid point array takes %.5g secs\n", t0-t1);
    qsort(grid_point_array, n_dense_grid_points, sizeof(unsigned long long), ullcompare);
    t1 = getTime();
    printf("Sorting grid point array takes %.5g secs\n", t1-t0);   

#if defined(_OPENMP)
# pragma omp parallel for collapse(2)
#endif
    for ( l_i = 0; l_i < M; l_i++ ) {
      for ( l_j = 0; l_j < K; l_j++ ) {
        /* float tmp = (float)libxsmm_rng_f64();
        if (tmp >= sparse_frac) {*/
        float *zero_pt = (float*)naive_filter + l_i * K + l_j; 
        if (is_dense_grid_point(l_i * K + l_j, n_dense_grid_points, grid_point_array)) {
          if ((*zero_pt) != 0.0) {
            #pragma omp atomic   
            nnz += 1;
          }
        } else {
          (*zero_pt) = 0.0; 
        }
      }
    }
    t0 = getTime();
    printf("Enforcing sparsity takes %.5g secs\n", t0-t1);     
    printf("Total sparsity enforced is %.3g\n", 100.0-100.0*nnz/(M*K));
    free(grid_point_array);

    compressed_filter = (DType*)libxsmm_aligned_malloc( nnz*sizeof(DType), 2097152);
    compressed_a_bitmap = (unsigned int*)libxsmm_aligned_malloc( (M/8)*K, 2097152);
    compressed_a_offsets = (long long*)libxsmm_aligned_malloc( Mb * sizeof(long long), 2097152);
  }

  if (int8_gemm > 0) {
    naive_input_i8  = (unsigned char*)libxsmm_aligned_malloc( K*N*sizeof(unsigned char ), 2097152);
    naive_output_i8 = (unsigned char*)libxsmm_aligned_malloc( M*N*sizeof(unsigned char ), 2097152);
    naive_output_opt_i8 = (unsigned char*)libxsmm_aligned_malloc( M*N*sizeof(unsigned char ), 2097152);
    naive_filter_i8 = (char*)libxsmm_aligned_malloc( M*K*sizeof(float), 2097152);
    for (i = 0; i < K*N; i++) naive_input_i8[i] = (unsigned char) (get_random_pos_p5_num() * 20.0);
    for (i = 0; i < M*N; i++) naive_output_i8[i] = (unsigned char) (get_random_pos_p5_num() * 20.0);
    for (i = 0; i < M*K; i++) naive_filter_i8[i] = (char) (get_random_posneg_p5_num() * 40.0);
    /* Use the following 8-bit formating routines */
    matrix_copy_NC_to_NCNC_bf8(  (libxsmm_bfloat8*)naive_input_i8,  (libxsmm_bfloat8*)ACT[0],     1, N, K, bn, bk );
    //matrix_copy_NC_to_NCNC_bf8(  (libxsmm_bfloat8*)naive_output_i8, (libxsmm_bfloat8*)ACT[n_layers],     1, N, M, bn, bm );
    for (i = 0; i < n_layers; i++) {
      matrix_copy_KC_to_KCCK_bf8( (libxsmm_bfloat8*)naive_filter_i8, (libxsmm_bfloat8*)WGT[i]       , K, M, bk, bm );
    }
  } else if ((sizeof(DType) == 1) && (int8_gemm == 0)) {
    libxsmm_rne_convert_fp32_bf8( naive_input,     (libxsmm_bfloat8*)naive_input_bf16,     N*K );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)naive_input_bf16, naive_input, N*K);
    libxsmm_rne_convert_fp32_bf8( naive_output,    (libxsmm_bfloat8*)naive_output_bf16,    N*M );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)naive_output_bf16, naive_output, N*M);
    libxsmm_rne_convert_fp32_bf8( naive_filter,    (libxsmm_bfloat8*)naive_filter_bf16,    M*K );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)naive_filter_bf16, naive_filter, M*K);
    matrix_copy_NC_to_NCNC_bf8(  (libxsmm_bfloat8*)naive_input_bf16,  (libxsmm_bfloat8*)ACT[0],     1, N, K, bn, bk );
    //matrix_copy_NC_to_NCNC_bf8(  (libxsmm_bfloat8*)naive_output_bf16, (libxsmm_bfloat8*)ACT[n_layers],     1, N, M, bn, bm );
    for (i = 0; i < n_layers; i++) {
      matrix_copy_KC_to_KCCK_bf8( (libxsmm_bfloat8*)naive_filter_bf16, (libxsmm_bfloat8*)WGT[i]       , K, M, bk, bm );
    }
  } else if (sizeof(DType) == 2) {
    if (f16_gemm > 0) {
      libxsmm_rne_convert_fp32_f16( naive_input,     (libxsmm_float16*)naive_input_bf16,     N*K );
      libxsmm_convert_f16_f32( (libxsmm_float16*)naive_input_bf16, naive_input, N*K);
      libxsmm_rne_convert_fp32_f16( naive_output,    (libxsmm_float16*)naive_output_bf16,    N*M );
      libxsmm_convert_f16_f32( (libxsmm_float16*)naive_output_bf16, naive_output, N*M);
      libxsmm_rne_convert_fp32_f16( naive_filter,    (libxsmm_float16*)naive_filter_bf16,    M*K );
      libxsmm_convert_f16_f32( (libxsmm_float16*)naive_filter_bf16, naive_filter, M*K);
      libxsmm_rne_convert_fp32_f16( naive_input,     (libxsmm_float16*)(libxsmm_bfloat16*)ACT[0],     N*K );
      for (i = 0; i < n_layers; i++) {
        matrix_copy_KC_to_KCCK_f16( (libxsmm_float16*)naive_filter_bf16, (libxsmm_float16*)WGT[i]       , K, M, bk, bm );
      }
      if (use_decompress_kernel > 0) {
        /* Compress weights and create bitmap along with aux structure */
        l_j = 0;
        l_c = 0;
        auto t1 = getTime();     
        compressed_a_offsets[0] = l_c;
        __m512i zero_vreg = _mm512_setzero_epi32();
        for (l_i = 0; l_i < Mb; l_i++) {
          for (i = 0; i < K*bm; i+=32 ) {
            int n_elts;
            __m512i  vreg_dense = _mm512_loadu_si512 ((DType*)WGT[0]+l_i * K * bm + i);
            __mmask32 cur_mask = _mm512_cmp_epu16_mask (zero_vreg, vreg_dense, _MM_CMPINT_NE);
            unsigned int int_bitmask = _cvtmask32_u32 (cur_mask);
            compressed_a_bitmap[l_j] = int_bitmask;
            l_j++;
            n_elts = _popcnt32(int_bitmask);
            _mm512_mask_compressstoreu_epi16((DType*)compressed_filter+l_c, cur_mask, vreg_dense);
            l_c += n_elts;
          }
          if (l_i + 1 < Mb) compressed_a_offsets[l_i+1] = l_c;
        }
        auto t0 = getTime();
        printf("Compressing weight takes %.5g secs\n", t0-t1);  
      }  
    } else {
      libxsmm_rne_convert_fp32_bf16( naive_input,     (libxsmm_bfloat16*)naive_input_bf16,     N*K );
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_input_bf16, naive_input, N*K);
      libxsmm_rne_convert_fp32_bf16( naive_output,    (libxsmm_bfloat16*)naive_output_bf16,    N*M );
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_output, N*M);
      libxsmm_rne_convert_fp32_bf16( naive_filter,    (libxsmm_bfloat16*)naive_filter_bf16,    M*K );
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_filter_bf16, naive_filter, M*K);
      matrix_copy_NC_to_NCNC_bf16(  (libxsmm_bfloat16*)naive_input_bf16,  (libxsmm_bfloat16*)ACT[0],     1, N, K, bn, bk );  
      //matrix_copy_NC_to_NCNC_bf16(  (libxsmm_bfloat16*)naive_output_bf16, (libxsmm_bfloat16*)ACT[n_layers],     1, N, M, bn, bm );
      for (i = 0; i < n_layers; i++) {
        matrix_copy_KC_to_KCCK_bf16( (libxsmm_bfloat16*)naive_filter_bf16, (libxsmm_bfloat16*)WGT[i]       , K, M, bk, bm );
      }
    }
  } else {
    matrix_copy_NC_to_NCNC( naive_input,     (float*)ACT[0],     1, N, K, bn, bk );
    //matrix_copy_NC_to_NCNC( naive_output,    (float*)ACT[n_layers],     1, N, M, bn, bm );
    for (i = 0; i < n_layers; i++) {
      matrix_copy_KC_to_KCCK( naive_filter,    (float*)WGT[i]       , K, M, bk, bm );
    }
  }
   
  // Setup TPP kernels
  auto l_flags    = ((sizeof(DType) == 2 && f16_gemm == 0) || sizeof(DType) == 1) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tc_flags = ((sizeof(DType) == 2 && f16_gemm == 0) || sizeof(DType) == 1) ? ( LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  auto l_tr_flags = ((sizeof(DType) == 2 && f16_gemm == 0) || sizeof(DType) == 1) ? ( LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'N');
  
  auto dtype      = (sizeof(DType) == 2) ? ((f16_gemm > 0) ? LIBXSMM_DATATYPE_F16 :  LIBXSMM_DATATYPE_BF16) : ((sizeof(DType) == 1) ? ((int8_gemm == 0) ? LIBXSMM_DATATYPE_BF8 : LIBXSMM_DATATYPE_I8) : LIBXSMM_DATATYPE_F32);
  auto l_shape = libxsmm_create_gemm_shape( bm, bn, bk, bm, bk, bm, dtype, dtype, (int8_gemm > 0) ? LIBXSMM_DATATYPE_F32 : dtype, (int8_gemm > 0) ? LIBXSMM_DATATYPE_I32 : LIBXSMM_DATATYPE_F32 );
  auto l_f16_shape = libxsmm_create_gemm_shape( bm, bn, K, bm, K, M, dtype, dtype, dtype, dtype );
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm*bk*sizeof(DType), bk*bn*sizeof(DType), brcount );
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bm*bn, 1, bm*bn, bm*bn, dtype, dtype, dtype);

  if (int8_gemm > 0) {
    l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_B_UNSIGNED;
    l_shape = libxsmm_create_gemm_shape( bm, bn, bk, bm, bk, bm, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_I32 );
    l_unary_shape = libxsmm_create_meltw_unary_shape(bm*bn, 1, bm*bn, bm*bn, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
  } else {
    if (brcount == Kb) l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  }

  if (use_decompress_kernel > 0) {
    l_flags |= LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK;
  }

  auto zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
  auto tileconfig_kernel  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
  auto tilerelease_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
  auto brgemm_kernel      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  auto f16_gemm_kernel    = (f16_gemm > 0) ? libxsmm_dispatch_gemm_v2( l_f16_shape, l_flags, l_prefetch_flags ) : tileconfig_kernel;
  
  // Setup fused TPP kernels
  libxsmm_gemmfunction_ext brgemm_kernel_fused;
  libxsmm_meltwfunction_unary copy_colbias_kernel;
  libxsmm_meltwfunction_unary relu_kernel;
  libxsmm_meltwfunction_unary quant_kernel;

  if (int8_gemm > 0) {
    auto l_quant_unary_shape = libxsmm_create_meltw_unary_shape(bm, bn, bm, bm, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_I8, LIBXSMM_DATATYPE_F32);
    // Create quant TPP
    quant_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_QUANT, l_quant_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

    if (fuse_bias > 0) {
      auto l_colbias_unary_shape = libxsmm_create_meltw_unary_shape(bm, bn, bm, bm, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
      // Create copy colbias TPP
      copy_colbias_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_colbias_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL);  
    }

    if (fuse_relu > 0) {
      auto l_relu_unary_shape = libxsmm_create_meltw_unary_shape(bm, bn, bm, bm, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
      // Create relu TPP
      relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU, l_relu_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
    }
  } else {
    if (fuse_bias > 0) {
      auto l_colbias_unary_shape = libxsmm_create_meltw_unary_shape(bm, bn, bm, bm, dtype, dtype, dtype);
      // Create copy colbias TPP
      copy_colbias_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_colbias_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL);  
    }

    if (fuse_relu > 0) {
      auto l_relu_unary_shape = libxsmm_create_meltw_unary_shape(bm, bn, bm, bm, dtype, dtype, LIBXSMM_DATATYPE_F32);
      // Create relu TPP
      relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU, l_relu_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
    }

    if (fuse_bias > 0 || fuse_relu > 0) {
      libxsmm_gemm_ext_unary_argops   l_argops;
      libxsmm_gemm_ext_binary_postops l_postops;
      auto l_flags_new = l_flags;
      // Create fused GEMM TPP
      l_flags_new |= LIBXSMM_GEMM_FLAG_BETA_0;
      memset( &l_argops,  0, sizeof(libxsmm_gemm_ext_unary_argops  ) );
      memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );
      if (fuse_bias > 0) {
        l_postops.d_in_type      = dtype;
        l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
        l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
        l_postops.ldd            = bm;
      }
      if (fuse_relu > 0) {
        l_argops.cp_unary_flags   = LIBXSMM_MELTW_FLAG_UNARY_NONE;
        l_argops.cp_unary_type    = LIBXSMM_MELTW_TYPE_UNARY_RELU;
        l_argops.ldcp             = bm;
      }
      brgemm_kernel_fused = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags_new, l_prefetch_flags, l_brconfig, l_argops, l_postops );
    }
  }

  // Compute reference if requested
  if (check_correctness) {
    naive_fullyconnected_t naive_param;
    naive_param.N = N;
    naive_param.C = K;
    naive_param.K = M;
    naive_param.fuse_type = 0;
    if (fuse_bias > 0 && fuse_relu == 0) {
      naive_param.fuse_type = 1;
    }
    if (fuse_bias == 0 && fuse_relu > 0) {
      naive_param.fuse_type = 2;
    }
    if (fuse_bias > 0 && fuse_relu > 0) {
      naive_param.fuse_type = 3;
    }
    for (i = 0; i < n_layers; i++) {
      if (int8_gemm > 0) {
        if (i % 2 == 0) {
          naive_fullyconnected_fused_int8(&naive_param, naive_input_i8, naive_output_i8, naive_filter_i8, (fuse_bias > 0) ? naive_bias[i] : NULL, &scf_quant[i], itm_f32_out );
        } else {
          naive_fullyconnected_fused_int8(&naive_param, naive_output_i8, naive_input_i8, naive_filter_i8, (fuse_bias > 0) ? naive_bias[i] : NULL, &scf_quant[i], itm_f32_out );
        }
      } else {
        if (i % 2 == 0) {
          naive_fullyconnected_fused_fp(&naive_param, naive_input, naive_output, naive_filter, (fuse_bias > 0) ? naive_bias[i] : NULL);
          /* Also downconvert and upconvert reference  */
          if (sizeof(DType) == 1) {
            libxsmm_rne_convert_fp32_bf8( naive_output,     (libxsmm_bfloat8*)naive_output_bf16,     N*M );
            libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)naive_output_bf16, naive_output, N*M);
          } else if (sizeof(DType) == 2) {
            if (f16_gemm > 0) {
              libxsmm_rne_convert_fp32_f16( naive_output,     (libxsmm_float16*)naive_output_bf16,     N*M );
              libxsmm_convert_f16_f32( (libxsmm_float16*)naive_output_bf16, naive_output, N*M);
            } else {
              libxsmm_rne_convert_fp32_bf16( naive_output,     (libxsmm_bfloat16*)naive_output_bf16,     N*M );
              libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_output, N*M);          
            }
          } 
        } else {
          naive_fullyconnected_fused_fp(&naive_param, naive_output, naive_input, naive_filter, (fuse_bias > 0) ? naive_bias[i] : NULL);
          /* Also downconvert and upconvert reference  */
          if (sizeof(DType) == 1) {
            libxsmm_rne_convert_fp32_bf8( naive_input,     (libxsmm_bfloat8*)naive_output_bf16,     N*M );
            libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)naive_output_bf16, naive_input, N*M);
          } else if (sizeof(DType) == 2) {
            if (f16_gemm > 0) {
              libxsmm_rne_convert_fp32_f16( naive_input,     (libxsmm_float16*)naive_output_bf16,     N*M );
              libxsmm_convert_f16_f32( (libxsmm_float16*)naive_output_bf16, naive_input, N*M);          
            } else {
              libxsmm_rne_convert_fp32_bf16( naive_input,     (libxsmm_bfloat16*)naive_output_bf16,     N*M );
              libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_input, N*M);
            }
          } 
        }
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
#if 0
  printf("K factors are: ");
  for (auto i = 0; i < k_factors.size(); i++) {
    printf("%d ", k_factors[i]);
  }
  printf("\n");
  printf("M factors are: ");
  for (auto i = 0; i < m_factors.size(); i++) {
    printf("%d ", m_factors[i]);
  }
  printf("\n");
  printf("N factors are: ");
  for (auto i = 0; i < n_factors.size(); i++) {
    printf("%d ", n_factors[i]);
  }
  printf("\n");
#endif
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
  auto gemm_loop = ThreadedLoop<3>({
      LoopSpecs{0, Kb, k_step, {l1_k_step, l0_k_step}},   // Logical K loop specs
      LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // Logical M loop specs
      LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // Logical N loop specs
      loop_specs_str);
  auto t1 = getTime();

  // Warmup iteration for i-caches
  if (f16_gemm > 0) {
    for (i = 0; i < n_layers; i++) {
      gemm_loop(
          [&](int* ind) {
            int i_k = ind[0], i_m = ind[1], i_n = ind[2];
              libxsmm_gemm_param gemm_param;
              if (use_decompress_kernel > 0) {
                gemm_param.a.primary = (void*)((DType*)compressed_filter + compressed_a_offsets[i_m]);
                gemm_param.a.secondary = (void*)((char*)compressed_a_bitmap + i_m * K * (bm/8));
              } else {
                gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm );
              }
              gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * bn * K);
              gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * bn * M + i_m  * bm );
              f16_gemm_kernel( &gemm_param );
          },
          [&]() {},
          [&]() {});
    }
  } else if (int8_gemm == 0) {
    for (i = 0; i < n_layers; i++) {
      gemm_loop(
          [&](int* ind) {
            int i_k = ind[0], i_m = ind[1], i_n = ind[2];
            if (fuse_bias > 0 || fuse_relu > 0) {
              if (brcount == Kb) {
                libxsmm_gemm_ext_param gemm_param_ext;
                gemm_param_ext.op.tertiary = (void*)&brcount;
                gemm_param_ext.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
                gemm_param_ext.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
                gemm_param_ext.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
                if (fuse_bias > 0) {
                  gemm_param_ext.d.primary = (void*)((DType*)BIAS[i] + i_m * bm );
                }
                brgemm_kernel_fused( &gemm_param_ext );             
              } else {
                libxsmm_gemm_param gemm_param;
                gemm_param.op.tertiary = (void*)&brcount;
                gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
                gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
                gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
                if (i_k == 0) {
                  if (fuse_bias > 0) {
                    libxsmm_meltw_unary_param copy_colbias_param;
                    copy_colbias_param.in.primary = (void*)(void*)((DType*)BIAS[i] + i_m * bm );
                    copy_colbias_param.out.primary = (void*)gemm_param.c.primary;
                    copy_colbias_kernel( &copy_colbias_param );
                  } else {
                    libxsmm_meltw_unary_param zero_param;
                    zero_param.out.primary = (void*)gemm_param.c.primary;
                    zero_kernel( &zero_param );
                  }
                }
                brgemm_kernel( &gemm_param );
                if (fuse_relu > 0) {
                  if (i_k + k_step >= Kb) {
                    libxsmm_meltw_unary_param relu_param;
                    relu_param.in.primary =  (void*)gemm_param.c.primary;
                    relu_param.out.primary = (void*)gemm_param.c.primary;
                    relu_kernel( &relu_param );
                  }
                }
              }
            } else {
              libxsmm_gemm_param gemm_param;
              gemm_param.op.tertiary = (void*)&brcount;
              gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
              gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
              gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
              if ((i_k == 0) && (brcount != Kb)) {
                libxsmm_meltw_unary_param zero_param;
                zero_param.out.primary = (void*)gemm_param.c.primary;
                zero_kernel( &zero_param );
              }
              brgemm_kernel( &gemm_param );
            }
          },
          [&]() {if (sizeof(DType) == 2) tileconfig_kernel(NULL);},
          [&]() {if (sizeof(DType) == 2) tilerelease_kernel(NULL);});
    }
  } else {
    for (i = 0; i < n_layers; i++) {
      gemm_loop(
          [&](int* ind) {
            int i_k = ind[0], i_m = ind[1], i_n = ind[2];
            const float float_one = 1.0f;
            libxsmm_gemm_param gemm_param;
            gemm_param.op.tertiary = (void*)&brcount;
            gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
            gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
            gemm_param.c.primary = (void*)((float*)itm_f32_out + i_n * M * bn + i_m * bn * bm );
            gemm_param.c.tertiary = (void*)&float_one;
            if (i_k == 0) {
              if (fuse_bias > 0) {
                libxsmm_meltw_unary_param copy_colbias_param;
                copy_colbias_param.in.primary = (void*)((float*)naive_bias[i] + i_m * bm );
                copy_colbias_param.out.primary = (void*)gemm_param.c.primary;
                copy_colbias_kernel( &copy_colbias_param );
              } else {
                libxsmm_meltw_unary_param zero_param;
                zero_param.out.primary = (void*)gemm_param.c.primary;
                zero_kernel( &zero_param );
              }
            }
            brgemm_kernel( &gemm_param );
            if (fuse_relu > 0) {
              if (i_k + k_step >= Kb) {
                libxsmm_meltw_unary_param relu_param;
                relu_param.in.primary =  (void*)gemm_param.c.primary;
                relu_param.out.primary = (void*)gemm_param.c.primary;
                relu_kernel( &relu_param );
              }
            }
            /* Quantize output to i8*/
            if (i_k + k_step >= Kb) {
              libxsmm_meltw_unary_param quant_param;
              quant_param.in.primary  = (void*)gemm_param.c.primary;
              quant_param.in.secondary= (void*)&scf_quant[i];
              quant_param.out.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
              quant_kernel( &quant_param );
            }
          },
          [&]() {tileconfig_kernel(NULL);},
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
  printf("Precision  : %s\n", prec_string);
  printf("Activation : %s\n", fuse_string);
  printf("Ring buffer: %d\n", use_ping_pong_bufs);
  if (check_correctness) {
    if (int8_gemm > 0) {
      matrix_copy_NCNC_to_NC_bf8( (libxsmm_bfloat8*)ACT[n_layers], (libxsmm_bfloat8*)naive_output_opt_i8, 1, N, M, bn, bm );
    } else if (sizeof(DType) == 1) {
      matrix_copy_NCNC_to_NC_bf8( (libxsmm_bfloat8*)ACT[n_layers], (libxsmm_bfloat8*)naive_output_bf16, 1, N, M, bn, bm );
      libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)naive_output_bf16, naive_output_opt, N*M );
    } else if (sizeof(DType) == 2) {
      if (f16_gemm > 0) {
        libxsmm_convert_f16_f32( (libxsmm_float16*)ACT[n_layers], naive_output_opt, N*M );  
      } else {
        matrix_copy_NCNC_to_NC_bf16( (libxsmm_bfloat16*)ACT[n_layers], (libxsmm_bfloat16*)naive_output_bf16, 1, N, M, bn, bm );
        libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)naive_output_bf16, naive_output_opt, N*M );
      }
    } else {
      matrix_copy_NCNC_to_NC( (float*)ACT[n_layers], naive_output_opt, 1, N, M, bn, bm );
    }
    printf("##########################################\n");
    printf("#           Correctness                  #\n");
    printf("##########################################\n");
    if (int8_gemm == 0) {
      if (n_layers % 2 == 1) {
        libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*M, 1, naive_output, naive_output_opt, 0, 0);
      } else {
        libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, N*M, 1, naive_input, naive_output_opt, 0, 0);
      }
    } else {
      if (n_layers % 2 == 1) {
        libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_I8, N*M, 1, naive_output_i8, naive_output_opt_i8, 0, 0);
      } else {
        libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_I8, N*M, 1, naive_input_i8, naive_output_opt_i8, 0, 0);
      }
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
  if (f16_gemm > 0) {
    for (long it = 0; it < n_iters; it++) {
      for (i = 0; i < n_layers; i++) {
        gemm_loop(
            [&](int* ind) {
              int i_k = ind[0], i_m = ind[1], i_n = ind[2];
                libxsmm_gemm_param gemm_param;
                if (use_decompress_kernel > 0) {
                  gemm_param.a.primary = (void*)((DType*)compressed_filter + compressed_a_offsets[i_m]);
                  gemm_param.a.secondary = (void*)((char*)compressed_a_bitmap + i_m * K * (bm/8));
                } else {
                  gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm );
                }
                gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * bn * K);
                gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * bn * M + i_m  * bm );
                f16_gemm_kernel( &gemm_param );
            },
            [&]() {},
            [&]() {});
      }
    }
  } else if (int8_gemm == 0) {
    for (long it = 0; it < n_iters; it++) {
      for (i = 0; i < n_layers; i++) {
        gemm_loop(
            [&](int* ind) {
              int i_k = ind[0], i_m = ind[1], i_n = ind[2];
              if (fuse_bias > 0 || fuse_relu > 0) {
                if (brcount == Kb) {
                  libxsmm_gemm_ext_param gemm_param_ext;
                  gemm_param_ext.op.tertiary = (void*)&brcount;
                  gemm_param_ext.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
                  gemm_param_ext.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
                  gemm_param_ext.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
                  if (fuse_bias > 0) {
                    gemm_param_ext.d.primary = (void*)((DType*)BIAS[i] + i_m * bm );
                  }
                  brgemm_kernel_fused( &gemm_param_ext );             
                } else {
                  libxsmm_gemm_param gemm_param;
                  gemm_param.op.tertiary = (void*)&brcount;
                  gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
                  gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
                  gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
                  if (i_k == 0) {
                    if (fuse_bias > 0) {
                      libxsmm_meltw_unary_param copy_colbias_param;
                      copy_colbias_param.in.primary = (void*)(void*)((DType*)BIAS[i] + i_m * bm );
                      copy_colbias_param.out.primary = (void*)gemm_param.c.primary;
                      copy_colbias_kernel( &copy_colbias_param );
                    } else {
                      libxsmm_meltw_unary_param zero_param;
                      zero_param.out.primary = (void*)gemm_param.c.primary;
                      zero_kernel( &zero_param );
                    }
                  }
                  brgemm_kernel( &gemm_param );
                  if (fuse_relu > 0) { 
                    if (i_k + k_step >= Kb) {
                      libxsmm_meltw_unary_param relu_param;
                      relu_param.in.primary =  (void*)gemm_param.c.primary;
                      relu_param.out.primary = (void*)gemm_param.c.primary;
                      relu_kernel( &relu_param );
                    }
                  }
                }
              } else {
                libxsmm_gemm_param gemm_param;
                gemm_param.op.tertiary = (void*)&brcount;
                gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
                gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
                gemm_param.c.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
                if ((i_k == 0) && (brcount != Kb)) {
                  libxsmm_meltw_unary_param zero_param;
                  zero_param.out.primary = (void*)gemm_param.c.primary;
                  zero_kernel( &zero_param );
                }
                brgemm_kernel( &gemm_param );
              }
            },
            [&]() {if (sizeof(DType) == 2) tileconfig_kernel(NULL);},
            [&]() {if (sizeof(DType) == 2) tilerelease_kernel(NULL);});
      }
    }
  } else {
    for (long it = 0; it < n_iters; it++) {
      for (i = 0; i < n_layers; i++) {
        gemm_loop(
            [&](int* ind) {
              int i_k = ind[0], i_m = ind[1], i_n = ind[2];
              const float float_one = 1.0f;
              libxsmm_gemm_param gemm_param;
              gemm_param.op.tertiary = (void*)&brcount;
              gemm_param.a.primary = (void*)((DType*)WGT[i] + i_m * K * bm + i_k * bk * bm );
              gemm_param.b.primary = (void*)((DType*)ACT[i] + i_n * K * bn + i_k * bk * bn );
              gemm_param.c.primary = (void*)((float*)itm_f32_out + i_n * M * bn + i_m * bn * bm );
              gemm_param.c.tertiary = (void*)&float_one;
              if (i_k == 0) {
                if (fuse_bias > 0) {
                  libxsmm_meltw_unary_param copy_colbias_param;
                  copy_colbias_param.in.primary = (void*)((float*)naive_bias[i] + i_m * bm );
                  copy_colbias_param.out.primary = (void*)gemm_param.c.primary;
                  copy_colbias_kernel( &copy_colbias_param );
                } else {
                  libxsmm_meltw_unary_param zero_param;
                  zero_param.out.primary = (void*)gemm_param.c.primary;
                  zero_kernel( &zero_param );
                }
              }
              brgemm_kernel( &gemm_param );
              if (fuse_relu > 0) {
                if (i_k + k_step >= Kb) {
                  libxsmm_meltw_unary_param relu_param;
                  relu_param.in.primary =  (void*)gemm_param.c.primary;
                  relu_param.out.primary = (void*)gemm_param.c.primary;
                  relu_kernel( &relu_param );
                }
              }
              /* Quantize output to i8*/
              if (i_k + k_step >= Kb) {
                libxsmm_meltw_unary_param quant_param;
                quant_param.in.primary  = (void*)gemm_param.c.primary;
                quant_param.in.secondary= (void*)&scf_quant[i];
                quant_param.out.primary = (void*)((DType*)ACT[i+1] + i_n * M * bn + i_m * bn * bm );
                quant_kernel( &quant_param );
              }
            },
            [&]() {tileconfig_kernel(NULL);},
            [&]() {tilerelease_kernel(NULL);});
      }
    }
  }
  auto t_end = getTime();
  
  // Model GEMM
  auto t_trace_start = getTime();
  double modeled_time = 0.0;
  if (use_model > 0) {
    set_tensor_metadata(bm, bn, bk, brcount, sizeof(DType), &tensor_metadata);
    for (i = 0; i < n_layers; i++) {
      gemm_loop(
          [&](int* ind) {
            int nc = ind[0], s1 = ind[1], nk = ind[2];
            char record[256];
            int my_thread_id = omp_get_thread_num();
            sprintf(record, "WGT%d[%d][%d]", i, s1, nc);
            std::string a_access(record);
            inp_trace[my_thread_id].push_back(a_access);
            sprintf(record, "ACT%d[%d][%d]", i,  nk, nc);
            std::string b_access(record);
            inp_trace[my_thread_id].push_back(b_access);
            sprintf(record, "ACT%d[%d][%d]", i+1,  nk, s1);
            std::string c_access(record);
            inp_trace[my_thread_id].push_back(c_access);
          },
          [&]() {},
          [&]() {});
    }
  }
  auto t2 = getTime();
  if (use_model > 0) {
    modeled_time = tensor_contraction_cost_estimator(
        PARALLEL_TRACES, inp_trace, tensor_metadata, my_platform);
  }
  auto t3 = getTime();

  // Print performance/model numbers
  double gflop = (2.0*(double)n_layers*(double)M*(double)N*(double)K) / (1000*1000*1000);
  printf("Time is %.5g ms (%.5g GFLOPS)\n", 1000.0*(t_end-t_start)/(1.0*n_iters), gflop/((t_end-t_start)/(1.0*n_iters)));
  printf("Effective A BW is %.5g GB/s\n", (((double)sizeof(DType)*(double)n_layers*(double)M*(double)K) / (1024*1024*1024))/((t_end-t_start)/(1.0*n_iters)));
  if (use_model > 0) {
    printf("Model time gemm is %.5g ms (%.5g GFLOPS)\n", modeled_time, gflop/(modeled_time/1000.0));
    printf("Tracing takes %.5g ms and modeling takes %.5g ms\n", 1000.0*(t2-t_trace_start), 1000.0*(t3-t2));
    printf("Compilation time is %.5g s\n", t1-t0);
    printf("MODELED %.5g %s_%d_%d_%d_%d_%d_%d_bf%d_threads%d\n", gflop/(modeled_time/1000.0), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads());
  }
  printf("MEASURE %.5g %s_%d_%d_%d_%d_%d_%d_bf%d_threads%d\n", gflop/((t_end-t_start)/(1.0*n_iters)), loop_specs_str, M, N, K, bm, bn, bk, kbf, omp_get_max_threads());

  // Free buffers
  libxsmm_free(itm_f32_out);
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_output_opt);
  libxsmm_free(naive_input_bf16);
  libxsmm_free(naive_output_bf16);
  libxsmm_free(naive_filter_bf16);

  if (use_decompress_kernel > 0) {
    libxsmm_free(compressed_filter);
    libxsmm_free(compressed_a_bitmap);
    libxsmm_free(compressed_a_offsets);
  }

  for (i = 0; i < (n_layers+1); i++) {
    if (use_ping_pong_bufs == 0) { 
      libxsmm_free(ACT[i]);
    } else {
      if (i < 2) {
        libxsmm_free(ACT[i]);
      }
    }
    if (i < n_layers) {
      if (fuse_bias > 0) {
        libxsmm_free(BIAS[i]);
        libxsmm_free(naive_bias[i]);
      }
      libxsmm_free(WGT[i]);
    }
  }
  if ((sizeof(DType) == 1) && (int8_gemm == 1)) {
    libxsmm_free(naive_input_i8);
    libxsmm_free(naive_output_i8);
    libxsmm_free(naive_output_opt_i8);
    libxsmm_free(naive_filter_i8);
  }
  free(ACT);
  free(WGT);
  free(BIAS);
  free(naive_bias);
  free(scf_quant);

  return 0;
}

int main(int argc, char** argv) {
  int use_prec_bf16 = 0;
  int cl_precision = 4;
  char cl_str_precision[256];
  const char* const env_prec_str = getenv("USE_BF16");
  if (0 == env_prec_str) {
    use_prec_bf16 = 0;
    if (argc > 13) {
      sprintf(cl_str_precision, "%s", argv[13]);
      if (strcmp(cl_str_precision, "FP32") == 0) {
        cl_precision = 4;
      } else if (strcmp(cl_str_precision, "BF16") == 0) {
        cl_precision = 2;
      } else if (strcmp(cl_str_precision, "BF8") == 0) {
        cl_precision = 1;
      } else if (strcmp(cl_str_precision, "INT8") == 0) {
        cl_precision = 5;
      } else if (strcmp(cl_str_precision, "F16") == 0) {
        cl_precision = 6;
      } else {
        printf("Unsupported precision... Supported precisions are FP32, BF16, F16, BF8, INT8. Exiting...\n");
        return 0;
      }
    }
  } else {
    use_prec_bf16 = atoi(env_prec_str);
    if (argc > 13) {
      cl_precision = atoi(argv[13]);
    }
  }
  if (use_prec_bf16 == 0) {
    if (cl_precision == 6) {
      return gemm_benchmark<libxsmm_float16>(argc, argv);
    } if (cl_precision == 5) {
      return gemm_benchmark<char>(argc, argv);
    } else if (cl_precision == 4) {
      return gemm_benchmark<float>(argc, argv);
    } else if (cl_precision == 2) {
      return gemm_benchmark<libxsmm_bfloat16>(argc, argv);
    } else if (cl_precision == 1) {
      return gemm_benchmark<libxsmm_bfloat8>(argc, argv);
    }
  } else {
    return gemm_benchmark<libxsmm_bfloat16>(argc, argv);  
  }
}

