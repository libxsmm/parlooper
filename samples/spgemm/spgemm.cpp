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

#define HARDWIRED_SPARSITY_KNOB

int ullcompare( const void* a , const void* b ) {
  const unsigned long long aull = *( const unsigned long* )a;
  const unsigned long long bull = *( const unsigned long* )b;
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

template<typename DType>
int spgemm_benchmark(int argc, char** argv) {
  // Setup default SPGEMM sizes
  int check_correctness = 1;
  char loop_specs_str[256] = "aBC";  
  long M = 128, N = 256, K = 512;
  long bm = 32, N_target_blocks = 8;
  long Mb = M/bm;
  long Nb = N_target_blocks;
  long Kb = 1;
  long long n_grid_points = 0, n_dense_grid_points = 0;
  double sparse_frac = 0.8;
  unsigned int use_bf16 = 1;
  unsigned int use_bcsc = 1;
  unsigned int bcsc_bk = 4;
  unsigned int bcsc_bn = 2;
  unsigned int sparse_block_bk = 4;
  unsigned int sparse_block_bn = 2;
  unsigned int hardwired_sparsity = 1;
  long k_block_size = (K/bcsc_bk)/Kb;
  unsigned int n_iters = 1;
  unsigned int use_ac_vnni = (use_bcsc > 0) ? 1 : 0;
  unsigned int vnni_block_size = (use_ac_vnni > 0) ? libxsmm_cpuid_dot_pack_factor(LIBXSMM_DATATYPE_BF16) : 1;
  unsigned int n_warmup_iters = 2;
  unsigned int c_flat = 0;
  unsigned int n_dynamic = 0;
  long i;
  unsigned long long l_start, l_end;
  double l_total;
  unsigned long long *grid_point_array;

  libxsmm_matdiff_info norms_csc, diff;
  libxsmm_matdiff_clear(&norms_csc);
  libxsmm_matdiff_clear(&diff);

  ifreq = 1.0 / getFreq();
  if (argc > 1) {
    sprintf(loop_specs_str, "%s", argv[1]);
  }
  if (argc > 2) {
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    bm = atoi(argv[5]);
    N_target_blocks = atoi(argv[6]);
    Kb = atoi(argv[7]);
    sparse_frac = atof(argv[8]);
    use_bf16 = atoi(argv[9]);
    use_bcsc = atoi(argv[10]);
    bcsc_bk = atoi(argv[11]);
    bcsc_bn = atoi(argv[12]);
    sparse_block_bk = atoi(argv[13]);
    sparse_block_bn = atoi(argv[14]);
    hardwired_sparsity = atoi(argv[15]);
    n_iters = atoi(argv[16]);
    if (argc > 17) {
      c_flat = atoi(argv[17]);
    }
    if (argc > 18) {
      n_dynamic = atoi(argv[18]);
    }
    use_ac_vnni = (use_bcsc > 0) ? 1 : 0;
    vnni_block_size = (use_ac_vnni > 0) ? ((use_bf16 > 0) ? libxsmm_cpuid_dot_pack_factor(LIBXSMM_DATATYPE_BF16) : 1) : 1;
    Mb = M/bm;
#ifdef HARDWIRED_SPARSITY_KNOB
    if (hardwired_sparsity == 0) {
      Kb = 1;
    }
#endif
    k_block_size = (K/bcsc_bk)/Kb;
  }

  n_grid_points = (N/sparse_block_bn) * (K/sparse_block_bk);
  grid_point_array = (unsigned long long *) malloc(n_grid_points * sizeof(unsigned long long));
  n_dense_grid_points = (long long) ((double)(1.0-sparse_frac) * n_grid_points);
  for (i = 0; i < n_grid_points; i++) {
    grid_point_array[i] = i;
  }
  /* Pemute array of n grid points and consider densifying on the ones with id <= n_dense_grid_points */
  shuffle_array(grid_point_array, n_grid_points);
  qsort(grid_point_array, n_dense_grid_points, sizeof(unsigned long long), ullcompare);

  // Kernel management specifics
  libxsmm_bitfield l_flags = (use_bf16 == 1) ? LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG : LIBXSMM_GEMM_FLAGS('N', 'N') ;
  libxsmm_bitfield l_tc_flags = (use_bf16 == 1) ? LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG : LIBXSMM_GEMM_FLAGS('N', 'N') ;
  libxsmm_bitfield l_tr_flags = (use_bf16 == 1) ? LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG : LIBXSMM_GEMM_FLAGS('N', 'N') ;
  if (c_flat == 0) {
    l_flags |= LIBXSMM_GEMM_FLAG_VNNI_C;
  }
  libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  
  if (hardwired_sparsity == 0) {
#ifdef HARDWIRED_SPARSITY_KNOB
    l_flags |=  LIBXSMM_GEMM_FLAG_NO_HARDWIRED_SPARSITY;
    l_tc_flags |= LIBXSMM_GEMM_FLAG_NO_HARDWIRED_SPARSITY;
    l_tr_flags |= LIBXSMM_GEMM_FLAG_NO_HARDWIRED_SPARSITY;
#else
    hardwired_sparsity = 1;
#endif
  }

  libxsmm_gemmfunction kernels_csc[(N_target_blocks+1)*Kb];
  //libxsmm_gemmfunction kernels_csc[N_target_blocks+1][Kb];
  libxsmm_gemmfunction tc_kernel;
  libxsmm_gemmfunction tr_kernel;
  libxsmm_meltwfunction_unary kernels_zero[N_target_blocks+1];

  // Allocate buffers
  unsigned int* l_colptr = NULL;
  unsigned int* l_rowidx = NULL;
  unsigned int* l_rowidx_tmp = NULL;
  float* l_b_de = (float*)libxsmm_aligned_malloc(sizeof(float) * K * N, 64);
  float* l_b_sp_csc = NULL;
  DType* l_b_sp_csc_bf16 = NULL;
  DType* l_b_sp_bcsc_bf16 = NULL;
  float* l_a = (float*)libxsmm_aligned_malloc(sizeof(float) * M * K, 64);
  libxsmm_bfloat16* l_a_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * M * K, 64);
  libxsmm_bfloat16* l_a_vnni_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * M * K, 64);
  float* l_c_gold = (float*)libxsmm_aligned_malloc(sizeof(float) * M * N, 64);
  float* l_c_asm_csc = (float*)libxsmm_aligned_malloc(sizeof(float) * M * N, 64);
  libxsmm_bfloat16* l_c_asm_csc_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * M * N, 64);
  libxsmm_bfloat16* l_c_vnni_asm_csc_bf16 = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * M * N, 64);
  libxsmm_blasint l_k, l_n;
  libxsmm_blasint l_i, l_j, l_jj;
  libxsmm_datatype dtype = (use_bf16 == 0) ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16;
  unsigned int nnz = 0;
  unsigned int *Nblocks_offsets = (unsigned int*)libxsmm_aligned_malloc(sizeof(unsigned int) * N_target_blocks, 64);

  LIBXSMM_VLA_DECL(2, float, l_p_b_de, l_b_de, K);
  LIBXSMM_VLA_DECL(3, float, l_p_a, l_a, K, bm);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, l_p_a_bf16, l_a_bf16, K, bm);
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, l_p_a_vnni_bf16, l_a_vnni_bf16, K/vnni_block_size, bm, vnni_block_size);
  LIBXSMM_VLA_DECL(3, float, l_p_c_asm_csc, l_c_asm_csc, N, bm);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, l_p_c_asm_csc_bf16, l_c_asm_csc_bf16, N, bm);
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, l_p_c_vnni_asm_csc_bf16, l_c_vnni_asm_csc_bf16, N/vnni_block_size, bm, vnni_block_size);
  LIBXSMM_VLA_DECL(3, float, l_p_c_gold, l_c_gold, N, bm);

  /* touch A */
  for ( l_i = 0; l_i < Mb; l_i++) {
    for ( l_j = 0; l_j < K; l_j++) {
      for ( l_k = 0; l_k < bm; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_a, l_i, l_j, l_k, K, bm) = (float)libxsmm_rng_f64();
        if (use_bf16 > 0) {
          libxsmm_rne_convert_fp32_bf16( &LIBXSMM_VLA_ACCESS(3, l_p_a, l_i, l_j, l_k, K, bm), &LIBXSMM_VLA_ACCESS(3, l_p_a_bf16, l_i, l_j, l_k, K, bm), 1);
          libxsmm_convert_bf16_f32( &LIBXSMM_VLA_ACCESS(3, l_p_a_bf16, l_i, l_j, l_k, K, bm), &LIBXSMM_VLA_ACCESS(3, l_p_a, l_i, l_j, l_k, K, bm), 1 );
          if (use_ac_vnni > 0) {
            LIBXSMM_VLA_ACCESS(4, l_p_a_vnni_bf16, l_i, l_j/vnni_block_size, l_k, l_j%vnni_block_size, K/vnni_block_size, bm, vnni_block_size) =
            LIBXSMM_VLA_ACCESS(3, l_p_a_bf16, l_i, l_j, l_k, K, bm);
          }
        }
      }
    }
  }

  /* touch dense B */
  for ( l_i = 0; l_i < N; l_i++ ) {
    for ( l_j = 0; l_j < K; l_j++ ) {
      LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K) = 0;
    }
  }

  /* Enforce sparsty pattern on dense B */
  if (use_bcsc > 0) {
    nnz = 0;
    for ( l_i = 0; l_i < N/sparse_block_bn; l_i++ ) {
      for ( l_j = 0; l_j < K/sparse_block_bk; l_j++ ) {
        /* float tmp = (float)libxsmm_rng_f64();
        if (tmp >= sparse_frac) {*/
        if (is_dense_grid_point(l_i * (K/sparse_block_bk) + l_j, n_dense_grid_points, grid_point_array)) {
          unsigned int l_ui = l_i * sparse_block_bn;
          unsigned int l_uj = l_j * sparse_block_bk;
          unsigned int l_di = 0, l_dj = 0;
          for (l_di = 0; l_di < sparse_block_bn; l_di++) {
            for (l_dj = 0; l_dj < sparse_block_bk; l_dj++) {
              float val = (float)libxsmm_rng_f64();
              while (val == 0) {
                val = (float)libxsmm_rng_f64();
              }
              LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_ui+l_di, l_uj+l_dj, K) = val;
            }
          }
          nnz += sparse_block_bn*sparse_block_bk;
        }
      }
    }
  } else {
    for ( l_i = 0; l_i < N; l_i++ ) {
      for ( l_j = 0; l_j < K; l_j++ ) {
        float tmp = (float)libxsmm_rng_f64();
        if ( tmp < sparse_frac ) {
          tmp = 0;
        }
        LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K) = tmp;
      }
    }
  }

  if (use_bcsc == 0) {
    nnz = 0;
    for ( l_i = 0; l_i < N; l_i++ ) {
      for ( l_j = 0; l_j < K; l_j++ ) {
        if (LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K) != 0) {
          LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K) = (float)libxsmm_rng_f64();
          while (LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K) == 0) {
            LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K) = (float)libxsmm_rng_f64();
          }
          nnz++;
        }
      }
    }
  } 

  printf("We just generated a %i x %i matrix (K x N) with %i NZ entries (%.3g sparsity)\n", K, N, nnz, 100.0-100.0*nnz/(N*K));

  /* touch C */
  for ( l_i = 0; l_i < Mb; l_i++) {
    for ( l_j = 0; l_j < N; l_j++) {
      for ( l_k = 0; l_k < bm; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N, bm) = 65535.0f;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc,  l_i, l_j, l_k, N, bm) = 65535.f;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc_bf16,  l_i, l_j, l_k, N, bm) = (libxsmm_bfloat16)65535;
        LIBXSMM_VLA_ACCESS(4, l_p_c_vnni_asm_csc_bf16,  l_i, l_j/vnni_block_size, l_k, l_j%vnni_block_size, N/vnni_block_size, bm, vnni_block_size) = (libxsmm_bfloat16)65535;
      }
    }
  }

  if (use_bcsc == 0) {
    /* create B, csc */
    l_colptr   = (unsigned int*) libxsmm_aligned_malloc( (N+1)*sizeof(unsigned int), 64 );
    l_rowidx   = (unsigned int*) libxsmm_aligned_malloc( nnz*sizeof(unsigned int),   64 );
    l_rowidx_tmp = (unsigned int*) libxsmm_aligned_malloc( nnz*sizeof(unsigned int),   64 ); 
    l_b_sp_csc = (float*       ) libxsmm_aligned_malloc( nnz*sizeof(float),          64 );
    l_b_sp_csc_bf16 = (DType*) libxsmm_aligned_malloc( nnz*sizeof(DType),          64 );
    l_k = 0;
    l_colptr[N] = nnz;
    for ( l_i = 0; l_i < N; l_i++ ) {
      l_colptr[l_i] = l_k;
      for ( l_j = 0; l_j < K; l_j++ ) {
        if ( LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K) != 0.0 ) {
          l_rowidx[l_k] = l_j;
          l_b_sp_csc[l_k] = LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K);
          if (use_bf16 > 0) {
            libxsmm_rne_convert_fp32_bf16( &l_b_sp_csc[l_k], (libxsmm_bfloat16*)&l_b_sp_csc_bf16[l_k], 1);
            libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)&l_b_sp_csc_bf16[l_k], &l_b_sp_csc[l_k], 1 );
            libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)&l_b_sp_csc_bf16[l_k], &LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K), 1 );
          }
          l_k++;
        }
      }
    }
  } else {
    /* Create B, BCSC if requested */
    unsigned int l_val_idx = 0;
    unsigned int l_nz_block_id = 0;
    l_colptr   = (unsigned int*) libxsmm_aligned_malloc( (N/bcsc_bn+1)*sizeof(unsigned int), 64 );
    l_rowidx   = (unsigned int*) libxsmm_aligned_malloc( nnz/(bcsc_bk*bcsc_bn)*sizeof(unsigned int),   64 );
    l_rowidx_tmp   = (unsigned int*) libxsmm_aligned_malloc( nnz/(bcsc_bk*bcsc_bn)*sizeof(unsigned int),   64 );
    l_b_sp_bcsc_bf16 = (DType*) libxsmm_aligned_malloc( nnz*sizeof(DType),          64 );
    l_nz_block_id = 0;
    l_colptr[N/bcsc_bn] = nnz/(bcsc_bk*bcsc_bn);
    for ( l_i = 0; l_i < N/bcsc_bn; l_i++ ) {
      l_colptr[l_i] = l_nz_block_id;
      for ( l_j = 0; l_j < K/bcsc_bk; l_j++ ) {
        unsigned int l_ui = l_i * bcsc_bn;
        unsigned int l_uj = l_j * bcsc_bk;
        /* It is a non-zero block, do something...  */
        if ( LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_ui, l_uj, K) != 0.0 ) {
          unsigned int l_di = 0, l_dj = 0;
          l_rowidx[l_nz_block_id] = l_j;
          for (l_di = 0; l_di < bcsc_bn; l_di++) {
            for (l_dj = 0; l_dj < bcsc_bk; l_dj++) {
              float val = LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_ui+l_di, l_uj+l_dj, K);
              if (use_bf16 > 0) {
                libxsmm_rne_convert_fp32_bf16( &val, (libxsmm_bfloat16*)&l_b_sp_bcsc_bf16[l_val_idx], 1);
              } else {
                l_b_sp_bcsc_bf16[l_val_idx] = val;     
              }
              l_val_idx++;
            }
          }
          l_nz_block_id++;
        }
      }
    }
  }

  /* Convert the BCSC to be in VNNI4T */
  if ((use_bcsc > 0) && (vnni_block_size == 4) && (use_bf16 > 0) && (bcsc_bk > vnni_block_size)) {
    unsigned int l_di = 0, l_dj = 0;  
    for ( l_i = 0; l_i < nnz/(bcsc_bk*bcsc_bn); l_i++) {
      libxsmm_bfloat16 tmp_block[bcsc_bk*bcsc_bn];
      memcpy(tmp_block, &l_b_sp_bcsc_bf16[l_i*(bcsc_bk*bcsc_bn)], (bcsc_bk*bcsc_bn)*sizeof(libxsmm_bfloat16));
      for (l_di = 0; l_di < bcsc_bn; l_di++) {
        for (l_dj = 0; l_dj < bcsc_bk; l_dj++) {
          l_b_sp_bcsc_bf16[l_i*(bcsc_bk*bcsc_bn) + (l_dj/vnni_block_size) * (bcsc_bn * vnni_block_size) + l_di * vnni_block_size + l_dj % vnni_block_size] = tmp_block[l_di * bcsc_bk + l_dj];    

        }
      }
    }
  }

  /* Logically partition the sparse B matrix */
  if (use_bcsc > 0) {
    unsigned int total_nnz_processed = 0;
    unsigned int nnz_entries_per_block = (nnz+N_target_blocks-1)/N_target_blocks;
    unsigned int all_done = 0;
    l_i = 0;
    l_j = 0;
    Nblocks_offsets[0] = 0;
    while (all_done == 0) {
      unsigned int nnz_so_far = 0;
      while ((nnz_so_far < nnz_entries_per_block) && (l_j < N/bcsc_bn)) {
        if (l_j + 2 <= N/bcsc_bn) {
          nnz_so_far += (l_colptr[l_j+2] - l_colptr[l_j]) * bcsc_bk * bcsc_bn;
          l_j += 2;
        } else if (l_j + 1 <= N/bcsc_bn) {
          nnz_so_far += (l_colptr[l_j+1] - l_colptr[l_j]) * bcsc_bk * bcsc_bn;
          l_j += 1;
        } else {
          /* Should not happen  */
        }
      }
      total_nnz_processed += nnz_so_far;
      l_i++;
      if (total_nnz_processed < nnz) {
        Nblocks_offsets[l_i] = l_j*bcsc_bn;
        if (l_j >= N/bcsc_bn) {
          all_done = 1; 
        }
      } else {
        Nblocks_offsets[l_i] = N;
        all_done = 1; 
      }
    }
    Nb = l_i;
    printf("Was targeting for %d logical N blocks, ended up with %d logical N blocks...\n", N_target_blocks, Nb);
  } else {
    unsigned int total_nnz_processed = 0;
    unsigned int nnz_entries_per_block = (nnz+N_target_blocks-1)/N_target_blocks;
    unsigned int all_done = 0;
    l_i = 0;
    l_j = 0;
    Nblocks_offsets[0] = 0;
    while (all_done == 0) {
      unsigned int nnz_so_far = 0;
      while ((nnz_so_far < nnz_entries_per_block) && (l_j < N)) {
        nnz_so_far += l_colptr[l_j+1] - l_colptr[l_j];
        l_j++;
      }
      total_nnz_processed += nnz_so_far;
      l_i++;
      if (total_nnz_processed < nnz) {
        Nblocks_offsets[l_i] = l_j;
        if (l_j >= N) {
          all_done = 1; 
        }
      } else {
        Nblocks_offsets[l_i] = N;
        all_done = 1; 
      }
    }
    Nb = l_i;
    printf("Was targeting for %d logical N blocks, ended up with %d logical N blocks...\n", N_target_blocks, Nb);
  }

  /* dense routine */
  memset(&LIBXSMM_VLA_ACCESS(3, l_p_c_gold, 0, 0, 0, N, bm), 0, Mb * bm * N * sizeof(float) );
  for ( l_i = 0; l_i < Mb; l_i++) {
    for ( l_j = 0; l_j < N; l_j++) {
      for ( l_jj = 0; l_jj < K; l_jj++) {
        LIBXSMM_PRAGMA_SIMD
        for (l_k = 0; l_k < bm; l_k++) {
          LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N, bm)
            +=   LIBXSMM_VLA_ACCESS(3, l_p_a, l_i, l_jj, l_k, K, bm)
               * l_b_de[(l_j*K)+l_jj];
        }
      }
    }
  }

  /* Create sparse routines */
  if (use_bcsc == 0) {
    for (l_j = 0; l_j < Kb; l_j++) {
      /* Offset rowidx values for K blocking in BCSC format  */
      for (l_n = 0; l_n < nnz; l_n++) {
        if (l_rowidx[l_n] < l_j*k_block_size) {
          l_rowidx_tmp[l_n] = (unsigned int)-1;
        } else {
          l_rowidx_tmp[l_n] = l_rowidx[l_n] - l_j*k_block_size;  
        }
      }
      for (l_i = 0; l_i < Nb; l_i++) {
        libxsmm_blasint cur_n_cols = Nblocks_offsets[l_i+1] - Nblocks_offsets[l_i];
        libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape( 1, cur_n_cols, K/Kb, K, 0, N, dtype, dtype, dtype, LIBXSMM_DATATYPE(float) );        
        kernels_csc[l_i * Kb + l_j] = libxsmm_create_packed_spgemm_csc_v2(gemm_shape, l_flags, l_prefetch_flags, bm, &l_colptr[Nblocks_offsets[l_i]], l_rowidx_tmp, (const void*)l_b_sp_csc);
        if (kernels_csc[l_i * Kb + l_j] == NULL) {
          printf("Could not generate CSC kernel[%d][%d] !!!\n", l_i, l_j);
          return 0;
        }
        if (l_j == 0) {
          auto l_unary_shape = libxsmm_create_meltw_unary_shape(bm*cur_n_cols, 1, bm*cur_n_cols, bm*cur_n_cols, dtype, dtype, dtype);
          kernels_zero[l_i]  = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE); 
          if (kernels_zero[l_i] == NULL) {
            printf("Could not generate zero kernel[%d] !!!\n", l_i);
            return 0;
          }  
        }
      }
    }
  } else {
    for (l_j = 0; l_j < Kb; l_j++) {
      /* Offset rowidx values for K blocking in BCSC format  */
      for (l_n = 0; l_n < nnz/(bcsc_bk*bcsc_bn); l_n++) {
        if (l_rowidx[l_n] < l_j*k_block_size) {
          l_rowidx_tmp[l_n] = (unsigned int)-1;
        } else {
          l_rowidx_tmp[l_n] = l_rowidx[l_n] - l_j*k_block_size;  
        }
      }
      for (l_i = 0; l_i < Nb; l_i++) {
        libxsmm_blasint cur_n_cols = Nblocks_offsets[l_i+1] - Nblocks_offsets[l_i];
        if (n_dynamic == 0) {
          libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape( 1, cur_n_cols, K/Kb, K, 0, N, dtype, dtype, dtype, LIBXSMM_DATATYPE(float) );
          kernels_csc[l_i * Kb + l_j] = libxsmm_create_packed_spgemm_bcsc(gemm_shape, l_flags, l_prefetch_flags, bm, bcsc_bk, bcsc_bn, &l_colptr[Nblocks_offsets[l_i]/bcsc_bn], l_rowidx_tmp);
          if (kernels_csc[l_i * Kb + l_j] == NULL) {   
            printf("Could not generate BCSC kernel[%d][%d] !!!\n", l_i, l_j);
            return 0; 
          }
        } else {
          if (l_i == 0 && l_j == 0) {         
            libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape( 1, 0, K/Kb, K, -1, N, dtype, dtype, dtype, LIBXSMM_DATATYPE(float) );
            kernels_csc[0] = libxsmm_create_packed_spgemm_bcsc(gemm_shape, l_flags, l_prefetch_flags, bm, bcsc_bk, bcsc_bn, NULL, NULL);
            if (kernels_csc[0] == NULL) {
              printf("Could not generate BCSC kernel[0] !!!\n");
              return 0;
            }
          }
        }
        if (l_j == 0) {
          auto l_unary_shape = libxsmm_create_meltw_unary_shape(bm*cur_n_cols, 1, bm*cur_n_cols, bm*cur_n_cols, dtype, dtype, dtype);
          kernels_zero[l_i]  = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE); 
          if (kernels_zero[l_i] == NULL) {
            printf("Could not generate zero kernel[%d] !!!\n", l_i);
            return 0;
          }  
        }
      }
    }
    if (use_bf16 > 0) {
      libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape( 1, Nb, K, K, 0, N, dtype, dtype, dtype, LIBXSMM_DATATYPE(float) );
      tc_kernel = libxsmm_create_packed_spgemm_bcsc(gemm_shape, l_tc_flags, l_prefetch_flags, bm, bcsc_bk, bcsc_bn, l_colptr, l_rowidx);
      tr_kernel = libxsmm_create_packed_spgemm_bcsc(gemm_shape, l_tr_flags, l_prefetch_flags, bm, bcsc_bk, bcsc_bn, l_colptr, l_rowidx);
    }
  }

  // JIT requested nested loop specs
  long k_step = k_block_size*bcsc_bk;
  long m_step = 1;
  long n_step = 1;
  // Prime factorization of trip-counts to find factors k0,m0 etc
  long k_trips = Kb;
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

  auto spgemm_loop = ThreadedLoop<3>({
      LoopSpecs{0, K,  k_step, {l1_k_step, l0_k_step}},   // Logical K loop specs
      LoopSpecs{0, Mb, m_step, {l1_m_step, l0_m_step}},   // Logical M loop specs
      LoopSpecs{0, Nb, n_step, {l1_n_step, l0_n_step}}},  // Logical N loop specs
      loop_specs_str);

  // Correctness run
  if (check_correctness > 0) {
    spgemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;

          if (use_bf16 == 0) {
            if (use_bcsc == 0) {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_a, i_m, i_k, 0, K, bm);
              gemm_param.b.primary = l_b_sp_csc;         
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_c_asm_csc, i_m, Nblocks_offsets[i_n], 0, N, bm);
            } else {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_a, i_m, i_k, 0, K, bm);;
              gemm_param.b.primary = l_b_sp_bcsc_bf16;
              gemm_param.b.secondary = &l_colptr[Nblocks_offsets[i_n]/bcsc_bn];
              gemm_param.b.tertiary  = l_rowidx;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_c_asm_csc, i_m, Nblocks_offsets[i_n], 0, N, bm);
            }
          } else {
            if (use_bcsc == 0) {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_a_bf16, i_m, i_k, 0, K, bm);              
              gemm_param.b.primary = l_b_sp_csc_bf16;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_c_asm_csc_bf16, i_m, Nblocks_offsets[i_n], 0, N, bm);             
            } else {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_a_vnni_bf16, i_m, i_k/vnni_block_size, 0, i_k%vnni_block_size, K/vnni_block_size, bm, vnni_block_size);
              gemm_param.b.primary = l_b_sp_bcsc_bf16;
              gemm_param.b.secondary = &l_colptr[Nblocks_offsets[i_n]/bcsc_bn];
              gemm_param.b.tertiary  = l_rowidx;
              if (c_flat == 0) {
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_c_vnni_asm_csc_bf16, i_m, Nblocks_offsets[i_n]/vnni_block_size, 0, Nblocks_offsets[i_n]%vnni_block_size, N/vnni_block_size, bm, vnni_block_size);
              } else {
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_c_asm_csc_bf16, i_m, Nblocks_offsets[i_n], 0, N, bm);
              }        
            }
          }

          if (i_k == 0) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)gemm_param.c.primary;
            kernels_zero[i_n]( &zero_param );
          }

          if (n_dynamic > 0) {
            unsigned long long cur_n_cols = (Nblocks_offsets[i_n+1] - Nblocks_offsets[i_n])/bcsc_bn;
            gemm_param.b.quaternary = &cur_n_cols;
            kernels_csc[0]( &gemm_param );
          } else {
            kernels_csc[i_n * Kb + i_k/(k_block_size*bcsc_bk)]( &gemm_param );
          }
        },
        [&]() { if(use_bcsc > 0 && use_bf16 > 0) tc_kernel(NULL);  },
        [&]() { if(use_bcsc > 0 && use_bf16 > 0) tr_kernel(NULL);  });
  }

  /* check for errors */
  if (use_bf16 > 0) {
    if (use_bcsc > 0) {
      for ( l_i = 0; l_i < Mb; l_i++) {
        for ( l_j = 0; l_j < N; l_j++) {
          for ( l_k = 0; l_k < bm; l_k++ ) {
            if (c_flat == 0) {
              LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc_bf16,  l_i, l_j, l_k, N, bm) =
                LIBXSMM_VLA_ACCESS(4, l_p_c_vnni_asm_csc_bf16, l_i, l_j/vnni_block_size, l_k, l_j%vnni_block_size, N/vnni_block_size, bm, vnni_block_size);
            }
          }
        }
      }
    }
    libxsmm_convert_bf16_f32( &LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc_bf16, 0, 0, 0, N, bm), &LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc, 0, 0, 0, N, bm), Mb * N * bm );
  }

  /* compare */
  libxsmm_matdiff(&norms_csc, LIBXSMM_DATATYPE_F32, Mb * N * bm, 1, l_c_gold, l_c_asm_csc, 0, 0);
  printf("L1 reference  : %.25g\n", norms_csc.l1_ref);
  printf("L1 test       : %.25g\n", norms_csc.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_csc.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_csc.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_csc.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_csc.linf_rel);
  printf("Check-norm    : %.24f\n", libxsmm_matdiff_epsilon(&norms_csc));
  libxsmm_matdiff_reduce(&diff, &norms_csc);

  // Warmup iteration for i-caches
  for (i = 0; i < n_warmup_iters; i++) {
    spgemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;
          if (use_bf16 == 0) {
            if (use_bcsc == 0) {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_a, i_m, i_k, 0, K, bm);
              gemm_param.b.primary = l_b_sp_csc;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_c_asm_csc, i_m, Nblocks_offsets[i_n], 0, N, bm);
            } else {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_a, i_m, i_k, 0, K, bm);;
              gemm_param.b.primary = l_b_sp_bcsc_bf16;
              gemm_param.b.secondary = &l_colptr[Nblocks_offsets[i_n]/bcsc_bn];
              gemm_param.b.tertiary  = l_rowidx;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_c_asm_csc, i_m, Nblocks_offsets[i_n], 0, N, bm);
            }
          } else {
            if (use_bcsc == 0) {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_a_bf16, i_m, i_k, 0, K, bm);              
              gemm_param.b.primary = l_b_sp_csc_bf16;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_c_asm_csc_bf16, i_m, Nblocks_offsets[i_n], 0, N, bm);             
            } else {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_a_vnni_bf16, i_m, i_k/vnni_block_size, 0, i_k%vnni_block_size, K/vnni_block_size, bm, vnni_block_size);
              gemm_param.b.primary = l_b_sp_bcsc_bf16;
              gemm_param.b.secondary = &l_colptr[Nblocks_offsets[i_n]/bcsc_bn];
              gemm_param.b.tertiary  = l_rowidx;    
              if (c_flat == 0) {
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_c_vnni_asm_csc_bf16, i_m, Nblocks_offsets[i_n]/vnni_block_size, 0, Nblocks_offsets[i_n]%vnni_block_size, N/vnni_block_size, bm, vnni_block_size);
              } else {
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_c_asm_csc_bf16, i_m, Nblocks_offsets[i_n], 0, N, bm);
              }            
            }
          }

          if (i_k == 0) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)gemm_param.c.primary;
            kernels_zero[i_n]( &zero_param );
          }
          if (n_dynamic > 0) {
            unsigned long long cur_n_cols = (Nblocks_offsets[i_n+1] - Nblocks_offsets[i_n])/bcsc_bn;
            gemm_param.b.quaternary = &cur_n_cols;
            kernels_csc[0]( &gemm_param );
          } else {
            kernels_csc[ i_n * Kb + i_k/(k_block_size*bcsc_bk)]( &gemm_param );
          }
        },
        [&]() { if(use_bcsc > 0 && use_bf16 > 0) tc_kernel(NULL);  },
        [&]() { if(use_bcsc > 0 && use_bf16 > 0) tr_kernel(NULL);  });
  }

  l_total = 0.0;
  l_start = libxsmm_timer_tick();
  for (i = 0; i < n_iters; i++) {
    spgemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;
          if (use_bf16 == 0) {
            if (use_bcsc == 0) {       
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_a, i_m, i_k, 0, K, bm);
              gemm_param.b.primary = l_b_sp_csc;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_c_asm_csc, i_m, Nblocks_offsets[i_n], 0, N, bm);
            } else {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_a, i_m, i_k, 0, K, bm);;
              gemm_param.b.primary = l_b_sp_bcsc_bf16;
              gemm_param.b.secondary = &l_colptr[Nblocks_offsets[i_n]/bcsc_bn];
              gemm_param.b.tertiary  = l_rowidx;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_c_asm_csc, i_m, Nblocks_offsets[i_n], 0, N, bm);
            }
          } else {
            if (use_bcsc == 0) {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_a_bf16, i_m, i_k, 0, K, bm);              
              gemm_param.b.primary = l_b_sp_csc_bf16;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_c_asm_csc_bf16, i_m, Nblocks_offsets[i_n], 0, N, bm);             
            } else {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_a_vnni_bf16, i_m, i_k/vnni_block_size, 0, i_k%vnni_block_size, K/vnni_block_size, bm, vnni_block_size);
              gemm_param.b.primary = l_b_sp_bcsc_bf16;
              gemm_param.b.secondary = &l_colptr[Nblocks_offsets[i_n]/bcsc_bn];
              gemm_param.b.tertiary  = l_rowidx;
              if (c_flat == 0) {
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_c_vnni_asm_csc_bf16, i_m, Nblocks_offsets[i_n]/vnni_block_size, 0, Nblocks_offsets[i_n]%vnni_block_size, N/vnni_block_size, bm, vnni_block_size);
              } else {
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_c_asm_csc_bf16, i_m, Nblocks_offsets[i_n], 0, N, bm);
              }              
            }
          }

          if (i_k == 0) {
            libxsmm_meltw_unary_param zero_param;
            zero_param.out.primary = (void*)gemm_param.c.primary;
            kernels_zero[i_n]( &zero_param );
          }
          if (n_dynamic > 0) {
            unsigned long long cur_n_cols = (Nblocks_offsets[i_n+1] - Nblocks_offsets[i_n])/bcsc_bn;
            gemm_param.b.quaternary = &cur_n_cols;
            kernels_csc[0]( &gemm_param );
          } else {
            kernels_csc[ i_n * Kb + i_k/(k_block_size*bcsc_bk)]( &gemm_param );
          }
        },
        [&]() { if(use_bcsc > 0 && use_bf16 > 0) tc_kernel(NULL);  },
        [&]() { if(use_bcsc > 0 && use_bf16 > 0) tr_kernel(NULL);  });
  }
  l_end = libxsmm_timer_tick();
  l_total += libxsmm_timer_duration(l_start, l_end);

  printf("%fs for sparse\n", l_total);
  printf("%f GFLOPS for sparse\n", ((double)((double)n_iters * (double)M * (double)K * (double)N) * 2.0) / (l_total * 1.0e9));

  /* free */
  libxsmm_free( l_b_de );
  libxsmm_free( l_a );
  libxsmm_free( l_a_bf16 );
  libxsmm_free( l_a_vnni_bf16 );
  libxsmm_free( l_c_gold );
  libxsmm_free( l_c_asm_csc );
  libxsmm_free( l_c_asm_csc_bf16 );
  libxsmm_free( l_c_vnni_asm_csc_bf16);
  if (use_bcsc == 0) {
    libxsmm_free( l_b_sp_csc );
    libxsmm_free( l_b_sp_csc_bf16 );
  } else {
    libxsmm_free( l_b_sp_bcsc_bf16 );
  }
  libxsmm_free( l_colptr );
  libxsmm_free( l_rowidx );
  libxsmm_free( l_rowidx_tmp );
  free( grid_point_array );

  return 0;
}

int main(int argc, char** argv) {
  int use_prec_bf16 = 0;
  if (argc > 2) {
    use_prec_bf16 = atoi(argv[9]);
  }
  if (use_prec_bf16 == 0) {
    return spgemm_benchmark<float>(argc, argv);  
  } else {
    return spgemm_benchmark<libxsmm_bfloat16>(argc, argv);  
  }
}

