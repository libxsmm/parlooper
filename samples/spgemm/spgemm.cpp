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

template<typename DType>
int spgemm_benchmark(int argc, char** argv) {
  // Setup default SPGEMM sizes
  int check_correctness = 1;
  char loop_specs_str[256] = "aBC";  
  long M = 128, N = 256, K = 512;
  long bm = 32, N_target_blocks = 8;
  long Mb = M/bm;
  long Nb = N_target_blocks;
  double sparse_frac = 0.8;
  unsigned int use_bf16 = 1;
  unsigned int use_bcsc = 1;
  unsigned int bcsc_bk = 4;
  unsigned int bcsc_bn = 2;
  unsigned int sparse_block_bk = 4;
  unsigned int sparse_block_bn = 2;
  unsigned int n_iters = 1;
  unsigned int use_ac_vnni = (use_bcsc > 0) ? 1 : 0;
  unsigned int vnni_block_size = (use_ac_vnni > 0) ? 4 : 1;
  unsigned int n_warmup_iters = 2;
  long i;
  unsigned long long l_start, l_end;
  double l_total;

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
    sparse_frac = atof(argv[7]);
    use_bf16 = atoi(argv[8]);
    use_bcsc = atoi(argv[9]);
    bcsc_bk = atoi(argv[10]);
    bcsc_bn = atoi(argv[11]);
    sparse_block_bk = atoi(argv[12]);
    sparse_block_bn = atoi(argv[13]);
    n_iters = atoi(argv[14]);
    use_ac_vnni = (use_bcsc > 0) ? 1 : 0;
    vnni_block_size = (use_ac_vnni > 0) ? 4 : 1;
    Mb = M/bm;
  }

  // Kernel management specifics
  const libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_gemmfunction kernels_csc[N_target_blocks+1];

  // Allocate buffers
  unsigned int* l_colptr = NULL;
  unsigned int* l_rowidx = NULL;
  float* l_b_de = (float*)libxsmm_aligned_malloc(sizeof(float) * K * N, 64);
  float* l_b_sp_csc = NULL;
  libxsmm_bfloat16* l_b_sp_csc_bf16 = NULL;
  libxsmm_bfloat16* l_b_sp_bcsc_bf16 = NULL;
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
        if (LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K) == 0) {
          float tmp = (float)libxsmm_rng_f64();
          if (tmp >= sparse_frac) {
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
        LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N, bm) = 0.f;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc,  l_i, l_j, l_k, N, bm) = 0.f;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc_bf16,  l_i, l_j, l_k, N, bm) = (libxsmm_bfloat16)0;
        LIBXSMM_VLA_ACCESS(4, l_p_c_vnni_asm_csc_bf16,  l_i, l_j/vnni_block_size, l_k, l_j%vnni_block_size, N/vnni_block_size, bm, vnni_block_size) = (libxsmm_bfloat16)0;
      }
    }
  }

  if (use_bcsc == 0) {
    /* create B, csc */
    l_colptr   = (unsigned int*) libxsmm_aligned_malloc( (N+1)*sizeof(unsigned int), 64 );
    l_rowidx   = (unsigned int*) libxsmm_aligned_malloc( nnz*sizeof(unsigned int),   64 );
    l_b_sp_csc = (float*       ) libxsmm_aligned_malloc( nnz*sizeof(float),          64 );
    l_b_sp_csc_bf16 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( nnz*sizeof(libxsmm_bfloat16),          64 );
    l_k = 0;
    l_colptr[N] = nnz;
    for ( l_i = 0; l_i < N; l_i++ ) {
      l_colptr[l_i] = l_k;
      for ( l_j = 0; l_j < K; l_j++ ) {
        if ( LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K) != 0.0 ) {
          l_rowidx[l_k] = l_j;
          l_b_sp_csc[l_k] = LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K);
          if (use_bf16 > 0) {
            libxsmm_rne_convert_fp32_bf16( &l_b_sp_csc[l_k], &l_b_sp_csc_bf16[l_k], 1);
            libxsmm_convert_bf16_f32( &l_b_sp_csc_bf16[l_k], &l_b_sp_csc[l_k], 1 );
            libxsmm_convert_bf16_f32( &l_b_sp_csc_bf16[l_k], &LIBXSMM_VLA_ACCESS(2, l_p_b_de, l_i, l_j, K), 1 );
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
    l_b_sp_bcsc_bf16 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( nnz*sizeof(libxsmm_bfloat16),          64 );
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
              libxsmm_rne_convert_fp32_bf16( &val, &l_b_sp_bcsc_bf16[l_val_idx], 1);
              l_val_idx++;
            }
          }
          l_nz_block_id++;
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
  l_total = 0.0;
  for ( l_n = 0; l_n < n_iters; l_n++) {
    l_start = libxsmm_timer_tick();
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
    l_end = libxsmm_timer_tick();
    l_total += libxsmm_timer_duration(l_start, l_end);
    if (l_n < n_iters-1) {
      memset(&LIBXSMM_VLA_ACCESS(3, l_p_c_gold, 0, 0, 0, N, bm), 0, N * Mb * bm * sizeof(float));
    }
  }  
  printf("%fs for dense\n", l_total);
  printf("%f GFLOPS for dense\n", ((double)((double)n_iters * (double)M * (double)K * (double)N) * 2.0) / (l_total * 1.0e9));

  /* Create sparse routines */
  if (use_bcsc == 0) {
    for (l_i = 0; l_i < Nb; l_i++) {
      libxsmm_blasint cur_n_cols = Nblocks_offsets[l_i+1] - Nblocks_offsets[l_i];
      libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape( 1, cur_n_cols, K, K, 0, K, dtype, dtype, dtype, LIBXSMM_DATATYPE(float) );
      kernels_csc[l_i] = libxsmm_create_packed_spgemm_csc_v2(gemm_shape, l_flags, l_prefetch_flags, bm, &l_colptr[Nblocks_offsets[l_i]], l_rowidx, (const void*)l_b_sp_csc);
      if (kernels_csc[l_i] == NULL) {
        printf("Could not generate BCSC kernel[%d]!!!\n", l_i);
        return 0;
      }
    }
  } else {
    for (l_i = 0; l_i < Nb; l_i++) {
      libxsmm_blasint cur_n_cols = Nblocks_offsets[l_i+1] - Nblocks_offsets[l_i];
      libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape( 1, cur_n_cols, K, K, 0, K, dtype, dtype, dtype, LIBXSMM_DATATYPE(float) );
      kernels_csc[l_i] = libxsmm_create_packed_spgemm_bcsc(gemm_shape, l_flags, l_prefetch_flags, bm, bcsc_bk, bcsc_bn, &l_colptr[Nblocks_offsets[l_i]/bcsc_bn], l_rowidx);
      if (kernels_csc[l_i] == NULL) {
        printf("Could not generate BCSC kernel[%d]!!!\n", l_i);
        return 0;
      }
    }
  }

  // JIT requested nested loop specs
  long k_step = K;
  long m_step = 1;
  long n_step = 1;
  // Prime factorization of trip-counts to find factors k0,m0 etc
  long k_trips = K/k_step;
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
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_a, i_m, i_k, 0, K, bm);
            gemm_param.b.primary = l_b_sp_csc;
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_c_asm_csc, i_m, Nblocks_offsets[i_n], 0, N, bm);
          } else {
            if (use_bcsc == 0) {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_a_bf16, i_m, i_k, 0, K, bm);              
              gemm_param.b.primary = l_b_sp_csc_bf16;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_c_asm_csc_bf16, i_m, Nblocks_offsets[i_n], 0, N, bm);             
            } else {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_a_vnni_bf16, i_m, i_k/vnni_block_size, 0, i_k%vnni_block_size, K/vnni_block_size, bm, vnni_block_size);
              gemm_param.b.primary = l_b_sp_bcsc_bf16;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_c_vnni_asm_csc_bf16, i_m, Nblocks_offsets[i_n]/vnni_block_size, 0, Nblocks_offsets[i_n]%vnni_block_size, N/vnni_block_size, bm, vnni_block_size);              
            }
          }
          kernels_csc[i_n]( &gemm_param );
        },
        [&]() {},
        [&]() {});
  }

  /* check for errors */
  if (use_bf16 > 0) {
    if (use_bcsc > 0) {
      for ( l_i = 0; l_i < Mb; l_i++) {
        for ( l_j = 0; l_j < N; l_j++) {
          for ( l_k = 0; l_k < bm; l_k++ ) {
            LIBXSMM_VLA_ACCESS(3, l_p_c_asm_csc_bf16,  l_i, l_j, l_k, N, bm) =
              LIBXSMM_VLA_ACCESS(4, l_p_c_vnni_asm_csc_bf16, l_i, l_j/vnni_block_size, l_k, l_j%vnni_block_size, N/vnni_block_size, bm, vnni_block_size);
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
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_a, i_m, i_k, 0, K, bm);
            gemm_param.b.primary = l_b_sp_csc;
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_c_asm_csc, i_m, Nblocks_offsets[i_n], 0, N, bm);
          } else {
            if (use_bcsc == 0) {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_a_bf16, i_m, i_k, 0, K, bm);              
              gemm_param.b.primary = l_b_sp_csc_bf16;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_c_asm_csc_bf16, i_m, Nblocks_offsets[i_n], 0, N, bm);             
            } else {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_a_vnni_bf16, i_m, i_k/vnni_block_size, 0, i_k%vnni_block_size, K/vnni_block_size, bm, vnni_block_size);
              gemm_param.b.primary = l_b_sp_bcsc_bf16;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_c_vnni_asm_csc_bf16, i_m, Nblocks_offsets[i_n]/vnni_block_size, 0, Nblocks_offsets[i_n]%vnni_block_size, N/vnni_block_size, bm, vnni_block_size);              
            }
          }
          kernels_csc[i_n]( &gemm_param );
        },
        [&]() {},
        [&]() {});
  }

  l_total = 0.0;
  l_start = libxsmm_timer_tick();
  for (i = 0; i < n_iters; i++) {
    spgemm_loop(
        [&](int* ind) {
          int i_k = ind[0], i_m = ind[1], i_n = ind[2];
          libxsmm_gemm_param gemm_param;
          if (use_bf16 == 0) {
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_a, i_m, i_k, 0, K, bm);
            gemm_param.b.primary = l_b_sp_csc;
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(float), l_c_asm_csc, i_m, Nblocks_offsets[i_n], 0, N, bm);
          } else {
            if (use_bcsc == 0) {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_a_bf16, i_m, i_k, 0, K, bm);              
              gemm_param.b.primary = l_b_sp_csc_bf16;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(3, sizeof(libxsmm_bfloat16), l_c_asm_csc_bf16, i_m, Nblocks_offsets[i_n], 0, N, bm);             
            } else {
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_a_vnni_bf16, i_m, i_k/vnni_block_size, 0, i_k%vnni_block_size, K/vnni_block_size, bm, vnni_block_size);
              gemm_param.b.primary = l_b_sp_bcsc_bf16;
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(4, sizeof(libxsmm_bfloat16), l_c_vnni_asm_csc_bf16, i_m, Nblocks_offsets[i_n]/vnni_block_size, 0, Nblocks_offsets[i_n]%vnni_block_size, N/vnni_block_size, bm, vnni_block_size);              
            }
          }
          kernels_csc[i_n]( &gemm_param );
        },
        [&]() {},
        [&]() {});
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
  
  return 0;
}

int main(int argc, char** argv) {
  int use_prec_bf16 = 0;
  if (argc > 2) {
    use_prec_bf16 = atoi(argv[8]);
  }
  if (use_prec_bf16 == 0) {
    return spgemm_benchmark<float>(argc, argv);  
  } else {
    return spgemm_benchmark<libxsmm_bfloat16>(argc, argv);  
  }
}

