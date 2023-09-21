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

void shuffle_array(unsigned int *array, int n) {
  if (n > 1)
  {
    int i;
    for (i = 0; i < n - 1; i++)
    {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned int t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

unsigned int random_mask_half_full(int sparsity_factor ) {
  int __i;
  unsigned int id_array[32];
  unsigned int cur_bitmap = 0;

  for (__i = 0; __i < 32; __i++) {
    id_array[__i] = __i;
  }
  shuffle_array(id_array, 32);

  for (__i = 0; __i < 32/sparsity_factor; __i++) {
    unsigned int cur_bit = (1 << id_array[__i]);
    cur_bitmap = cur_bitmap | cur_bit;
  }

  return cur_bitmap;
}

template<typename DType>
int decompress_benchmark(int argc, char** argv) {
  long M = 1024;
  long i = 0, j = 0, k = 0;
  int sparsity_factor = 1;
  long iters = 1000;
  long n_warmup_iters = 1;
  long n_threads = omp_get_max_threads();
  ifreq = 1.0 / getFreq();
  if (argc > 1) {
    M = atoi(argv[1]);
    sparsity_factor = atoi(argv[2]); 
    iters = atoi(argv[3]); 
  }

  /* Make sure M is multiple of 32 */
  long size = M*M;
  long chunks = size/32;
  long chunks_per_thread = (chunks + n_threads - 1)/n_threads;

  printf("Size is %d elements, chunks are %d, and perthread are %d\n", size, chunks, chunks_per_thread);

  DType* full_array = (DType*)libxsmm_aligned_malloc(sizeof(DType) * size, 64);
  DType* compact_array = (DType*)libxsmm_aligned_malloc(sizeof(DType) * (size/sparsity_factor), 64);
  unsigned int *bitmap= (unsigned int*)libxsmm_aligned_malloc( size/8, 64);
  init_buf( (float*)full_array, size/(4/sizeof(DType)), 0, 0 );

  for (i = 0; i < size; i+=32 ) {
    unsigned int  cur_mask_int    = random_mask_half_full(sparsity_factor);
    __mmask32     cur_mask        = _cvtu32_mask32(cur_mask_int);
    __m512i       vreg_dense      = _mm512_loadu_si512 ((DType*)full_array+i);
    bitmap[k] = cur_mask_int;
    _mm512_mask_storeu_epi16 ((DType*)full_array+i, cur_mask, vreg_dense);
    _mm512_mask_compressstoreu_epi16((DType*)compact_array+j, cur_mask, vreg_dense);
    k += 1;
    j += 32/sparsity_factor;
  }

  unsigned int result[16];

  /* Warmup iteration for i-caches */
  for (j = 0; j < n_warmup_iters; j++) {
#if defined(_OPENMP)
# pragma omp parallel
#endif 
    {  
      int tid = omp_get_thread_num();
      long my_chunk_start = (tid * chunks_per_thread < chunks) ? tid * chunks_per_thread : chunks_per_thread; 
      long my_chunk_end = ((tid+1) * chunks_per_thread < chunks) ? (tid+1) * chunks_per_thread : chunks_per_thread; 
      long chunk_id = 0;
      DType *load_ptr = (DType*)compact_array + my_chunk_start * (32/sparsity_factor);
      __m512i cur_vreg;
#pragma unroll(4)    
      for (chunk_id = my_chunk_start; chunk_id < my_chunk_end; chunk_id++) {
        /* Load mask from bitmask */
        unsigned int umask = bitmap[chunk_id];
        __mmask32 cur_mask = _cvtu32_mask32(umask);  
        /* Load+Decompress based on bitmask */
        cur_vreg = _mm512_mask_expandloadu_epi16 (cur_vreg, cur_mask, load_ptr);
        cur_vreg = _mm512_add_epi32 (cur_vreg, cur_vreg);    
        _mm_prefetch ((char*)load_ptr+16*64, _MM_HINT_T0);   
        /* Popcount on mask bits */
        int n_elts = _popcnt32(umask); 
        /* Advance compact index by popcount */
        load_ptr = (DType*) load_ptr + n_elts;
      } 
      if (j == iters-1 && tid == 0) {
        _mm512_storeu_epi16 (result, cur_vreg);
      }
    }
  }

  auto l_total = 0.0;
  auto l_start = libxsmm_timer_tick();
  for (j = 0; j < iters; j++) {
#if defined(_OPENMP)
# pragma omp parallel
#endif 
    {  
      int tid = omp_get_thread_num();
      long my_chunk_start = (tid * chunks_per_thread < chunks) ? tid * chunks_per_thread : chunks_per_thread; 
      long my_chunk_end = ((tid+1) * chunks_per_thread < chunks) ? (tid+1) * chunks_per_thread : chunks_per_thread; 
      long chunk_id = 0;
      //long total_elts = 0;
      DType *load_ptr = (DType*)compact_array + my_chunk_start * (32/sparsity_factor);
      __m512i cur_vreg;
#pragma unroll(4)
      for (chunk_id = my_chunk_start; chunk_id < my_chunk_end; chunk_id++) {
        /* Load mask from bitmask */
        unsigned int umask = bitmap[chunk_id];
        __mmask32 cur_mask = _cvtu32_mask32(umask);  
        /* Load+Decompress based on bitmask */
        cur_vreg = _mm512_mask_expandloadu_epi16 (cur_vreg, cur_mask, load_ptr);
        cur_vreg = _mm512_add_epi32 (cur_vreg, cur_vreg);
        _mm_prefetch ((char*)load_ptr+16*64, _MM_HINT_T0);
        /* Popcount on mask bits */
        int n_elts = _popcnt32(umask); 
        /* Advance compact index by popcount */
        load_ptr = (DType*) load_ptr + n_elts;
      }
      if (j == iters-1 && tid == 0) {
        _mm512_storeu_epi16 (result, cur_vreg);
      }
#if 0
      if (tid == 0 && j == 0) {
        printf("Read in total %d elements\n", total_elts);
      } 
#endif
    }
  }
  printf("A val is %d\n", result[0]);

  auto l_end = libxsmm_timer_tick();
  l_total += libxsmm_timer_duration(l_start, l_end);
  printf("Achieved GB/s is %.5g\n", (double)((double)iters * (double)(M/sparsity_factor) * (double)M * (sizeof(DType) + 0.125)/1024.0/1024.0/1024.0/(l_total)));
  printf("Effective GB/s is %.5g\n", (double)((double)iters * (double)M * (double)M * sizeof(DType)/1024.0/1024.0/1024.0/(l_total)));


  libxsmm_free(full_array);
  libxsmm_free(compact_array);
  libxsmm_free(bitmap);

  return 0;
}

int main(int argc, char** argv) {
  int use_prec = 0;
  return decompress_benchmark<libxsmm_bfloat16>(argc, argv);  
#if 0
  if (use_prec == 0) {
    return decompress_benchmark<float>(argc, argv);  
  } else if (use_prec == 1) {  
    return decompress_benchmark<libxsmm_bfloat16>(argc, argv);  
  } else if (use_prec == 2) { 
    return decompress_benchmark<char>(argc, argv);  
  } else {
    return 1;
  }
#endif
}

