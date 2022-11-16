#!/bin/bash

# Source the compiler and OpenMP RT library

echo "Usage: ./run_conv_bwd_resnet50_all.sh <#omp> <batch>"
echo "Note: OMP_NUM_THREADS and USE_BF16 are set inside the script"
echo "Note: loop strings and parameters are optimized for 56-core SPR SP with 56 threads and batch = 56"

omp=$1
batch=$2
use_bf16=1

 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd BAfgcedb $batch 7 7 512 2048  1 1  1 1  0 0  32 32  1 1 4 1 7  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd BAfgcedb $batch 7 7 512 512  3 3  1 1  1 1  32 32  1 1 4 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd BAfgcedb $batch 7 7 2048 512  1 1  1 1  0 0  32 32  1 1 4 1 7  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd BAfgcedb $batch 7 7 512 2048  1 1  1 1  0 0  32 32  1 1 4 1 7  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd BAfgcedb $batch 7 7 512 512  3 3  1 1  1 1  32 32  1 1 4 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd BAfgcedb $batch 7 7 2048 512  1 1  1 1  0 0  32 32  1 1 4 1 7  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 7 7 512 2048  1 1  1 1  0 0  32 32  1 1 8 1 7  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 512 512  3 3  2 2  1 1  32 32  1 1 8 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 1024 512  1 1  1 1  0 0  32 32  1 1 4 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 1024 2048  1 1  2 2  0 0  32 32  1 1 16 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 256  3 3  1 1  1 1  32 32  1 1 1 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbdeb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 256  3 3  1 1  1 1  32 32  1 1 1 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbdeb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 256  3 3  1 1  1 1  32 32  1 1 1 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbdeb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 256  3 3  1 1  1 1  32 32  1 1 1 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbdeb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 14 14 256 256  3 3  1 1  1 1  32 32  1 1 1 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbdeb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1 1 8 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcebd $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1 1 4 1 2  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcebd $batch 28 28 256 256  3 3  2 2  1 1  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcebd $batch 28 28 512 256  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcebd $batch 28 28 512 1024  1 1  2 2  0 0  32 32  1 1 4 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcdeb $batch 28 28 128 512  1 1  1 1  0 0  32 32  1 1 2 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcdeb $batch 28 28 128 128  3 3  1 1  1 1  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcdeb $batch 28 28 512 128  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcdeb $batch 28 28 128 512  1 1  1 1  0 0  32 32  1 1 2 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcdeb $batch 28 28 128 128  3 3  1 1  1 1  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcdeb $batch 28 28 512 128  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcdeb $batch 28 28 128 512  1 1  1 1  0 0  32 32  1 1 2 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcdeb $batch 28 28 128 128  3 3  1 1  1 1  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcdeb $batch 28 28 512 128  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 28 28 128 512  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 56 56 128 128  3 3  2 2  1 1  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 56 56 256 128  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 56 56 256 512  1 1  2 2  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 56 56 64 256  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 56 56 64 64  3 3  1 1  1 1  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 56 56 256 64  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 56 56 64 256  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 56 56 64 64  3 3  1 1  1 1  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbde $batch 56 56 256 64  1 1  1 1  0 0  32 32  1 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbded $batch 56 56 64 256  1 1  1 1  0 0  32 32  7 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbded $batch 56 56 64 64  3 3  1 1  1 1  32 32  7 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbded $batch 56 56 64 64  1 1  1 1  0 0  32 32  7 1 1 1 1  1000
 OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_bwd Afgcbded $batch 56 56 64 256  1 1  1 1  0 0  32 32  7 1 1 1 1  1000
