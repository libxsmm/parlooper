#!/bin/bash

# Source the compiler and OpenMP RT library

echo "Usage: ./run_conv_fwd_resnet50_all.sh <#omp> <batch>"
echo "Note: OMP_NUM_THREADS and USE_BF16 are set inside the script"
echo "Note: loop strings and parameters are optimized for 56-core SPR SP with 56 threads and batch = 56"

omp=$1
batch=$2
use_bf16=1

# first convolution (the one outside bottleneckes), 224x224, with logical padding
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdced $batch 224 224 4 64 7 7 2 2 3 3    4 64 1 2 1 1 1 0 1000 1 # 1935, 9 ms

# all convs except the first one
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Abcdefg $batch 56 56 64 64 1 1 1 1 0 0 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Abcdefg $batch 56 56 64 64 3 3 1 1 1 1 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Abcdefg $batch 56 56 64 256 1 1 1 1 0 0 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Abcdefg $batch 56 56 64 256 1 1 1 1 0 0 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdecd $batch 56 56 256 64 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdecd $batch 56 56 64 64 3 3 1 1 1 1 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdecd $batch 56 56 64 256 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdecd $batch 56 56 256 64 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdecd $batch 56 56 64 64 3 3 1 1 1 1 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdecd $batch 56 56 64 256 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdced $batch 56 56 256 128 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdced $batch 56 56 128 128 3 3 2 2 1 1 64 64 4 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdced $batch 56 56 256 512 1 1 2 2 0 0 64 64 4 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbdced $batch 28 28 128 512 1 1 1 1 0 0 64 64 4 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdced $batch 28 28 512 128 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdced $batch 28 28 128 128 3 3 1 1 1 1 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdced $batch 28 28 128 512 1 1 1 1 0 0 64 64 7 1 1 2 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdced $batch 28 28 512 128 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdced $batch 28 28 128 128 3 3 1 1 1 1 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdced $batch 28 28 128 512 1 1 1 1 0 0 64 64 7 1 1 2 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdced $batch 28 28 512 128 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdced $batch 28 28 128 128 3 3 1 1 1 1 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdced $batch 28 28 128 512 1 1 1 1 0 0 64 64 7 1 1 2 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcde $batch 28 28 512 256 1 1 1 1 0 0 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcde $batch 28 28 256 256 3 3 2 2 1 1 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcde $batch 28 28 512 1024 1 1 2 2 0 0 64 64 1 1 1 1 2 1 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcde $batch 14 14 256 1024 1 1 1 1 0 0 64 64 1 1 1 1 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 1024 256 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 256 3 3 1 1 1 1 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 1024 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 1024 256 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 256 3 3 1 1 1 1 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 1024 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 1024 256 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 256 3 3 1 1 1 1 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 1024 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 1024 256 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 256 3 3 1 1 1 1 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 1024 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 1024 256 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 256 3 3 1 1 1 1 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcdec $batch 14 14 256 1024 1 1 1 1 0 0 64 64 1 1 1 4 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcecd $batch 14 14 1024 512 1 1 1 1 0 0 64 64 1 1 1 8 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcecd $batch 14 14 512 512 3 3 2 2 1 1 64 64 1 1 1 8 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcecd $batch 14 14 1024 2048 1 1 2 2 0 0 64 64 1 1 1 16 7 1 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd Afgbcecd $batch 7 7 512 2048 1 1 1 1 0 0 64 64 1 1 1 8 7 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd A{C:14}C{R:4}fgbcde $batch 7 7 2048 512 1 1 1 1 0 0 64 64 1 1 1 1 7 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd A{C:14}C{R:4}fgbcde $batch 7 7 512 512 3 3 1 1 1 1 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd A{C:14}C{R:4}fgbcde $batch 7 7 512 2048 1 1 1 1 0 0 64 64 1 1 1 1 7 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd A{C:14}C{R:4}fgbcde $batch 7 7 2048 512 1 1 1 1 0 0 64 64 1 1 1 1 7 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd A{C:14}C{R:4}fgbcde $batch 7 7 512 512 3 3 1 1 1 1 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_fwd A{C:14}C{R:4}fgbcde $batch 7 7 512 2048 1 1 1 1 0 0 64 64 1 1 1 1 7 0 1000



