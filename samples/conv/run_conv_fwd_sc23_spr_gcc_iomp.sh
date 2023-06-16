#!/bin/bash

# Source compiler and OpenMP RT

# Set paths to LIBXSMM and LIBXSMM-DNN

#export LIBXSMM_ROOT=<path to LIBXSMM>
#export LIBXSMM_DNN_ROOT=<path to LIBXSMM-DNN>
#export LD_LIBRARY_PATH=$LIBXSMM_ROOT/lib:$LD_LIBRARY_PATH

make clean && make LIBXSMM_ROOT=${LIBXSMM_ROOT} LIBXSMM_DNN_ROOT=$LIBXSMM_DNN_ROOT PARLOOPER_COMPILER=gcc


# Preload Intel OpenMP RT (as it increases perf compared to GNU OpenMP)
export LD_PRELOAD=/swtools/intel/oneapi/compiler/latest/linux/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD

export KMP_AFFINITY=granularity=fine,compact,1,0 # if HT is on
#export KMP_AFFINITY=granularity=fine,compact # if HT is off

preamble="numactl -C 0-56"

omp=56
batch=56
use_bf16=1

OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbcde $batch 56 56 64 256 1 1 1 1 0 0 64 64 1 1 1 1 4 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbedc $batch 56 56 64 64 1 1 1 1 0 0 64 64 1 1 1 1 4 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbdced $batch 56 56 64 64 3 3 1 1 1 1 64 64 14 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbcde $batch 56 56 256 64 1 1 1 1 0 0 64 64 1 1 1 1 4 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbcde $batch 56 56 256 512 1 1 2 2 0 0 64 64 1 1 1 1 7 1 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbedc $batch 56 56 256 128 1 1 2 2 0 0 64 64 1 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbdced $batch 56 56 128 128 3 3 2 2 1 1 64 64 4 1 1 1 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbcded $batch 28 28 128 512 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbdced $batch 28 28 512 128 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbecd $batch 28 28 512 1024 1 1 2 2 0 0 64 64 1 1 1 1 2 1 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbecd $batch 28 28 512 256 1 1 2 2 0 0 64 64 1 1 1 1 2 1 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbedc $batch 14 14 256 256 3 3 1 1 1 1 64 64 1 1 1 1 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbcecd $batch 14 14 256 1024 1 1 1 1 0 0 64 64 1 1 1 8 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbedcdc $batch 14 14 1024 256 1 1 1 1 0 0 64 64 2 1 1 2 2 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbecd $batch 14 14 1024 2048 1 1 2 2 0 0 64 64 1 1 1 1 7 1 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd Afgbcdec $batch 14 14 1024 512 1 1 2 2 0 0 64 64 1 1 1 8 7 1 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd C{R:8}A{C:7}edcfgb $batch 7 7 512 512 3 3 1 1 1 1 64 64 1 1 1 1 7 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd A{C:4}C{R:14}ecfgbd $batch 7 7 512 2048 1 1 1 1 0 0 64 64 1 1 1 1 7 0 1000
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ${preamble} ./conv_fwd A{C:4}C{R:14}cdefgb $batch 7 7 2048 512 1 1 1 1 0 0 64 64 1 1 1 1 7 0 1000

exit