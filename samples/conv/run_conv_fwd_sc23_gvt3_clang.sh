#!/bin/bash

# Source the clang compiler and clang OpenMP RT library

# Set paths to LIBXSMM

#export LIBXSMM_ROOT=<path to LIBXSMM>
#export LIBXSMM_DNN_ROOT=<path to LIBXSMM-DNN>
#export LD_LIBRARY_PATH=$LIBXSMM_ROOT/lib:$LD_LIBRARY_PATH


make clean && make LIBXSMM_ROOT=${LIBXSMM_ROOT} LIBXSMM_DNN_ROOT=${LIBXSMM_DNN_ROOT} PARLOOPER_COMPILER=clang
#export LD_PRELOAD=/usr/lib64/libomp.so:$LD_PRELOAD

echo "Usage: ./run_conv_fwd_resnet50_sc23_clang.sh <#omp> <batch>"
echo "With a suggested post-processing via grep PERFDUMP <output> for perf or grep Validation <output> for correctness"
echo "Note: OMP_NUM_THREADS and USE_BF16 are set inside the script"
echo "Note: loop strings and parameters are optimized for 56-core SPR SP with 56 threads and batch = 56"

export KMP_AFFINITY=compact

export cols=16
export rows=4

OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 64 64 1 1 1 1 0 0 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 64 64 3 3 1 1 1 1 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 64 256 1 1 1 1 0 0 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 64 256 1 1 1 1 0 0 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 256 64 1 1 1 1 0 0 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 64 64 3 3 1 1 1 1 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 64 256 1 1 1 1 0 0 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 256 64 1 1 1 1 0 0 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 64 64 3 3 1 1 1 1 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdced 64 56 56 64 256 1 1 1 1 0 0 32 32 4 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbedc 64 56 56 256 128 1 1 1 1 0 0 32 32 1 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbedc 64 56 56 128 128 3 3 2 2 1 1 32 32 1 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbedc 64 56 56 256 512 1 1 2 2 0 0 32 32 1 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbedc 64 28 28 128 512 1 1 1 1 0 0 32 32 1 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdecd 64 28 28 512 128 1 1 1 1 0 0 32 32 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdecd 64 28 28 128 128 3 3 1 1 1 1 32 32 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdecd 64 28 28 128 512 1 1 1 1 0 0 32 32 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdecd 64 28 28 512 128 1 1 1 1 0 0 32 32 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdecd 64 28 28 128 128 3 3 1 1 1 1 32 32 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdecd 64 28 28 128 512 1 1 1 1 0 0 32 32 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdecd 64 28 28 512 128 1 1 1 1 0 0 32 32 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdecd 64 28 28 128 128 3 3 1 1 1 1 32 32 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbdecd 64 28 28 128 512 1 1 1 1 0 0 32 32 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcdce 64 28 28 512 256 1 1 1 1 0 0 32 32 1 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcdce 64 28 28 256 256 3 3 2 2 1 1 32 32 1 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcdce 64 28 28 512 1024 1 1 2 2 0 0 32 32 1 1 1 8 2 1 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcdce 64 14 14 256 1024 1 1 1 1 0 0 32 32 1 1 1 8 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 1024 256 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 256 3 3 1 1 1 1 32 32 1 1 1 1 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 1024 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 1024 256 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 256 3 3 1 1 1 1 32 32 1 1 1 1 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 1024 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 1024 256 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 256 3 3 1 1 1 1 32 32 1 1 1 1 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 1024 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 1024 256 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 256 3 3 1 1 1 1 32 32 1 1 1 1 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 1024 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 1024 256 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 256 3 3 1 1 1 1 32 32 1 1 1 1 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 256 1024 1 1 1 1 0 0 32 32 1 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 14 1024 512 1 1 1 1 0 0 32 32 1 1 1 8 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 16 512 512 3 3 2 2 1 1 32 32 1 1 1 4 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 14 16 1024 2048 1 1 2 2 0 0 32 32 1 1 1 4 7 1 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd Afgbcecd 64 7 8 512 2048 1 1 1 1 0 0 32 32 1 1 1 8 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd ACfgbdec 64 7 8 2048 512 1 1 1 1 0 0 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd ACfgbdec 64 7 8 512 512 3 3 1 1 1 1 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd ACfgbdec 64 7 8 512 2048 1 1 1 1 0 0 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd ACfgbdec 64 7 8 2048 512 1 1 1 1 0 0 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd ACfgbdec 64 7 8 512 512 3 3 1 1 1 1 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd ACfgbdec 64 7 8 512 2048 1 1 1 1 0 0 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd "A{C:$cols}fgbC{R:$rows}ed" 64 7 8 512 2048 1 1 1 1 0 0 32 32 1 1 1 8 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd "A{C:$cols}C{R:$rows}fgbde" 64 7 8 2048 512 1 1 1 1 0 0 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd "A{C:$cols}C{R:$rows}fgbde" 64 7 8 512 512 3 3 1 1 1 1 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd "A{C:$cols}C{R:$rows}fgbde" 64 7 8 512 2048 1 1 1 1 0 0 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd "A{C:$cols}C{R:$rows}fgbde" 64 7 8 2048 512 1 1 1 1 0 0 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd "A{C:$cols}C{R:$rows}fgbde" 64 7 8 512 512 3 3 1 1 1 1 32 32 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=1 ./conv_fwd "A{C:$cols}C{R:$rows}fgbde" 64 7 8 512 2048 1 1 1 1 0 0 32 32 1 1 1 1 7 0 1000

# 19 shapes (but some tunings from the set above can provide better results for same shapes)
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbcde 64 56 56 64 256 1 1 1 1 0 0 64 64 1 1 1 1 4 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbedc 64 56 56 64 64 1 1 1 1 0 0 64 64 1 1 1 1 4 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbdced 64 56 56 64 64 3 3 1 1 1 1 64 64 14 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbcde 64 56 56 256 64 1 1 1 1 0 0 64 64 1 1 1 1 4 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbcde 64 56 56 256 512 1 1 2 2 0 0 64 64 1 1 1 1 7 1 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbedc 64 56 56 256 128 1 1 2 2 0 0 64 64 1 1 1 1 1 0 1000 # !!!
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbdced 64 56 56 128 128 3 3 1 1 1 1 64 64 4 1 1 1 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbcded 64 28 28 128 512 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbdced 64 28 28 512 128 1 1 1 1 0 0 64 64 7 1 1 1 1 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbecd 64 28 28 512 1024 1 1 2 2 0 0 64 64 1 1 1 1 2 1 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbecd 64 28 28 512 256 1 1 2 2 0 0 64 64 1 1 1 1 2 1 1000 # !!!
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbedc 64 14 14 256 256 3 3 1 1 1 1 64 64 1 1 1 1 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbcecd 64 14 14 256 1024 1 1 1 1 0 0 64 64 1 1 1 8 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbedcdc 64 14 14 1024 256 1 1 1 1 0 0 64 64 2 1 1 2 2 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbecd 64 14 14 1024 2048 1 1 2 2 0 0 64 64 1 1 1 1 7 1 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd Afgbcdec 64 14 14 1024 512 1 1 2 2 0 0 64 64 1 1 1 8 7 1 1000 # !!!
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd C{R:8}A{C:8}edcfgb 64 7 7 512 512 3 3 1 1 1 1 64 64 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd A{C:4}C{R:16}ecfgbd 64 7 7 512 2048 1 1 1 1 0 0 64 64 1 1 1 1 7 0 1000
OMP_NUM_THREADS=64 USE_BF16=$use_bf16 ./conv_fwd A{C:4}C{R:16}cdefgb 64 7 7 2048 512 1 1 1 1 0 0 64 64 1 1 1 1 7 0 1000