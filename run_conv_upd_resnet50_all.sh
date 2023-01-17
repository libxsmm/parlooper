#!/bin/bash

# Source the compiler and OpenMP RT library

echo "Usage: ./run_conv_upd_resnet50_all.sh <#omp> <batch>"
echo "Note: OMP_NUM_THREADS and USE_BF16 are set inside the script"
echo "Note: loop strings and parameters are optimized for 56-core SPR SP with 56 threads and batch = 56"

omp=$1
batch=$2
use_bf16=1

# all except the first one
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcdb $batch 7 7 512 2048  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
rows=14
cols=4
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd A{R:$rows}C{C:$cols}dbef $batch 7 7 512 2048 1 1 1 1 0 0 32 32 1000 1 0 0   0 0 1  1 $rows $cols  0 0 1
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Adbcef $batch 7 7 512 512  3 3  1 1  1 1  32 32  1000  1 0 0 0 0 0  0 1 1  1 0
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Adbcef $batch 7 7 512 512  3 3  1 1  1 1  32 32  1000  1 0 0 0 0 0  0 1 1  0 0 1
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd ABEFcd $batch 7 7 512 512  3 3  1 1  1 1  32 32  1000   0 0 0 0 0 0  0 1 1  0 0 1
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcdb $batch 7 7 2048 512  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
rows=14
cols=4
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd A{R:$rows}C{C:$cols}dbef $batch 7 7 2048 512 1 1 1 1 0 0 32 32 1000 1 0 0   0 0 1  1 $rows $cols  0 0 1 # best performing
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcdb $batch 7 7 512 2048  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
rows=14
cols=4
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd A{R:$rows}C{C:$cols}dbef $batch 7 7 512 2048 1 1 1 1 0 0 32 32 1000 1 0 0   0 0 1  1 $rows $cols  0 0 1
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Adbcef $batch 7 7 512 512  3 3  1 1  1 1  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd ABEFcd $batch 7 7 512 512  3 3  1 1  1 1  32 32  1000   0 0 0 0 0 0  0 1 1  0 0 1
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcdb $batch 7 7 2048 512  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
rows=14
cols=4
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd A{R:$rows}C{C:$cols}dbef $batch 7 7 2048 512 1 1 1 1 0 0 32 32 1000 1 0 0   0 0 1  1 $rows $cols  0 0 1 # best performing
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdbc $batch 7 7 512 2048  1 1  1 1  0 0  32 32  1000  1 0 1 0 0 0 0 1 1 1 0

rows=14
cols=4
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd A{R:$rows}C{C:$cols}dbef $batch 7 7 512 2048 1 1 1 1 0 0 32 32 1000 1 0 0   0 0 1  1 $rows $cols  0 0 1
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd abEfdc $batch 14 14 512 512  3 3  2 2  1 1  32 32  1000  0 0 1 0 1 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd ABEFcd $batch 14 14 512 512  3 3  2 2  1 1  32 32  1000  0 0 1 0 1 0  0 1 1  1 0 1
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdbc $batch 14 14 1024 512  1 1  1 1  0 0  32 32  1000  1 0 1 0 0 0 0 1 1 1 0
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdbc $batch 14 14 1024 2048  1 1  2 2  0 0  32 32  1000  1 0 1 0 1 1  0 1 1  1 0
rows=8
cols=7
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd A{R:$rows}C{C:$cols}dbef $batch 14 14 1024 2048 1 1 2 2 0 0 32 32 1000  1 0 0 0 1 1  1 $rows $cols  1 0 1

OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Abcdef $batch 14 14 256 256  3 3  1 1  1 1  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Abcdef $batch 14 14 256 256  3 3  1 1  1 1  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Abcdef $batch 14 14 256 256  3 3  1 1  1 1  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Abcdef $batch 14 14 256 256  3 3  1 1  1 1  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Abcdef $batch 14 14 256 256  3 3  1 1  1 1  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdcb $batch 14 14 1024 256  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefbcd $batch 14 14 256 1024  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd abEfcd $batch 28 28 256 256  3 3  2 2  1 1  32 32  1000  0 1 0 0 1 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd cAEBfd $batch 28 28 256 256  3 3  2 2  1 1  32 32  1000  0 0 1 0 1 0  0 1 1  0 0 1
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefbcd $batch 28 28 512 256  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefbcd $batch 28 28 512 1024  1 1  2 2  0 0  32 32  1000  1 0 0 0 1 1 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcbd $batch 28 28 128 512  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Acdbef $batch 28 28 128 128  3 3  1 1  1 1  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
rows=56
cols=1
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd A{R:$rows}C{C:$cols}dbef $batch 28 28 128 128  3 3  1 1  1 1  32 32  1000  1 0 0 0 0 1  1 $rows $cols  0 0 1 # ~ GFLOP/s
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcbd $batch 28 28 512 128  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcbd $batch 28 28 128 512  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Acdbef $batch 28 28 128 128  3 3  1 1  1 1  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
rows=56
cols=1
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd A{R:$rows}C{C:$cols}dbef $batch 28 28 128 128  3 3  1 1  1 1  32 32  1000  1 0 0 0 0 1  1 $rows $cols  0 0 1 # ~ GFLOP/s
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcbd $batch 28 28 512 128  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcbd $batch 28 28 128 512  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Acdbef $batch 28 28 128 128  3 3  1 1  1 1  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
rows=56
cols=1
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd A{R:$rows}C{C:$cols}dbef $batch 28 28 128 128  3 3  1 1  1 1  32 32  1000  1 0 0 0 0 1  1 $rows $cols  0 0 1 # ~ GFLOP/s
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcbd $batch 28 28 512 128  1 1  1 1  0 0  32 32  1000  1 1 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcdb $batch 28 28 128 512  1 1  1 1  0 0  32 32  1000  1 0 1 0 0 0 0 1 1 1 0
#OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd abEFdc $batch 56 56 128 128  3 3  2 2  1 1  32 32  1000  0 0 1 0 1 0 0 1 1 1 0
rows=4
cols=14
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd C{C:$cols}A{R:$rows}bdef $batch 56 56 128 128  3 3  2 2  1 1  32 32  1000  0 0 1 1 1 0  0 $rows $cols  0 0 1 # ?????
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcdb $batch 56 56 256 128  1 1  1 1  0 0  32 32  1000  1 0 1 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcdb $batch 56 56 256 512  1 1  2 2  0 0  32 32  1000  1 0 1 0 1 1 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdbc $batch 56 56 64 256  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Acbdef $batch 56 56 64 64  3 3  1 1  1 1  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdbc $batch 56 56 256 64  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdbc $batch 56 56 64 256  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Acbdef $batch 56 56 64 64  3 3  1 1  1 1  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefdbc $batch 56 56 256 64  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcbd $batch 56 56 64 256  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Acdbef $batch 56 56 64 64  3 3  1 1  1 1  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcbd $batch 56 56 64 64  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0 0 1 1 1 0
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd Aefcbd $batch 56 56 64 256  1 1  1 1  0 0  32 32  1000  1 0 0 0 0 0  0 1 1  1 0

# first convolution (the one outside bottleneckes), 224x224, with logical padding
rows=1
cols=56
OMP_NUM_THREADS=$omp USE_BF16=$use_bf16 ./conv_upd C{C:$cols}A{R:$rows}bdef $batch 224 224 4 64   7 7  2 2 3 3  4 32  1000  0 0 1 1 1 0  0 $rows $cols  0 0 1  1
