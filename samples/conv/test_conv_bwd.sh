#!/bin/bash

N=${OMP_NUM_THREADS}
loop_string="Abcdefg"
bc=32
bk=32
niters=100

#./conv_bwd $N "${loop_string}" ... $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

#set -e
#set -o pipefail

h_block=1
w_block=1
c_block=1
k_block=1
h_in_gemm=1

# 52 convolutions from Resnet50-v1.5
./conv_bwd "${loop_string}" $N 56 56 64 64      1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N 56 56 64 256     1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N 56 56 64 256     1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N 56 56 256 64     1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N 56 56 64 256     1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N 56 56 256 64     1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N 56 56 64 256     1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N 56 56 256 128    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 128 512   1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  56 56 256 512   1 1 2 2 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 512 128   1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 128 512   1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 512 128   1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 128 512   1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 512 128   1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 128 512   1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 512 256   1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 1024  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 512 1024  1 1 2 2 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 256  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 1024  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 256  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 1024  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 256  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 1024  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 256  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 1024  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 256  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 1024  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 512  1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 512 2048    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 2048 1 1 2 2 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 2048 512    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 512 2048    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 2048 512    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 512 2048    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  56 56 64 64     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  56 56 64 64     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  56 56 128 128   3 3 2 2 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  56 56 64 64     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 128 128   3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 128 128   3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 128 128   3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  28 28 256 256   3 3 2 2 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 256   3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 256   3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 256   3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 256   3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 256 256   3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 512 512   3 3 2 2 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 512 512     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 512 512     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

h_block=4
w_block=1
c_block=1
k_block=1
h_in_gemm=1

./conv_bwd "${loop_string}" $N  56 56 64 256    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 2048 1 1 2 2 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 512 512   3 3 2 2 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 512 512     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

h_block=1
w_block=2
c_block=1
k_block=1
h_in_gemm=1
./conv_bwd "${loop_string}" $N  56 56 64 256    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

# with w_block = 2 it is questionable why H=W=14 does not work (both fp32! and bf16)
h_block=1
w_block=7
c_block=1
k_block=1
h_in_gemm=1
./conv_bwd "${loop_string}" $N  14 14 1024 2048 1 1 2 2 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 512 512   3 3 2 2 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 512 512     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

h_block=1
w_block=1
c_block=4
k_block=1
h_in_gemm=1

./conv_bwd "${loop_string}" $N  56 56 64 256    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 2048 1 1 2 2 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 512 512   3 3 2 2 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 512 512     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

h_block=1
w_block=1
c_block=1
k_block=4
h_in_gemm=1

./conv_bwd "${loop_string}" $N  56 56 64 256    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 1024 2048 1 1 2 2 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 512 512   3 3 2 2 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  7 7 512 512     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

h_block=1
w_block=1
c_block=1
k_block=1
h_in_gemm=2

./conv_bwd "${loop_string}" $N  56 56 64 256    1 1 1 1 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
# These two should early exit as strided convs do not support h_in_gemm != 1
./conv_bwd "${loop_string}" $N  14 14 1024 2048 1 1 2 2 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 512 512   3 3 2 2 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

h_block=1
w_block=1
c_block=1
k_block=1
h_in_gemm=7
./conv_bwd "${loop_string}" $N  7 7 512 512     3 3 1 1 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

h_block=1
w_block=2
c_block=1
k_block=1
h_in_gemm=1
./conv_bwd "${loop_string}" $N  14 14 1024 2048 1 1 2 2 0 0 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters
./conv_bwd "${loop_string}" $N  14 14 512 512   3 3 2 2 1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

h_block=1
w_block=1
c_block=1
k_block=1
h_in_gemm=2
./conv_bwd "${loop_string}" $N 14 14 256 256  3 3  1 1  1 1 $bc $bk $h_block $w_block $c_block $k_block $h_in_gemm $niters

#./conv_bwd $@
#export OMP_NUM_THREADS=28
#gdb --args ./conv_bwd $@
