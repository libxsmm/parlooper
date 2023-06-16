#!/bin/bash

# Source compiler and OpenMP RT (Intel OpenMP)

#export LD_PRELOAD=/usr/lib64/libstdc++.so.6:$LD_PRELOAD

omp=56
batch=56
build=# build directory for oneDNN build for SPR

export KMP_AFFINITY=granularity=fine,compact,1,0 # if HT is on
#export KMP_AFFINITY=granularity=fine,compact # if HT is off

#export ONEDNN_JIT_DUMP=1
#export ONEDNN_VERBOSE=2

# bf16, training, fwd, no bias, no relu, parlooper SC23 paper
# File shapes_resnet_50_v1_5_parlooper should be copied from the repo to tests/benchdnn/inputs/conv prior to running benchdnn
OMP_NUM_THREADS=$omp numactl -l \
  ./$build/tests/benchdnn/benchdnn --conv --cfg=bf16bf16bf16 --mode=p --dir=FWD_D --mb=56 --batch=tests/benchdnn/inputs/conv/shapes_resnet_50_v1_5_parlooper
exit