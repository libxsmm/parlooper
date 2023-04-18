#!/bin/bash

#export LD_PRELOAD=/usr/lib64/libstdc++.so.6:$LD_PRELOAD

omp=64
batch=64
build=# build directory for oneDNN build with ACL (ACL should be built for arch=arch=armv8.6-a-sve at least


# bf16, training, fwd, no bias, no relu, parlooper SC23 paper
# File shapes_resnet_50_v1_5_parlooper should be copied from the repo to tests/benchdnn/inputs/conv prior to running benchdnn
omp=64
#ONEDNN_JIT_DUMP=1 \
OMP_NUM_THREADS=$omp LD_PRELOAD=/usr/lib64/libomp.so:$LD_PRELOAD \
  ONEDNN_DEFAULT_FPMATH_MODE=BF16 \
    ./$build/tests/benchdnn/benchdnn --conv --cfg=f32 --attr-fpmath=bf16 --mode=p --dir=FWD_D --mb=$batch \
      --batch=tests/benchdnn/inputs/conv/shapes_resnet_50_v1_5_parlooper
exit


