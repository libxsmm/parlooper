###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
#!/bin/bash
CC_USE=gcc
CXX_USE=g++
if [[ -z "${PARLOOPER_COMPILER}" ]]; then
  CC_USE=gcc
  CXX_USE=g++
elif [[ "${PARLOOPER_COMPILER}" == "icc" ]]; then
  CC_USE=icc
  CXX_USE=icpc
elif [[ "${PARLOOPER_COMPILER}" == "icx" ]]; then
  CC_USE=icx
  CXX_USE=icpx
elif [[ "${PARLOOPER_COMPILER}" == "clang" ]]; then
  CC_USE=clang
  CXX_USE=clang++
elif [[ "${PARLOOPER_COMPILER}" == "gcc" ]]; then
  CC_USE=gcc
  CXX_USE=g++
else
  CC_USE=gcc
  CXX_USE=g++
fi
echo "Using compiler ${CC_USE} and ${CXX_USE} to build LIBXSMM and LIBXSMM_DNN"
BRANCH=main
BRANCHDNN=main
if [ $# -eq 2 ]; then
  BRANCH=$1
  BRANCHDNN=$2
fi
echo "Building PARLOOPER with libxsmm branch $BRANCH and libxsmm-dnn branch $BRANCHDNN"

#clone LIBXSMM
if [ ! -d "libxsmm" ]; then
  echo "libxsmm not exist, clone one from remote repo ..."
  git clone https://github.com/libxsmm/libxsmm.git  libxsmm
  cd libxsmm
else
  echo "libxsmm exists, just updating ..."
  cd libxsmm
  git pull
fi
echo "building LIBXSMM..."
echo "switching to $BRANCH branch ..."
git checkout $BRANCH
git pull
make realclean && make CC=${CC_USE} CXX=${CXX_USE} FC= -j16
echo "done building LIBXSMM"
cd ..

#clone LIBXSMM_DNN
if [ ! -d "libxsmm_dnn" ]; then
  echo "libxsmm_dnn not exist, clone one from remote repo ..."
  git clone https://github.com/libxsmm/libxsmm_dnn.git  libxsmm_dnn
  cd libxsmm_dnn
else
  echo "libxsmm_dnn exists, just updating ..."
  cd libxsmm_dnn
  git pull
fi
echo "building LIBXSMM_DNN..."
echo "switching to $BRANCHDNN branch ..."
git checkout $BRANCHDNN
git pull
export LIBXSMMROOT=../libxsmm
#make realclean && make CC=${CC_USE} CXX=${CXX_USE} FC= -j16
#echo "done building LIBXSMM_DNN"
cd ..
