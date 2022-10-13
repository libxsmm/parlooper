###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
#!/bin/bash

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
make realclean && make CC=clang CXX=clang++ FC= -j16
echo "done building LIBXSMM"
cd ..

#clone LIBXSMM_DNN
if [ ! -d "libxsmm_dnn" ]; then
  echo "libxsmm_dnn not exist, clone one from remote repo ..."
  git clone https://github.com/libxsmm/libxsmm_dnn.git  libxsmm_dnn
  cd libxsmm_dnn
  git checkout f309a040e46a968d86e24cc7da16325e756b303b
else
  echo "libxsmm_dnn exists, just updating ..."
  cd libxsmm_dnn
  git pull
  git checkout f309a040e46a968d86e24cc7da16325e756b303b
fi
echo "building LIBXSMM_DNN..."
export LIBXSMMROOT=../libxsmm
make realclean && make CC=clang CXX=clang++ FC= -j16
echo "done building LIBXSMM_DNN"
cd ..
