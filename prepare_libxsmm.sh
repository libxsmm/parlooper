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
make realclean && make CC=gcc CXX=g++ FC= -j16
echo "done building LIBXSMM"
cd ..

