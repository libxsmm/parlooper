###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
#! /bin/bash

threads=$1
platform=$2

for m in 512 1024 2048; do
  for n in 512 1024 2048; do
    for k in 512 1024 2048; do
      benchmark_out_name="${m}_${n}_${k}_bench_results"
      cat ${benchmark_out_name} | grep MEASURE | cut -d' ' -f2-3 | sort -r -k1 -n > measured
      cat ${benchmark_out_name} | grep MODELED | cut -d' ' -f2-3 | sort -r -k1 -n > modeled
      awk 'FNR==NR {x2[$2] = $0; next} $2 in x2 {print x2[$2]}' measured modeled > sorted
      python gemm_ploter.py sorted modeled ${m}_${n}_${k}_${threads}_${platform}
      rm sorted modeled measured
    done
  done
done


