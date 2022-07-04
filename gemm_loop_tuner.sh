###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
#! /bin/bash

#SBATCH -J gemm_tune_job
#SBATCH --get-user-env
#SBATCH --partition=clx
#SBATCH --nodes=1
#SBATCH --time=23:59:00

source /swtools/gcc/9.4.0/gcc_vars.sh

export OMP_NUM_THREADS=28
export GOMP_CPU_AFFINITY="0-27"

loop_names=()
loop_specs=()

#For now we don't block additionally K dim, handled by BF of BR dimension
for i in 0; do
  for j in 0 1 2; do
    for k in 0 1 2; do
      loop_names+=("gemm${i}${j}${k}")
      loop_specs+=("a_${i}_K,b_${j}_M,c_${k}_N")
    done
  done
done

for i in ${!loop_specs[@]}; do
  ./loop_permute_generator "${loop_names[$i]}" "${loop_specs[$i]}"
done

cat *bench_configs.txt > uber_config.txt
rm -rf *bench_configs.txt
awk '!seen[$0]++' uber_config.txt > tuner_config.txt
rm uber_config.txt

#Try various blocking factors for BR dimension
for m in 512 1024 2048; do
  for n in 512 1024 2048; do
    for k in 512 1024 2048; do
      benchmark_out_name="${m}_${n}_${k}_bench_results"
      echo -e "Performance" > ${benchmark_out_name}
      KBFS=()
      if [ $k -eq 512 ]; then
        KBFS+=("1")
      fi
      if [ $k -eq 1024 ]; then
        KBFS+=("1")
      fi
      if [ $k -eq 2048 ]; then
        KBFS+=("1" "2")
      fi
      if [ $k -eq 4096 ]; then
        KBFS+=("1" "2" "4")
      fi
      for b in "${KBFS[@]}"; do
        loopArray=()
        nLoops=0
        while IFS= read -r line || [[ "$line" ]]; do
          loopArray+=("$line")
          let "nLoops+=1"
        done < tuner_config.txt
        for (( j = 0 ; j < $nLoops ; j++)); do
          line=${loopArray[$j]}
          export OMP_NUM_THREADS=28
          export GOMP_CPU_AFFINITY="0-27"
          unset LIBXSMM_VERBOSE
          ./gemm ${line} ${m} ${n} ${k} 32 32 32 ${b} 1 400 >> ${benchmark_out_name}
        done
      done
    done
  done
done


