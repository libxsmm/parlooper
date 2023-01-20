###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
#! /bin/bash

source /swtools/gcc/latest/gcc_vars.sh

export OMP_NUM_THREADS=24
export GOMP_CPU_AFFINITY="0-23"

loop_names=()
loop_specs=()

#For now we don't block additionally K dim, handled by BF of BR dimension
#for i in 0 1; do
#  for j in 0 1; do
for i in 0; do
  for j in 1; do
    loop_names+=("conv1")
    loop_specs+=("a_0_K,b_0_K,c_${i}_M,d_${j}_N,e_0_N,f_0_K,g_0_K")
  done
done

for i in ${!loop_specs[@]}; do
  ./loop_permute_generator "${loop_names[$i]}" "${loop_specs[$i]}"
done

cat *bench_configs.txt > uber_config.txt
rm -rf *bench_configs.txt
awk '!seen[$0]++' uber_config.txt > tuner_config.txt
rm uber_config.txt

loopArray=()
nLoops=0

cat tuner_config.txt | sort > SORTED
cat SORTED > tuner_config.txt
rm SORTED

while IFS= read -r line || [[ "$line" ]]; do
  loopArray+=("$line")
  let "nLoops+=1"
done < tuner_config.txt

nConvs=20

for l in 2 3 7 12; do
  N=1
  bc=32
  bk=32
  niters=100
  H=56
  W=56
  C=64
  K=256
  R=1
  S=1
  str=1
  pad=0
  CBFS=("1" "2")
  WBFS=("1" "2" "4")

  if [ $l -eq 0 ]; then
    H=224
    W=224
    C=3
    K=64
    R=7
    S=7
    str=2
    pad=3
    bc=3
    CBFS=("1")
    WBFS=("1" "2" "4" "7")
  fi

  if [ $l -eq 1 ]; then
    H=56
    W=56
    C=64
    K=256
    R=1
    S=1
    str=1
    pad=0
    CBFS=("1" "2")
    WBFS=("1" "2")
  fi

  #GFLOPS 1106.11 afgbEDc_hb=1_wb=2_cb=1_kb=1
  if [ $l -eq 2 ]; then
    H=56
    W=56
    C=64
    K=64
    R=1
    S=1
    str=1
    pad=0
    CBFS=("1")
    WBFS=("1" "2" "4" "8")
    HBFS=("2" "4" "7" "14")
  fi

  #GFLOPS 1402.66 afgbCDE_hb=1_wb=2_cb=1_kb=1
  if [ $l -eq 3 ]; then
    H=56
    W=56
    C=64
    K=64
    R=3
    S=3
    str=1
    pad=1
    CBFS=("1")
    WBFS=("1" "2" "4" "8")
    HBFS=("2" "4" "7" "14")
  fi

  if [ $l -eq 4 ]; then
    H=56
    W=56
    C=256
    K=64
    R=1
    S=1
    str=1
    pad=0
    CBFS=("1" "2" "8")
    WBFS=("1" "2")
  fi

  if [ $l -eq 5 ]; then
    H=56
    W=56
    C=256
    K=512
    R=1
    S=1
    str=2
    pad=0
    CBFS=("1" "2" "8")
    WBFS=("1" "2")
  fi

  if [ $l -eq 6 ]; then
    H=56
    W=56
    C=256
    K=128
    R=1
    S=1
    str=2
    pad=0
    CBFS=("1" "2" "8")
    WBFS=("1" "2")
  fi

  #GFLOPS 1409.27 afgbCDE_hb=1_wb=2_cb=1_kb=1
  if [ $l -eq 7 ]; then
    H=28 
    W=28
    C=128
    K=128
    R=3
    S=3
    str=1
    pad=1
    CBFS=("1")
    WBFS=("1" "2" "4")
    HBFS=("2" "4" "7" "14")
  fi

  if [ $l -eq 8 ]; then
    H=28 
    W=28
    C=128
    K=512
    R=1
    S=1
    str=1
    pad=0
    CBFS=("1" "2" "4")
    WBFS=("1" "2")
  fi

  if [ $l -eq 9 ]; then
    H=28 
    W=28
    C=512
    K=128
    R=1
    S=1
    str=1
    pad=0
    CBFS=("1" "4" "16")
    WBFS=("1" "2")
  fi

  if [ $l -eq 10 ]; then
    H=28 
    W=28
    C=512
    K=1024
    R=1
    S=1
    str=2
    pad=0
    CBFS=("1" "4" "16")
    WBFS=("1" "2")
  fi

  if [ $l -eq 11 ]; then
    H=28 
    W=28
    C=512
    K=256
    R=1
    S=1
    str=2
    pad=0
    CBFS=("1" "4" "16")
    WBFS=("1" "2")
  fi

  #GFLOPS 1318.95 afgbCDE_hb=1_wb=2_cb=1_kb=1
  if [ $l -eq 12 ]; then
    H=14 
    W=14
    C=256
    K=256
    R=3
    S=3
    str=1
    pad=1
    CBFS=("1" "2")
    WBFS=("1" "2")
    HBFS=("2" "7")
  fi

  if [ $l -eq 13 ]; then
    H=14 
    W=14
    C=256
    K=1024
    R=1
    S=1
    str=1
    pad=0
    CBFS=("1" "4" "8")
    WBFS=("1" "2")
  fi

  if [ $l -eq 14 ]; then
    H=14 
    W=14
    C=1024
    K=256
    R=1
    S=1
    str=1
    pad=0
    CBFS=("1" "8" "16")
    WBFS=("1" "2")
  fi

  if [ $l -eq 15 ]; then
    H=14 
    W=14
    C=1024
    K=2048
    R=1
    S=1
    str=2
    pad=0
    CBFS=("1" "8" "16")
    WBFS=("1")
  fi

  if [ $l -eq 16 ]; then
    H=14 
    W=14
    C=1024
    K=512
    R=1
    S=1
    str=2
    pad=0
    CBFS=("1" "8" "16")
    WBFS=("1")
  fi

  if [ $l -eq 17 ]; then
    H=7 
    W=7
    C=512
    K=512
    R=3
    S=3
    str=1
    pad=1
    CBFS=("1" "8" "16")
    WBFS=("1")
  fi

  if [ $l -eq 18 ]; then
    H=7 
    W=7
    C=512
    K=2048
    R=1
    S=1
    str=1
    pad=0
    CBFS=("1" "8" "16")
    WBFS=("1")
  fi

  if [ $l -eq 19 ]; then
    H=7 
    W=7
    C=2048
    K=512
    R=1
    S=1
    str=1
    pad=0
    CBFS=("1" "8" "16")
    WBFS=("1")
  fi

  benchmark_out_name="${H}_${W}_${C}_${K}_${R}_${S}_${str}_${pad}_adl_conv_bench_results"
  echo "Tuning convs..." > ${benchmark_out_name}
  
  for (( j = 0 ; j < $nLoops ; j++)); do
    line=${loopArray[$j]}
    #echo ${line}
    lowerline=$(echo ${line} | tr '[:upper:]' '[:lower:]')
    #echo ${lowerline}
    KBFcount=$( echo ${lowerline} | tr -d -c 'c' | awk '{ print length; }' )
    #echo "C count is ${KBFcount}"
    HBFcount=$( echo ${lowerline} | tr -d -c 'd' | awk '{ print length; }' )
    #echo "D count is ${HBFcount}"
    if [[ ${HBFcount} -eq 1 ]]
    then
      continue
    fi
   #echo "HBF count is ${HBFcount}"
    
    if [ $KBFcount -eq 2 ]; then
      if [ $HBFcount -eq 2 ]; then
        for cb in "${CBFS[@]}"; do
          for kb in "${KBFS[@]}"; do
            for wb in "${WBFS[@]}"; do
              for hb in "${HBFS[@]}"; do
                export OMP_NUM_THREADS=24     
                export GOMP_CPU_AFFINITY="0-23"
                unset LIBXSMM_VERBOSE
                ./conv_fwd ${line} ${N} ${H} ${W} ${C} ${K} ${R} ${S} ${str} ${str} ${pad} ${pad} ${bc} ${bk} ${hb} ${wb} ${cb} ${kb} ${niters}  >> ${benchmark_out_name}
              done
            done
          done
        done
      fi
    fi

    if [ $KBFcount -eq 2 ]; then
      if [ $HBFcount -eq 1 ]; then
        for cb in "${CBFS[@]}"; do
          for kb in "${KBFS[@]}"; do
            for wb in "${WBFS[@]}"; do
              hb=1
              export OMP_NUM_THREADS=24     
              export GOMP_CPU_AFFINITY="0-23"
              unset LIBXSMM_VERBOSE
              ./conv_fwd ${line} ${N} ${H} ${W} ${C} ${K} ${R} ${S} ${str} ${str} ${pad} ${pad} ${bc} ${bk} ${hb} ${wb} ${cb} ${kb} ${niters}  >> ${benchmark_out_name}
            done
          done
        done
      fi
    fi

    if [ $KBFcount -eq 1 ]; then
      if [ $HBFcount -eq 2 ]; then
        for cb in "${CBFS[@]}"; do
          for wb in "${WBFS[@]}"; do
            for hb in "${HBFS[@]}"; do
              kb=1
              export OMP_NUM_THREADS=24     
              export GOMP_CPU_AFFINITY="0-23"
              unset LIBXSMM_VERBOSE
              #echo "Testing config ${line}"            
              ./conv_fwd ${line} ${N} ${H} ${W} ${C} ${K} ${R} ${S} ${str} ${str} ${pad} ${pad} ${bc} ${bk} ${hb} ${wb} ${cb} ${kb} ${niters}  >> ${benchmark_out_name}
            done
          done
        done
      fi
    fi

    if [ $KBFcount -eq 1 ]; then
      if [ $HBFcount -eq 1 ]; then
        for cb in "${CBFS[@]}"; do
          for wb in "${WBFS[@]}"; do
            kb=1
            hb=1
            export OMP_NUM_THREADS=24     
            export GOMP_CPU_AFFINITY="0-23"
            unset LIBXSMM_VERBOSE
            ./conv_fwd ${line} ${N} ${H} ${W} ${C} ${K} ${R} ${S} ${str} ${str} ${pad} ${pad} ${bc} ${bk} ${hb} ${wb} ${cb} ${kb} ${niters}  >> ${benchmark_out_name}
          done
        done
      fi
    fi
  done

done

