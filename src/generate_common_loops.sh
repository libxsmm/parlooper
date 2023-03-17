#!/bin/bash


loop_names=()
loop_specs=()

#For now we don't block additionally K dim, handled by BF of BR dimension
for i in 0; do
  for j in 0 1 2 3; do
    for k in 0 1 2 3; do
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

./common_loop_generator tuner_config.txt TEST

