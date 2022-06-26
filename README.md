# Threaded loops infrastructure with exemplary GEMM/MLP/CONV implementations

## Contributors
* Evangelos Georganas (Intel Corp.)
* Dhiraj Kalamkar (Intel Corp.)

## Build instructions
```
source /swtools/gcc/latest/gcc_vars.sh 
bash prepare_libxsmm.sh 
make
```

## Exemplary run of test matmul and forward convolution
```
salloc --nodes=1 --partition=clx --time=03:59:00
export OMP_NUM_THREADS=28
export GOMP_CPU_AFFINITY="0-27"
srun ./test aCBc 2048 2048 2048 32 32 32 2 1 400
srun ./conv_fwd Abcdefg 28 14 14 64 64 3 3 1 1 1 1 32 32 1 1 1 1 1 400
 ```
