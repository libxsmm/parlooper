git clone https://github.com/libxsmm/parlooper.git
cd parlooper
git checkout jit_with_clang
bash ./prepare_libxsmm.sh
make -j16 

export LIBXSMM_TARGET=neov1
export OMP_NUM_THREADS=64
export KMP_AFFINITY=compact
export USE_BF16=1

#Convs
bash RN50_parlooper_graviton3_bench.sh 64 64 16 4 > RN50_mb64_thr64
cat RN50_mb64_thr64 | grep GFLOPS | cut -d' ' -f2

bash RN50_parlooper_graviton3_bench.sh 64 1 1 1 > RN50_mb1_thr64
cat RN50_mb1_thr64 | grep GFLOPS | cut -d' ' -f2

bash RN50_parlooper_graviton3_bench.sh 16 1 1 1  > RN50_mb1_thr16
cat RN50_mb1_thr16 | grep GFLOPS | cut -d' ' -f2

bash RN50_parlooper_graviton3_bench.sh 8 1 1 1 > RN50_mb1_thr8
cat RN50_mb1_thr8 | grep GFLOPS | cut -d' ' -f2

#MLP
#MB=C=K
./gemm "aB{R:8}C{C:8}" 512 512 512 32 32 32 1 200 100 > mlp_var_mb_bench
./gemm "aB{R:8}C{C:8}" 1024 1024 1024 32 32 32 1 20 100 >>  mlp_var_mb_bench
./gemm "aBCbc" 2048 2048 2048 32 32 32 2 20 100 >>  mlp_var_mb_bench
./gemm "aBCbc" 4096 4096 4096 32 32 32 4 20 100 >>  mlp_var_mb_bench
./gemm "aBCbc" 8192 8192 8192 32 32 32 8 20 100 >>  mlp_var_mb_bench
cat mlp_var_mb_bench | grep MEASURE | cut -d' ' -f2

#MB=512
./gemm "aBC" 512 512 512 32 32 32 1 200 100 > mlp_512_mb_bench
./gemm "aBC" 1024 512 1024 32 32 32 1 20 100 >> mlp_512_mb_bench
./gemm "aBCbc" 2048 512 2048 32 32 32 4 20 100 >> mlp_512_mb_bench
./gemm "aBCbc" 4096 512 4096 32 32 32 4 20 100 >> mlp_512_mb_bench
./gemm "aBCbc" 8192 512 8192 32 32 32 8 20 100 >> mlp_512_mb_bench
cat mlp_512_mb_bench | grep MEASURE | cut -d' ' -f2

#MB=256
./gemm "aBC" 512 256 512 32 32 32 1 200 100 > mlp_256_mb_bench
./gemm "aBC" 1024 256 1024 32 32 32 1 20 100 >> mlp_256_mb_bench
./gemm "aBC" 2048 256 2048 32 32 32 4 20 100 >> mlp_256_mb_bench
./gemm "aBCbc" 4096 256 4096 32 32 32 8 20 100 >> mlp_256_mb_bench
./gemm "aBCbc" 8192 256 8192 32 32 32 8 20 100 >> mlp_256_mb_bench
cat mlp_256_mb_bench | grep MEASURE | cut -d' ' -f2


