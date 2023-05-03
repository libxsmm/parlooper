export OMP_NUM_THREADS=56
export KMP_AFFINITY=granularity=fine,compact,1,0 
export LIBXSMM_TARGET=spr
./test_spmm_spr_amx_bf16_prec.sh 256 aBC 1 > SPR_AMX_BF16_results
export LIBXSMM_TARGET=cpx
./test_spmm_spr_avx512_all_prec.sh 256 aBC 1 > SPR_AVX512_BF16_results
./test_spmm_spr_avx512_all_prec.sh 256 aBC 0 > SPR_AVX512_FP32_results

