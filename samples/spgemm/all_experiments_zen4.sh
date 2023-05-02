export OMP_NUM_THREADS=16
export KMP_AFFINITY=granularity=fine,compact,1,0 
./test_spmm_zen4_all_prec.sh 256 aBC 0 > ZEN4_FP32_results_latest
./test_spmm_zen4_all_prec.sh 256 aBC 1 > ZEN4_BF16_results_latest

