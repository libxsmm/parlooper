export OMP_NUM_THREADS=64
export KMP_AFFINITY=compact 
./test_spmm_gvt3_dot_all_prec.sh 256 aBC 0 > DOT_FP32_results_latest
./test_spmm_gvt3_dot_all_prec.sh 256 aBC 1 > DOT_BF16_results_latest
./test_spmm_gvt3_mmla_bf16_prec.sh 256 aBC 1 > MMLA_BF16_results_latest

