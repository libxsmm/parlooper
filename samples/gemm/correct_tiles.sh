unset LIBXSMM_X86_AMX_GEMM_PANEL_SW_PIPELINE_GRANULARITY
export LIBXSMM_X86_AMX_GEMM_STREAMING_A=1
export LIBXSMM_X86_AMX_GEMM_STREAMING_B=1

KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 4096 512 4096 64 256 32 2 1 10 0 0 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 4096 512 4096 64 256 32 2 1 10 FLAT_A 0 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 4096 512 4096 64 256 32 2 1 10 FLAT_A TRA 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 4096 512 4096 128 256 32 16 1 10 FLAT_A 0 TRB 1

KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 8192 512 4096 128 256 32 8 1 10 0 0 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 8192 512 4096 128 256 32 8 1 10 FLAT_A 0 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 8192 512 4096 128 256 32 8 1 10 FLAT_A TRA 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 8192 512 4096 128 256 32 8 1 10 FLAT_A 0 TRB 1

KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 16384 512 4096 256 256 32 8 1 10 0 0 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 16384 512 4096 256 256 32 8 1 10 FLAT_A 0 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 16384 512 4096 256 256 32 8 1 10 FLAT_A TRA 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 16384 512 4096 256 256 32 8 1 10 FLAT_A 0 TRB 1

KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 32768 512 4096 128 512 32 4 1 10 0 0 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 32768 512 4096 128 512 32 4 1 10 FLAT_A 0 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 32768 512 4096 128 512 32 4 1 10 FLAT_A TRA 0 1
KMP_AFFINITY=granularity=fine,compact,1,0 OMP_NUM_THREADS=64 srun numactl -m 0 -C 0-63 ./gemm SFC 32768 512 4096 256 512 32 8 1 10 FLAT_A 0 TRB 1




