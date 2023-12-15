SN=$1
# M,K dimensions of the MLP (weights)
D=5120
# N dimension of the MLP (acts)
A=1024
# Number of cores
NC=40
# blocking factor for accumulation chain (K)
BF=2
# Number of layers
L=40
# Number of benchmark iters
IT=400
# Number of teams dividing M
PW=10
# Number of teams dividing N
PA=4
# Core offsets for SNC0, SNC1, SNC2
OFF0=0
OFF1=40
OFF2=80
# BRGEMM block size
BS=32

export LIBXSMM_X86_AMX_GEMM_STREAMING_TILELOAD=1

if [ $(($SN)) == 3 ]; then
  echo "Run 3 SNC"
  OMP_NUM_THREADS=$NC numactl -m 0 -C $OFF0-$((NC-1)) ./gemm aB{R:$PW}C{C:$PA} $D $A $D $BS $BS $BS $BF $L $IT 0 0 BF16 &
  OMP_NUM_THREADS=$NC numactl -m 1 -C $OFF1-$(($OFF1+NC-1)) ./gemm aB{R:$PW}C{C:$PA} $D $A $D $BS $BS $BS $BF $L $IT 0 0 BF16  &
  OMP_NUM_THREADS=$NC numactl -m 2 -C $OFF2-$(($OFF2+NC-1)) ./gemm aB{R:$PW}C{C:$PA} $D $A $D $BS $BS $BS $BF $L $IT 0 0 BF16
fi

if [ $(($SN)) == 2 ]; then
  echo "Run 2 SNC"
  OMP_NUM_THREADS=$NC numactl -m 0 -C $OFF0-$((NC-1)) ./gemm aB{R:$PW}C{C:$PA} $D $A $D $BS $BS $BS $BF $L $IT 0 0 BF16 &
  OMP_NUM_THREADS=$NC numactl -m 1 -C $OFF1-$(($OFF1+NC-1)) ./gemm aB{R:$PW}C{C:$PA} $D $A $D $BS $BS $BS $BF $L $IT 0 0 BF16
fi

if [ $(($SN)) == 1 ]; then
  echo "Run 1 SNC"
  OMP_NUM_THREADS=$NC numactl -m 0 -C $OFF0-$((NC-1)) ./gemm aB{R:$PW}C{C:$PA} $D $A $D $BS $BS $BS $BF $L $IT 0 0 BF16
fi


