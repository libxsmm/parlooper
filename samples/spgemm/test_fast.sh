M=$1
N=$2
K=$3
STRING=$4
BF16=$5
TARGET_NBLOCKS=$6
THREADS=$7
THREADS_MINUS_ONE=$((THREADS-1))

for sp_frac in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99
do
    CMD="OMP_NUM_THREADS=$THREADS KMP_AFFINITY=\"granularity=fine,compact,1,0\" numactl -m 0 -C 0-$THREADS_MINUS_ONE  ./spgemm ${STRING} $M $N $K 1 ${TARGET_NBLOCKS} ${sp_frac} ${BF16} 8 8 8 8 0 100 0"
    echo $CMD
    eval $CMD
done
