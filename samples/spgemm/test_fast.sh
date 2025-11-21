M=$1
N=$2
K=$3
STRING=$4
BF16=$5
TARGET_NBLOCKS=$6
THREADS=$7
THREADS_MINUS_ONE=$((THREADS-1))
# bm should be 1 if M =1 otherwise it should be 32
if [ $M -eq 1 ]; then
    bm=1
else
    bm=32
fi

for bs in 8 16 32 64
do
    for sp_frac in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99
    do
        CMD="OMP_NUM_THREADS=$THREADS KMP_AFFINITY=\"granularity=fine,compact,1,0\" numactl -m 0 -C 0-$THREADS_MINUS_ONE  ./spgemm ${STRING} $M $N $K ${bm} ${TARGET_NBLOCKS} ${sp_frac} ${BF16} ${bs} ${bs} ${bs} ${bs} 0 100 0"
        echo $CMD
        eval $CMD
    done
done