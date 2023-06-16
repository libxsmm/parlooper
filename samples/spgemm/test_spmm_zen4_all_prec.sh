MB=$1
STRING=$2
BF16=$3

if [ $BF16 -eq 0 ]
then
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.0 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.1 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.2 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.3 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.4 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.5 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.6 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.7 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.8 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.9 ${BF16} 1 1 1 1 1000
  ./spgemm ${STRING} ${MB} 2048 2048 64 16 0.99 ${BF16} 1 1 1 1 1000
fi

./spgemm ${STRING} ${MB} 2048 2048 64 16 0.0 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.1 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.2 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.3 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.4 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.5 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.6 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.7 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.8 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.9 ${BF16} 2 1 2 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.99 ${BF16} 2 1 2 1 1000

./spgemm ${STRING} ${MB} 2048 2048 64 16 0.0 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.1 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.2 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.3 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.4 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.5 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.6 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.7 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.8 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.9 ${BF16} 4 1 4 1 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.99 ${BF16} 4 1 4 1 1000

./spgemm ${STRING} ${MB} 2048 2048 64 16 0.0 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.1 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.2 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.3 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.4 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.5 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.6 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.7 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.8 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.9 ${BF16} 2 2 2 2 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.99 ${BF16} 2 2 2 2 1000

./spgemm ${STRING} ${MB} 2048 2048 64 16 0.0 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.1 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.2 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.3 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.4 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.5 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.6 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.7 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.8 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.9 ${BF16} 4 4 4 4 1000
./spgemm ${STRING} ${MB} 2048 2048 64 16 0.99 ${BF16} 4 4 4 4 1000

./spgemm ${STRING} ${MB} 2048 2048 32 16 0.0 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.1 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.2 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.3 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.4 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.5 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.6 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.7 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.8 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.9 ${BF16} 8 8 8 8 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.99 ${BF16} 8 8 8 8 1000

./spgemm ${STRING} ${MB} 2048 2048 32 16 0.0 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.1 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.2 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.3 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.4 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.5 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.6 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.7 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.8 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.9 ${BF16} 16 8 16 16 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.99 ${BF16} 16 8 16 16 1000

./spgemm ${STRING} ${MB} 2048 2048 32 16 0.0 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.1 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.2 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.3 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.4 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.5 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.6 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.7 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.8 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.9 ${BF16} 32 8 32 32 1000
./spgemm ${STRING} ${MB} 2048 2048 32 16 0.99 ${BF16} 32 8 32 32 1000

