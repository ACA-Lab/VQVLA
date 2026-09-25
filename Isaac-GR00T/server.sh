export CUDA_VISIBLE_DEVICES=1

dataset=${1:-"1"}

if [ ${dataset} -eq 1 ]
then
bash spatial-server.sh
fi

if [ ${dataset} -eq 2 ]
then
bash object-server.sh
fi

if [ ${dataset} -eq 3 ]
then
bash goal-server.sh
fi

if [ ${dataset} -eq 4 ]
then
bash 10-server.sh
fi