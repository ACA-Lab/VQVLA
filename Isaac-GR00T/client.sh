export CUDA_VISIBLE_DEVICES=1

dataset=${1:-"1"}

if [ ${dataset} -eq 1 ]
then
bash spatial-client.sh > logs/libero-spatial.log
fi

if [ ${dataset} -eq 2 ]
then
bash object-client.sh > logs/libero-object.log
fi

if [ ${dataset} -eq 3 ]
then
bash goal-client.sh > logs/libero-goal.log
fi

if [ ${dataset} -eq 4 ]
then
bash 10-client.sh > logs/libero-10.log
fi