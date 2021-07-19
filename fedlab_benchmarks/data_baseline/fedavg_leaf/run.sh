#!/bin/bash

# python server.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3
# CUDA_VISIBLE_DEVICES=2 python client.py --server_ip 127.0.0.1 --server_port 3002 --local_rank 1 --world_size 3
# CUDA_VISIBLE_DEVICES=3 python client.py --server_ip 127.0.0.1 --server_port 3002 --local_rank 2 --world_size 3

echo "server started"
python server_fedavg.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3  --dataset shakespeare &

for ((i=1; i<=2; i++))
do
{
    echo "client ${i} started"
    CUDA_VISIBLE_DEVICES=3 python client_fedavg.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3 --local_rank ${i} --dataset shakespeare
} &
done
wait

#declare -A dataset_client_num=(['femnist']=5 ['shakespeare']=5)
#
#for key in ${!dataset_client_num[*]}
#do
#  echo "${key} client_num is ${dataset_client_num[${key}]}"
#
#  echo "server started"
#  python server_fedavg.py --server_ip 127.0.0.1 --server_port 3002 --world_size ${dataset_client_num[${key}]}  --dataset ${key} &
#
#  for ((i=1; i<=${dataset_client_num[$key]}; i++))
#  do
#  {
#      echo "client ${i} started"
#      CUDA_VISIBLE_DEVICES=3 python client_fedavg.py --server_ip 127.0.0.1 --server_port 3002 --world_size ${dataset_client_num[${key}]} --local_rank ${i} --dataset ${key}
#  } &
#  done
#  wait
#
#  echo "${key} experiment end"
#done
