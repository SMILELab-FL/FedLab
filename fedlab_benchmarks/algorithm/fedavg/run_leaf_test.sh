#!/bin/bash

#echo "server started"
#python server_fedavg.py --server_ip 127.0.0.1 --server_port 3004 --world_size 4  --dataset shakespeare &
#
#for ((i=1; i<=3; i++))
#do
#{
#    echo "client ${i} started"
#    CUDA_VISIBLE_DEVICES=3 python client_fedavg.py --server_ip 127.0.0.1 --server_port 3004 --world_size 4 --local_rank ${i} --dataset shakespeare
#} &
#done
#wait

# get world size dicts for all datasets and run all client processes foe each dataset
declare -A dataset_world_size=(['femnist']=15 ['shakespeare']=10)  # real split client number is ['femnist']=55 ['shakespeare']=122)

for key in ${!dataset_world_size[*]}
do
  echo "${key} client_num is ${dataset_world_size[${key}]}"

  echo "server started"
  python server.py --server_ip 127.0.0.1 --server_port 3002 --world_size ${dataset_world_size[${key}]}  --dataset ${key} &

  for ((i=1; i<${dataset_world_size[$key]}; i++))
  do
  {
      echo "client ${i} started"
      CUDA_VISIBLE_DEVICES=3 python client.py --server_ip 127.0.0.1 --server_port 3002 --world_size ${dataset_world_size[${key}]} --rank ${i} --dataset ${key}
  } &
  done
  wait

  echo "${key} experiment end"
done