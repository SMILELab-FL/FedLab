#!/bin/bash

# get world size dicts for all datasets and run all client processes foe each dataset
declare -A dataset_world_size=(['femnist']=15 ['shakespeare']=10)  # real split client number is ['femnist']=55 ['shakespeare']=122)

for key in ${!dataset_world_size[*]}
do
  echo "${key} client_num is ${dataset_world_size[${key}]}"

  echo "server started"
  python server.py --ip 127.0.0.1 --port 3002 --world_size ${dataset_world_size[${key}]}  --dataset ${key} &

  for ((i=1; i<${dataset_world_size[$key]}; i++))
  do
  {
      echo "client ${i} started"
      python client.py --ip 127.0.0.1 --port 3002 --world_size ${dataset_world_size[${key}]} --rank ${i} --dataset ${key} --epoch 2
  } &
  done
  wait

  echo "${key} experiment end"
done