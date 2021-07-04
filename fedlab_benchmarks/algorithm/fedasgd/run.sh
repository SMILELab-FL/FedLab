#!/bin/bash

echo "server started"
python server_fedasgd.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3  &

for ((i=1; i<=2; i++))
do
{
    echo "client ${i} started"
    CUDA_VISIBLE_DEVICES=3 python client_fedasgd.py --server_ip 127.0.0.1 --server_port 3002 --local_rank ${i} --world_size 3
} & 
done
wait
