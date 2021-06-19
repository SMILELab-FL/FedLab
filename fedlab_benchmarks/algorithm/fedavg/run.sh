#!/bin/bash

# python server.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3
# CUDA_VISIBLE_DEVICES=2 python client.py --server_ip 127.0.0.1 --server_port 3002 --local_rank 1 --world_size 3
# CUDA_VISIBLE_DEVICES=3 python client.py --server_ip 127.0.0.1 --server_port 3002 --local_rank 2 --world_size 3

echo "server started"
python server.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3  &

for ((i=1; i<=2; i++))
do
{
    echo "client ${i} started"
    python client.py --server_ip 127.0.0.1 --server_port 3002 --local_rank ${i} --world_size 3
}& 
done
wait
