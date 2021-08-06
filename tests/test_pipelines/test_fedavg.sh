#!/bin/bash


echo "server started"

python server.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3  --dataset mnist &

for ((i=1; i<=2; i++))
do
{
    echo "client ${i} started"
    python client.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3 --dataset mnist --rank ${i} --epoch 1
} & 
done

wait
