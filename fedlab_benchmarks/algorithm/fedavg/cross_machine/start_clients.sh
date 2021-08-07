#!/bin/bash

# start a group of client. continuous rank is required.
# example: bash start_clients.sh 1 5
#           start client from rank 1-5

ip = $1
port = $2
world_size = $3

for ((i=$4; i<=$5; i++))
do
{
    echo "client ${i} started"
    python client.py --server_ip $ip --server_port $port --world_size $world_size --rank ${i} --dataset $6 --epoch 2
} & 
done
wait