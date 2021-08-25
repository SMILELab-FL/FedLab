#!/bin/bash

# start a group of client. continuous rank is required.
# example: bash start_clients.sh ip port wolrd_size 1 5 dataset ethernet epoch
#           start client from rank 1-5

echo "Connecting server:($1:$2), world_size $3, rank $4-$5"


for ((i=$4; i<=$5; i++))
do
{
    echo "client ${i} started"
    python client.py --ip $1 --port $2 --world_size $3 --rank ${i} &
    sleep 2s
}
done
wait


