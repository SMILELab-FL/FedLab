#!/bin/bash

python server.py --world_size 11 --round 3 &
sleep 2s

for ((i=1; i<=10; i++))
do
{
    echo "client ${i} started"
    python client.py --world_size 11 --rank ${i} &
    sleep 2s
}
done

wait