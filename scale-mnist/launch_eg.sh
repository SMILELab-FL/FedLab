#!/bin/bash

# python server.py --ip 10.249.47.145 --world_size 11 --round 100 --ethernet enp5s0 &
# sleep 2s

for ((i=1; i<=5; i++))
do
{
    echo "client ${i} started"
    python client.py --ip 10.249.47.145 --world_size 11 --rank ${i} --ethernet enp5s0 &
    sleep 2s
}
done

wait