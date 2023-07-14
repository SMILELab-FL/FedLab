#!/bin/bash


rm -rf ./.fedboard

python server.py --world_size 11 --round 10 &
echo "server started"
sleep 2

for ((i=1; i<=10; i++))
do
{
    echo "client ${i} started"
    python client.py --world_size 11 --rank ${i} &
    sleep 1
}
done

python board.py &
echo "board started"
sleep 5

wait