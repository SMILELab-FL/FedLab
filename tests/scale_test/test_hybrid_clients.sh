#!/bin/bash

python server.py &
sleep 2s

python client_alone.py --world_size 4 --rank 1 &
sleep 2s

python client.py --world_size 4 --rank 2 --client_num 14 &
sleep 2s

python client.py --world_size 4 --rank 3 --client_num 15 &
sleep 2s

wait