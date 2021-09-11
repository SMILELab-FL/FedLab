#!/bin/bash

python client_alone.py --world_size 4 --rank 1 &
#python client.py --world_size 4 --rank 1 --client_num 5 &
sleep 2s

python client.py --world_size 4 --rank 2 --client_num 14 &
#python client.py --world_size 4 --rank 2 --client_num 10 &
sleep 2s

python client.py --world_size 4 --rank 3 --client_num 15 &
sleep 2s

wait