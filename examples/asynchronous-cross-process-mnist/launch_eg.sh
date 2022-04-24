#!/bin/bash


python server.py --world_size 3 &

python client.py --world_size 3 --rank 1 --epoch 10 &
python client.py --world_size 3 --rank 2 --epoch 5 &

wait