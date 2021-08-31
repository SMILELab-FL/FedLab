#!bin/bash

python server.py --ip 127.0.0.1 --port 3002 --world_size 3 --dataset mnist --round 3 &

python client.py --ip 127.0.0.1 --port 3002 --world_size 3 --rank 1 --dataset mnist &

python client.py --ip 127.0.0.1 --port 3002 --world_size 3 --rank 2 --dataset mnist &

wait