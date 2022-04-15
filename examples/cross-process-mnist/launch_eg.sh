#!bin/bash

python server.py --ip 127.0.0.1 --port 3001 --world_size 3 --round 3 &

python client.py --ip 127.0.0.1 --port 3001 --world_size 3 --rank 1 &

python client.py --ip 127.0.0.1 --port 3001 --world_size 3 --rank 2  &

wait