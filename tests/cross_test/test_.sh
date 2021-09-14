#!bin/bash

python server.py &
sleep 1s

python client.py --rank 1 &
sleep 1s

python client.py --rank 2 &
sleep 1s

wait