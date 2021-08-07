#!/bin/bash


ip = $1
port = $2
world_size = $3

dataset = $4
round = $5

python server.py --server_ip $ip --server_port $port --world_size $world_size --round $round --dataset $dataset