#!/bin/bash


echo "start server\n ip:port ${1}:${2}, world_size ${3}, dataset $4, round $5"


python server.py --server_ip ${1} --server_port ${2} --world_size ${3} --round ${5} --dataset ${4} --ethernet $6