#!/bin/bash


# 参数 client_num cuda
# 单机 1 server 2 client
# 联邦平均测试 
python server.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3  &

for ((i=1; i<=2; i++))
do
{
    echo "client ${i} started"
    CUDA_VISIBLE_DEVICES=2 python client.py --server_ip 127.0.0.1 --server_port 3002 --local_rank ${i} --world_size 3
}& 

done
wait
