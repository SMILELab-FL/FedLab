
# 单机多进程测试


python命令开启一个进程， server.py对应server进程；

client.py对应client进程，client进程可开多个，但一定要和world_size大小匹配，且local_rank不可重复；

示例为一个server两个client的联邦启动指令，在单个物理机上开启3个进程。

ip: localhost, port: 3002

window1:

`python server.py --server_ip 127.0.0.1 --server_port 3002 --world_size 3`

window2:

`CUDA_VISIBLE_DEVICES=2 python client.py --server_ip 127.0.0.1 --server_port 3002 --local_rank 1 --world_size 3`

window3:

`CUDA_VISIBLE_DEVICES=3 python client.py --server_ip 127.0.0.1 --server_port 3002 --local_rank 2 --world_size 3`
