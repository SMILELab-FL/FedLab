#!bin/bash

python middle_server.py  --ip_u 127.0.0.1 --port_u 3002 --world_size_u 3 --rank_u 2 --ip_l 127.0.0.1 --port_l 3003 --world_size_l 3 --rank_l 0 &

python single_client.py --ip 127.0.0.1 --port 3003 --world_size 3 --rank 1 &

python serial_client.py --ip 127.0.0.1 --port 3003 --world_size 3 --rank 2 &

wait