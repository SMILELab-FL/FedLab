
server:
    machine 9: eth eno1, ip 192.168.1.202
            python server.py --server_ip 192.168.1.202 --server_port 3002 --world_size 40 --dataset mnist --round 10 --ethernet eno1 --sample 0.5

client:
    machine 11: eth enp129s0f0
         bash start_clients.sh 192.168.1.202 3002 40 1 20 mnist enp129s0f0 5
    machine 13: eth enp129s0f0
         bash start_clients.sh 192.168.1.202 3002 40 21 39 mnist enp129s0f0 5

Start a server:

`python server.py --server_ip ${1} --server_port ${2} --world_size ${3} --dataset ${4} --round ${5} --ethernet ${6} --sample ${7}`


Start clients from rank a to rank b continuous.

`bash start_clients ip port world_size rank_a rank_b dataset ethernet epoch gpu`

