
server:
    machine 9: eth eno1, ip 192.168.1.202
            python server.py --server_ip 192.168.1.202 --server_port 3002 --world_size 20 --dataset mnist --round 10 --ethernet eno1 --sample 0.5

client:
    machine 11: eth enp129s0f0
         bash start_clients.sh 192.168.1.202 3002 20 1 10 mnist enp129s0f0 5 0,1,2,3,4,5,6,7
    machine 13: eth enp129s0f0
         bash start_clients.sh 192.168.1.202 3002 20 11 19 mnist enp129s0f0 5 0,1,2,3,4,5,6,7

Start a server:

`python server.py --ip ${1} --port ${2} --world_size ${3} --dataset ${4} --round ${5} --ethernet ${6} --sample ${7}`


Start clients from rank a to rank b continuous.

`bash start_clients ip port world_size rank_a rank_b dataset ethernet epoch gpu`

