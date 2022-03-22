# Test your network configuration

*Example*: a FL simulation system with **k** client and **1** server

server process run server.py:
> python server.py --ip 121.23.54.65 (random example) --port 12345 --world_size k+1 --ethernet lo1 (ethernet name)

client 1 process run client.py
> pthon client.py --ip 121.23.54.65(server ip) --port 12345 --world_size k+1 --rank 1 --ethernet eno (ethernet name)

client 2 process run client.py
> pthon client.py --ip 121.23.54.65(server ip) --port 12345 --world_size k+1 --rank 2 --ethernet docker0 (ethernet name)

...

client k process run client.py
> pthon client.py --ip 121.23.54.65(server ip) --port 12345 --world_size k+1 --rank k --ethernet docker2 (ethernet name)

Check the ethernet name by:
> $ ifconfig

