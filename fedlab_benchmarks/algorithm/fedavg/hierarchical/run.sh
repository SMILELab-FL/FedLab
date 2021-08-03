#!/bin/bash

# 1 client 1 middle 1 server

python client_hi.py &
python server_hi.py &
python scheduler.py &