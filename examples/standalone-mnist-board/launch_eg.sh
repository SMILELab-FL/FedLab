#!/bin/bash

python standalone.py --total_client 100 --com_round 10 --sample_ratio 0.1 --batch_size 128 --epochs 3 --lr 0.1 --port 8040
