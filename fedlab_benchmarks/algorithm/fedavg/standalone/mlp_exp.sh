#!/bin/bash

python standalone.py --com_round 10 --sample_ratio 0.1 --batch_size 10 --epochs 5 --partition iid --name test1 --model mlp --lr 0.02 &
sleep 2s
python standalone.py --com_round 800 --sample_ratio 0.5 --batch_size 10 --epochs 5 --partition noniid --name test2 --model mlp --lr 0.02 &
sleep 2s
python standalone.py --com_round 800 --sample_ratio 0.5 --batch_size 10 --epochs 5 --partition iid --name test3 --model mlp --lr 0.02 &
sleep 2s
python standalone.py --com_round 800 --sample_ratio 0.5 --batch_size 10 --epochs 5 --partition noniid --name test4 --model mlp --lr 0.02 &
