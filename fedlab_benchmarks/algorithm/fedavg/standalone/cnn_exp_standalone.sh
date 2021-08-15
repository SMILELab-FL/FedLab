#!/bin/bash



python standalone.py --sample_ratio 0.1 --batch_size 600 --epochs 1 --partition iid --name exp1_iid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 600 --epochs 1 --partition noniid --name exp1_noniid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 600 --epochs 5 --partition iid --name exp2_iid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 600 --epochs 5 --partition noniid --name exp2_noniid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 50 --epochs 1 --partition iid --name exp3_iid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 50 --epochs 1 --partition noniid --name exp3_noniid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 600 --epochs 20 --partition iid --name exp4_iid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 600 --epochs 20 --partition noniid --name exp4_noniid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 1 --partition iid --name exp5_iid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 1 --partition noniid --name exp5_noniid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 50 --epochs 5 --partition iid --name exp6_iid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 50 --epochs 5 --partition noniid --name exp6_noniid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 50 --epochs 20 --partition iid --name exp7_iid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 50 --epochs 20 --partition noniid --name exp7_noniid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 5 --partition iid --name exp8_iid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 5 --partition noniid --name exp8_noniid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 20 --partition iid --name exp9_iid --model cnn --lr 0.02 &
sleep 2s
python standalone.py --sample_ratio 0.1 --batch_size 10 --epochs 20 --partition noniid --name exp9_noniid --model cnn --lr 0.02 &



