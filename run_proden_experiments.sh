#!/bin/bash

echo "Starting PRODEN + FixMatch experiments..."
echo "Running SVHN (1000, p=0.3)"
python3 ProdenFixmatch.py --dataset svhn --num_labeled 1000 --partial_rate 0.3 --use_amp

echo "Running CIFAR-10 (1000, p=0.7)..."
python3 ProdenFixmatch.py --dataset cifar10 --num_labeled 1000 --partial_rate 0.7 --use_amp

echo "Running Fashion-MNIST (1000, p=0.3)..."
python3 ProdenFixmatch.py --dataset fashion_mnist --num_labeled 1000 --partial_rate 0.3 --use_amp


echo "All experiments completed!"
