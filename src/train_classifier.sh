#!/bin/sh

# VANILLA
python3 train_classifier.py --rho 0.999 --crit vanilla --root ~/mnist
python3 train_classifier.py --rho 0.997 --crit vanilla --root ~/mnist
python3 train_classifier.py --rho 0.995 --crit vanilla --root ~/mnist
python3 train_classifier.py --rho 0.990 --crit vanilla --root ~/mnist

# ReBias
python3 train_classifier.py --rho 0.999 --crit rebias --root ~/mnist
python3 train_classifier.py --rho 0.997 --crit rebias --root ~/mnist
python3 train_classifier.py --rho 0.995 --crit rebias --root ~/mnist
python3 train_classifier.py --rho 0.990 --crit rebias --root ~/mnist

# rubi
python3 train_classifier.py --rho 0.999 --crit rubi --root ~/mnist
python3 train_classifier.py --rho 0.997 --crit rubi --root ~/mnist
python3 train_classifier.py --rho 0.995 --crit rubi --root ~/mnist
python3 train_classifier.py --rho 0.990 --crit rubi --root ~/mnist

# learned-mixin
python3 train_classifier.py --rho 0.999 --crit learned-mixin --root ~/mnist
python3 train_classifier.py --rho 0.997 --crit learned-mixin --root ~/mnist
python3 train_classifier.py --rho 0.995 --crit learned-mixin --root ~/mnist
python3 train_classifier.py --rho 0.990 --crit learned-mixin --root ~/mnist

# EnD
python3 train_classifier.py --rho 0.999 --crit end --root ~/mnist
python3 train_classifier.py --rho 0.997 --crit end --root ~/mnist
python3 train_classifier.py --rho 0.995 --crit end --root ~/mnist
python3 train_classifier.py --rho 0.990 --crit end --root ~/mnist