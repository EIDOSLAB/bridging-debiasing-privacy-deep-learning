#!/bin/sh

# EnD
python3 train_classifier.py --rho 0.999 --crit end --root ~/mnist
python3 train_classifier.py --rho 0.997 --crit end --root ~/mnist
python3 train_classifier.py --rho 0.995 --crit end --root ~/mnist
python3 train_classifier.py --rho 0.990 --crit end --root ~/mnist