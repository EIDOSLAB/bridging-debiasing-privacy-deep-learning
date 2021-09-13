#!/bin/sh

### REBIAS
mkdir -p checkpoints/rebias

python main_biased_mnist.py --root ~/mnist --train_correlation 0.999
mv checkpoints/best.pth checkpoints/rebias/best0.999.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.997
mv checkpoints/best.pth checkpoints/rebias/best0.997.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.995
mv checkpoints/best.pth checkpoints/rebias/best0.995.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.99
mv checkpoints/best.pth checkpoints/rebias/best0.990.pth


### Learned Mixin
mkdir -p checkpoints/learned-mixin

python main_biased_mnist.py --root ~/mnist --train_correlation 0.999 --outer_criterion LearnedMixin --g_lambda_inner 0 --n_g_pretrain_epochs 5 --n_g_update 0
mv checkpoints/best.pth checkpoints/learned-mixin/best0.999.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.997 --outer_criterion LearnedMixin --g_lambda_inner 0 --n_g_pretrain_epochs 5 --n_g_update 0
mv checkpoints/best.pth checkpoints/learned-mixin/best0.997.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.995 --outer_criterion LearnedMixin --g_lambda_inner 0 --n_g_pretrain_epochs 5 --n_g_update 0
mv checkpoints/best.pth checkpoints/learned-mixin/best0.995.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.99 --outer_criterion LearnedMixin --g_lambda_inner 0 --n_g_pretrain_epochs 5 --n_g_update 0
mv checkpoints/best.pth checkpoints/learned-mixin/best0.990.pth

### RUBi
mkdir -p checkpoints/rubi

python main_biased_mnist.py --root ~/mnist --train_correlation 0.999 --outer_criterion RUBi --g_lambda_inner 0
mv checkpoints/best.pth checkpoints/rubi/best0.999.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.997 --outer_criterion RUBi --g_lambda_inner 0
mv checkpoints/best.pth checkpoints/rubi/best0.997.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.995 --outer_criterion RUBi --g_lambda_inner 0
mv checkpoints/best.pth checkpoints/rubi/best0.995.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.99 --outer_criterion RUBi --g_lambda_inner 0
mv checkpoints/best.pth checkpoints/rubi/best0.990.pth

### VANILLA
mkdir -p checkpoints/vanilla

python main_biased_mnist.py --root ~/mnist --train_correlation 0.999 --f_lambda_outer 0 --g_lambda_inner 0
mv checkpoints/best.pth checkpoints/vanilla/best0.999.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.997 --f_lambda_outer 0 --g_lambda_inner 0
mv checkpoints/best.pth checkpoints/vanilla/best0.997.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.995 --f_lambda_outer 0 --g_lambda_inner 0
mv checkpoints/best.pth checkpoints/vanilla/best0.995.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.99 --f_lambda_outer 0 --g_lambda_inner 0
mv checkpoints/best.pth checkpoints/vanilla/best0.990.pth
