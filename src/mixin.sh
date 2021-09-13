#!/bin/sh

mkdir -p checkpoints/learned-mixin

python main_biased_mnist.py --root ~/mnist --train_correlation 0.999 --outer_criterion LearnedMixin --g_lambda_inner 0 --n_g_pretrain_epochs 5 --n_g_update 0
mv checkpoints/best.pth checkpoints/learned-mixin/best0.999.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.997 --outer_criterion LearnedMixin --g_lambda_inner 0 --n_g_pretrain_epochs 5 --n_g_update 0
mv checkpoints/best.pth checkpoints/learned-mixin/best0.997.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.995 --outer_criterion LearnedMixin --g_lambda_inner 0 --n_g_pretrain_epochs 5 --n_g_update 0
mv checkpoints/best.pth checkpoints/learned-mixin/best0.995.pth

python main_biased_mnist.py --root ~/mnist --train_correlation 0.99 --outer_criterion LearnedMixin --g_lambda_inner 0 --n_g_pretrain_epochs 5 --n_g_update 0
mv checkpoints/best.pth checkpoints/learned-mixin/best0.990.pth
