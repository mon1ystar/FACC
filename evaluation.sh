#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"


python train.py -project savc \
    -dataset classroom \
    -train_mode resnet_a1 \
    -base_mode 'ft_cos' \
    -new_mode 'avg_cos' \
    -lr_base 0.1 \
    -lr_new 0.001 \
    -decay 0.0005 \
    -epochs_base 0 \
    -schedule Cosine \
    -gpu 0 \
    -temperature 16 \
    -moco_dim 128 \
    -moco_k 8192 \
    -mlp \
    -moco_t 0.07 \
    -moco_m 0.995 \
    -size_crops 56 28 \
    -min_scale_crops 0.9 0.2 \
    -max_scale_crops 1.0 0.7 \
    -num_crops 2 4 \
    -alpha 0.2 \
    -beta 0.8 \
    -constrained_cropping \
    -fantasy rotation2 \
    -model_dir checkpoint/classroom/savc/ft_cos-avg_cos-data_init-start_0/Cosine-Epo_200-Lr_0.1000-T_16.00-fantasy_rotation2-alpha_0.20-beta_0.80/session0_max_acc.pth \
    
