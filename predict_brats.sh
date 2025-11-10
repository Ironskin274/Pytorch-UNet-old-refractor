#!/bin/bash
# BraTS2020预测脚本示例

# 预测单个病例
python predict.py \
    --brats-mode \
    --model checkpoints/checkpoint_epoch50.pth \
    --input /data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001 \
    --output ./predictions \
    --scale 1.0

# 批量预测多个病例
# python predict.py \
#     --brats-mode \
#     --model checkpoints/checkpoint_epoch50.pth \
#     --input /data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_* \
#     --output ./predictions \
#     --scale 1.0

