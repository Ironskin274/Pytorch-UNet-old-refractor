#!/bin/bash
# BraTS2020训练脚本示例

# 基本训练配置
python train.py \
    --use-brats \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --scale 1.0 \
    --amp \
    --bilinear

# 如果遇到内存不足，可以尝试以下配置：
# python train.py \
#     --use-brats \
#     --epochs 50 \
#     --batch-size 4 \
#     --learning-rate 1e-4 \
#     --scale 0.5 \
#     --amp \
#     --bilinear

# 从检查点继续训练：
# python train.py \
#     --use-brats \
#     --epochs 100 \
#     --batch-size 8 \
#     --learning-rate 1e-4 \
#     --scale 1.0 \
#     --amp \
#     --load checkpoints/checkpoint_epoch50.pth

