# BraTS2020 快速入门指南

## 5分钟开始训练

### 第一步：验证数据集

```bash
python verify_brats_dataset.py
```

这个脚本会检查：
- ✅ 数据目录是否存在
- ✅ train_list.txt 和 valid_list.txt 是否存在
- ✅ 病例文件是否完整
- ✅ 数据集是否能正常加载

### 第二步：开始训练

```bash
python train.py --use-brats --epochs 50 --batch-size 8 --learning-rate 1e-4 --amp
```

**参数说明：**
- `--use-brats`: 使用BraTS2020数据集（必需）
- `--epochs 50`: 训练50个epoch
- `--batch-size 8`: 批次大小为8（根据GPU调整）
- `--learning-rate 1e-4`: 学习率
- `--amp`: 使用混合精度训练（节省内存）

**如果遇到内存不足：**
```bash
python train.py --use-brats --epochs 50 --batch-size 4 --learning-rate 1e-4 --scale 0.5 --amp
```

### 第三步：监控训练

训练开始后，会自动创建Weights & Biases链接，可以在线监控：
- 训练损失曲线
- 验证Dice分数
- 预测结果可视化
- 权重和梯度直方图

### 第四步：进行预测

训练完成后，使用训练好的模型进行预测：

```bash
python predict.py --brats-mode \
    --model checkpoints/checkpoint_epoch50.pth \
    --input /data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001 \
    --output ./predictions
```

预测结果会保存为 `predictions/BraTS20_Training_001_prediction.nii.gz`

## 常见问题

### Q: 如何查看预测结果？

A: 使用ITK-SNAP或3D Slicer等医学图像查看器：
```bash
# 安装ITK-SNAP (Ubuntu)
sudo apt-get install itksnap

# 打开预测结果
itksnap predictions/BraTS20_Training_001_prediction.nii.gz
```

### Q: 训练需要多长时间？

A: 取决于：
- GPU型号（推荐RTX 3090或更好）
- 批次大小
- 数据集大小

参考时间：
- RTX 3090, batch_size=8: ~2-3小时/epoch
- 推荐训练50+ epochs

### Q: 如何调整超参数？

A: 关键超参数：
```bash
--learning-rate 1e-4    # 学习率（1e-5到1e-3）
--batch-size 8          # 批次大小（4-16）
--scale 1.0            # 图像缩放（0.5-1.0）
--epochs 50            # 训练轮数（50-200）
```

### Q: 如何从checkpoint继续训练？

A:
```bash
python train.py --use-brats --epochs 100 \
    --load checkpoints/checkpoint_epoch50.pth \
    --batch-size 8 --learning-rate 1e-4 --amp
```

### Q: 如何批量预测多个病例？

A:
```bash
# 方法1：在命令行列出所有病例
python predict.py --brats-mode \
    --model checkpoints/checkpoint_epoch50.pth \
    --input case1_dir case2_dir case3_dir \
    --output ./predictions

# 方法2：使用通配符（需要shell支持）
python predict.py --brats-mode \
    --model checkpoints/checkpoint_epoch50.pth \
    --input /path/to/cases/BraTS20_Training_* \
    --output ./predictions
```

## 数据路径配置

如果你的数据集在不同位置，需要修改 `train.py` 中的路径：

```python
# 在 train.py 第23-25行
dir_brats_train = '/你的/数据集/路径/MICCAI_BraTS2020_TrainingData/'
train_list_file = '/你的/数据集/路径/train_list.txt'
valid_list_file = '/你的/数据集/路径/valid_list.txt'
```

## 性能优化建议

### 1. 使用混合精度训练
```bash
--amp  # 减少50%内存使用，加速1.5-2倍
```

### 2. 调整num_workers
修改 `train.py` 第77行：
```python
loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
```

### 3. 使用更大的批次大小
如果GPU内存充足：
```bash
--batch-size 16  # 或更大
```

### 4. 使用双线性上采样
```bash
--bilinear  # 减少参数量，节省内存
```

## 下一步

1. 📊 查看 [BraTS2020_README.md](BraTS2020_README.md) 了解详细文档
2. 📝 查看 [CHANGELOG_BraTS2020.md](CHANGELOG_BraTS2020.md) 了解技术细节
3. 🔍 运行 `verify_brats_dataset.py` 确保数据集配置正确
4. 🚀 开始训练！

## 获取帮助

```bash
# 查看训练选项
python train.py -h

# 查看预测选项
python predict.py -h
```

祝训练愉快！🎉

