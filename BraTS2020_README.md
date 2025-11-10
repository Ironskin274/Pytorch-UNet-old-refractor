# BraTS2020 数据集使用说明

本项目已重构以支持BraTS2020脑肿瘤分割数据集。

## 数据集结构

BraTS2020数据集包含多模态MRI图像（t1, t1ce, t2, flair）和分割标注。数据集路径：
- 训练数据：`/data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/`
- 训练集划分文件：`/data/ssd2/liying/Datasets/BraTS2020/train_list.txt`
- 验证集划分文件：`/data/ssd2/liying/Datasets/BraTS2020/valid_list.txt`

## 标签说明

BraTS2020的标注包含以下类别：
- **0**: 背景
- **1**: 坏死/非增强肿瘤核心 (NCR/NET)
- **2**: 水肿 (ED)
- **3**: 增强肿瘤 (ET) (原始数据中为4，代码会自动转换)

## 安装依赖

首先安装所需的依赖包：

```bash
pip install -r requirements.txt
```

主要新增依赖：
- `nibabel`: 用于读取NIfTI格式的医学图像
- `SimpleITK`: 用于医学图像处理
- `scipy`: 用于科学计算

## 训练模型

使用BraTS2020数据集训练模型：

```bash
python train.py --use-brats --epochs 50 --batch-size 8 --learning-rate 1e-4 --scale 1.0 --amp
```

### 训练参数说明

- `--use-brats`: 使用BraTS2020数据集（必需）
- `--epochs`: 训练轮数（推荐50+）
- `--batch-size`: 批次大小（根据GPU内存调整，推荐4-16）
- `--learning-rate`: 学习率（推荐1e-4）
- `--scale`: 图像缩放因子（1.0表示使用原始尺寸240x240）
- `--amp`: 使用混合精度训练以节省内存
- `--bilinear`: 使用双线性上采样代替转置卷积

### 模型配置

使用`--use-brats`参数时，模型会自动配置为：
- **输入通道数**: 4（t1, t1ce, t2, flair四个模态）
- **输出类别数**: 4（背景 + 3种肿瘤类型）

## 预测

### 预测BraTS病例

预测单个或多个BraTS病例：

```bash
python predict.py --brats-mode \
    --model checkpoints/checkpoint_epoch50.pth \
    --input /data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001 \
    --output ./predictions \
    --scale 1.0
```

### 批量预测

可以一次预测多个病例：

```bash
python predict.py --brats-mode \
    --model checkpoints/checkpoint_epoch50.pth \
    --input case1_dir case2_dir case3_dir \
    --output ./predictions \
    --scale 1.0
```

预测结果将保存为NIfTI格式（.nii.gz），可以使用ITK-SNAP等工具可视化。

## 数据集特点

### 数据预处理

`BraTS2020Dataset`类自动处理以下预处理步骤：

1. **多模态融合**: 将4个模态（t1, t1ce, t2, flair）合并为4通道输入
2. **2D切片提取**: 从3D体数据中提取2D切片进行训练
3. **智能切片选择**: 只选择包含肿瘤的切片，避免大量背景切片
4. **归一化**: 对每个模态使用百分位数归一化（1%-99%）
5. **标签映射**: 将BraTS的标签4映射为3，以便于网络学习

### 切片范围

默认使用切片范围`(2, 150)`，可以在实例化数据集时修改：

```python
train_set = BraTS2020Dataset(
    data_dir=dir_brats_train,
    list_file=train_list_file,
    scale=img_scale,
    slice_range=(10, 140)  # 自定义切片范围
)
```

## 评估

评估脚本会自动计算多类别Dice分数，支持BraTS的4类分割。评估会在训练过程中自动执行。

## 模型架构

U-Net架构配置：
- **输入**: 4通道（4个MRI模态）
- **输出**: 4类分割图
- **图像尺寸**: 240×240（BraTS标准尺寸）
- **上采样方式**: 转置卷积（默认）或双线性插值

## 训练建议

1. **学习率**: 推荐从1e-4开始，使用ReduceLROnPlateau调度器
2. **批次大小**: 根据GPU内存调整，8-16为佳
3. **数据增强**: 可以考虑添加旋转、翻转等增强
4. **训练时长**: 推荐至少50个epoch
5. **混合精度**: 使用`--amp`可以显著减少内存使用并加速训练

## 参考

本实现参考了Kaggle上的BraTS2020 U-Net实现：
https://www.kaggle.com/code/yug201/u-net-on-brats-2020-tumor-segmentation-in-2d-mri

## 注意事项

1. **数据路径**: 确保train_list.txt和valid_list.txt中的病例名称与实际文件夹名称匹配
2. **内存使用**: BraTS数据集较大，建议使用GPU训练并启用混合精度
3. **切片数量**: 每个病例包含多个切片，总训练样本数会远多于病例数
4. **标签映射**: 预测结果会自动将类别3映射回4，以符合BraTS标准

## 文件结构

```
utils/
  └── data_loading.py         # 包含BraTS2020Dataset类
train.py                      # 训练脚本，支持--use-brats参数
predict.py                    # 预测脚本，支持--brats-mode参数
evaluate.py                   # 评估脚本（自动支持多类别）
requirements.txt              # 依赖包列表
```

