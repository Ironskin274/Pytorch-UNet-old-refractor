# BraTS2020 重构变更日志

## 概述

本次重构为项目添加了对BraTS2020脑肿瘤分割数据集的完整支持，同时保持了对原始Carvana数据集的向后兼容。

## 主要变更

### 1. 新增文件

#### 数据集和工具
- `utils/data_loading.py` - 新增 `BraTS2020Dataset` 类
  - 支持多模态MRI数据加载（t1, t1ce, t2, flair）
  - 3D NIfTI格式到2D切片的自动转换
  - 智能切片选择（仅保留包含肿瘤的切片）
  - 自动归一化（使用百分位数方法）
  - BraTS标签映射（0, 1, 2, 4 -> 0, 1, 2, 3）

#### 文档
- `BraTS2020_README.md` - 完整的BraTS2020使用文档
- `CHANGELOG_BraTS2020.md` - 本变更日志
- `README.md` - 更新主README，添加BraTS2020部分

#### 脚本
- `train_brats.sh` - BraTS2020训练示例脚本
- `predict_brats.sh` - BraTS2020预测示例脚本
- `verify_brats_dataset.py` - 数据集配置验证工具

### 2. 修改的文件

#### `requirements.txt`
新增依赖：
- `nibabel>=3.2.0` - 用于读取NIfTI格式文件
- `SimpleITK>=2.1.0` - 用于医学图像处理
- `scipy>=1.9.0` - 用于科学计算

#### `train.py`
- 添加 `--use-brats` 参数以启用BraTS2020模式
- 修改 `train_model()` 函数以支持固定的训练/验证集划分
- 自动配置BraTS2020的输入通道数（4）和输出类别数（4）
- 更新checkpoint保存逻辑以正确处理不同数据集的mask_values

主要修改：
```python
# 新增BraTS数据集路径配置
dir_brats_train = '/data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/'
train_list_file = '/data/ssd2/liying/Datasets/BraTS2020/train_list.txt'
valid_list_file = '/data/ssd2/liying/Datasets/BraTS2020/valid_list.txt'

# train_model函数新增use_brats参数
def train_model(..., use_brats: bool = True):
    if use_brats:
        train_set = BraTS2020Dataset(...)
        val_set = BraTS2020Dataset(...)
    else:
        # 原始逻辑...
```

#### `predict.py`
- 添加 `--brats-mode` 参数以启用BraTS预测模式
- 新增 `predict_brats_case()` 函数用于预测完整的3D病例
- 自动将预测结果映射回BraTS标准标签空间
- 支持输出NIfTI格式的预测结果

主要新增功能：
- 逐切片预测3D体数据
- 保留原始NIfTI的仿射矩阵和header信息
- 自动标签映射（3 -> 4）

#### `evaluate.py`
- 无需修改（已支持多类别分割）
- 自动计算4类分割的Dice分数

## 数据集配置

### 数据路径
- 训练数据：`/data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/`
- 训练列表：`/data/ssd2/liying/Datasets/BraTS2020/train_list.txt`
- 验证列表：`/data/ssd2/liying/Datasets/BraTS2020/valid_list.txt`

### 数据格式
每个病例包含5个NIfTI文件：
- `{case_name}_t1.nii.gz` - T1加权图像
- `{case_name}_t1ce.nii.gz` - T1增强图像
- `{case_name}_t2.nii.gz` - T2加权图像
- `{case_name}_flair.nii.gz` - FLAIR图像
- `{case_name}_seg.nii.gz` - 分割标注

### 标签定义
- 0: 背景
- 1: 坏死/非增强肿瘤核心 (NCR/NET)
- 2: 水肿 (ED)
- 3/4: 增强肿瘤 (ET) - 训练时使用3，预测时映射回4

## 使用方法

### 数据集验证
```bash
python verify_brats_dataset.py
```

### 训练
```bash
# 使用BraTS2020数据集
python train.py --use-brats --epochs 50 --batch-size 8 --learning-rate 1e-4 --amp

# 使用原始数据集（向后兼容）
python train.py --epochs 5 --batch-size 1 --learning-rate 1e-5 --amp
```

### 预测
```bash
# BraTS模式
python predict.py --brats-mode --model checkpoints/checkpoint_epoch50.pth \
    --input /path/to/case_dir --output ./predictions

# 标准模式（向后兼容）
python predict.py -i image.jpg -o output.jpg --model MODEL.pth
```

## 技术细节

### BraTS2020Dataset类

**关键特性：**
1. **多模态处理**：自动加载并合并4个MRI模态
2. **智能切片选择**：
   - 默认切片范围：(2, 150)
   - 仅选择包含肿瘤标签的切片
   - 大幅减少训练数据中的背景切片比例

3. **归一化策略**：
   - 对每个模态独立归一化
   - 使用1%-99%百分位数进行裁剪
   - 映射到[0, 1]范围

4. **标签处理**：
   - 自动将标签4映射为3（训练时）
   - 预测时自动映射回4（BraTS标准）

### 内存优化

- 支持图像缩放（`--scale`参数）
- 支持混合精度训练（`--amp`参数）
- 推荐batch_size：4-16（取决于GPU内存）

### 性能建议

1. **训练配置**：
   - 学习率：1e-4
   - 批次大小：8
   - Epochs：50+
   - 使用AMP和双线性上采样

2. **数据加载**：
   - 利用多进程加载（自动配置）
   - Pin memory加速GPU传输

## 向后兼容性

- ✅ 完全保留原始Carvana数据集支持
- ✅ 原有命令行参数保持不变
- ✅ 新功能通过可选参数启用
- ✅ 所有原有功能正常工作

## 测试建议

1. 运行 `verify_brats_dataset.py` 验证数据集配置
2. 尝试加载单个样本确认数据加载正确
3. 使用小的epoch数（如5）进行快速测试训练
4. 检查checkpoint文件是否正确保存
5. 测试预测功能确认输出格式正确

## 已知限制

1. 验证集（MICCAI_BraTS2020_ValidationData）没有标签，因此不直接使用
2. 训练数据按train_list.txt和valid_list.txt固定划分
3. 只支持2D切片训练（不支持3D卷积）
4. 预测时逐切片处理，没有利用3D上下文信息

## 未来改进方向

1. 添加数据增强（旋转、翻转、弹性变形等）
2. 支持3D U-Net架构
3. 添加更多评估指标（Hausdorff距离、IoU等）
4. 实现在线hard example mining
5. 支持多尺度训练和测试
6. 添加模型集成功能

## 参考资料

- BraTS2020挑战赛：https://www.med.upenn.edu/cbica/brats2020/
- Kaggle参考实现：https://www.kaggle.com/code/yug201/u-net-on-brats-2020-tumor-segmentation-in-2d-mri
- U-Net原始论文：https://arxiv.org/abs/1505.04597

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

---

**最后更新**: 2025-11-10
**版本**: 1.0.0

