#!/usr/bin/env python
"""
验证BraTS2020数据集配置是否正确
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 数据集路径配置
data_dir = '/data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/'
train_list = '/data/ssd2/liying/Datasets/BraTS2020/train_list.txt'
valid_list = '/data/ssd2/liying/Datasets/BraTS2020/valid_list.txt'

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        logging.info(f"✓ {description} 存在: {filepath}")
        return True
    else:
        logging.error(f"✗ {description} 不存在: {filepath}")
        return False

def check_case(case_name, data_dir):
    """检查单个病例的文件完整性"""
    case_dir = Path(data_dir) / case_name
    
    if not case_dir.exists():
        logging.warning(f"  ✗ 病例目录不存在: {case_dir}")
        return False
    
    # 检查所需的文件
    required_files = ['t1', 't1ce', 't2', 'flair', 'seg']
    missing_files = []
    
    for modality in required_files:
        file_path = case_dir / f'{case_name}_{modality}.nii.gz'
        if not file_path.exists():
            missing_files.append(modality)
    
    if missing_files:
        logging.warning(f"  ✗ {case_name} 缺少文件: {missing_files}")
        return False
    else:
        logging.info(f"  ✓ {case_name} 文件完整")
        return True

def main():
    logging.info("="*60)
    logging.info("BraTS2020 数据集验证")
    logging.info("="*60)
    
    # 检查数据目录
    logging.info("\n1. 检查数据目录...")
    if not check_file_exists(data_dir, "数据目录"):
        return
    
    # 检查列表文件
    logging.info("\n2. 检查列表文件...")
    train_exists = check_file_exists(train_list, "训练列表")
    valid_exists = check_file_exists(valid_list, "验证列表")
    
    if not train_exists or not valid_exists:
        return
    
    # 读取列表文件
    logging.info("\n3. 读取病例列表...")
    with open(train_list, 'r') as f:
        train_cases = [line.strip() for line in f if line.strip()]
    
    with open(valid_list, 'r') as f:
        valid_cases = [line.strip() for line in f if line.strip()]
    
    logging.info(f"训练集病例数: {len(train_cases)}")
    logging.info(f"验证集病例数: {len(valid_cases)}")
    
    # 检查是否有重复
    overlap = set(train_cases) & set(valid_cases)
    if overlap:
        logging.warning(f"警告: 训练集和验证集有重叠病例: {overlap}")
    else:
        logging.info("✓ 训练集和验证集无重叠")
    
    # 检查训练集病例（最多检查前5个）
    logging.info("\n4. 验证训练集病例（前5个）...")
    for i, case_name in enumerate(train_cases[:5]):
        check_case(case_name, data_dir)
    
    if len(train_cases) > 5:
        logging.info(f"  ... 省略其余 {len(train_cases) - 5} 个训练病例")
    
    # 检查验证集病例（最多检查前5个）
    logging.info("\n5. 验证验证集病例（前5个）...")
    for i, case_name in enumerate(valid_cases[:5]):
        check_case(case_name, data_dir)
    
    if len(valid_cases) > 5:
        logging.info(f"  ... 省略其余 {len(valid_cases) - 5} 个验证病例")
    
    # 尝试加载数据集
    logging.info("\n6. 尝试加载数据集...")
    try:
        from utils.data_loading import BraTS2020Dataset
        
        logging.info("正在初始化训练数据集...")
        train_dataset = BraTS2020Dataset(
            data_dir=data_dir,
            list_file=train_list,
            scale=1.0
        )
        logging.info(f"✓ 训练数据集加载成功！总切片数: {len(train_dataset)}")
        
        # 尝试加载一个样本
        logging.info("尝试加载第一个样本...")
        sample = train_dataset[0]
        logging.info(f"  图像形状: {sample['image'].shape}")
        logging.info(f"  掩码形状: {sample['mask'].shape}")
        logging.info(f"  图像数据类型: {sample['image'].dtype}")
        logging.info(f"  掩码数据类型: {sample['mask'].dtype}")
        logging.info(f"  掩码唯一值: {sample['mask'].unique().tolist()}")
        
        logging.info("\n正在初始化验证数据集...")
        valid_dataset = BraTS2020Dataset(
            data_dir=data_dir,
            list_file=valid_list,
            scale=1.0
        )
        logging.info(f"✓ 验证数据集加载成功！总切片数: {len(valid_dataset)}")
        
    except Exception as e:
        logging.error(f"✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logging.info("\n" + "="*60)
    logging.info("✓ 所有检查通过！数据集配置正确。")
    logging.info("="*60)
    logging.info("\n现在可以开始训练：")
    logging.info("  python train.py --use-brats --epochs 50 --batch-size 8 --learning-rate 1e-4 --amp")

if __name__ == '__main__':
    main()

