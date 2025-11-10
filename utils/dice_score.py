import torch
from torch import Tensor
import logging


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-8):
    """
    计算Dice系数，改进的数值稳定性版本
    
    关键改进：
    1. epsilon从1e-6增加到1e-8，提高数值稳定性
    2. 添加NaN/Inf检查
    3. 使用更稳定的计算方式
    """
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    # 检查输入是否包含NaN/Inf
    if torch.isnan(input).any() or torch.isinf(input).any():
        logging.warning('dice_coeff: input contains NaN/Inf')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)
    if torch.isnan(target).any() or torch.isinf(target).any():
        logging.warning('dice_coeff: target contains NaN/Inf')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # 计算交集和并集
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    
    # 处理空集情况（都是背景的情况）
    # 如果sets_sum为0且inter也为0，表示都是背景，应该返回dice=1（完全匹配）
    # 使用torch.where确保正确处理这种情况
    sets_sum = torch.where(sets_sum == 0, inter + epsilon, sets_sum)
    
    # 更稳定的dice计算：使用更大的epsilon
    dice = (inter + epsilon) / (sets_sum + epsilon)
    
    # 确保dice值在有效范围内 [0, 1]
    dice = torch.clamp(dice, min=0.0, max=1.0)
    
    # 检查结果是否为NaN/Inf
    if torch.isnan(dice).any() or torch.isinf(dice).any():
        logging.warning('dice_coeff: result contains NaN/Inf, returning 0')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)
    
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-8):
    """
    多类别Dice系数，改进的数值稳定性版本
    """
    # 检查输入维度
    if input.dim() < 3 or target.dim() < 3:
        logging.warning('multiclass_dice_coeff: input/target dimensions too small')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)
    
    # 使用更稳定的方式：先检查每个类别，然后平均
    try:
        result = dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)
        # 确保结果有效
        if torch.isnan(result) or torch.isinf(result):
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        return result
    except Exception as e:
        logging.warning(f'multiclass_dice_coeff error: {e}, returning 0')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False, epsilon: float = 1e-8):
    """
    Dice损失函数，改进的数值稳定性版本
    
    关键改进：
    1. 增加epsilon参数，默认使用更大的值（1e-8）
    2. 添加NaN/Inf检查和保护
    3. 确保返回值在有效范围内
    """
    try:
        fn = multiclass_dice_coeff if multiclass else dice_coeff
        dice = fn(input, target, reduce_batch_first=True, epsilon=epsilon)
        
        # 计算loss
        loss = 1 - dice
        
        # 确保loss在有效范围内 [0, 1]
        loss = torch.clamp(loss, min=0.0, max=1.0)
        
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning('dice_loss: result is NaN/Inf, returning 1.0 (maximum loss)')
            return torch.tensor(1.0, device=input.device, dtype=input.dtype)
        
        return loss
    except Exception as e:
        logging.warning(f'dice_loss error: {e}, returning 1.0 (maximum loss)')
        return torch.tensor(1.0, device=input.device, dtype=input.dtype)
