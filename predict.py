import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import nibabel as nib
from pathlib import Path

from utils.data_loading import BasicDataset, BraTS2020Dataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def predict_brats_case(net, case_dir, device, scale_factor=1, output_dir=None):
    """
    预测BraTS病例的所有切片
    
    Args:
        net: 训练好的模型
        case_dir: 病例文件夹路径
        device: 设备
        scale_factor: 缩放因子
        output_dir: 输出目录
    """
    net.eval()
    case_dir = Path(case_dir)
    case_name = case_dir.name
    
    logging.info(f'Processing case: {case_name}')
    
    # 加载所有模态
    modalities = ['t1', 't1ce', 't2', 'flair']
    nii_data = {}
    
    for modality in modalities:
        file_path = case_dir / f'{case_name}_{modality}.nii.gz'
        if not file_path.exists():
            raise FileNotFoundError(f'Modality file not found: {file_path}')
        nii_data[modality] = nib.load(str(file_path))
    
    # 获取数据形状
    data_shape = nii_data['t1'].shape
    logging.info(f'Data shape: {data_shape}')
    
    # 创建输出数组
    predictions = np.zeros(data_shape, dtype=np.uint8)
    
    # 逐切片预测
    for slice_idx in range(data_shape[2]):
        # 加载该切片的所有模态
        slice_data = []
        for modality in modalities:
            slice_img = nii_data[modality].get_fdata()[:, :, slice_idx]
            slice_data.append(slice_img)
        
        # 合并模态 (H, W, C)
        image = np.stack(slice_data, axis=-1)
        
        # 归一化
        image = BraTS2020Dataset.normalize_image(image)
        
        # 预处理
        if scale_factor != 1.0:
            h, w = image.shape[:2]
            new_h, new_w = int(scale_factor * h), int(scale_factor * w)
            channels = []
            for i in range(image.shape[-1]):
                pil_img = Image.fromarray((image[:, :, i] * 255).astype(np.uint8))
                pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
                channels.append(np.array(pil_img) / 255.0)
            image = np.stack(channels, axis=-1)
        
        # 转换维度 (H, W, C) -> (C, H, W)
        image = image.transpose((2, 0, 1))
        
        # 转换为tensor
        img_tensor = torch.from_numpy(image).float().unsqueeze(0)
        img_tensor = img_tensor.to(device=device)
        
        # 预测
        with torch.no_grad():
            output = net(img_tensor)
            output = F.interpolate(output, (data_shape[0], data_shape[1]), mode='bilinear')
            mask = output.argmax(dim=1).cpu().numpy()[0]
        
        # 将预测结果映射回原始标签空间 (0, 1, 2, 3) -> (0, 1, 2, 4)
        mask_original = mask.copy()
        mask_original[mask == 3] = 4
        
        predictions[:, :, slice_idx] = mask_original
    
    # 保存预测结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用原始NIfTI的affine和header
        output_nii = nib.Nifti1Image(predictions, nii_data['t1'].affine, nii_data['t1'].header)
        output_path = output_dir / f'{case_name}_prediction.nii.gz'
        nib.save(output_nii, str(output_path))
        logging.info(f'Prediction saved to: {output_path}')
    
    return predictions


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--brats-mode', action='store_true', default=False,
                        help='Use BraTS prediction mode (input should be case directories)')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    # 根据模式确定通道数和类别数
    if args.brats_mode:
        n_channels = 4
        n_classes = 4
        logging.info('Using BraTS prediction mode')
    else:
        n_channels = 3
        n_classes = args.classes

    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=args.bilinear)
    net.to(device=device)
    
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    if args.brats_mode:
        # BraTS模式：处理病例目录
        in_dirs = args.input
        out_dir = args.output[0] if args.output else './predictions'
        
        for case_dir in in_dirs:
            logging.info(f'Processing BraTS case: {case_dir}')
            try:
                predict_brats_case(
                    net=net,
                    case_dir=case_dir,
                    device=device,
                    scale_factor=args.scale,
                    output_dir=out_dir if not args.no_save else None
                )
            except Exception as e:
                logging.error(f'Error processing {case_dir}: {e}')
    else:
        # 标准模式：处理单张图像
        in_files = args.input
        out_files = get_output_filenames(args)

        for i, filename in enumerate(in_files):
            logging.info(f'Predicting image {filename} ...')
            img = Image.open(filename)

            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask)
