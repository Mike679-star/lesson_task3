import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y


# 生成训练集
def train(args):
    h5_file = h5py.File(args.output_path, 'w')
    #  #用于存储低分辨率和高分辨率的patch
    lr_patches = []
    hr_patches = []

    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        # 将照片转换为RGB通道
        hr = pil_image.open(image_path).convert('RGB')
        # 取放大倍数的倍数, width, height为可被scale整除的训练数据尺寸
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        # 图像大小调整,得到高分辨率图像Hr
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        # 低分辨率图像缩小
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        # 低分辨率图像放大，得到低分辨率图像Lr
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        # 转换为浮点并取ycrcb中的y通道
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)
        # 将数据分割
        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    # 创建数据集,把得到的数据转化为数组类型
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()


# 下同,生成测试集
def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--eval', action='store_true')  # store_flase就是存储一个bool值true，也就是说在该参数在被激活时它会输出store存储的值true。
    args = parser.parse_args()

    # 决定使用哪个函数来生成h5文件，因为有俩个不同的函数train和eval生成对应的h5文件。
    if not args.eval:
        train(args)
    else:
        eval(args)
