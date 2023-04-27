import argparse
import os
import copy

import numpy as np
from torch import Tensor
import torch
from torch import nn
import torch.optim as optim

# gpu加速库
import torch.backends.cudnn as cudnn

from torch.utils.data.dataloader import DataLoader

# 进度条
from tqdm import tqdm

import SRGAN
import SRGAN_loss
from datasets_SRGAN import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

##需要修改的参数
# epoch.pth
# losslog
# psnrlog
# best.pth

'''
python train.py --train-file "path_to_train_file" \
                --eval-file "path_to_eval_file" \
                --outputs-dir "path_to_outputs_file" \
                --scale 3 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 400 \
                --num-workers 0 \
                --seed 123  
'''
if __name__ == '__main__':

    # 初始参数设定
    parser = argparse.ArgumentParser()  # argparse是python用于解析命令行参数和选项的标准模块
    parser.add_argument('--train-file', type=str, required=True, )  # 训练 h5文件目录
    parser.add_argument('--eval-file', type=str, required=True)  # 测试 h5文件目录
    parser.add_argument('--outputs-dir', type=str, required=True)  # 模型 .pth保存目录
    parser.add_argument('--scale', type=int, default=3)  # 放大倍数
    parser.add_argument('--lr', type=float, default=1e-4)  # 学习率
    parser.add_argument('--batch-size', type=int, default=16)  # 一次处理的图片大小
    parser.add_argument('--num-workers', type=int, default=0)  # 线程数
    parser.add_argument('--num-epochs', type=int, default=400)  # 训练次数
    parser.add_argument('--seed', type=int, default=123)  # 随机种子
    args = parser.parse_args()

    # 输出放入固定文件夹里
    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))
    # 没有该文件夹就新建一个文件夹
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    # benckmark模式，加速计算，但寻找最优配置，计算的前馈结果会有差异
    cudnn.benchmark = True

    # gpu或者cpu模式，取决于当前cpu是否可用
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 每次程序运行生成的随机数固定
    torch.manual_seed(args.seed)

    # 构建生成网络
    gnet = SRGAN.Generator().to(device)

    # 构建辨别网络
    dnet = SRGAN.Discriminator().to(device)

    # 恢复训练，从之前结束的那个地方开始
    # model.load_state_dict(torch.load('outputs/x3/epoch_173.pth'))

    # 定义感知损失函数 为content loss 和adversarial loss 的加权和
    criterion_g = SRGAN_loss.PerceptualLoss(device)

    # 正则项
    regularization = SRGAN_loss.RegularizationLoss()
    criterion_d = torch.nn.BCELoss()

    # 定义优化器
    optimizer_d = torch.optim.Adam(dnet.parameters(), lr=args.lr)
    optimizer_g = torch.optim.Adam(gnet.parameters(), lr=args.lr * 0.1)

    real_label = torch.ones([args.batch_size, 1, 1, 1]).to(device)
    fake_label = torch.zeros([args.batch_size, 1, 1, 1]).to(device)

    # 预处理训练集
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(
        # 数据
        dataset=train_dataset,
        # 分块
        batch_size=args.batch_size,
        # 数据集数据洗牌,打乱后取batch
        shuffle=True,
        # 工作进程，像是虚拟存储器中的页表机制
        num_workers=args.num_workers,
        # 锁页内存，不换出内存，生成的Tensor数据是属于内存中的锁页内存区
        pin_memory=True,
        # 不取余，丢弃不足batchSize大小的图像
        drop_last=True)
    # 预处理验证集
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # 拷贝权重
    best_weights_g = copy.deepcopy(gnet.state_dict())
    best_weights_d = copy.deepcopy(dnet.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    # 画图用
    gnet_lossLog = []
    dnet_lossLog = []
    psnrLog = []

    # 恢复训练
    # for epoch in range(args.num_epochs):
    for epoch in range(1, args.num_epochs + 1):
        # for epoch in range(174, 400):
        # 模型训练入口
        gnet.train()
        dnet.train()

        train_loss_all_d = 0.
        train_loss_all_g = 0.
        total = 0

        # 进度条，就是不要不足batchsize的部分
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            # t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs - 1))
            t.set_description('epoch:{}/{}'.format(epoch, args.num_epochs))

            # 每个batch计算一次
            for i, (inputs, labels) in enumerate(train_dataloader):

                train_loss_d = 0.
                train_loss_g = 0.
                total += inputs.shape[0]

                inputs = inputs.to(device)
                labels = labels.to(device)

                fake_img = gnet(inputs)
                loss_g = criterion_g(fake_img, labels, dnet(fake_img)) + 2e-8 * regularization(fake_img)

                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

                if i % 2 == 0:
                    real_out = dnet(labels)
                    fake_out = dnet(fake_img.detach())
                    loss_d = criterion_d(real_out, real_label) + criterion_d(fake_out, fake_label)
                    optimizer_d.zero_grad()
                    loss_d.backward()
                    optimizer_d.step()

                    train_loss_d += loss_d.item()
                    train_loss_all_d += loss_d.item()
                train_loss_g += loss_g.item()
                train_loss_all_g += loss_g.item()

                # 进度条更新
                t.set_postfix(gnet_loss='{:.6f}'.format(train_loss_g/args.batch_size), dnet_loss='{:.6f}'.format(train_loss_d/args.batch_size))
                t.update(len(inputs))
        # 记录lossLog 方面画图
        gnet_lossLog.append(np.array(train_loss_all_g / total))
        dnet_lossLog.append(np.array(train_loss_all_d / total))
        # 可以在前面加上路径
        np.savetxt("gnet_lossLog.txt", gnet_lossLog)
        np.savetxt("dnet_lossLog.txt", dnet_lossLog)

        # 保存模型
        param_dict = {
            "dnet_dict": dnet.state_dict(),
            "gnet_dict": gnet.state_dict()
        }
        torch.save(param_dict, os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        # 是否更新当前最好参数
        dnet.eval()
        gnet.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # 验证不用求导
            with torch.no_grad():
                fake_img = gnet(inputs).clamp(0.0, 1.0)
                loss = criterion_g(fake_img, labels, dnet(fake_img))

            epoch_psnr.update(calc_psnr(fake_img, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        # 记录psnr
        psnrLog.append(Tensor.cpu(epoch_psnr.avg))
        np.savetxt('psnrLog.txt', psnrLog)
        # 找到更好的权重参数，更新
        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights_g = copy.deepcopy(gnet.state_dict())
            best_weights_d = copy.deepcopy(dnet.state_dict())

        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))

        param_dict = {
            "dnet_dict": best_weights_d,
            "gnet_dict": best_weights_g
        }
        torch.save(param_dict, os.path.join(args.outputs_dir, 'best.pth'))

    # print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    #
    # torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
