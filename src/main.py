from typing import Union, Tuple

import cv2
import os
import random
import zipfile
import numpy as np
from copy import deepcopy
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC

import paddle
from paddle import nn
from paddle.framework import ParamAttr
from paddle.io import DataLoader, Dataset
from paddle.nn import initializer as I, functional as F
from paddle.optimizer import Adam, AdamW
from paddle.optimizer.lr import CosineAnnealingDecay
from param import*

if not os.path.isdir(DATA_PATH["img"]) or not os.path.isdir(DATA_PATH["lab"]):
    z = zipfile.ZipFile(SRC_PATH, "r")  # 以只读模式打开zip文件
    z.extractall(path=DST_PATH)  # 解压zip文件至目标路径
    z.close()
print("The dataset has been unpacked successfully!")

train_list, test_list = [], []  # 存放图像路径与标签路径的映射
images = os.listdir(DATA_PATH["img"])  # 统计数据集下的图像文件

for idx, img in enumerate(images):
    lab = os.path.join(DATA_PATH["lab"], img.replace(".jpg", ".png"))
    img = os.path.join(DATA_PATH["img"], img)
    if idx % 10 != 0:  # 按照1:9的比例划分数据集
        train_list.append((img, lab))
    else:
        test_list.append((img, lab))


def random_brightness(img, lab, low=0.5, high=1.5):
    ''' 随机改变亮度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Brightness(img).enhance(x)
    return img, lab


def random_contrast(img, lab, low=0.5, high=1.5):
    ''' 随机改变对比度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Contrast(img).enhance(x)
    return img, lab


def random_color(img, lab, low=0.5, high=1.5):
    ''' 随机改变饱和度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Color(img).enhance(x)
    return img, lab


def random_sharpness(img, lab, low=0.5, high=1.5):
    ''' 随机改变清晰度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Sharpness(img).enhance(x)
    return img, lab


def random_rotate(img, lab, low=0, high=360):
    ''' 随机旋转图像(0~360度) '''
    angle = random.choice(range(low, high))
    img, lab = img.rotate(angle), lab.rotate(angle)
    return img, lab


def random_flip(img, lab, prob=0.5):
    """ 随机翻转图像(p=0.5) """
    if random.random() < prob:  # 上下翻转
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        lab = lab.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < prob:  # 左右翻转
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
    return img, lab


def random_noise(img, lab, low=0, high=10):
    ''' 随机加高斯噪声(0~10) '''
    img = np.asarray(img)
    sigma = np.random.uniform(low, high)
    noise = np.random.randn(img.shape[0], img.shape[1], 3) * sigma
    img = img + np.round(noise).astype('uint8')
    # 将矩阵中的所有元素值限制在0~255之间：
    img[img > 255], img[img < 0] = 255, 0
    img = Image.fromarray(img)
    return img, lab


def image_augment(img, lab, prob=0.5):
    ''' 叠加多种数据增强方法 '''
    opts = [random_brightness, random_contrast, random_color, random_flip,
            random_noise, random_rotate, random_sharpness, ]  # 数据增强方法
    for func in opts:
        if random.random() < prob:
            img, lab = func(img, lab)  # 处理图像和标签
    return img, lab


class MyDataset(Dataset):
    ''' 自定义的数据集类
    * `label_list`: 图像路径和标签路径的映射列表
    * `transform`: 图像处理函数
    * `augment`: 数据增强函数
    '''

    def __init__(self, label_list, transform, augment=None):
        super(MyDataset, self).__init__()
        random.shuffle(label_list)  # 打乱映射列表
        self.label_list = label_list
        self.transform = transform
        self.augment = augment

    def __getitem__(self, index):
        ''' 根据位序获取对应数据 '''
        img_path, lab_path = self.label_list[index]
        img, lab = self.transform(img_path, lab_path, self.augment)
        return img, lab

    def __len__(self):
        ''' 获取数据集的样本总数 '''
        return len(self.label_list)


def data_mapper(img_path, lab_path, augment=None):
    ''' 图像处理函数 '''
    img = Image.open(img_path).convert("RGB")
    lab = cv2.cvtColor(cv2.imread(lab_path), cv2.COLOR_RGB2GRAY)
    # 将标签文件进行灰度二值化：
    _, lab = cv2.threshold(src=lab,  # 待处理图片
                           thresh=170,  # 起始阈值
                           maxval=255,  # 最大阈值
                           type=cv2.THRESH_BINARY_INV)  # 算法类型
    lab = Image.fromarray(lab).convert("L")  # 转换为PIL.Image
    # 将图像缩放为IMG_SIZE大小的高质量图像：
    img = img.resize(IMG_SIZE, Image.ANTIALIAS)
    lab = lab.resize(IMG_SIZE, Image.ANTIALIAS)
    if augment is not None:  # 数据增强
        img, lab = augment(img, lab)
    # 将图像转为numpy数组，并转换图像的格式：
    img = np.array(img).astype("float32").transpose((2, 0, 1))
    lab = np.array(lab).astype("int32")[np.newaxis, ...]
    # 将图像数据归一化，并转换成Tensor格式：
    img = paddle.to_tensor(img / 255.0)
    lab = paddle.to_tensor(lab // 255)
    return img, lab


train_dataset = MyDataset(train_list, data_mapper, image_augment)  # 训练集
test_dataset = MyDataset(test_list, data_mapper, augment=None)  # 测试集

train_loader = DataLoader(train_dataset,  # 训练数据集
                          batch_size=BATCH_SIZE,  # 每批次的样本数
                          num_workers=4,  # 加载数据的子进程数
                          shuffle=True,  # 打乱数据集
                          drop_last=False)  # 不丢弃不完整的样本批次

test_loader = DataLoader(test_dataset,  # 测试数据集
                         batch_size=1,  # 每批次的样本数
                         num_workers=4,  # 加载数据的子进程数
                         shuffle=False,  # 不打乱数据集
                         drop_last=False)  # 不丢弃不完整的样本批次


def init_weights(net, init_type="normal"):
    ''' 初始化网络的权重与偏置
    * `net`: 需要初始化的神经网络层
    * `init_type`: 初始化机制（normal/xavier/kaiming/truncated）
    '''
    if init_type == "normal":
        attr = ParamAttr(initializer=I.Normal())
    elif init_type == "xavier":
        attr = ParamAttr(initializer=I.XavierNormal())
    elif init_type == "kaiming":
        attr = ParamAttr(initializer=I.KaimingNormal())
    elif init_type == "truncated":
        attr = ParamAttr(initializer=I.TruncatedNormal())
    else:
        error = "Initialization method [%s] is not implemented!"
        raise NotImplementedError(error % init_type)
    # 初始化网络层net的权重系数和偏置系数：
    net.param_attr, net.bias_attr = attr, deepcopy(attr)


class Encoder(nn.Layer):
    ''' 用于构建编码器模块
    * `in_size`: 输入通道数
    * `out_size`: 输出通道数
    * `is_batchnorm`: 是否批正则化
    * `n`: 卷积层数量（默认为2）
    * `ks`: 卷积核大小（默认为3）
    * `s`: 卷积运算步长（默认为1）
    * `p`: 卷积填充大小（默认为1）
    '''

    def __init__(self, in_size, out_size, is_batchnorm,
                 n=2, ks=3, s=1, p=1):
        super(Encoder, self).__init__()
        self.n = n

        for i in range(1, self.n + 1):  # 定义多层卷积神经网络
            if is_batchnorm:
                block = nn.Sequential(nn.Conv2D(in_size, out_size, ks, s, p),
                                      nn.BatchNorm2D(out_size),
                                      nn.ReLU())
            else:
                block = nn.Sequential(nn.Conv2D(in_size, out_size, ks, s, p),
                                      nn.ReLU())
            setattr(self, "block%d" % i, block)
            in_size = out_size

        for m in self.children():  # 初始化各层网络的系数
            init_weights(m, init_type="kaiming")

    def forward(self, x):
        for i in range(1, self.n + 1):
            block = getattr(self, "block%d" % i)
            x = block(x)  # 进行前向传播运算
        return x


class Decoder(nn.Layer):
    ''' 用于构建解码器模块
    * `cur_stage`(int): 当前解码器所在层数
    * `cat_size`(int): 统一后的特征图通道数
    * `up_size`(int): 特征融合后的通道总数
    * `filters`(list): 各卷积网络的卷积核数
    * `ks`: 卷积核大小（默认为3）
    * `s`: 卷积运算步长（默认为1）
    * `p`: 卷积填充大小（默认为1）
    '''

    def __init__(self, cur_stage, cat_size, up_size,
                 filters, ks=3, s=1, p=1):
        super(Decoder, self).__init__()
        self.n = len(filters)  # 卷积网络模块的个数

        for idx, num in enumerate(filters):
            idx += 1  # 待处理输出所在层数
            if idx < cur_stage:
                # he[idx]_PT_hd[cur_stage], Pool [ps] times
                ps = 2 ** (cur_stage - idx)
                block = nn.Sequential(nn.MaxPool2D(ps, ps, ceil_mode=True),
                                      nn.Conv2D(num, cat_size, ks, s, p),
                                      nn.BatchNorm2D(cat_size),
                                      nn.ReLU())
            elif idx == cur_stage:
                # he[idx]_Cat_hd[cur_stage], Concatenate
                block = nn.Sequential(nn.Conv2D(num, cat_size, ks, s, p),
                                      nn.BatchNorm2D(cat_size),
                                      nn.ReLU())
            else:
                # hd[idx]_UT_hd[cur_stage], Upsample [us] times
                us = 2 ** (idx - cur_stage)
                num = num if idx == 5 else up_size
                block = nn.Sequential(nn.Upsample(scale_factor=us, mode="bilinear"),
                                      nn.Conv2D(num, cat_size, ks, s, p),
                                      nn.BatchNorm2D(cat_size),
                                      nn.ReLU())
            setattr(self, "block%d" % idx, block)

        # fusion(he[]_PT_hd[], ..., he[]_Cat_hd[], ..., hd[]_UT_hd[])
        self.fusion = nn.Sequential(nn.Conv2D(up_size, up_size, ks, s, p),
                                    nn.BatchNorm2D(up_size),
                                    nn.ReLU())

        for m in self.children():  # 初始化各层网络的系数
            init_weights(m, init_type="kaiming")

    def forward(self, inputs):
        outputs = []  # 记录各层的输出，以便于拼接起来
        for i in range(self.n):
            block = getattr(self, "block%d" % (i + 1))
            outputs.append(block(inputs[i]))
        hd = self.fusion(paddle.concat(outputs, 1))
        return hd


class UNet3Plus(nn.Layer):
    ''' UNet3+ with Deep Supervision and Class-guided Module
    * `in_channels`: 输入通道数（默认为3）
    * `n_classes`: 物体的分类种数（默认为2）
    * `is_batchnorm`: 是否批正则化（默认为True）
    * `deep_sup`: 是否开启深度监督机制（Deep Supervision）
    * `set_cgm`: 是否设置分类引导模块（Class-guided Module）
    '''

    def __init__(self, in_channels=3, n_classes=2,
                 is_batchnorm=True, deep_sup=True, set_cgm=True):
        super(UNet3Plus, self).__init__()
        self.deep_sup = deep_sup
        self.set_cgm = set_cgm
        filters = [64, 128, 256, 512, 1024]  # 各模块的卷积核大小
        cat_channels = filters[0]  # 统一后的特征图通道数
        cat_blocks = 5  # 编（解）码器的层数
        up_channels = cat_channels * cat_blocks  # 特征融合后的通道数

        # ====================== Encoders ======================
        self.conv_e1 = Encoder(in_channels, filters[0], is_batchnorm)
        self.pool_e1 = nn.MaxPool2D(kernel_size=2)
        self.conv_e2 = Encoder(filters[0], filters[1], is_batchnorm)
        self.pool_e2 = nn.MaxPool2D(kernel_size=2)
        self.conv_e3 = Encoder(filters[1], filters[2], is_batchnorm)
        self.pool_e3 = nn.MaxPool2D(kernel_size=2)
        self.conv_e4 = Encoder(filters[2], filters[3], is_batchnorm)
        self.pool_e4 = nn.MaxPool2D(kernel_size=2)
        self.conv_e5 = Encoder(filters[3], filters[4], is_batchnorm)

        # ====================== Decoders ======================
        self.conv_d4 = Decoder(4, cat_channels, up_channels, filters)
        self.conv_d3 = Decoder(3, cat_channels, up_channels, filters)
        self.conv_d2 = Decoder(2, cat_channels, up_channels, filters)
        self.conv_d1 = Decoder(1, cat_channels, up_channels, filters)

        # ======================= Output =======================
        if self.set_cgm:
            # -------------- Class-guided Module ---------------
            self.cls = nn.Sequential(nn.Dropout(p=0.5),
                                     nn.Conv2D(filters[4], 2, 1),
                                     nn.AdaptiveMaxPool2D(1),
                                     nn.Sigmoid())
        if self.deep_sup:
            # -------------- Bilinear Upsampling ---------------
            self.upscore5 = nn.Upsample(scale_factor=16, mode="bilinear")
            self.upscore4 = nn.Upsample(scale_factor=8, mode="bilinear")
            self.upscore3 = nn.Upsample(scale_factor=4, mode="bilinear")
            self.upscore2 = nn.Upsample(scale_factor=2, mode="bilinear")
            # ---------------- Deep Supervision ----------------
            self.outconv5 = nn.Conv2D(filters[4], n_classes, 3, 1, 1)
            self.outconv4 = nn.Conv2D(up_channels, n_classes, 3, 1, 1)
            self.outconv3 = nn.Conv2D(up_channels, n_classes, 3, 1, 1)
            self.outconv2 = nn.Conv2D(up_channels, n_classes, 3, 1, 1)
        self.outconv1 = nn.Conv2D(up_channels, n_classes, 3, 1, 1)

        # ================= Initialize Weights =================
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D) or isinstance(m, nn.BatchNorm):
                init_weights(m, init_type='kaiming')

    def dot_product(self, seg, cls):
        B, N, H, W = seg.shape
        seg = seg.reshape((B, N, H * W))
        clssp = paddle.ones((1, N))
        ecls = (cls * clssp).reshape((B, N, 1))
        final = (seg * ecls).reshape((B, N, H, W))
        return final

    def forward(self, x):
        # ====================== Encoders ======================
        e1 = self.conv_e1(x)  # e1: 320*320*64
        e2 = self.pool_e1(self.conv_e2(e1))  # e2: 160*160*128
        e3 = self.pool_e2(self.conv_e3(e2))  # e3: 80*80*256
        e4 = self.pool_e3(self.conv_e4(e3))  # e4: 40*40*512
        e5 = self.pool_e4(self.conv_e5(e4))  # e5: 20*20*1024

        # ================ Class-guided Module =================
        if self.set_cgm:
            cls_branch = self.cls(e5).squeeze(3).squeeze(2)
            cls_branch_max = cls_branch.argmax(axis=1)
            cls_branch_max = cls_branch_max[:, np.newaxis].astype("float32")

        # ====================== Decoders ======================
        d5 = e5
        d4 = self.conv_d4((e1, e2, e3, e4, d5))
        d3 = self.conv_d3((e1, e2, e3, d4, d5))
        d2 = self.conv_d2((e1, e2, d3, d4, d5))
        d1 = self.conv_d1((e1, d2, d3, d4, d5))

        # ======================= Output =======================
        if self.deep_sup:
            y5 = self.upscore5(self.outconv5(d5))  # 16 => 256
            y4 = self.upscore4(self.outconv4(d4))  # 32 => 256
            y3 = self.upscore3(self.outconv3(d3))  # 64 => 256
            y2 = self.upscore2(self.outconv2(d2))  # 128 => 256
            y1 = self.outconv1(d1)  # 256
            if self.set_cgm:
                y5 = self.dot_product(y5, cls_branch_max)
                y4 = self.dot_product(y4, cls_branch_max)
                y3 = self.dot_product(y3, cls_branch_max)
                y2 = self.dot_product(y2, cls_branch_max)
                y1 = self.dot_product(y1, cls_branch_max)
            return F.sigmoid(y1), F.sigmoid(y2), F.sigmoid(y3), \
                F.sigmoid(y4), F.sigmoid(y5)
        else:
            y1 = self.outconv1(d1)  # 320*320*n_classes
            if self.set_cgm:
                y1 = self.dot_product(y1, cls_branch_max)
            return F.sigmoid(y1)


model = UNet3Plus(n_classes=N_CLASSES, deep_sup=False, set_cgm=False)


# paddle.Model(model).summary((1, 3) + IMG_SIZE)  # 可视化模型结构


class DiceLoss(nn.Layer):
    ''' Dice Loss for Segmentation Tasks'''

    def __init__(self,
                 n_classes: int = 2,
                 smooth: Union[float, Tuple[float, float]] = (0, 1e-6),
                 sigmoid_x: bool = False,
                 softmax_x: bool = True,
                 onehot_y: bool = True,
                 square_xy: bool = True,
                 include_bg: bool = True,
                 reduction: str = "mean"):
        ''' Args:
        * `n_classes`: number of classes.
        * `smooth`: smoothing parameters of the dice coefficient.
        * `sigmoid_x`: whether using `sigmoid` to process the result.
        * `softmax_x`: whether using `softmax` to process the result.
        * `onehot_y`: whether using `one-hot` to encode the label.
        * `square_xy`: whether using squared result and label.
        * `include_bg`: whether taking account of bg-class when computering dice.
        * `reduction`: reduction function of dice loss.
        '''
        super(DiceLoss, self).__init__()
        if reduction not in ["mean", "sum"]:
            raise NotImplementedError(
                "`reduction` of dice loss should be 'mean' or 'sum'!"
            )
        if isinstance(smooth, float):
            self.smooth = (smooth, smooth)
        else:
            self.smooth = smooth

        self.n_classes = n_classes
        self.sigmoid_x = sigmoid_x
        self.softmax_x = softmax_x
        self.onehot_y = onehot_y
        self.square_xy = square_xy
        self.include_bg = include_bg
        self.reduction = reduction

    def forward(self, pred, mask):
        (sm_nr, sm_dr) = self.smooth

        if self.sigmoid_x:
            pred = F.sigmoid(pred)
        if self.n_classes > 1:
            if self.softmax_x and self.n_classes == pred.shape[1]:
                pred = F.softmax(pred, axis=1)
            if self.onehot_y:
                mask = mask if mask.ndim < 4 else mask.squeeze(axis=1)
                mask = F.one_hot(mask.astype("int64"), self.n_classes)
                mask = mask.transpose((0, 3, 1, 2))
            if not self.include_bg:
                pred = pred[:, 1:] if pred.shape[1] > 1 else pred
                mask = mask[:, 1:] if mask.shape[1] > 1 else mask
        if pred.ndim != mask.ndim or pred.shape[1] != mask.shape[1]:
            raise ValueError(
                f"The shape of `pred`({pred.shape}) and " +
                f"`mask`({mask.shape}) should be the same."
            )

        # only reducing spatial dimensions:
        reduce_dims = paddle.arange(2, pred.ndim).tolist()
        insersect = paddle.sum(pred * mask, axis=reduce_dims)
        if self.square_xy:
            pred, mask = paddle.pow(pred, 2), paddle.pow(mask, 2)
        pred_sum = paddle.sum(pred, axis=reduce_dims)
        mask_sum = paddle.sum(mask, axis=reduce_dims)
        loss = 1. - (2 * insersect + sm_nr) / (pred_sum + mask_sum + sm_dr)

        if self.reduction == "sum":
            loss = paddle.sum(loss)
        else:
            loss = paddle.mean(loss)
        return loss


def dice_func(pred: np.ndarray, mask: np.ndarray,
              n_classes: int, ignore_bg: bool = False):
    ''' compute dice (for NumpyArray) '''

    def sub_dice(x: paddle.Tensor, y: paddle.Tensor, sm: float = 1e-6):
        intersect = np.sum(np.sum(np.sum(x * y)))
        y_sum = np.sum(np.sum(np.sum(y)))
        x_sum = np.sum(np.sum(np.sum(x)))
        return (2 * intersect + sm) / (x_sum + y_sum + sm)

    assert pred.shape == mask.shape
    assert isinstance(ignore_bg, bool)
    return [
        sub_dice(pred == i, mask == i)
        for i in range(int(ignore_bg), n_classes)
    ]


fig = plt.figure(figsize=[10, 5])
