
BATCH_SIZE = 1  # 每批次的样本数
EPOCHS = 16      # 模型训练的总轮数
LOG_GAP = 100    # 输出训练信息的间隔

N_CLASSES = 2  # 图像分类种类数量
IMG_SIZE = (512, 512)  # 图像缩放尺寸

INIT_LR = 3e-4  # 初始学习率

SRC_PATH = "./data/data69911/mydataset1.zip"  # 压缩包路径
DST_PATH = "./data/mydataset1"  # 解压路径
DATA_PATH = {  # 实验数据集路径
    "img": DST_PATH + "/image",  # 正常图像
    "lab": DST_PATH + "/label",  # 分割图像
}
INFER_PATH = {  # 预测数据集路径
    "img": ["./work/test3.png"],# "./work/test8.png"],  # 正常图像
    "lab": ["./work/1.png"],# "./work/2.png"],  # 分割图像
}
MODEL_PATH = "UNet3+.pdparams"  # 模型参数保存路径