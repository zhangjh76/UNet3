from main import *



def show_result(img_path , pred):
    ''' 展示原图、标签以及预测结果 '''
    img = Image.open(img_path).resize(IMG_SIZE)
    #lab = Image.open(lab_path).resize(IMG_SIZE)
    pred = pred.argmax(axis=1).numpy().reshape(IMG_SIZE)
    plt.figure(figsize=(12, 6))
    add_subimg(img, 121, "Image")
    #add_subimg(lab, 132, "Label")
    add_subimg(pred, 122, "Predict", colormap())
    plt.tight_layout()
    plt.show()
    plt.close()


def add_subimg(img, loc, title, cmap=None):
    ''' 添加子图以展示图像 '''
    plt.subplot(loc)
    plt.title(title)
    plt.imshow(img, cmap)
    plt.xticks([])  # 去除X刻度
    plt.yticks([])  # 去除Y刻度


def colormap(colors=[ '#000000' , '#FFFFFF']):
    ''' 自定义ColorMap '''
    return LSC.from_list('cmap', colors, 256)


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

model.eval()  # 开启评估模式
model.set_state_dict(
    paddle.load(MODEL_PATH)
)  # 载入预训练模型参数



for i in range(len(INFER_PATH["img"])):
    img_path, lab_path = INFER_PATH["img"][i], INFER_PATH["lab"][i]
    img, lab = data_mapper(img_path, lab_path)  # 处理预测图像
    pred = model(img[np.newaxis, ...])  # 开始模型预测
    show_result(img_path, pred)#, lab_path

