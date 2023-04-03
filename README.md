# UNet3
.
├── data
│   └── data69911
│      └──mydataset1.zip
├── result
│      ├──  1
│       ...
├──  work    
│      ├──  test1.png
│      ...
├── src
│   ├──  main.py
│   ├──  param.py
│   ├──  train.py
│   └──  predict.py
├──  README
├──  UNet3+.txt
└──  dataset.txt

python train.py
//开始训练，训练结束后以折线图的形式显示模型评估结果
//超参数batch_size=5,img_size=(256,256)

python predict.py
//模型保存后，根据路径中的图片可进行预测，图片路径在param.py中设置
//预测时img_size=(512,512)(提高输入图像分辨率)，batch_size=1（修改img_size后显存不够）

/*
data用于保存训练集的zip文件
work用于保存预测图片
result中是训练好的模型预测的图片
src中保存源码：
    main.py中以图像预处理函数和UNet3+网络结构为主
    param.py设置训练的超参数
    train.py进行模型训练，并在结束后进行模型评估
    predict.py进行图像的预测
*/
