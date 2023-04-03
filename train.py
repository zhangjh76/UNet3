from main import *
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.optimizer import AdamW
from param import*

model.train()  # 开启训练模式
scheduler = CosineAnnealingDecay(
    learning_rate=INIT_LR,
    T_max=EPOCHS,
)  # 定义学习率衰减器
optimizer = AdamW(
    learning_rate=scheduler,
    parameters=model.parameters(),
    weight_decay=1e-5
)  # 定义Adam优化器
dice_loss = DiceLoss(n_classes=N_CLASSES)
loss_list = []  # 用于可视化




for ep in range(EPOCHS):
    ep_loss_list = []
    for batch_id, data in enumerate(train_loader()):
        image, label = data
        pred = model(image)  # 预测结果
        loss = dice_loss(pred, label)  # 计算损失函数值
        if batch_id % LOG_GAP == 0:  # 定期输出训练结果
            print("Epoch：%2d，Batch：%3d，Loss：%.5f" % (ep, batch_id, loss))
        ep_loss_list.append(loss.item())
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()  # 衰减一次学习率
    loss_list.append(np.mean(ep_loss_list))
    print("【Train】Epoch：%2d，Loss：%.5f" % (ep, loss_list[-1]))
paddle.save(model.state_dict(), MODEL_PATH)  # 保存训练好的模型

model.eval()                 # 开启评估模式
model.set_state_dict(
    paddle.load(MODEL_PATH)
)   # 载入预训练模型参数
dice_accs = []

for batch_id, data in enumerate(test_loader()):
    image, label = data
    pred = model(image)                          # 预测结果
    pred = pred.argmax(axis=1).squeeze(axis=0).cpu().numpy()
    label = label.squeeze(0).squeeze(0).cpu().numpy()
    dice = dice_func(pred, label, N_CLASSES)     # 计算损失函数值
    dice_accs.append(dice)
print("Eval \t Dice: %.5f" % (np.mean(dice_accs)))

# 训练误差图像：
ax = fig.add_subplot(111, facecolor="#E8E8F8")
ax.set_xlabel("Steps", fontsize=18)
ax.set_ylabel("Loss", fontsize=18)
plt.tick_params(labelsize=14)
ax.plot(range(len(loss_list)), loss_list, color="orangered")
ax.grid(linewidth=1.5, color="white")  # 显示网格

fig.tight_layout()
plt.show()
plt.close()


