import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入自定义模块
from config import Config
from model import EEGEmotionSexClassifier
from SEEDVDataset import SEEDVDataset

class PrimarySecondaryOptimizer:
    """
    主次任务优化器
    主任务：情绪分类
    次任务：性别分类
    """
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.lr = lr
        self.params = list(model.parameters())

    def step(self, loss_emotion, loss_sex):
        """执行优化步骤"""
        # 清零梯度
        self.model.zero_grad()
        
        # 计算主任务梯度
        with torch.no_grad():
            grads_emotion = torch.autograd.grad(
                loss_emotion, 
                self.params, 
                retain_graph=True, 
                create_graph=False, 
                allow_unused=True
            )
            
            # 计算次任务梯度
            grads_sex = torch.autograd.grad(
                loss_sex, 
                self.params, 
                retain_graph=True, 
                create_graph=False, 
                allow_unused=True
            )
        
        # 计算梯度内积和范数平方
        dot_product = 0.0
        norm_sq = 0.0
        
        for g_e, g_s in zip(grads_emotion, grads_sex):
            if g_e is not None and g_s is not None:
                dot_product += (g_e * g_s).sum()
                norm_sq += (g_s ** 2).sum()
        
        # 计算系数
        if norm_sq.item() == 0:
            coef = 0.0
        else:
            coef = torch.relu(-dot_product) / (norm_sq + 1e-8)
        
        # 构建参数更新方向 Δθ
        delta_theta = []
        for g_e, g_s in zip(grads_emotion, grads_sex):
            if g_e is not None and g_s is not None:
                # 主次任务梯度更新
                delta = g_e + coef * g_s
                delta = -self.lr * delta
                delta_theta.append(delta)
            elif g_e is not None:
                # 仅主任务梯度
                delta_theta.append(-self.lr *g_e)
            elif g_s is not None:
                # 仅次任务梯度
                delta_theta.append(-self.lr * coef * g_s)
            else:
                delta_theta.append(torch.zeros_like(self.params[0]))

        # 计算内积 <Δθ, ∇L₁> 和 <Δθ, ∇L₂>
        inner_product_L1 = 0.0
        inner_product_L2 = 0.0

        for delta, g_e, g_s in zip(delta_theta, grads_emotion, grads_sex):
            if g_e is not None:
                inner_product_L1 += (delta * g_e).sum()
            if g_s is not None:
                inner_product_L2 += (delta * g_s).sum()

        # 每一步打印内积信息
        print(f"<Δθ, ∇L₁> = {inner_product_L1.item():.6f}, <Δθ, ∇L₂> = {inner_product_L2.item():.6f}")

        # 更新参数
        with torch.no_grad():
            for param, delta in zip(self.params, delta_theta):
                param.data += delta


def main():
    # 1. 加载配置
    config = Config()
    print(f"使用的设备: {config.DEVICE}")
    print("使用主次任务优化器")

    # 2. 准备数据
    subject_list = list(range(1, 17))  # 1~16被试
    trial_list = list(range(0, 45))    # 0~44试次

    train_dataset = SEEDVDataset(
        subject_list, 
        trial_list, 
        config.ROOT_DIR, 
        config.FEATURE_DIR, 
        seg_len=config.TIME_LEN, 
        skip=1
    )
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )

    # 3. 初始化模型、损失函数和优化器
    model = EEGEmotionSexClassifier(config).to(config.DEVICE)
    
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_sex = nn.CrossEntropyLoss()
    
    # 使用自定义优化器
    optimizer = PrimarySecondaryOptimizer(model, lr=config.LEARNING_RATE)

    # 4. 训练循环
    print("开始训练...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        sex_loss = 0.0
        emotion_loss = 0.0
        
        for i, (inputs, labels_emotion, labels_sex) in enumerate(train_loader):
            # 将数据移动到指定设备
            inputs = inputs.to(config.DEVICE)
            labels_emotion = labels_emotion.to(config.DEVICE)
            labels_sex = labels_sex.to(config.DEVICE)
            
            # 前向传播
            emotion_outputs, sex_outputs = model(inputs)
            
            # 计算损失
            loss_emotion = criterion_emotion(emotion_outputs, labels_emotion)
            loss_sex = criterion_sex(sex_outputs, labels_sex)
            
            # 使用自定义优化器更新参数
            optimizer.step(loss_emotion, loss_sex)
            
            # 打印日志
            sex_loss += loss_sex.item()
            emotion_loss += loss_emotion.item()
            
            # if (i + 1) % 20 == 0: # 每20个batch打印一次
            print(f'轮数 [{epoch+1}/{config.NUM_EPOCHS}], 步数 [{i+1}/{len(train_loader)}], 性别损失: {loss_sex.item():.4f}, 情绪损失：{loss_emotion.item():.4f}')
        
        print(f'--- 轮数 [{epoch+1}/{config.NUM_EPOCHS}] 结束, 性别损失: {sex_loss / len(train_loader):.4f} , 情绪损失：{emotion_loss / len(train_loader):.4f} ---')

    print("训练结束.")

    # 5. 保存模型
    # torch.save()用于保存模型的状态字典（state_dict），即模型的参数。
    # 只保存state_dict是一种推荐的做法，因为它更灵活，与模型定义解耦。
    torch.save(model.state_dict(), 'eeg_emotion_sex_model_my_optim.pth')
    print("模型已经保存到 eeg_emotion_sex_model_my_optim.pth")

if __name__ == '__main__':
    main()
