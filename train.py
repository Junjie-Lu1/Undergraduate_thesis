import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 导入我们自定义的模块
from config import Config
from model import EEGEmotionSexClassifier
from SEEDVDataset import SEEDVDataset

def main():
    # 1. 加载配置
    config = Config()
    print(f"使用的设备: {config.DEVICE}")

    # 2. 准备数据
    # --------------------------------------------------------------------
    subject_list = list(range(1, 17))  # 1~16被试
    trial_list = list(range(0, 45))    # 0~44试次

    train_dataset = SEEDVDataset(subject_list, trial_list, config.ROOT_DIR, config.FEATURE_DIR, seg_len=config.TIME_LEN, skip=1)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)  # 创建数据加载器
    # 数据准备结束，您的真实数据应该以类似的方式被train_loader加载

    # 3. 初始化模型、损失函数和优化器
    # 实例化我们的模型，并将其移动到指定的设备（GPU或CPU）
    model = EEGEmotionSexClassifier(config).to(config.DEVICE)
    
    # nn.CrossEntropyLoss 是PyTorch中用于多分类任务的常用损失函数。
    # 它内部包含了LogSoftmax和NLLLoss，所以模型的输出层不需要加Softmax。
    # 它期望的输入是模型的原始输出，标签是类别索引。
    criterion_emotion = nn.CrossEntropyLoss() 
    criterion_sex = nn.CrossEntropyLoss()
    
    # optim.Adam 需要调整的参数是模型的参数（model.parameters()）和学习率。
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 4. 训练循环
    print("开始训练...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()  # 将模型设置为训练模式，这会启用Dropout和BatchNorm等层
        running_loss = 0.0
        emotion_loss = 0.0
        
        # DataLoader会按批次返回数据
        for i, (inputs, labels_emotion, labels_sex) in enumerate(train_loader):
            # 将数据移动到指定设备
            inputs = inputs.to(config.DEVICE)
            labels_emotion = labels_emotion.to(config.DEVICE)
            labels_sex = labels_sex.to(config.DEVICE)
            
            # --- 梯度清零 ---
            # 在每次反向传播之前，需要将上一轮的梯度清零。
            # 因为PyTorch默认会累积梯度。
            optimizer.zero_grad()
            
            # --- 前向传播 ---
            # 将输入数据送入模型，得到两个任务的输出
            emotion_outputs, sex_outputs = model(inputs)
            
            # --- 计算损失 ---
            # 分别计算两个任务的损失
            # 假设 emotion_logits 是模型输出，labels_emotion 是真实标签
  
            loss_emotion = criterion_emotion(emotion_outputs, labels_emotion)
            loss_sex = criterion_sex(sex_outputs, labels_sex)
            
            # 总损失是两个任务损失的平均值
            total_loss = (loss_emotion + loss_sex) / 2
            
            # --- 反向传播 ---
            # .backward()会自动计算所有参数的梯度
            total_loss.backward()
            
            # --- 更新权重 ---
            # .step()会根据计算出的梯度来更新模型的参数
            optimizer.step()
            
            # 打印日志
            running_loss += total_loss.item()
            emotion_loss += loss_emotion.item()
            if (i + 1) % 20 == 0: # 每20个batch打印一次
                print(f'轮数 [{epoch+1}/{config.NUM_EPOCHS}], 步数 [{i+1}/{len(train_loader)}], 总损失: {total_loss.item():.4f}, 情绪损失：{loss_emotion.item():.4f}')
        
        print(f'--- Epoch [{epoch+1}/{config.NUM_EPOCHS}] 结束, 总损失: {running_loss / len(train_loader):.4f} , 情绪损失：{emotion_loss / len(train_loader):.4f} ---')

    print("训练结束.")

    # 5. 保存模型
    # torch.save()用于保存模型的状态字典（state_dict），即模型的参数。
    # 只保存state_dict是一种推荐的做法，因为它更灵活，与模型定义解耦。
    torch.save(model.state_dict(), 'eeg_emotion_sex_model.pth')
    print("模型已经保存到 eeg_emotion_sex_model.pth")

if __name__ == '__main__':
    main()
