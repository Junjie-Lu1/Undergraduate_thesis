# config.py

import torch

# 这是一个配置类，用于存储模型和训练所需的所有超参数。
# 将所有配置集中管理，使得修改参数非常方便，而无需在代码中多处查找。
class Config:
    """模型和训练的配置参数"""
    
    # --- 数据相关参数 ---
    BATCH_SIZE = 64  # 训练时的批次大小
    # 输入数据的维度，这些是固定的，由数据预处理决定
    TIME_LEN = 3     # 时间步长度
    NUM_CHANNELS = 62# EEG通道数
    FREQ_BANDS = 5   # 频谱带数量

    # --- 模型结构参数 ---
    # Transformer多头注意力的头数。更多的头可以让模型关注不同方面的信息。
    NUM_HEADS = 8

    UNIFIED_EMBED_DIM  = 256 # 统一的嵌入
    
    # 分类头（全连接层）的隐藏层维度。
    # 经过注意力模块后，数据会被展平，然后送入这个全连接层。
    CLASSIFIER_HIDDEN_DIM = 256
    
    # 情绪分类任务的类别数
    NUM_EMOTION_CLASSES = 5
    # 性别分类任务的类别数
    NUM_SEX_CLASSES = 2

    # --- 训练相关参数 ---
    LEARNING_RATE = 1e-4  # 优化器（如Adam）的学习率
    NUM_EPOCHS = 100      # 训练的总轮数

    # --- 其他参数 ---
    # 设备配置：优先使用GPU（CUDA），如果没有则使用CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 是否使用旋转位置编码。这是一个实验性功能。
    # RoPE可以为序列中的元素提供相对位置信息，在某些任务中可能有效。
    USE_ROPE = False # 默认设置为False，您可以尝试将其改为True

