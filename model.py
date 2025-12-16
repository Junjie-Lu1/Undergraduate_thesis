# model.py

import torch
import torch.nn as nn
import math
from config import Config

# ------------------- 辅助模块：RoPE (可选) -------------------

class RoPE(nn.Module):
    """
    旋转位置编码。
    它不是一个标准的nn.Module，更像一个函数，但为了封装性，我们将其写成一个模块。
    它通过旋转矩阵来编码绝对位置信息，同时保持内积不变，从而能更好地处理相对位置。
    """
    def __init__(self, dim):
        """
        初始化RoPE。
        :param dim: 特征的维度，需要是偶数。
        """
        super().__init__()
        # dim必须是偶数，因为我们要将特征按奇偶索引拆分
        assert dim % 2 == 0, "RoPE 要求输入的维度是偶数."
        self.dim = dim

    def forward(self, x):
        """
        应用RoPE。
        :param x: 输入张量，形状为 [seq_len, batch_size, dim]
        """
        seq_len, _, dim = x.shape
        
        # 1. 将特征维度按奇偶索引拆分
        # x[..., 0::2] 表示从第0个索引开始，每隔2个取一个元素，即偶数索引
        # x[..., 1::2] 表示从第1个索引开始，每隔2个取一个元素，即奇数索引
        # 这样我们就得到了 dim/2 个二维向量
        x_even = x[..., 0::2] # 偶数维度
        x_odd = x[..., 1::2] # 奇数维度
        # x_even和x_odd的形状: [seq_len, batch_size, dim/2]

        # 2. 计算旋转角度
        # 根据RoPE论文，θ_i = 10000^(-2i/d)，其中 i 的范围是 0 到 dim/2 - 1
        # 这里我们使用torch.arange来生成i的序列
        i = torch.arange(0, dim // 2, device=x.device, dtype=torch.float32)
        theta = 1. / (10000 ** (2 * i / dim))
        
        # 创建位置索引 m
        # m的范围是0到seq_len - 1
        m = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        # 使用外积计算 m * θ
        # rotary_pos的形状: [seq_len, dim/2]
        rotary_pos = m.unsqueeze(1) * theta.unsqueeze(0)
        
        # 计算sin和cos值
        sin_val = torch.sin(rotary_pos) # [seq_len, dim/2]
        cos_val = torch.cos(rotary_pos) # [seq_len, dim/2]
        
        # 为了能和x_even, x_odd进行广播运算，我们需要在cos_val和sin_val上增加一个维度
        # x_even的形状: [seq_len, batch_size, dim/2]
        # cos_val的形状需要从 [seq_len, dim/2] 变为 [seq_len, 1, dim/2]
        sin_val = sin_val.unsqueeze(1)
        cos_val = cos_val.unsqueeze(1)

        # 3. 应用旋转公式
        # x'_even = x_even * cos - x_odd * sin
        # x'_odd = x_even * sin + x_odd * cos
        x_rotated_even = x_even * cos_val - x_odd * sin_val
        x_rotated_odd = x_even * sin_val + x_odd * cos_val
        # x_rotated_even和x_rotated_odd的形状: [seq_len, batch_size, dim/2]

        # 4. 将旋转后的奇偶部分交错合并
        # 我们先使用torch.stack将它们在新的维度上堆叠，然后reshape
        # stack后的形状: [seq_len, batch_size, 2, dim/2]
        # reshape后的形状: [seq_len, batch_size, dim]
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).reshape(seq_len, -1, dim)

        return x_rotated

# ------------------- 核心模块：三种注意力机制 -------------------

class ChannelAttention(nn.Module):
    """
    通道注意力模块。
    它的目标是学习哪些EEG通道对于情绪和性别识别更重要。
    """
    def __init__(self, time_len, freq_bands, num_heads, unified_embed_dim, use_rope=False):
        super().__init__()
        self.original_dim = time_len * freq_bands
        # embed_dim是每个“通道”向量的特征长度，即时间维度和频谱维度的乘积
        self.embed_dim = unified_embed_dim
        # 为了统一维度，引入线性层
        self.qkv_proj = nn.Linear(self.original_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.original_dim)

        # nn.MultiheadAttention是PyTorch中实现多头注意力机制的模块。
        # 它非常强大，是Transformer模型的核心组件。
        # 参数 embed_dim: 输入特征的维度。
        # 参数 num_heads: 注意力的“头”数。多头注意力允许模型同时关注来自不同表示子空间的信息。
        # 参数 batch_first: 默认为False，意味着输入和输出张量的形状是。我们设置为False，并在forward中手动调整维度。
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=False)
        
        # nn.LayerNorm是一种归一化技术，对每个样本的所有特征进行归一化。
        # 它有助于稳定训练过程，加速收敛。
        # 参数 normalized_shape: 要归一化的特征维度。
        self.norm = nn.LayerNorm(self.original_dim) # 注意不是 embed_dim
        
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(self.embed_dim)

    def forward(self, x):
        # x的初始形状: [batch_size, time_len, num_channels, freq_bands]
        batch_size, time_len, num_channels, freq_bands = x.shape
        
        # 为了应用通道注意力，我们将通道维度 视为序列长度。
        # 因此，我们需要将x的形状从 变为 [num_channels, batch_size, time_len * freq_bands]
        # 这样，每个通道就是一个序列中的“词”，其特征是T*F维的向量。
        # .permute(2, 0, 1, 3) 将维度从 变为
        # .reshape(num_channels, batch_size, self.embed_dim) 将后三维展平
        x_reshaped = x.permute(2, 0, 1, 3).reshape(num_channels, batch_size, self.original_dim)
        # x_reshaped的形状: [num_channels, batch_size, embed_dim]
        x_proj = self.qkv_proj(x_reshaped)
        
        if self.use_rope:
            # 应用RoPE
            x_proj = self.rope(x_proj)

        # nn.MultiheadAttention的forward方法
        # query, key, value: 在自注意力中，这三个通常是同一个输入。
        # 返回值:
        #   attn_output: 注意力机制的输出，形状与输入相同。
        #   attn_output_weights: 注意力权重，形状为 [batch_size, num_heads, seq_len, seq_len]。
        attn_output, _ = self.mha(query=x_proj, key=x_proj, value=x_proj)
        # attn_output的形状: [num_channels, batch_size, embed_dim]
        attn_output = self.out_proj(attn_output)
        # attn_output形状: [num_channels, batch_size, original_dim]
        # 残差连接和层归一化
        # 这是一种常见的技巧，有助于防止梯度消失，并使训练更稳定。
        # 我们将注意力输出与原始输入相加，然后进行归一化。
        x_norm = self.norm(x_reshaped + attn_output)
        # x_norm的形状: [num_channels, batch_size, embed_dim]
        
        # 将数据恢复到原始的4维形状，以便传递给下一个注意力模块。
        # .reshape(num_channels, batch_size, time_len, freq_bands) 恢复展平的维度
        # .permute(1, 2, 0, 3) 将维度从 变回
        output = x_norm.reshape(num_channels, batch_size, time_len, freq_bands).permute(1, 2, 0, 3)
        # output的形状: [batch_size, time_len, num_channels, freq_bands]
        
        return output

class TemporalAttention(nn.Module):
    """
    时间注意力模块。
    它的目标是学习在哪些时间点上，EEG信号的特征最为关键。
    """
    def __init__(self, num_channels, freq_bands, num_heads, unified_embed_dim, use_rope=False):
        super().__init__()
        self.original_dim = num_channels * freq_bands
        self.embed_dim = unified_embed_dim
        self.qkv_proj = nn.Linear(self.original_dim, self.embed_dim) 
        self.out_proj = nn.Linear(self.embed_dim, self.original_dim) 
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=False)
        self.norm = nn.LayerNorm(self.original_dim)
        
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(self.embed_dim)

    def forward(self, x):
        # x的初始形状: [batch_size, time_len, num_channels, freq_bands]
        batch_size, time_len, num_channels, freq_bands = x.shape
        
        # 将时间维度 视为序列长度。
        # 将x的形状从 变为 [time_len, batch_size, num_channels * freq_bands]
        x_reshaped = x.permute(1, 0, 2, 3).reshape(time_len, batch_size, self.original_dim)
        # x_reshaped的形状: [time_len, batch_size, embed_dim]
        x_proj = self.qkv_proj(x_reshaped)

        if self.use_rope:
            x_proj = self.rope(x_proj)

        attn_output, _ = self.mha(query=x_proj, key=x_proj, value=x_proj)
        # attn_output的形状: [time_len, batch_size, embed_dim]
        attn_output = self.out_proj(attn_output)
        

        x_norm = self.norm(x_reshaped + attn_output)
        # x_norm的形状: [time_len, batch_size, embed_dim]
        
        # 恢复原始形状
        output = x_norm.reshape(time_len, batch_size, num_channels, freq_bands).permute(1, 0, 2, 3)
        # output的形状: [batch_size, time_len, num_channels, freq_bands]
        
        return output

class SpectralAttention(nn.Module):
    """
    频谱注意力模块。
    它的目标是学习哪些频谱带（如Alpha, Beta波）对于当前任务最具判别力。
    """
    def __init__(self, time_len, num_channels, num_heads, unified_embed_dim, use_rope=False):
        super().__init__()
        # embed_dim是每个“频谱带”向量的特征长度，即时间维度和通道维度的乘积
        self.original_dim = time_len * num_channels
        self.embed_dim = unified_embed_dim
        self.qkv_proj = nn.Linear(self.original_dim, self.embed_dim) 
        self.out_proj = nn.Linear(self.embed_dim, self.original_dim)
        self.mha = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads, batch_first=False)
        self.norm = nn.LayerNorm(self.original_dim)
        
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RoPE(self.embed_dim)

    def forward(self, x):
        # x的初始形状: [batch_size, time_len, num_channels, freq_bands]
        batch_size, time_len, num_channels, freq_bands = x.shape
        
        # 将频谱维度 视为序列长度。
        # 将x的形状从 变为 [freq_bands, batch_size, time_len * num_channels]
        x_reshaped = x.permute(3, 0, 1, 2).reshape(freq_bands, batch_size, self.original_dim)
        # x_reshaped的形状: [freq_bands, batch_size, embed_dim]
        x_proj = self.qkv_proj(x_reshaped)

        if self.use_rope:
            x_proj = self.rope(x_proj)

        attn_output, _ = self.mha(query=x_proj, key=x_proj, value=x_proj)
        # attn_output的形状: [freq_bands, batch_size, embed_dim]
        attn_output = self.out_proj(attn_output)

        x_norm = self.norm(x_reshaped + attn_output)
        # x_norm的形状: [freq_bands, batch_size, embed_dim]
        
        # 恢复原始形状
        output = x_norm.reshape(freq_bands, batch_size, time_len, num_channels).permute(1, 2, 3, 0)
        # output的形状: [batch_size, time_len, num_channels, freq_bands]
        
        return output

# ------------------- 主模型 -------------------

class EEGEmotionSexClassifier(nn.Module):
    """
    主模型，用于情绪和性别分类。
    它按顺序集成了通道、时间和频谱注意力模块，并有两个分类头。
    """
    def __init__(self, config):
        super().__init__()
        
        # 1. 初始化三个独立的注意力模块
        # 顺序：通道 -> 时间 -> 频谱
        self.channel_attn = ChannelAttention(
            time_len=config.TIME_LEN, 
            freq_bands=config.FREQ_BANDS, 
            num_heads=config.NUM_HEADS,
            unified_embed_dim=config.UNIFIED_EMBED_DIM,
            use_rope=config.USE_ROPE
        )
        self.temporal_attn = TemporalAttention(
            num_channels=config.NUM_CHANNELS, 
            freq_bands=config.FREQ_BANDS, 
            num_heads=config.NUM_HEADS,
            unified_embed_dim=config.UNIFIED_EMBED_DIM,
            use_rope=config.USE_ROPE
        )
        self.spectral_attn = SpectralAttention(
            time_len=config.TIME_LEN, 
            num_channels=config.NUM_CHANNELS, 
            num_heads=config.NUM_HEADS,
            unified_embed_dim=config.UNIFIED_EMBED_DIM,
            use_rope=config.USE_ROPE
        )
        
        # 2. 计算展平后的特征维度
        # 经过三个注意力模块后，数据的维度仍然是
        flattened_dim = config.TIME_LEN * config.NUM_CHANNELS * config.FREQ_BANDS
        
        # 3. 定义分类头
        # 这是一个共享的特征提取层，将注意力处理后的特征映射到一个更低的维度
        self.shared_classifier = nn.Sequential(
            nn.Linear(flattened_dim, config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            # 在全连接层后添加Dropout可以有效防止过拟合。
            # 它在训练期间随机将一部分神经元的输出置为零，迫使网络学习更鲁棒的特征。
            # # Dropout(0.5), # 过拟合风险点，如果模型在训练集上表现很好但在验证集上很差，可以取消此行注释
        )
        
        # 情绪分类的专用输出层
        self.emotion_classifier = nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_EMOTION_CLASSES)
        # 性别分类的专用输出层
        self.sex_classifier = nn.Linear(config.CLASSIFIER_HIDDEN_DIM, config.NUM_SEX_CLASSES)

    def forward(self, x):
        # x的初始形状: [batch_size, time_len, num_channels, freq_bands]
        
        # 依次通过三个注意力模块
        x = self.channel_attn(x)
        # x的形状: [batch_size, time_len, num_channels, freq_bands] (维度不变，但内容已被注意力加权)
        
        x = self.temporal_attn(x)
        # x的形状: [batch_size, time_len, num_channels, freq_bands]
        
        x = self.spectral_attn(x)
        # x的形状: [batch_size, time_len, num_channels, freq_bands]
        
        # 将4D张量展平为2D，以便送入全连接层
        # .view(x.size(0), -1) 是一种常见的展平方式，-1表示自动计算该维度的大小
        x_flat = x.reshape(x.size(0), -1)
        # x_flat的形状: [batch_size, time_len * num_channels * freq_bands]
        
        # 通过共享的特征提取层
        shared_features = self.shared_classifier(x_flat)
        # shared_features的形状: [batch_size, CLASSIFIER_HIDDEN_DIM]
        
        # 分别通过两个分类头，得到两个任务的原始输出
        emotion_logits = self.emotion_classifier(shared_features)
        # emotion_logits的形状: [batch_size, NUM_EMOTION_CLASSES]
        
        sex_logits = self.sex_classifier(shared_features)
        # sex_logits的形状: [batch_size, NUM_SEX_CLASSES]
        
        return emotion_logits, sex_logits

