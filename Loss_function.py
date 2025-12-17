import torch
import torch.nn as nn
import numpy as np

class MultiTaskLoss(nn.Module):
    def __init__(self, config, method='simple', alpha1=1.0, alpha2=1.0, gamma=2.0):
        """
        多任务损失函数封装
        Args:
            config: 配置对象
            method: 损失计算方法 ('simple', 'init', 'prior', 'dynamic', 'generalized')
            alpha1, alpha2: 简单加权平均的权重
            gamma: 广义平均损失的超参数
        """
        super().__init__()
        self.method = method
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.gamma = gamma
        
        # 使用默认先验分布（5类情绪均匀分布，2类性别均匀分布）
        self.prior_loss_emotion = self._compute_prior_loss(config.NUM_EMOTION_CLASSES)
        self.prior_loss_sex = self._compute_prior_loss(config.NUM_SEX_CLASSES)
        
        # 初始化动态权重存储
        self.register_buffer('init_loss_emotion', torch.tensor(1.0))
        self.register_buffer('init_loss_sex', torch.tensor(1.0))
        self.initialized = False

    def _compute_prior_loss(self, num_classes):
        """计算均匀先验分布的熵"""
        # 均匀分布的熵: -sum(1/n * log(1/n)) = log(n)
        return np.log(num_classes)

    def forward(self, loss_emotion, loss_sex):
        if not self.initialized and self.method == 'init':
            # 首次运行时保存初始损失
            self.init_loss_emotion.copy_(loss_emotion.detach())
            self.init_loss_sex.copy_(loss_sex.detach())
            self.initialized = True

        if self.method == 'simple':
            return self.alpha1 * loss_emotion + self.alpha2 * loss_sex
        
        elif self.method == 'init':
            return (loss_emotion / (self.init_loss_emotion + 1e-8) + 
                   loss_sex / (self.init_loss_sex + 1e-8))
        
        elif self.method == 'prior':
            return (loss_emotion / self.prior_loss_emotion + 
                   loss_sex / self.prior_loss_sex)
        
        elif self.method == 'dynamic':
            return (loss_emotion / (loss_emotion.detach() + 1e-8) + 
                   loss_sex / (loss_sex.detach() + 1e-8))
        
        elif self.method == 'generalized':
            return (loss_emotion ** self.gamma + 
                   loss_sex ** self.gamma) ** (1/self.gamma)
        
        else:
            raise ValueError(f"Unknown loss method: {self.method}")

    @staticmethod
    def compute_grad_norm(loss, model):
        """计算损失对模型参数的梯度范数"""
        grads = torch.autograd.grad(loss, model.parameters(), 
                                  create_graph=False, retain_graph=True, allow_unused=True)
        grad_norm = torch.sqrt(sum(g.norm()**2 for g in grads if g is not None))
        return grad_norm

    def grad_normalized_loss(self, loss_emotion, loss_sex, model):
        """梯度归一化损失"""
        with torch.no_grad():
            grad_norm1 = self.compute_grad_norm(loss_emotion, model)
            grad_norm2 = self.compute_grad_norm(loss_sex, model)
        
        # 防止除零
        grad_norm1 = torch.clamp(grad_norm1, min=1e-8)
        grad_norm2 = torch.clamp(grad_norm2, min=1e-8)
        
        return loss_emotion / grad_norm1 + loss_sex / grad_norm2

    
