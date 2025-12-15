'''
代码功能说明：SEED-V数据集加载器

数据集信息：16名参与者，每人3个会话，每个会话15个试次

有个评分文件，记录每个试次的被试自我评分

label_dict = {0:'Disgust', 1:'Fear', 2:'Sad', 3:'Neutral', 4:'Happy'}   数字标签与情感类别的对应关系
'''

import os  # 导入os库，用于文件路径操作
import numpy as np  # 导入numpy库
import pickle  # 导入pickle库，用于反序列化数据
import torch  # 导入PyTorch库
from torch.utils.data import Dataset, DataLoader  # 从PyTorch导入数据集和数据加载器基类

sex_list = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0]
class SEEDVDataset(Dataset):  # 定义SEEDVDataset类，继承自PyTorch的Dataset
    def __init__(self, subject_list, trial_list, ROOT_DIR, FEATURE_DIR, SCORE_PATH=None, seg_len=None, skip=1):  # 类的初始化方法
        """
        Args:  # 参数说明
            subject_list: List of subject IDs  # 被试ID列表
            trial_list: List of trial indices  # 试次索引列表
            ROOT_DIR: Root directory path  # 根目录路径
            FEATURE_DIR: Feature directory path  # 特征数据目录路径
            seg_len: Segment length for slicing along time dimension. If None, use full sequence  # 时间维度切割的片段长度，为None时使用完整序列
            skip: Step size for sliding window (default: 1)  # 滑动窗口的步长（默认1）
        """
        self.X_data = []  # 初始化存储特征数据的列表
        self.Y_data = []  # 初始化存储标签数据的列表
        self.Y_lvl=[] # 初始化存储试次置信度评分的列表，lvl：level
        self.Y_sex = [] # 初始化性别列表
        self.seg_len = seg_len  # 保存片段长度参数
        self.skip = skip  # 保存滑动窗口步长参数
        self.SCORE_PATH = SCORE_PATH if SCORE_PATH is not None else os.path.join(ROOT_DIR, 'Scores.csv')  # 评分文件路径，默认在根目录下的Scores.csv
        
        Trial_scores = np.loadtxt(self.SCORE_PATH, delimiter=',').reshape((16, 3, 15)).reshape((16, 45))  # 加载评分文件，重塑为(16被试, 45试次)的数组
        # numpy的reshape逻辑是从尾部填充，所以根据Score.csv的排列规则，不能用（3，16，15）

        for subject in subject_list:  # 遍历每个被试
            data_npz = np.load(os.path.join(FEATURE_DIR, f'{subject}_123.npz'))  # 加载当前被试的特征文件（文件名格式：被试ID_123.npz）
            X = pickle.loads(data_npz['data'])  # 反序列化npz文件中的'data'字段，获取EEG特征数据，反序列是把字节流变为原始的数据结构
            Y = pickle.loads(data_npz['label'])  # 反序列化npz文件中的'label'字段，获取标签数据
            # print(X[1].shape)
            # print(X[2].shape)
            # print(X[40].shape)
            # print(X[42].shape)
            # print(Y[1].shape)
            # print(Y[2].shape)
            # print(Y[40].shape)
            # print(Y[42].shape)
            # print(Y)
            # X,Y是元组，里面的key是0，1，2，...，44，一共45个试次，每个试次的时长都不一样的，Y为了和X的形状一致，重复了很多次
            for trial in trial_list:  # 遍历每个试次
                trial_data = X[trial].reshape((-1, 62, 5))  # 将试次数据重塑为(时间步T, 62通道, 5特征)的形状
                
                if seg_len is not None and seg_len > 0:  # 如果指定了片段长度且长度为正数
                    # Slice into segments with sliding window  # 用滑动窗口切割序列
                    T = trial_data.shape[0]  # 获取时间步总数
                    for start_idx in range(0, T - seg_len + 1, skip):  # 滑动窗口的起始索引（从0开始，步长为skip，直到能完整取到seg_len长度）
                        segment = trial_data[start_idx:start_idx + seg_len]  # 截取片段，形状为(seg_len, 62, 5)
                        self.X_data.append(segment)  # 将片段添加到特征列表
                        self.Y_data.append(Y[trial][0], ) # 同一试次的所有片段标签相同，取第一个元素作为标签
                        self.Y_lvl.append(Trial_scores[int(subject)-1, trial])
                        self.Y_sex.append(sex_list[int(subject)-1])  # 添加当前被试当前试次的置信度评分
                else:  # 如果不切割（使用完整序列）
                    # Use full sequence  # 使用完整序列
                    self.X_data.append(trial_data)  # 添加完整试次数据到特征列表
                    self.Y_data.append(Y[trial][0])  # 添加试次标签
                    self.Y_lvl.append(Trial_scores[int(subject)-1, trial])  # 添加置信度评分
                    self.Y_sex.append(sex_list[int(subject)-1])

    def __len__(self):  # 定义数据集长度方法
        return len(self.X_data)  # 返回样本总数（特征列表的长度）

    def __getitem__(self, idx):  # 定义根据索引获取样本的方法
        # 将数据转换为PyTorch张量并返回，分别对应特征、标签、置信度评分
        return torch.tensor(self.X_data[idx], dtype=torch.float32), torch.tensor(self.Y_data[idx], dtype=torch.long), torch.tensor(self.Y_lvl[idx], dtype=torch.float32), torch.tensor(self.Y_sex[idx], dtype=torch.long)
    


if __name__ == "__main__":  
    ROOT_DIR = 'E:/毕业论文/EEGdataset/SEED-V/'  # 定义数据集根目录
    FEATURE_DIR = os.path.join(ROOT_DIR, 'EEG_DE_features/')  # 定义特征数据目录
    # 示例用法
    subject_list = ['8']  # 被试列表（此处仅包含被试'1'）
    trial_list = [1, 2, 3]  # 试次列表（此处包含试次1、2、3）
    dataset = SEEDVDataset(subject_list, trial_list, ROOT_DIR, FEATURE_DIR, 
                        seg_len=3, skip=1)  # 实例化数据集，片段长度3，步长1
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # 创建数据加载器，批大小4，打乱数据

    X, Y, Y_lvl, Y_sex= next(iter(dataloader))  # 获取一个批次的数据
    print(X.shape, Y.shape, Y_lvl.shape)  # 打印特征、标签、置信度评分的形状
    print(Y_sex.shape)