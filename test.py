import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 导入我们自定义的模块
from config import Config
from model import EEGEmotionSexClassifier

def evaluate(model, data_loader, config):
    """
    评估模型在给定数据加载器上的性能。
    :param model: 训练好的模型实例
    :param data_loader: 测试数据的DataLoader
    :param config: 配置对象
    """
    # 将模型设置为评估模式
    # 这会关闭Dropout和BatchNorm等层在训练时的特定行为
    model.eval()
    
    # 初始化性能指标
    correct_emotion = 0
    correct_sex = 0
    total = 0
    
    # torch.no_grad() 是一个上下文管理器，在此代码块中，PyTorch不会计算梯度。
    # 这可以显著减少内存消耗，并加速计算，因为在评估时我们不需要梯度。
    with torch.no_grad():
        for inputs, labels_emotion, labels_sex in data_loader:
            # 将数据移动到指定设备
            inputs = inputs.to(config.DEVICE)
            labels_emotion = labels_emotion.to(config.DEVICE)
            labels_sex = labels_sex.to(config.DEVICE)
            
            # 前向传播，获取模型输出
            emotion_outputs, sex_outputs = model(inputs)
            
            # 获取预测结果
            # torch.max()返回两个值：最大值和最大值的索引。
            # 我们只关心索引，也就是预测的类别。
            # dim=1表示在类别维度上找最大值。
            _, predicted_emotion = torch.max(emotion_outputs.data, 1)
            _, predicted_sex = torch.max(sex_outputs.data, 1)
            
            # 统计总样本数
            total += labels_emotion.size(0)
            
            # 统计预测正确的样本数
            # .sum()会计算True值的数量（True被当作1，False被当作0）
            # .item()将只有一个元素的张量转换为Python标量
            correct_emotion += (predicted_emotion == labels_emotion).sum().item()
            correct_sex += (predicted_sex == labels_sex).sum().item()
    
    # 计算准确率
    emotion_accuracy = 100 * correct_emotion / total
    sex_accuracy = 100 * correct_sex / total
    
    print(f'模型在测试集上的精确度:')
    print(f'  - 情绪分类: {emotion_accuracy:.2f} %')
    print(f'  - 性别分类: {sex_accuracy:.2f} %')
    
    return emotion_accuracy, sex_accuracy

def main():
    # 1. 加载配置
    config = Config()
    print(f"使用设备: {config.DEVICE}")

    # 2. 准备测试数据 (这里使用模拟数据，您需要替换为您自己的测试数据)
    # --------------------------------------------------------------------
    # 假设我们有200个测试样本
    num_test_samples = 200
    mock_X_test = torch.randn(num_test_samples, config.TIME_LEN, config.NUM_CHANNELS, config.FREQ_BANDS)
    mock_Y_label_test = torch.randint(0, config.NUM_EMOTION_CLASSES, (num_test_samples,))
    mock_Y_sex_test = torch.randint(0, config.NUM_SEX_CLASSES, (num_test_samples,))

    test_dataset = TensorDataset(mock_X_test, mock_Y_label_test, mock_Y_sex_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    # --------------------------------------------------------------------

    # 3. 加载模型
    # 首先，需要实例化一个和保存时结构完全相同的模型
    model = EEGEmotionSexClassifier(config).to(config.DEVICE)
    
    # 然后，使用torch.load()加载保存的状态字典
    # map_location=torch.device(config.DEVICE) 确保即使模型是在CPU上保存的，也能加载到GPU上
    model.load_state_dict(torch.load('eeg_emotion_sex_model.pth', map_location=torch.device(config.DEVICE)))
    
    print("模型加载成功.")

    # 4. 开始评估
    evaluate(model, test_loader, config)

if __name__ == '__main__':
    main()
