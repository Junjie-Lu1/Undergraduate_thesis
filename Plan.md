# 基于性别-情绪双任务学习的情绪EEG识别


**关键词**：情绪识别，EEG，双任务学习，性别差异

## 数据集
[SEED系列 数据集](https://bcmi.sjtu.edu.cn/home/seed/index.html)；
[DEAP数据集](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/);

## 方法
采用双任务学习的框架，同时进行性别分类和情绪识别。具体步骤如下：
1. **数据预处理**：使用已经预处理好的特征，使用较长时间窗口（>=4秒），如果资源可以，也可以尝试自己进行数据预处理。

   [这是EEG的处理库](https://github.com/braindecode/braindecode/tree/master)
2. **模型设计**：复现论文中的模型作为基座，再设计两个分类头，分别进行性别分类和情绪识别。
   
    参考以下：
   - https://blog.csdn.net/mantoudamahou/article/details/134990549
   - https://zhuanlan.zhihu.com/p/667108069
   - https://github.com/eeyhsong/EEG-Conformer
   - https://github.com/braindecode/braindecode/tree/master
   - https://www.scholat.com/teamwork/showPostMessage.html?id=17215
     
   这块的模型挺多的，Github里面有很多仓库，还是要确认一下效果，有个CNN+BiLSTM+Attention的架构，据说能够有 99% 以上的准确率，这块暂时还没看到开源的，但感觉实现起来应该不会很难。
   还有个EEG-Conformer，是处理 EEG 的潜在向量表示的，后面接个头就可以预测分类。
4. **损失函数**设计联合损失函数，综合考虑性别分类和情绪识别的准确性，促进模型学习到更具代表性的特征，因为主要是情绪识别，所以可以设置情绪识别的损失权重稍微高一些。
5. **评估与验证**：
   - 通过交叉验证或独立测试集评估模型的性能(可以考虑改动版本subject-dependent cross trial的实验，即随机匹配一位男性和女性的数据放一起训练)，比较联合预测的任务是否优于单一任务的预测效果（准确率、F1-score等指标外，也需要讨论对表示学习的影响）。
   - 关于表示学习观察，可以把模型中间层的特征提取，用降维算法可视化，观察聚类效果，可以体现模型表示学习的质量，还可以使用类内、类间的距离这种量化指标。除此之外，也可以冻结前面层，专门训练其他分类器，和之前的对比，如果效果更好，说明是表示学习效果增强的功劳。

   - 可以进行鲁棒性测试，对数据增加些许噪声，观察模型的鲁棒性。
   
   - 分析任务相关性，衡量不同性别的EEG的差异，差异大则说明任务有用的可能性大。
6. **结果分析**：分析模型在性别和情绪两个任务上的表现，探讨性别差异对情绪识别的影响，并提出相应的改进建议。
