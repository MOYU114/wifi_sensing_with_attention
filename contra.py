# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:19:29 2024

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 创建一个简单的孪生网络结构
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # 三层的2D卷积层用于对输入的数据进行编码
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3) #shape(batch,256,44)

        # 全连接层
        self.fc1 = nn.Linear(11264, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 假设有10个动作类别

    def forward_one_branch(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1) # shape（batch，11264）
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one_branch(input1)
        output2 = self.forward_one_branch(input2)
        return output1, output2

# 构造数据集
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

# 创建一个虚构的数据集
num_samples = 100
data = np.random.rand(num_samples, 1, 50)
labels = np.random.randint(0, 10, num_samples)

# 定义超参数和模型
batch_size = 16
learning_rate = 0.001
num_epochs = 10

model = SiameseNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 创建数据加载器
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        optimizer.zero_grad()
        output1, output2 = model(batch_data, batch_data)

        loss = criterion(output1, batch_labels.squeeze()) + criterion(output2, batch_labels.squeeze())
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# 模型训练完成后，你可以使用训练好的模型进行相似性的评估，以及在新样本上进行预测。
