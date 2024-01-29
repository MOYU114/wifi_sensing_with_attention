# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:04:17 2023

@author: Administrator
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import csv
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

if (torch.cuda.is_available()):
    print("Using GPU for training.")
    device = torch.device("cuda:0")
else:
    print("Using CPU for training.")
    device = torch.device("cpu")

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x += residual
        x = self.relu(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dim = in_features // n_heads

        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)

        self.fc_out = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        # Linearly project input
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Split the input into multiple heads
        query = query.view(x.shape[0], -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(x.shape[0], -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(x.shape[0], -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate scaled dot-product attention
        energy = torch.einsum('nqhd,nkhd->nhqk', [query, key]) / (self.head_dim ** 0.5)

        # Apply attention mask if needed

        # Normalize attention scores
        attention = torch.nn.functional.softmax(energy, dim=-1)

        # Apply dropout
        attention = self.dropout(attention)

        # Weighted sum of values
        x = torch.einsum('nhql,nlhd->nqhd', [attention, value]).permute(0, 2, 1, 3).contiguous()

        # Combine heads
        x = x.view(x.shape[0], -1, self.n_heads * self.head_dim)

        # Linearly project the outputs
        x = self.fc_out(x)

        # Apply dropout and residual connection
        x = self.dropout(x)
        x = self.norm(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, in_features, n_heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(in_features, n_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(in_features, 4*in_features),
            nn.ReLU(),
            nn.Linear(4*in_features, in_features)
        )
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attention_output = self.attention(x)
        x = x + attention_output
        x = self.dropout(x)
        x = self.norm(x)

        feedforward_output = self.feedforward(x)
        x = x + feedforward_output
        x = self.dropout(x)
        x = self.norm(x)

        return x

class MyModel(nn.Module):
    def __init__(self, n_heads=8):
        super(MyModel, self).__init__()

        # Upsample layer
        self.upsample = UpsampleBlock(1, 64)  # Assuming 1 input channel

        # Convolutional layers
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.residual1 = ResidualBlock(64, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.residual2 = ResidualBlock(256, 256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.residual3 = ResidualBlock(512, 512)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Transformer layer
        self.transformer = TransformerBlock(512*14*12, n_heads)

        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU()
        )

        # Output layer
        self.output_layer = nn.Linear(128, 2*14)

    def forward(self, x):
        # Upsample
        x = self.upsample(x)

        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.residual2(x)
        x = self.conv5(x)
        x = self.residual3(x)

        # Flatten
        x = x.repeat(1, 1, 1, 3)  # Repeat the last layer 3 times
        x = self.flatten(x)

        # Transformer
        x = self.transformer(x)

        # Bottleneck layer
        x = x.view(-1, 512, 17, 12)
        x = self.bottleneck(x)

        # Output layer
        x = torch.mean(x, dim=(2, 3))  # Average pooling over spatial dimensions
        x = self.output_layer

def reshape_and_average(x):
    num_rows = x.shape[0]
    averaged_data = np.zeros((num_rows, 50))
    for i in trange(num_rows):
        row_data = x.iloc[i].to_numpy()
        reshaped_data = row_data.reshape(-1, 50)
        reshaped_data = pd.DataFrame(reshaped_data).replace({None: np.nan}).values
        reshaped_data = pd.DataFrame(reshaped_data).dropna().values
        non_empty_rows = np.any(reshaped_data != '', axis=1)
        filtered_arr = reshaped_data[non_empty_rows]
        reshaped_data = np.asarray(filtered_arr, dtype=np.float64)
        averaged_data[i] = np.nanmean(reshaped_data, axis=0)  # Compute column-wise average
    averaged_df = pd.DataFrame(averaged_data, columns=None)
    return averaged_df
        
# Assuming you have a dataset and dataloader for training
# train_dataset = YourTrainingDataset(...)
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 假设你有一个自定义的数据集类，名为YourTrainingDataset
class YourTrainingDataset(Dataset):
    def __init__(self, data):
        self.data = torch.Tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
CSI_test = "./data/static/data/device/CSI_mov_6C.csv"
Video_test = "./data/static/data/device/points_mov_6C.csv"
with open(CSI_test, "r") as csvfilee:
    csvreadere = csv.reader(csvfilee)
    data2 = list(csvreadere)  # 将读取的数据转换为列表
csi_test = pd.DataFrame(data2)
test_bb = reshape_and_average(csi_test)
test_bb = test_bb.values.astype('float32')
csi_test = test_bb / np.max(test_bb)
video_test = pd.read_csv(Video_test, header=None)
video_test = video_test.values.astype('float32')
video_test = video_test.reshape(len(video_test), 14, 2)
video_test = video_test / [1280, 720]
video_test = video_test.reshape(len(video_test), -1)
data = np.hstack((Video_test, CSI_test))

# b = torch.from_numpy(csi_test).double()
# b = b.view(len(b),int(len(csi_test[0])/10),10)
# g = torch.from_numpy(video_test).double()

original_length = video_test.shape[0]

# 创建伪训练数据集实例
# train_dataset = YourTrainingDataset(data)

batch_size = 32
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = MyModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lambda_lr = lambda epoch: 0.001 * (0.5 ** (epoch // 10))
scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
num_epochs = 20

for epoch in range(num_epochs):
    random_indices = np.random.choice(original_length, size=batch_size, replace=False)
    f = torch.from_numpy(video_test[random_indices, :]).to(device)#.double()
    a = torch.from_numpy(csi_test[random_indices, :]).to(device)#.double()
    f = f.view(batch_size, 2, 14)
    a = a.view(batch_size, 1, 5, 10)
    
    optimizer.zero_grad()
    outputs = model(a)
    loss = criterion(outputs, f)
    loss.backward()
    optimizer.step()
    scheduler.step()

# for epoch in range(num_epochs):
#     model.train()

#     for batch_idx, (inputs, targets) in enumerate(train_dataloader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#     # Learning rate scheduling step
#     scheduler.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

print("Training complete.")

