# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:26:11 2023

@author: Administrator
"""
import torch
import torch.nn as nn
import pandas as pd
import csv
import numpy as np
from torch.cuda.amp import autocast

# 定义自编码器
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 50, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]
        h = h.unsqueeze(3)
        v = self.conv(h)
        # encoded = self.encoder(x)
        decoded = self.decoder(v)
        return decoded
    
class EncoderEs(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderEs, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.conv = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]  # Get the hidden state of the last LSTM unit
        h = h.unsqueeze(2).unsqueeze(3)  # Add dimensions for 2D convolution
        v = self.conv(h)
        # print(v.shape)
        return v

class DecoderDv(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(DecoderDv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(32, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(50)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        return x
    
class StudentModel(nn.Module):
    def __init__(self, dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim):
        super(StudentModel, self).__init__()
        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim).double()

    def forward(self, re, fa):
        s = self.student_encoder_es(re)
        z = self.student_encoder_es(fa)
        y = self.student_decoder_ds(z)
        return s, z, y
    
# 把读取的对应每一帧的CSI，取平均值为一个CSI，即从50*n维变为50维
def reshape_and_average(x):
    num_rows = x.shape[0]
    averaged_data = np.zeros((num_rows, 50))
    for i in range(num_rows):
        row_data = x.iloc[i].to_numpy()
        reshaped_data = row_data.reshape(-1, 50)
        reshaped_data = pd.DataFrame(reshaped_data).replace({None: np.nan}).values
        reshaped_data = pd.DataFrame(reshaped_data).dropna().values
        reshaped_data = np.asarray(reshaped_data, dtype=np.float64)
        averaged_data[i] = np.nanmean(reshaped_data, axis=0)  # Compute column-wise average
    averaged_df = pd.DataFrame(averaged_data, columns=None)
    return averaged_df

latent_dim = 64
input_dim = 10
hidden_dim = 300
output_dim = 50

# 创建自编码器实例
# autoencoder = Autoencoder()
model = StudentModel(output_dim, input_dim, hidden_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
criterion1 = nn.MSELoss()
criterion2 = nn.BCEWithLogitsLoss()

path_in = "./data/CSI_in_wave1.csv"
path_out = "./data/CSI_out_wave1.csv"

with open(path_in, "r") as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)  # 将读取的数据转换为列表
CSIin = pd.DataFrame(data1)
with open(path_out, "r") as csvfile:
    csvreader = csv.reader(csvfile)
    data2 = list(csvreader)  # 将读取的数据转换为列表
CSIout = pd.DataFrame(data2)

# CSIin = pd.read_csv(path_in, header=None)
# CSIout = pd.read_csv(path_out, header=None)

averagein = reshape_and_average(CSIin)
averageout = reshape_and_average(CSIout)

CSI_in = averagein / averagein.max()
CSI_out = averageout / averageout.max()
data = np.hstack((CSI_in, CSI_out))  # merge(V,S)
batch_size = 300
# np.random.shuffle(data)  # 打乱data顺序，体现随机

in_lable = data[:, :50]
out_fake = data[:, 50:]
original_length = in_lable.shape[0]

num_epochs = 150
for epoch in range(num_epochs):
    random_indices = np.random.choice(original_length, size=batch_size, replace=False)
    raw_re = torch.from_numpy(in_lable[random_indices, :]).double()
    raw_fa = torch.from_numpy(out_fake[random_indices, :]).double()
    re = raw_re.view(batch_size, 5, 10)  # .shape(batch_size,28,1,1)
    fa = raw_fa.view(batch_size, 5, 10)

    optimizer.zero_grad()
    real_latent, fake_latent, fake = model(re, fa)
    fake = fake.reshape(batch_size,50)

    # if (torch.cuda.is_available()):
    #     f = re.cuda()
    #     a = fa.cuda()
    # try:
    #     with autocast():
    #         real_latent, fake_latent, fake = model(re, fa)
    # except RuntimeError as exception:
    #     if "out of memory" in str(exception):
    #         print('WARNING: out of memory')
    #         if hasattr(torch.cuda, 'empty_cache'):
    #             torch.cuda.empty_cache()
    #         else:
    #             raise exception
     
    loss_latent = criterion1(real_latent, fake_latent)
    loss_gen = criterion1(raw_re, fake)
    loss_total = loss_latent + loss_gen
    
    # target = model.teacher_discriminator_c(f)
    # label = torch.ones_like(target)
    # real_loss = criterion2(target, label)
    # print(real_loss)

    # target2 = 1 - model.teacher_discriminator_c(y)
    # label2 = torch.ones_like(target2)
    # fake_loss = criterion2(target2, label2)
    # teacher_loss = criterion1(y, f) + 0.5 * (real_loss + fake_loss)
    # student_loss = 0.5 * criterion1(v, z) + criterion1(s, y)
    # total_loss = teacher_loss + student_loss
    # # loss_values.append(total_loss) #记录损失值

    loss_total.backward()
    optimizer.step()

    # 打印训练信息
    print(
        f"GANModel training:Epoch [{epoch + 1}/{num_epochs}], Latent Loss: {loss_latent.item():.6f}, Gen Loss: {loss_gen.item():.4f}")