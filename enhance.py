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
    
class EncoderEv(nn.Module):
    def __init__(self, embedding_dim, input_dim=50):
        super(EncoderEv, self).__init__()
        self.L1=nn.Sequential(
            nn.Linear(input_dim,25),
            nn.LeakyReLU(),
            nn.Linear(25, embedding_dim),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        y=self.L1(x)
        # y = self.L1(x.to(self.L1[0].weight.dtype))

        return y


class DecoderDv(nn.Module):
    def __init__(self, embedding_dim, output_dim=50):
        super(DecoderDv, self).__init__()
        self.L2=nn.Sequential(
            nn.Linear(embedding_dim, 25),
            nn.LeakyReLU(),
            nn.Linear(25, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.L2(x)

        return x
    
class TeacherModel(nn.Module):
    def __init__(self, input_dim,  output_dim, embedding_dim=64):
        super(TeacherModel, self).__init__()
        self.Ev = EncoderEv(embedding_dim, input_dim)
        self.Dv = DecoderDv(embedding_dim, output_dim)
    def forward(self,r,f):
        s = self.Ev(r)
        z = self.Ev(f)
        y = self.Dv(z)
        return s,z,y

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

class DecoderDs(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(DecoderDs, self).__init__()
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
        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).float()
        self.student_decoder_ds = DecoderDs(ev_latent_dim, dv_output_dim).float()

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
        non_empty_rows = np.any(reshaped_data != '', axis=1)
        filtered_arr = reshaped_data[non_empty_rows]
        reshaped_data = np.asarray(filtered_arr, dtype=np.float64)
        averaged_data[i] = np.nanmean(reshaped_data, axis=0)  # Compute column-wise average
    averaged_df = pd.DataFrame(averaged_data, columns=None)
    return averaged_df

latent_dim = 64
input_dim = 50
hidden_dim = 300
output_dim = 50
embedding_dim = 10

# 创建自编码器实例
# autoencoder = Autoencoder()
# model = StudentModel(output_dim, input_dim, hidden_dim, latent_dim)
model= TeacherModel(input_dim, input_dim, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
criterion1 = nn.MSELoss()
criterion2 = nn.BCEWithLogitsLoss()

path_in = "./data/inout/CSI_wave_in_2m2.csv"
path_out = "./data/inout/CSI_wave_out_2m2.csv"

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
batch_size = 256
# np.random.shuffle(data)  # 打乱data顺序，体现随机

in_real = data[:, :50]
out_fake = data[:, 50:]
original_length = in_real.shape[0]

num_epochs = 3000
for epoch in range(num_epochs):
    random_indices = np.random.choice(original_length, size=batch_size, replace=False)
    raw_re = torch.from_numpy(in_real[random_indices, :]).float()
    raw_fa = torch.from_numpy(out_fake[random_indices, :]).float()
    # re = raw_re.view(batch_size, 5, 10)  # .shape(batch_size,28,1,1)
    # fa = raw_fa.view(batch_size, 5, 10)
    re = raw_re.view(batch_size, 50)  
    fa = raw_fa.view(batch_size, 50)

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
