# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:33:30 2024

@author: Administrator
"""
import math
import csv
import os

from tqdm import tqdm,trange
import torch, gc
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.init as init
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

if (torch.cuda.is_available()):
    print("Using GPU for training.")
    device = torch.device("cuda:0")
else:
    print("Using CPU for training.")
    device = torch.device("cpu")


# Teacher Model Components
class EncoderEv(nn.Module):
    def __init__(self, input_dim):
        super(EncoderEv, self).__init__()
        self.gen = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
            # nn.Conv2d(input_dim, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            # nn.LeakyReLU(),
            # nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            # nn.LeakyReLU(),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.gen(x)
        # print(x.shape)
        return x


class DecoderDv(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(DecoderDv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.LeakyReLU()
        self.deconv2 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()
        self.deconv3 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU()
        self.deconv4 = nn.ConvTranspose2d(64, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(28)
        self.relu = nn.LeakyReLU()
        # self.deconv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.LeakyReLU()
        # self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.relu = nn.LeakyReLU()
        # self.deconv3 = nn.ConvTranspose2d(32, 30, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(30)
        # self.relu = nn.LeakyReLU()
        # self.deconv4 = nn.ConvTranspose2d(30, output_dim, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(28)
        # self.relu1 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        # print(x.shape)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # 对应Excitation操作
        return x * y.expand_as(x)    

class TeacherModel_G(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, dv_output_dim):
        super(TeacherModel_G, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim).double()

        # self.CBAM = CBAM(ev_latent_dim).double()
        # self.Transformer = Transformer(ev_latent_dim).double()
        self.selayer = SELayer(ev_latent_dim).double()
    def forward(self, f):
        z = self.teacher_encoder_ev(f)
        z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z_atti)

        return y
    
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

def poseloss(real, fake):
    # 生成权重值数组
    weights = torch.zeros((28,))
    indices = [6, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 21, 24, 25, 26, 27]
    weights[indices] = 1.0

    # 将 Tensor 转换为一维张量
    real_flat = real.view(real.size(0), -1)
    fake_flat = fake.view(fake.size(0), -1)

    # 计算两个 Tensor 对应位置的差值的平方
    squared_diff = (real_flat - fake_flat) ** 2

    # 分别乘以权重值
    weighted_diff = squared_diff * weights

    # 求和
    result = torch.sum(weighted_diff, dim=1)  # 沿着第二个维度求和
    result_mean = torch.mean(result)
    return result_mean

ev_input_dim = 28
ev_latent_dim = 8
es_input_dim = 10
es_hidden_dim = 300
dv_output_dim = 28
# CSI_PATH = "./data/CSI_out_static_abb.csv"
# Video_PATH = "./data/points_static_abb.csv"
# CSI_test = "./data/static/data/device/CSI_armleft_device_test.csv"
# Video_test = "./data/static/data/device/points_armleft_device_test.csv"
# CSI_PATH = "./data/static/data/room/CSI_sta_C205_1.csv"
# Video_PATH = "./data/static/data/device/points_static.csv"
# CSI_PATH = "./data/CSI_move.csv"
# Video_PATH = "./data/points_move.csv"

CSI_PATH = "./data/static/data/device/CSI_static_6C.csv"
Video_PATH = "./data/static/data/device/points_mov_6C.csv"
CSI_test = "./data/CSI_mov_room.csv"
Video_test = "./data/points_mov_room.csv"
CSI_OUTPUT_PATH = "./data/output/CSI_merged_output.csv"
Video_OUTPUT_PATH = "./data/output/points_merged_output.csv"

#aa = pd.read_csv(CSI_PATH, header=None,low_memory=False,encoding="utf-8-sig")
with open(CSI_PATH, "r", encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)
aa = pd.DataFrame(data1)                             #读取CSI数据到aa
ff = pd.read_csv(Video_PATH, header=None)            #读取骨架关节点数据到ff
print("data has loaded.")

bb = reshape_and_average(aa)     #把多个CSI数据包平均为一个数据包，使一帧对应一个CSI数据包
Video_train = ff.values.astype('float32')
CSI_train = bb.values.astype('float32')

# merged_index = group_list(Video_train)
# Video_train = Video_train[merged_index,:]
# CSI_train = CSI_train[merged_index,:]

CSI_train = CSI_train / np.max(CSI_train)
Video_train = Video_train.reshape(len(Video_train), 14, 2)  # 分成990组14*2(x,y)的向量
Video_train = Video_train / [1280, 720]        #输入的图像帧是1280×720的，所以分别除以1280和720归一化。
Video_train = Video_train.reshape(len(Video_train), -1)

data = np.hstack((Video_train, CSI_train))
np.random.shuffle(data)
data_length = len(data)
train_data_length = int(data_length * 0.9)
test_data_length = int(data_length - train_data_length)

frame_train = data[0:train_data_length, 0:28]
csi_train = data[0:train_data_length, 28:78]
# a = torch.from_numpy(data[0:100,50:800])
# f = torch.from_numpy(data[0:100,0:50])
# f = f.view(100,50,1,1,1)
# a = a.view(100,50,10)
original_length = frame_train.shape[0]

#训练Teacher模型
LR_G = 0.002
# LR_D = 0.001
teacher_model_G=TeacherModel_G(ev_input_dim, ev_latent_dim, dv_output_dim).to(device)
# teacher_model_D=TeacherModel_D(ev_input_dim).to(device)
criterion1 = nn.MSELoss()
criterion2 = nn.BCELoss()
optimizer_G = torch.optim.Adam(teacher_model_G.parameters(), lr=LR_G)
# optimizer_D = torch.optim.Adam(teacher_model_D.parameters(), lr=LR_D)

# # 随机初始化生成器和鉴别器的参数
# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         init.xavier_uniform_(m.weight.data)  # 使用Xavier均匀分布初始化权重
#         if m.bias is not None:
#             init.constant_(m.bias.data, 0.1)  # 初始化偏置为0.1

# teacher_model_G.apply(weights_init)

torch.autograd.set_detect_anomaly(True)
Teacher_num_epochs = 650
teacher_batch_size = 256
epoch_losses0 = []
epoch_losses1 = []
for epoch in range(Teacher_num_epochs):
    random_indices = np.random.choice(original_length, size=teacher_batch_size, replace=False)
    f = torch.from_numpy(frame_train[random_indices, :]).double()
    f = f.view(teacher_batch_size, 28, 1, 1)  # .shape(batch_size,28,1,1)
    y = teacher_model_G(f)
    # if (torch.cuda.is_available()):
    #     f = f.cuda()
    # try:
    #     with autocast():
    #         z, y = teacher_model_G(f)
    # except RuntimeError as exception:
    #     if "out of memory" in str(exception):
    #         print('WARNING: out of memory')
    #         if hasattr(torch.cuda, 'empty_cache'):
    #             torch.cuda.empty_cache()
    #         else:
    #             raise exception
    # # 进行对抗学习
    # optimizer_D.zero_grad()
    
    # real_labels = torch.ones(teacher_batch_size, 1).double()
    # fake_labels = torch.zeros(teacher_batch_size, 1).double()
    
    # real_target = teacher_model_D.teacher_discriminator_c(f)
    # real_target = real_target.view(teacher_batch_size,1)
    # real_loss = criterion2(real_target, real_labels)
    
    # fake_target = teacher_model_D.teacher_discriminator_c(y)
    # fake_target = fake_target.view(teacher_batch_size,1)
    # fake_loss=criterion2(fake_target, fake_labels) #+ 1e-6 #防止log0导致结果为-inf
    
    # eps = 1e-8#平滑值，防止出现log0
    # # teacher_loss = torch.mean(torch.abs(real_target.mean(0) - fake_target.mean(0)))
    # # teacher_loss = -torch.mean(torch.log(real_target + eps) + torch.log(1 - fake_target + eps))
    
    # teacher_loss = real_loss + fake_loss
    
    # #训练鉴别器
    # teacher_loss.backward(retain_graph=True)
    # optimizer_D.step()
    
    # #训练生成器
    # optimizer_G.zero_grad()
    # real_output = teacher_model_D.teacher_discriminator_c(f)
    # real_output = real_target.view(teacher_batch_size,1)
    # fake_output = teacher_model_D.teacher_discriminator_c(y)
    # fake_output = fake_output.view(teacher_batch_size,1)
    # # gen_loss = torch.mean(torch.abs(real_output.mean(0) - fake_output.mean(0)))
    # gen_loss = -torch.mean(torch.log(fake_output + eps))
    # # gen_loss = criterion(fake_output, real_labels)
    gen_loss = criterion1(y,f)
    # gen_loss = poseloss(y, f)
    gen_loss.backward()
    optimizer_G.step()
    # epoch_losses0.append(teacher_loss.item())
    epoch_losses1.append(gen_loss.item())
    # 打印训练信息
    print(
        f"TeacherModel training:Epoch [{epoch + 1}/{Teacher_num_epochs}], Teacher_G Loss: {gen_loss.item():.4f}")#,Teacher_D Loss: {teacher_loss.item():.4f}")
# plt.plot(epoch_losses0, label='dis Loss')
plt.plot(epoch_losses1, label='gen Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

f = f.cpu()
fnp = f.detach().numpy()
fnp=fnp.squeeze()
y = y.cpu()
ynp = y.detach().numpy()
ynp=ynp.squeeze()
np.savetxt("./data/output/CSI_merged_output_training.csv", ynp, delimiter=',')
np.savetxt("./data/output/real_output_training.csv", fnp, delimiter=',')

