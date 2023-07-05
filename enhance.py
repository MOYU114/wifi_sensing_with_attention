import math
import csv
import os
from PIL import Image

from tqdm import tqdm
import torch, gc
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast
from torch.utils.data import random_split, DataLoader,Dataset,RandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

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
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.gen(x)
        x = self.avgpool(x)
        # print(x.shape)
        return x

class DecoderDv(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(DecoderDv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(32, output_dim, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        return x

class DiscriminatorC(nn.Module):
    def __init__(self, input_dim):
        super(DiscriminatorC, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.out(x)
        return x
    
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
    
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
    

class GANModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(GANModel, self).__init__()
        self.teacher_encoder_ev = EncoderEv(input_dim).double()
        self.teacher_decoder_dv = DecoderDv(latent_dim, output_dim).double()
        self.teacher_discriminator_c = DiscriminatorC(input_dim).double()

        self.CBAM = CBAM(latent_dim).double()
        # self.Transformer = Transformer(latent_dim).double()
        # self.selayer = SELayer(latent_dim).double()
    def forward(self, f, a):
        z = self.teacher_encoder_ev(f)
        y = self.teacher_decoder_dv(z)
        # test = self.teacher_discriminator_c(f)

        return z, y


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



input_dim = 10
latent_dim = 64
output_dim = 50
learning_rate = 0.001

path_in = "./data/CSI_in_wave1.csv"
path_out = "./data/CSI_out_wave1.csv"

CSIin = pd.read_csv(path_in, header=None)
CSIout = pd.read_csv(path_out, header=None)

averagein = reshape_and_average(CSIin)
averageout = reshape_and_average(CSIout)

model = GANModel(input_dim, latent_dim, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion1 = nn.MSELoss()
