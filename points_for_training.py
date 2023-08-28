# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:43:47 2023

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
            nn.Conv2d(input_dim, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.gen(x)
        # print(x.shape)
        return x


class DecoderDv(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(DecoderDv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()
        self.deconv3 = nn.ConvTranspose2d(32, 30, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(30)
        self.relu = nn.LeakyReLU()
        self.deconv4 = nn.ConvTranspose2d(30, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(28)
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        return x


class DiscriminatorC(nn.Module):
    def __init__(self, input_dim):
        super(DiscriminatorC, self).__init__()
        self.f0 = nn.Sequential(
            nn.Conv2d(input_dim, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.out(x)
        return x


# Student Model Components
class EncoderEs(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderEs, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.conv = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h=self.relu(h)
        h = h[-1]  # Get the hidden state of the last LSTM unit
        h = h.unsqueeze(2).unsqueeze(3)  # Add dimensions for 2D convolution
        v = self.conv(h)
        # print(v.shape)
        return v


# for transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, hidden_dim):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=16, num_encoder_layers=12)

        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=0)
        self.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.Linear_to_1 = nn.Linear(hidden_dim, 1)

    def forward(self, z, v):
        # 去除多余的维度
        z = z.squeeze()
        v = v.squeeze()

        # 对src和tgt进行编码
        src = self.Linear(v)
        tgt = self.Linear(z)

        out = []
        # 转置输入张量
        pbar = tqdm(total=len(src))
        for i in range(len(src)):
            # 给src和tgt的token增加位置信息
            srci = self.positional_encoding(src[i])
            tgti = self.positional_encoding(tgt[i])

            # 将准备好的数据送给transformer
            outi = self.transformer(srci, tgti)
            outi = self.Linear_to_1(outi)
            # outi=self.softmax(outi)
            # min_max_scaler = MinMaxScaler()
            # outi = min_max_scaler.fit_transform(outi)
            out.append(outi)
            pbar.update(1)
            # 调整输出张量的形状
            gc.collect()
            torch.cuda.empty_cache()
        # 将列表中的所有张量拼接成一个大张量
        out = torch.stack(out)
        out = out.unsqueeze(-1)

        return out


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


class TeacherStudentModel(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim):
        super(TeacherStudentModel, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim).double()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()

        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = self.teacher_decoder_dv
        #self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim).double()

        self.CBAM = CBAM(ev_latent_dim).double()
        self.Transformer = Transformer(ev_latent_dim).double()
        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, f, a):
        z = self.teacher_encoder_ev(f)
        z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z_atti)
        # test = self.teacher_discriminator_c(f)

        v = self.student_encoder_es(a)
        v_atti = self.selayer(v)
        # v_atti = v
        s = self.teacher_decoder_dv(v_atti)

        return z, y, v, s

class StudentModel(nn.Module):
    def __init__(self, dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim):
        super(StudentModel, self).__init__()
        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim).double()
        self.CBAM = CBAM(ev_latent_dim).double()
        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, x):
        v = self.student_encoder_es(x)
        # v_atti=self.selayer(v)
        v_atti = v
        s = self.student_decoder_ds(v_atti)
        return s

class TeacherModel_G(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, dv_output_dim):
        super(TeacherModel_G, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim).double()

        self.CBAM = CBAM(ev_latent_dim).double()
        self.Transformer = Transformer(ev_latent_dim).double()
        self.selayer = SELayer(ev_latent_dim).double()
    def forward(self, f):
        z = self.teacher_encoder_ev(f)
        # z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z)

        return y
class TeacherModel_D(nn.Module):
    def __init__(self, ev_input_dim):
        super(TeacherModel_D, self).__init__()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()
    def forward(self, input):
        output = self.teacher_discriminator_c(input)
        return output

class TeacherModel(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim):
        super(TeacherModel, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim).double()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()
        self.CBAM = CBAM(ev_latent_dim).double()
        self.Transformer = Transformer(ev_latent_dim).double()
        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, f):
        z = self.teacher_encoder_ev(f)
        z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z_atti)
        return z, y

# 换种思路，不是填充长度，而是求平均值，把每行n个50变成一个50，对应video的每一帧points
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

def fillna_with_previous_values(s):
    non_nan_values = s[s.notna()].values
    nan_indices = s.index[s.isna()]
    n_fill = len(nan_indices)
    n_repeat = int(np.ceil(n_fill / len(non_nan_values)))
    fill_values = np.tile(non_nan_values, n_repeat)[:n_fill]
    s.iloc[nan_indices] = fill_values
    return s

# 通过关节点的制约关系得到wave，leg和stand的索引，然后返回相同数量的三种类别的索引
def group_list(frame_value):
    leg_index = []
    wave_index = []
    stand_index = []

    for i in range(len(frame_value)):
        if frame_value[i,9]-frame_value[i,5] < 50:
            wave_index.append(i)
        elif frame_value[i,26]-frame_value[i,20] > 160:
            leg_index.append(i)
        elif frame_value[i,26]-frame_value[i,20] < 100 and frame_value[i,9]-frame_value[i,5] > 150:
            stand_index.append(i)
        else:
            continue
        
    length_min = min(len(wave_index),len(leg_index),len(stand_index))
    leg_index = leg_index[0:length_min]
    wave_index = wave_index[0:length_min]
    stand_index = stand_index[0:length_min]
    merged_index = leg_index + wave_index + stand_index
    return merged_index

ev_input_dim = 28
ev_latent_dim = 64
es_input_dim = 10
es_hidden_dim = 300
dv_output_dim = 28

#points_in的准确率有60左右

CSI_PATH = "./data/CSI_in.csv"
Video_PATH = "./data/points_in.csv"
CSI_test = "./data/CSI_test_legwave_25.csv"
Video_test = "./data/points_test_legwave.csv"
CSI_OUTPUT_PATH = "./data/output/CSI_merged_output.csv"
Video_OUTPUT_PATH = "./data/output/points_merged_output.csv"

#aa = pd.read_csv(CSI_PATH, header=None,low_memory=False,encoding="utf-8-sig")
with open(CSI_PATH, "r", encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)
aa = pd.DataFrame(data1)                             #读取CSI数据到aa
ff = pd.read_csv(Video_PATH, header=None)            #读取骨架关节点数据到ff
print("data has loaded.")

bb = reshape_and_average(aa)                        #把多个CSI数据包平均为一个数据包，使一帧对应一个CSI数据包
Video_train = ff.values.astype('float32')
CSI_train = bb.values.astype('float32')

# merged_index = group_list(Video_train)
# Video_train = Video_train[merged_index,:]
# CSI_train = CSI_train[merged_index,:]

CSI_train = CSI_train / np.max(CSI_train)
Video_train = Video_train.reshape(len(Video_train), 14, 2)  # 分成990组14*2(x,y)的向量
Video_train = Video_train / [1280, 720]            #输入的图像帧是1280×720的，所以分别除以1280和720归一化。
Video_train = Video_train.reshape(len(Video_train), -1)

data = np.hstack((Video_train, CSI_train))
np.random.shuffle(data)
data_length = len(data)
train_data_length = int(data_length * 0.9)
test_data_length = int(data_length - train_data_length)

f_train = data[0:train_data_length, 0:28]
a_train = data[0:train_data_length, 28:78]
# a = torch.from_numpy(data[0:100,50:800])
# f = torch.from_numpy(data[0:100,0:50])
# f = f.view(100,50,1,1,1)
# a = a.view(100,50,10)
original_length = f_train.shape[0]

# 剩余作为测试
g = torch.from_numpy(data[train_data_length:data_length,0:28]).double()
b = torch.from_numpy(data[train_data_length:data_length,28:78]).double()
b = b.view(len(b),int(len(a_train[0])/10),10)#输入的维度可能不同，需要对输入大小进行动态调整


# 记录损失值
# loss_values = []
# '''
#训练Teacher模型
LR_G = 0.001
LR_D = 0.001
teacher_model_G=TeacherModel_G(ev_input_dim, ev_latent_dim, dv_output_dim).to(device)
teacher_model_D=TeacherModel_D(ev_input_dim).to(device)
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(teacher_model_G.parameters(), lr=LR_G)
optimizer_D = torch.optim.Adam(teacher_model_D.parameters(), lr=LR_D)

# # 创建生成器和鉴别器实例
# generator = Generator(input_dim, output_dim)
# discriminator = Discriminator(output_dim)

# 随机初始化生成器和鉴别器的参数
def weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)  # 使用Xavier均匀分布初始化权重
        if m.bias is not None:
            init.constant_(m.bias.data, 0.1)  # 初始化偏置为0.1

teacher_model_G.apply(weights_init)
teacher_model_D.apply(weights_init)

# # 输出生成器和鉴别器的模型结构和参数
# print("Generator:")
# print(generator)
# print("Discriminator:")
# print(discriminator)

torch.autograd.set_detect_anomaly(True)
Teacher_num_epochs = 1000
teacher_batch_size = 128
for epoch in range(Teacher_num_epochs):
    random_indices = np.random.choice(original_length, size=teacher_batch_size, replace=False)
    f = torch.from_numpy(f_train[random_indices, :]).double()
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
    # 进行对抗学习
    optimizer_D.zero_grad()
    
    # real_labels = torch.ones(teacher_batch_size, 1).double()
    # fake_labels = torch.zeros(teacher_batch_size, 1).double()
    
    real_target = teacher_model_D.teacher_discriminator_c(f)
    real_target = real_target.view(teacher_batch_size,1)
    # real_loss = criterion(real_target, real_labels)
    
    fake_target = teacher_model_D.teacher_discriminator_c(y)
    fake_target = fake_target.view(teacher_batch_size,1)
    # fake_loss=criterion(fake_target, fake_labels) #+ 1e-6 #防止log0导致结果为-inf
    
    eps = 1e-8#平滑值，防止出现log0
    teacher_loss = torch.mean(torch.abs(real_target.mean(0) - fake_target.mean(0)))
    # teacher_loss = -torch.mean(torch.log(real_target + eps) + torch.log(1 - fake_target + eps))
    
    # teacher_loss = - real_loss + fake_loss
    
    #训练鉴别器
    teacher_loss.backward(retain_graph=True)
    optimizer_D.step()
    
    #训练生成器
    optimizer_G.zero_grad()
    real_output = teacher_model_D.teacher_discriminator_c(f)
    real_output = real_target.view(teacher_batch_size,1)
    fake_output = teacher_model_D.teacher_discriminator_c(y)
    fake_output = fake_output.view(teacher_batch_size,1)
    # gen_loss = torch.mean(torch.abs(real_output.mean(0) - fake_output.mean(0)))
    gen_loss = -torch.mean(torch.log(fake_output + eps))
    # gen_loss = criterion(fake_output, real_labels)
    
    gen_loss.backward()
    optimizer_G.step()
    # 打印训练信息
    print(
        f"TeacherModel training:Epoch [{epoch + 1}/{Teacher_num_epochs}], Teacher_G Loss: {gen_loss.item():.4f},Teacher_D Loss: {teacher_loss.item():.4f}")


# Training configuration
# 学习率scheduling;
learning_rate = 0.01
beta1 = 0.5
beta2 = 0.999
teacher_weights = {"wadv": 0.5, "wY": 1.0}
student_weights = {"wV": 0.5, "wS": 1.0}

# Initialize models
# 所有参数进行grid-search.

model = TeacherStudentModel(ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
criterion1 = nn.MSELoss()
criterion2 = nn.BCELoss()

model.teacher_encoder_ev.load_state_dict(teacher_model_G.teacher_encoder_ev.state_dict())
model.teacher_decoder_dv.load_state_dict(teacher_model_G.teacher_decoder_dv.state_dict())
model.teacher_discriminator_c.load_state_dict(teacher_model_D.teacher_discriminator_c.state_dict())

# 25条子载波否则会报错ParserError: Error tokenizing data. C error: Expected 75 fields in line 20, saw 100

# 50条子载波
# csi_test = pd.read_csv(CSI_test, header=None)
# 25条子载波
# with open(CSI_test, "r") as csvfilee:
#     csvreadere = csv.reader(csvfilee)
#     data2 = list(csvreadere)  # 将读取的数据转换为列表
# csi_test = pd.DataFrame(data2)
# print(csi_test.shape)

# video_test = pd.read_csv(Video_test, header=None)
# print(aa.shape)



# aa = aa.apply(fillna_with_previous_values, axis=1)
# csi_test = csi_test.apply(fillna_with_previous_values, axis=1)

# array_length = 50
# result_array = np.zeros(array_length, dtype=int)

# for i in range(array_length):
#     if i % 2 == 0:
#         result_array[i] = 3 * (i // 2)
#     else:
#         result_array[i] = 3 * (i // 2) + 1
# if(os.path.exists('./data/CSI_avg.csv')!=True):
#     bb = reshape_and_average(aa)
#     np.savetxt('./data/CSI_avg.csv', bb, delimiter=',')
# else:
#     with open('./data/CSI_avg.csv', "r", encoding='utf-8-sig') as csvfile:
#         csvreader = csv.reader(csvfile)
#         data1 = list(csvreader)  # 将读取的数据转换为列表
    # bb = pd.DataFrame(data1)
# bb = reshape_and_average(aa)   
# Video_train = ff.values.astype('float32')  # 共990行，每行28个数据，为关键点坐标，按照xi，yi排序
# CSI_train = bb.values.astype('float32')

# csi_test = csi_test.values.astype('float32')
# video_test = video_test.values.astype('float32')

# CSI_train = CSI_train / np.max(CSI_train)
# Video_train = Video_train.reshape(len(Video_train), 14, 2)  # 分成990组14*2(x,y)的向量
# Video_train = Video_train / [1280, 720]  # 输入的图像帧是1280×720的，所以分别除以1280和720归一化。
# Video_train = Video_train.reshape(len(Video_train), -1)

# csi_test = csi_test / np.max(csi_test)
# video_test = video_test.reshape(len(video_test), 14, 2)
# video_test = video_test / [1280, 720]
# video_test = video_test.reshape(len(video_test), -1)

# data = DataLoader(data, batch_size=500, shuffle=True)

# scaler = MinMaxScaler()
# Video_train = scaler.fit_transform(Video_train)
# CSI_train = scaler.fit_transform(CSI_train)

# Divide the training set and test set
# data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)
# data = np.hstack((Video_train, CSI_train))  # merge(V,S)
# data_length = len(data)
# train_data_length = int(data_length * 0.9)
# test_data_length = int(data_length - train_data_length)
# batch_size = 300
# np.random.shuffle(data)  # 打乱data顺序，体现随机

# 视频帧是20帧每秒，每秒取一帧数据进行训练，缓解站立数据过多对训练数据造成的不平衡
# f_train = data[19::20, 0:28]
# # f = torch.from_numpy(data[0:100,0:50])
# # f = f.view(100,50,1,1,1)
# a_train = data[19::20, 28:78]
# # a = torch.from_numpy(data[0:100,50:800])
# # a = a.view(100,50,10)
# original_length = f_train.shape[0]

# # 剩余作为测试
# g = torch.from_numpy(data[11::20,0:28]).double()
# b = torch.from_numpy(data[11::20,28:78]).double()
# b = b.view(len(b),int(len(a_train[0])/10),10)#输入的维度可能不同，需要对输入大小进行动态调整

# 训练模型 1000 lr=0.01
# selayer 800 0.0023
# CBAM 1000 0.0022
# 非注意力机制训练的模型结果不稳定，使用注意力机制的模型训练结果变化不大，考虑训练样本的多元


# 1. 原来的教师损失函数导致教师模型的损失不变，但是效果似乎比使用新的损失效果好。
# 2. 教师模型中使用z_atti = self.CBAM(z)和v_atti = self.CBAM(v)，但是在学生模型中不使用CBAM模块，最终loss更低，比在学生模型中也使用该模块效果要好；
# 3. 在教师模型中使用selayer似乎比使用CBAM模块效果要好。
# 4. 教师模型和学生模型中都使用selayer似乎效果不错。
# 5. 在变化不大的素材里，过多的使用注意力机制会导致输出结果趋于一个取平均的状态

num_epochs =1000
batch_size = 128
# arr_loss = np.
# 开始打印discrimination的参数
for epoch in range(num_epochs):
    random_indices = np.random.choice(original_length, size=batch_size, replace=False)
    f = torch.from_numpy(f_train[random_indices, :]).double()
    a = torch.from_numpy(a_train[random_indices, :]).double()
    f = f.view(batch_size, 28, 1, 1)  # .shape(batch_size,28,1,1)
    a = a.view(batch_size, int(len(a_train[0]) / 10), 10)

    optimizer.zero_grad()
    z, y, v, s = model(f, a)

    # if (torch.cuda.is_available()):
    #     f = f.cuda()
    #     a = a.cuda()
    # try:
    #     with autocast():
    #         z, y, v, s = model(f, a)
    # except RuntimeError as exception:
    #     if "out of memory" in str(exception):
    #         print('WARNING: out of memory')
    #         if hasattr(torch.cuda, 'empty_cache'):
    #             torch.cuda.empty_cache()
    #         else:
    #             raise exception
    # 计算教师模型的损失
    '''
    # target = model.teacher_discriminator_c(f)
    # label = torch.ones_like(target)
    # real_loss = criterion2(target, label)
    # # print(real_loss)

    # target2 = 1 - model.teacher_discriminator_c(y)
    # label2 = torch.ones_like(target2)
    #      #label2 = torch.zeros_like(target2)
    # fake_loss = criterion2(target2, label2)
    # # print(fake_loss)
    # teacher_loss = criterion1(y, f) + 0.5 * (real_loss + fake_loss)
    '''


    # eps = 1e-8#平滑值，防止出现log0
    # real_prob = model.teacher_discriminator_c(f)
    # fake_prob = model.teacher_discriminator_c(y)
    # d_loss = -torch.mean(torch.log(real_prob + eps) + torch.log(1 - fake_prob + eps))
    # g_loss = -torch.mean(torch.log(fake_prob + eps))
    # Ladv = d_loss + g_loss
    # teacher_loss = 0.5*Ladv+criterion1(f,y)
    teacher_loss = criterion1(f,y)

    #teacher_loss.backward()
    #optimizer.step()


    # 计算学生模型的损失
    student_loss =0.5 *criterion1(v, z) +criterion1(s, y)

    total_loss = teacher_loss + student_loss
    # optimizer.zero_grad()
    # 计算梯度
    total_loss.backward()
    # 更新模型参数
    optimizer.step()

    # 打印训练信息
    print(
        f"training:Epoch [{epoch + 1}/{num_epochs}], Teacher Loss: {teacher_loss.item():.4f}, Student Loss: {student_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

# loss_values = np.array(loss_values)   #把损失值变量保存为numpy数组

# # 查看训练集效果
# f = f.cpu()
# y = y.cpu()
# s = s.cpu()
# ynp = y.detach().numpy()
# snp = s.detach().numpy()
# fnp = f.detach().numpy()
# ynp=ynp.squeeze()
# snp=snp.squeeze()
# fnp=fnp.squeeze()
# np.savetxt("./data/output/CSI_merged_output_training.csv", ynp, delimiter=',')
# np.savetxt("./data/output/points_merged_output_training.csv", snp, delimiter=',')
# np.savetxt("./data/output/real_output_training.csv", fnp, delimiter=',')

# 参数传递
student_model = StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim).to(device)
student_model.student_encoder_es.load_state_dict(model.student_encoder_es.state_dict())
student_model.student_decoder_ds.load_state_dict(model.teacher_decoder_dv.state_dict())
# student_model.student_decoder_ds.load_state_dict(model.student_decoder_ds.state_dict())
# 在测试阶段只有学生模型的自编码器工作
with torch.no_grad():
    b = b.to(device)
    g = g.to(device)
    r = student_model(b)
    r = r.view(np.size(r, 0), np.size(r, 1))

    loss = criterion1(r, g)
    # df = pd.DataFrame(r.numpy())
    # df.to_excel("result.xls", index=False)
    print("loss:", loss)
    g = g.cpu()
    r = r.cpu()
    gnp = g.numpy()
    rnp = r.numpy()
    np.savetxt(Video_OUTPUT_PATH, gnp, delimiter=',')
    np.savetxt(CSI_OUTPUT_PATH, rnp, delimiter=',')
    

