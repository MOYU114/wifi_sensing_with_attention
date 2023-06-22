# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:43:47 2023

@author: Administrator
"""
import math
from tqdm import tqdm, trange
import torch,gc
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
if(torch.cuda.is_available()):
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
            nn.Conv3d(input_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        x = self.gen(x)
        # print(x.shape)
        return x

class DecoderDv(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(DecoderDv, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(latent_dim, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()
        self.deconv3 = nn.ConvTranspose3d(32, output_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(28)
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
            nn.Conv3d(input_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2)
            )
        self.f2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2)
            )
        self.out = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.out(x)
        return x

# Student Model Components
class EncoderEs(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderEs, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.conv = nn.Conv3d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = h[-1]  # Get the hidden state of the last LSTM unit
        h = h.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Add dimensions for 3D convolution
        v = self.conv(h)
        # print(v.shape)
        return v
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=5000):
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
    def __init__(self,hidden_dim):
        super(Transformer,self).__init__()
        self.transformer = nn.Transformer(d_model=hidden_dim,nhead=16, num_encoder_layers=12)

        self.positional_encoding = PositionalEncoding(hidden_dim, dropout=0)
        self.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.Linear_to_1= nn.Linear(hidden_dim, 1)
    def forward(self, z,v):
        # 去除多余的维度
        z = z.squeeze()
        v = v.squeeze()

        # 对src和tgt进行编码
        src = self.Linear(v)
        tgt = self.Linear(z)

        out=[]
        # 转置输入张量
        pbar = tqdm(total=len(src))
        for i in range(len(src)):
            # 给src和tgt的token增加位置信息
            srci = self.positional_encoding(src[i])
            tgti = self.positional_encoding(tgt[i])

            # 将准备好的数据送给transformer
            outi = self.transformer(srci, tgti)
            outi=self.Linear_to_1(outi)
            #outi=self.softmax(outi)
            #min_max_scaler = MinMaxScaler()
            #outi = min_max_scaler.fit_transform(outi)
            out.append(outi)
            pbar.update(1)
        # 调整输出张量的形状
            gc.collect()
            torch.cuda.empty_cache()
        # 将列表中的所有张量拼接成一个大张量
        out = torch.stack(out)
        out = out.unsqueeze(-1)

        return out

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs, decoder_hidden):
        encoder_outputs = encoder_outputs.view(len(encoder_outputs),64)
        decoder_hidden = decoder_hidden.view(len(decoder_hidden),64)
        # print(encoder_outputs.shape)
        # decoder_hidden = decoder_hidden.unsqueeze(0)
        # print(decoder_hidden.shape)
        energy = self.linear(torch.cat((encoder_outputs, decoder_hidden), dim=1))
        attention_weights = self.softmax(energy)
        # print(attention_weights.shape)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        # print(context_vector.shape)
        return context_vector

class TeacherStudentModel(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim):
        super(TeacherStudentModel, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim).double()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()

        self.attention = Attention(ev_latent_dim).double()
        self.Transformer = Transformer(ev_latent_dim).double()
        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim).double() #分为了两个DS

    def forward(self, f, a):
        z = self.teacher_encoder_ev(f)
        y = self.teacher_decoder_dv(z)
        # test = self.teacher_discriminator_c(f)

        v = self.student_encoder_es(a)
        v_trans = self.Transformer(z,v)
        #v_trans = v
        s = self.student_decoder_ds(v_trans)

        return z, y, v_trans, s

class StudentModel(nn.Module):
    def __init__(self, dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim):
        super(StudentModel, self).__init__()
        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim).double()

    def forward(self, x):
        v = self.student_encoder_es(x)
        s = self.student_decoder_ds(v)
        return s

# # Loss functions
# def compute_teacher_loss(real_videos, fake_videos, discriminator_outputs_real, discriminator_outputs_fake):
#     lf = torch.mean(torch.log(discriminator_outputs_real))
#     ly = torch.mean(torch.log(1 - discriminator_outputs_fake))
#     ladv = torch.min(torch.stack(discriminator_outputs[:2]))  # Ev,Dv
#     nn.BCELoss()

#     return lf + ly + ladv

# def compute_student_loss(real_videos, fake_videos):
#     msev = torch.mean((real_videos - fake_videos[0]) ** 2)
#     mses = torch.mean((fake_videos[0] - fake_videos[1]) ** 2)

#     return msev + mses

# Training configuration
epochs = 1
learning_rate = 0.01
beta1 = 0.5
beta2 = 0.999
teacher_weights = {"wadv": 0.5, "wY": 1.0}
student_weights = {"wV": 0.5, "wS": 1.0}

# Initialize models
ev_input_dim = 28
ev_latent_dim = 64
es_input_dim = 15
es_hidden_dim = 400
dv_output_dim = 28

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = TeacherStudentModel(ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim).to(device)
student_model = StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim).to(device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
criterion1 = nn.MSELoss()
criterion2 = nn.BCELoss()

# aa=pd.read_csv('raw_data/CSI_ampt.csv',header=None)
# ff=pd.read_csv('raw_data/human_points.csv',header=None)
aa = pd.read_csv("CSI_ampt.csv", header=None)
ff = pd.read_csv("test2_result.csv", header=None)

def fillna_with_previous_values(s):
    non_nan_values = s[s.notna()].values
    # Gets the location of the missing value
    nan_indices = s.index[s.isna()]
    # Calculate the number of elements to fill
    n_fill = len(nan_indices)
    # Count the number of repetitions required
    n_repeat = int(np.ceil(n_fill / len(non_nan_values)))
    # Generate the fill value
    fill_values = np.tile(non_nan_values, n_repeat)[:n_fill]
    # Fill missing value
    s.iloc[nan_indices] = fill_values
    return s
aa=aa.apply(fillna_with_previous_values,axis=1)

# array_length = 50
# result_array = np.zeros(array_length, dtype=int)

# for i in range(array_length):
#     if i % 2 == 0:
#         result_array[i] = 3 * (i // 2)
#     else:
#         result_array[i] = 3 * (i // 2) + 1

Video_train = ff.values.astype('float32')#共990行，每行28个数据，为关键点坐标，按照xi，yi排序
# Video_train = Video_train[:,result_array]
CSI_train = aa.values.astype('float32')

CSI_train = CSI_train/np.max(CSI_train)
Video_train = Video_train.reshape(len(Video_train),14,2)#分成990组14*2(x,y)的向量
Video_train = Video_train/[1280,720] #输入的图像帧是1280×720的，所以分别除以1280和720归一化。
Video_train = Video_train.reshape(len(Video_train),-1)

# data = DataLoader(data, batch_size=500, shuffle=True)

# scaler = MinMaxScaler()
# Video_train = scaler.fit_transform(Video_train)
# CSI_train = scaler.fit_transform(CSI_train)

#Divide the training set and test set
# data_train, data_test = train_test_split(data, test_size=0.2, random_state=0)

data = np.hstack((Video_train,CSI_train))#merge(V,S)

#将数据转换为PyTorch张量
f_train = data[0:800,0:28]#只取了前800行数据
# f = torch.from_numpy(data[0:100,0:50])
# f = f.view(100,50,1,1,1)
a_train = data[0:800,28:778]
# a = torch.from_numpy(data[0:100,50:800])
# a = a.view(100,50,15)
original_length = f_train.shape[0]
batch_size = 200#如果调整训练集测试集大小，大小记得调整数值
#剩余作为测试
g = torch.from_numpy(data[800:900,0:28]).double()
b = torch.from_numpy(data[800:900,28:778]).double()
b = b.view(100,50,15)#如果调整训练集测试集大小，大小记得调整数值


# 训练模型
num_epochs = 5

for epoch in range(num_epochs):
    
    random_indices = np.random.choice(original_length, size=batch_size, replace=False)
    f = torch.from_numpy(f_train[random_indices,:]).double()
    a = torch.from_numpy(a_train[random_indices,:]).double()
    f = f.view(batch_size,28,1,1,1)#.shape(batch_size,28,1,1,1)
    a = a.view(batch_size,50,15)
    if(torch.cuda.is_available()):
        f = f.cuda()
        a = a.cuda()
    try:
        z, y, v, s = model(f, a)
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print('WARNING: out of memory')
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            else:
                raise exception
    # 计算教师模型的损失
    target = model.teacher_discriminator_c(f)
    label = torch.ones_like(target)
    real_loss = criterion2(target, label)
    # print(real_loss)
    
    target2 = 1-model.teacher_discriminator_c(y)
    label2 = torch.ones_like(target2)
    # label2 = torch.zeros_like(target2)
    fake_loss = criterion2(target2, label2)
    # print(fake_loss)
    teacher_loss = criterion1(y, f) + 0.5*(real_loss + fake_loss)

    # 计算学生模型的损失
    student_loss = 0.5*criterion1(v, z) + criterion1(s, y)

    # 计算总体损失
    total_loss = teacher_loss + student_loss

    # 反向传播和优化
    optimizer.zero_grad()
    # teacher_loss.backward()
    total_loss.backward()
    optimizer.step()

    # 打印训练信息
    print(f"Epoch [{epoch+1}/{num_epochs}], Teacher Loss: {teacher_loss.item():.4f}, Student Loss: {student_loss.item():.4f}")

# 在测试阶段只有学生模型的自编码器工作
with torch.no_grad():
    b = b.to(device)
    g = g.to(device)
    r = student_model(b)
    r = r.view(np.size(r,0),np.size(r,1))

    loss = criterion1(r, g)
    # df = pd.DataFrame(r.numpy())
    # df.to_excel("result.xls", index=False)
    print("loss:",loss)
