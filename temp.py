# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 16:35:23 2023

@author: Administrator
"""
import math
import csv
import os

from tqdm import tqdm, trange
import torch, gc
import torch.nn as nn
import numpy as np
import pandas as pd
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
    def __init__(self, embedding_dim, input_dim=28):
        super(EncoderEv, self).__init__()
        self.L1=nn.Sequential(
            nn.Linear(input_dim,14),
            nn.LeakyReLU(),
            nn.Linear(14, embedding_dim),
            nn.LeakyReLU(),
        )
    def forward(self, x):
        x=self.L1(x)
        # print(x.shape)
        return x


class DecoderDv(nn.Module):
    def __init__(self, embedding_dim, output_dim=28):
        super(DecoderDv, self).__init__()
        self.L2=nn.Sequential(
            nn.Linear(embedding_dim, 14),
            nn.LeakyReLU(),
            nn.Linear(14, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.L2(x)

        return x

'''
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
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.f0(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.out(x)
        return x

'''
# Student Model Components
class EncoderEs(nn.Module):
    def __init__(self, embedding_dim, csi_input_dim=50):
        super(EncoderEs, self).__init__()
        self.L3 = nn.Sequential(
            nn.Linear(csi_input_dim, 25),
            nn.LeakyReLU(),
            nn.Linear(25, embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.L3(x)
        return x


# 换种思路，不是填充长度，而是求平均值，把每行n个50变成一个50，对应video的每一帧points

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

class TeacherModel(nn.Module):
    def __init__(self, input_dim,  output_dim, embedding_dim=64):
        super(TeacherModel, self).__init__()
        self.Ev = EncoderEv(embedding_dim, input_dim)
        self.Dv = DecoderDv(embedding_dim, output_dim)
        #self.selayer = SELayer(embedding_dim).double()
    def forward(self,f):
        z = self.Ev(f)
        #z_atti = self.selayer(z)
        y = self.Dv(z)
        # test = self.teacher_discriminator_c(f)
        return z,y
class TeacherStudentModel(nn.Module):
    def __init__(self, csi_input_dim,input_dim, embedding_dim, output_dim):
        super(TeacherStudentModel, self).__init__()
        self.Ev = EncoderEv(embedding_dim, input_dim)
        self.Dv = DecoderDv(embedding_dim, output_dim)
        self.Es = EncoderEs(embedding_dim, csi_input_dim)
        self.Ds = self.Dv
        #self.selayer = SELayer(embedding_dim).double()
    def forward(self,f, a):
        z = self.Ev(f)
        #z_atti = self.selayer(z)
        y = self.Dv(z)
        # test = self.teacher_discriminator_c(f)

        v = self.Es(a)
        #v_atti = self.selayer(v)
        # v_atti = v
        s = self.Ds(v)
        return z,y,v,s
class StudentModel(nn.Module):
    def __init__(self,csi_input_dim, embedding_dim, output_dim):
        super(StudentModel, self).__init__()
        self.Es = EncoderEs(embedding_dim, csi_input_dim)
        self.Ds = DecoderDv(embedding_dim, output_dim)
        #self.selayer = SELayer(embedding_dim).double()
    def forward(self,a):
        v = self.Es(a)
        #v_atti = self.selayer(v)
        # v_atti = v
        s = self.Ds(v)
        return s
    
# 通过关节点的制约关系得到wave，leg和stand的索引，然后返回相同数量的三种类别的索引
def group_list(frame_value):
    leg_index = []
    wave_index = []
    stand_index = []

    for i in range(len(frame_value)):
        if frame_value[i,9]-frame_value[i,5] < 60:
            wave_index.append(i)
        elif frame_value[i,26]-frame_value[i,20] > 180:
            leg_index.append(i)
        elif frame_value[i,26]-frame_value[i,20] < 110 and frame_value[i,9]-frame_value[i,5] > 140:
            stand_index.append(i)
        else:
            continue
        
    length_min = min(len(leg_index),len(stand_index))#len(wave_index),
    leg_index = leg_index[0:length_min*6]
    wave_index = wave_index[0:length_min*4]
    stand_index = stand_index[0:length_min]
    merged_index = leg_index + stand_index + wave_index
    return merged_index



# Training configuration
learning_rate = 0.001
beta1 = 0.5
beta2 = 0.999
teacher_weights = {"wadv": 0.5, "wY": 1.0}
student_weights = {"wV": 0.5, "wS": 1.0}

# Initialize models
csi_input_dim = 50
input_dim = 28#最后输出长度
n_features = 1
embedding_dim=7

teacher_model= TeacherModel(input_dim, input_dim, embedding_dim).to(device)
model = TeacherStudentModel(csi_input_dim,input_dim, embedding_dim, input_dim).to(device)

#points_in的准确率有80左右，merged只有64
CSI_PATH = "./data/CSI_out_static_wden2.csv"
Video_PATH = "./data/points_static.csv"
# CSI_test = "./data/CSI_test_legwave_25.csv"
# Video_test = "./data/points_test_legwave.csv"
CSI_OUTPUT_PATH = "./data/output/CSI_merged_output.csv"
Video_OUTPUT_PATH = "./data/output/points_merged_output.csv"

# 25条子载波否则会报错ParserError: Error tokenizing data. C error: Expected 75 fields in line 20, saw 100
# aa = pd.read_csv(CSI_PATH, header=None,low_memory=False,encoding="utf-8-sig")
with open(CSI_PATH, "r", encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)  # 将读取的数据转换为列表
aa = pd.DataFrame(data1)

ff = pd.read_csv(Video_PATH, header=None)
print("data has loaded.")


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


# bb = aa.apply(fillna_with_previous_values, axis=1)
# csi_test = csi_test.apply(fillna_with_previous_values, axis=1)

# array_length = 50
# result_array = np.zeros(array_length, dtype=int)

# for i in range(array_length):
#     if i % 2 == 0:
#         result_array[i] = 3 * (i // 2)
#     else:
#         result_array[i] = 3 * (i // 2) + 1
# if (os.path.exists('./data/static data/CSI_avg.csv') != True):
#     bb = reshape_and_average(aa)
#     np.savetxt('./data/static data/CSI_avg.csv', bb, delimiter=',')
# else:
#     with open('./data/static data/CSI_avg.csv', "r", encoding='utf-8-sig') as csvfile:
#         csvreader = csv.reader(csvfile)
#         data1 = list(csvreader)  # 将读取的数据转换为列表
#     bb = pd.DataFrame(data1)
bb = reshape_and_average(aa)
# bb = bb.iloc[:,0:25]
Video_train = ff.values.astype('float32')  # 共990行，每行28个数据，为关键点坐标，按照xi，yi排序
CSI_train = bb.values.astype('float32')

# merged_index = group_list(Video_train)
# Video_train = Video_train[merged_index,:]
# CSI_train = CSI_train[merged_index,:]

# csi_test = csi_test.values.astype('float32')
# video_test = video_test.values.astype('float32')

CSI_train = CSI_train / np.max(CSI_train)
Video_train = Video_train.reshape(len(Video_train), 14, 2)  # 分成990组14*2(x,y)的向量
Video_train = Video_train / [1280, 720]  # 输入的图像帧是1280×720的，所以分别除以1280和720归一化。
Video_train = Video_train.reshape(len(Video_train), -1)

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
data = np.hstack((Video_train, CSI_train))  # merge(V,S)
data_length = len(data)
train_data_length = int(data_length * 0.9)
test_data_length = int(data_length - train_data_length)

np.random.shuffle(data)  # 打乱data顺序，体现随机


f_train = data[0:train_data_length, 0:28]
a_train = data[0:train_data_length, 28:]
original_length = f_train.shape[0]

# 剩余作为测试
g = data[train_data_length:,0:28]
b = data[train_data_length:,28:]
#teacher_model_train
# teacher_f=pd.read_csv("./data/points_in.csv", header=None)
# teacher_f=teacher_f.values.astype('float32')
# teacher_f = teacher_f.reshape(len(teacher_f), 14, 2)
# teacher_f = teacher_f/ [1280, 720]  # 除以800和700归一化。
# teacher_f = teacher_f.reshape(len(teacher_f), -1)
teacher_f = Video_train
teacher_length = teacher_f.shape[0]

'''
# 训练模型 1000 lr=0.01
# selayer 800 0.0023
# CBAM 1000 0.0022
# 非注意力机制训练的模型结果不稳定，使用注意力机制的模型训练结果变化不大，考虑训练样本的多元
'''

'''
# 1. 原来的教师损失函数导致教师模型的损失不变，但是效果似乎比使用新的损失效果好。
# 2. 教师模型中使用z_atti = self.CBAM(z)和v_atti = self.CBAM(v)，但是在学生模型中不使用CBAM模块，最终loss更低，比在学生模型中也使用该模块效果要好；
# 3. 在教师模型中使用selayer似乎比使用CBAM模块效果要好。
# 4. 教师模型和学生模型中都使用selayer似乎效果不错。
# 5. 在变化不大的素材里，过多的使用注意力机制会导致输出结果趋于一个取平均的状态
'''

optimizer = torch.optim.Adam(model.Es.parameters(), lr=learning_rate, betas=(beta1, beta2))
teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=learning_rate, betas=(beta1, beta2))
criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss(reduction='sum')
teacher_num_epochs = 1500
teacher_batch_size = 128
num_epochs = 2000
batch_size = 128
#teacher_training
for epoch in range(teacher_num_epochs):

    random_indices = np.random.choice(teacher_length, size=teacher_batch_size, replace=False)
    f = torch.from_numpy(teacher_f[random_indices,:]).float()
    f = f.view(teacher_batch_size, len(f_train[0]))  # .shape(batch_size,28)

    if (torch.cuda.is_available()):
        f = f.cuda()

    with autocast():
        teacher_optimizer.zero_grad()
        seq_true = f
        z, seq_pred = teacher_model(f)
        teacher_loss = criterion1(seq_pred, seq_true)
        # 计算梯度
        teacher_loss.backward()
        # 更新模型参数
        teacher_optimizer.step()
        # print(f"{teacher_loss.item():.4f}")

    # 打印训练信息
    print(
        f"teacher_training:Epoch [{epoch + 1}/{teacher_num_epochs}], Teacher Loss: {teacher_loss.item():.4f}")

# loss_values = np.array(loss_values)   #把损失值变量保存为numpy数组
model.Dv.load_state_dict(teacher_model.Dv.state_dict())
model.Ev.load_state_dict(teacher_model.Ev.state_dict())

for epoch in range(num_epochs):

    random_indices = np.random.choice(original_length, size=batch_size, replace=False)
    f = torch.from_numpy(f_train[random_indices, :]).float()
    a = torch.from_numpy(a_train[random_indices, :]).float()
    f = f.view(batch_size, len(f_train[0]))
    a = a.view(batch_size, len(a_train[0]))

    if (torch.cuda.is_available()):
        f = f.cuda()
        a = a.cuda()
    with autocast():
        optimizer.zero_grad()
        if epoch==99:
            print()
        z, y, v, s = model(f, a)

        teacher_loss = criterion1(y, f)
        # 计算学生模型的损失
        student_loss = 0.5 * criterion1(s, y) + criterion1(v, z) + 0.6 * criterion1(s, f)

        total_loss = teacher_loss + student_loss

        # 计算梯度
        total_loss.backward()
        # 更新模型参数
        optimizer.step()
    print(
        f"training:Epoch [{epoch + 1}/{num_epochs}], Teacher Loss: {teacher_loss.item():.4f}, Student Loss: {student_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

# loss_values = np.array(loss_values)   #把损失值变量保存为numpy数组

# 查看训练集效果
f = f.cpu()
y = y.cpu()
s = s.cpu()
ynp = y.detach().numpy()
snp = s.detach().numpy()
fnp = f.detach().numpy()
ynp = ynp.squeeze()
snp = snp.squeeze()
fnp = fnp.squeeze()
np.savetxt("./data/output/CSI_merged_output_training.csv", ynp, delimiter=',')
np.savetxt("./data/output/points_merged_output_training.csv", snp, delimiter=',')
np.savetxt("./data/output/real_output_training.csv", fnp, delimiter=',')

# 参数传递
student_model = StudentModel(csi_input_dim, embedding_dim, input_dim).to(device)
student_model.Es.load_state_dict(model.Es.state_dict())
student_model.Ds.load_state_dict(model.Ds.state_dict())
# 在测试阶段只有学生模型的自编码器工作
#student_batch_size=10
#random_indices = np.random.choice(len(b), size=student_batch_size, replace=False)
#b= torch.from_numpy(b[random_indices, :]).float()
#g= torch.from_numpy(g[random_indices, :]).float()
with torch.no_grad():
    #b=b.view(student_batch_size, len(b[0]))
    b= torch.from_numpy(b).float()
    g= torch.from_numpy(g).float()
    b = b.to(device)
    g = g.to(device)

    r = student_model(b)

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

# # 计算正样本和负样本的输出
# output1, output2 = model(x1, x2)
# output1, output3 = model(x1, x3)

# # 计算欧氏距离损失
# euclidean_distance = F.pairwise_distance(output1, output2)
# print("正样本距离:", euclidean_distance.item())

# euclidean_distance = F.pairwise_distance(output1, output3)
# print("负样本距离:", euclidean_distance.item())

# # 计算对比损失
# margin = 1.0  # 对比损失的边界
# loss_contrastive = torch.mean((1 - F.cosine_similarity(output1, output2)) +
#                               (F.relu(margin - F.cosine_similarity(output1, output3))))
# print("对比损失:", loss_contrastive.item())

