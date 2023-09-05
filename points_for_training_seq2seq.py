# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:43:47 2023

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
    def __init__(self, input_dim=28, embedding_dim=7):
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
    def __init__(self, embedding_dim=7, output_dim=28):
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
'''
class EncoderEs(nn.Module):
    def __init__(self, csi_input_dim=50, embedding_dim=7):
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
'''

class EncoderEs(nn.Module):
    def __init__(self, csi_input_dim=50, embedding_dim=7):
        super(EncoderEs, self).__init__()
        self.gru=nn.GRU(csi_input_dim, 25, num_layers=1,batch_first=True)
        self.relu=nn.LeakyReLU()
        self.L3 = nn.Sequential(
            nn.Linear(25, embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        output, h = self.gru(x)
        h = h[-1]
        h=self.relu(h)
        v = self.L3(h)
        v = v.squeeze()
        return v
class seqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(seqEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,batch_first=True)

    def forward(self, input):
        output, hidden = self.gru(input)
        return output, hidden


class seqDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(seqDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.LeakyReLU()
        self.into =nn.Linear(1,hidden_size)
    def forward(self, input, hidden):
        output = self.into(input)
        output = output.unsqueeze(1)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, self.hidden_size)
class seq2seq(nn.Module):
    def __init__(self,input_size, hidden_size,output_size, num_layers=1):
        super(seq2seq,self).__init__()
        self.encoder = seqEncoder(input_size,hidden_size,num_layers)
        self.decoder = seqDecoder(hidden_size, output_size,num_layers)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
    def forward(self, a):
        a=a.reshape((1, 1, self.input_size))
        # 编码器读取输入序列
        encoder_output, encoder_hidden = self.encoder(a)

        # 解码器使用编码器的隐藏状态作为其初始隐藏状态
        decoder_hidden = encoder_hidden

        # 解码器生成输出序列
        decoder_outputs = []
        batch_size = encoder_output.size(0)

        #需要构建一个(batchsize,1,hidden_dim)的输入,均为0
        decoder_input = torch.empty(batch_size, 1, dtype=torch.float, device=device)
        for di in range(self.output_size):  # max_length是输出序列的最大长度
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)#选择最大可能
            decoder_outputs.append(topv.item())
            decoder_input = topv


        return torch.tensor(decoder_outputs).cuda()
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
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=10, num_encoder_layers=12)

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

class TeacherModel(nn.Module):
    def __init__(self, input_dim,output_dim,embedding_dim=64):
        super(TeacherModel, self).__init__()
        self.Ev = EncoderEv(input_dim, embedding_dim)
        self.Dv = DecoderDv(embedding_dim, output_dim)
        #self.selayer = SELayer(embedding_dim).double()
    def forward(self,f):
        z = self.Ev(f)
        #z_atti = self.selayer(z)
        y = self.Dv(z)
        # test = self.teacher_discriminator_c(f)
        return z,y
class TeacherStudentModel(nn.Module):
    def __init__(self, csi_input_dim,input_dim, output_dim,hidden_dim=10, embedding_dim=7):
        super(TeacherStudentModel, self).__init__()
        self.Ev = EncoderEv(input_dim, embedding_dim)
        self.Dv = DecoderDv(embedding_dim, output_dim)
        #self.Es = EncoderEs(csi_input_dim, embedding_dim)
        self.Ds = self.Dv
        #self.Trans = Transformer(csi_input_dim).double()
        #self.selayer = SELayer(embedding_dim).double()
        self.seq2seq=seq2seq(csi_input_dim,hidden_dim,embedding_dim)#hidden=10
    def forward(self,f, a):
        z = self.Ev(f)
        #z_atti = self.selayer(z)
        y = self.Dv(z)
        # test = self.teacher_discriminator_c(f)
        v = self.seq2seq(a)
        #v = self.Es(v)
        #v_atti = self.selayer(v)
        # v_atti = v
        s = self.Ds(v)
        return z,y,v,s
class StudentModel(nn.Module):
    def __init__(self,csi_input_dim, output_dim, hidden_dim=10,embedding_dim=7):
        super(StudentModel, self).__init__()
        #self.Es = EncoderEs(csi_input_dim, embedding_dim)
        self.Ds = DecoderDv(embedding_dim, output_dim)
        #self.Trans = Transformer(csi_input_dim).double()
        #self.selayer = SELayer(embedding_dim).double()
        self.seq2seq = seq2seq(csi_input_dim, hidden_dim, embedding_dim)  # hidden=10
    def forward(self,a):
        v = self.seq2seq(a)
        #v = self.Es(v)
        #v_atti = self.selayer(v)
        # v_atti = v
        s = self.Ds(v)
        return s



# Training configuration
learning_rate = 0.001
beta1 = 0.5
beta2 = 0.999
teacher_weights = {"wadv": 0.5, "wY": 1.0}
student_weights = {"wV": 0.5, "wS": 1.0}

# Initialize models

csi_input_dim = 50
input_dim = 28

n_features = 1
embedding_dim=20
hidden_dim=10
teacher_model= TeacherModel(input_dim, input_dim, embedding_dim).to(device)
#添加EOS导致dim+1
model = TeacherStudentModel(csi_input_dim,input_dim, input_dim,hidden_dim, embedding_dim).to(device)


CSI_PATH = "./data/inout/move/CSI_leg_right_out1.csv"
Video_PATH = "./data/inout/move/points_legright1.csv"
CSI_AVG_PATH = "./data/inout/move/CSI_leg_right_out1_avg.csv"
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


# aa = aa.apply(fillna_with_previous_values, axis=1)
# csi_test = csi_test.apply(fillna_with_previous_values, axis=1)

# array_length = 50
# result_array = np.zeros(array_length, dtype=int)

# for i in range(array_length):
#     if i % 2 == 0:
#         result_array[i] = 3 * (i // 2)
#     else:
#         result_array[i] = 3 * (i // 2) + 1
if (os.path.exists(CSI_AVG_PATH) != True):
    bb = reshape_and_average(aa)
    np.savetxt(CSI_AVG_PATH, bb, delimiter=',')
else:
    with open(CSI_AVG_PATH, "r", encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile)
        data1 = list(csvreader)  # 将读取的数据转换为列表
    bb = pd.DataFrame(data1)
Video_train = ff.values.astype('float32')  # 共990行，每行28个数据，为关键点坐标，按照xi，yi排序
CSI_train = bb.values.astype('float32')

# csi_test = csi_test.values.astype('float32')
# video_test = video_test.values.astype('float32')

CSI_train = CSI_train / np.max(CSI_train)
Video_train = Video_train.reshape(len(Video_train), 14, 2)  # 分成990组14*2(x,y)的向量
Video_train = Video_train / [800, 700]  # 输入的图像帧是1280×720的，所以分别除以1280和720归一化。
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

#np.random.shuffle(data)  # 打乱data顺序，体现随机


f_train = data[0:train_data_length, 0:28]
a_train = data[0:train_data_length, 28:]
original_length = f_train.shape[0]

# 剩余作为测试
g = data[train_data_length:,0:28]
b = data[train_data_length:,28:]
#teacher_model_train
teacher_f=pd.read_csv(Video_PATH, header=None)
teacher_f=teacher_f.values.astype('float32')
np.random.shuffle(teacher_f)
teacher_f = teacher_f.reshape(len(teacher_f), 14, 2)
teacher_f = teacher_f/ [800, 700]  # 除以800和700归一化。
teacher_f = teacher_f.reshape(len(teacher_f), -1)
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

optimizer = torch.optim.Adam(model.seq2seq.parameters(), lr=learning_rate, betas=(beta1, beta2))
teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=learning_rate, betas=(beta1, beta2))
criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss(reduction='sum')
teacher_num_epochs = 6000
teacher_batch_size = 50
num_epochs = 10000
batch_size = 50
#teacher_training
for epoch in range(teacher_num_epochs):

    random_indices_t = np.random.choice(original_length, size=teacher_batch_size, replace=False)
    f = torch.from_numpy(teacher_f[random_indices_t, :]).float()
    f = f.view(teacher_batch_size, len(teacher_f[0]))  # .shape(batch_size,28)

    if (torch.cuda.is_available()):
        f = f.cuda()
    if epoch == 4000:
        print()
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

seq_true = f.cpu()
seq_pred = seq_pred.cpu()

seq_true_np = seq_true.detach().numpy()
seq_pred_np = seq_pred.detach().numpy()
seq_true_np = seq_true_np.squeeze()
seq_pred_np = seq_pred_np.squeeze()

np.savetxt("./data/output/points_merged_output_training.csv", seq_pred_np, delimiter=',')
np.savetxt("./data/output/real_output_training.csv", seq_true_np, delimiter=',')

# loss_values = np.array(loss_values)   #把损失值变量保存为numpy数组
model.Dv.load_state_dict(teacher_model.Dv.state_dict())
model.Ev.load_state_dict(teacher_model.Ev.state_dict())
window_size=10
def create_dataset(X, y, P=10):  # 设置时间窗口P=10
    features = []
    targets = []
    for i in range(len(X) - P):
        data = X[i:i + P]
        label = y[i + P]
        # 保存
        features.append(data)
        targets.append(label)

    return torch.stack(features),torch.stack(targets) #x.size=(batchsize-P, P,feature) y.size=(batchsize-P,target)

def PCA(X_arry,q):
    X_pca=[]
    for X in X_arry:
        X=torch.transpose(X,0,1)
        X_mean = torch.mean(X, 0)
        X = X - X_mean

        U, S, V = torch.pca_lowrank(X, q)
        X_pca_rev=torch.mm(X, V)
        X_pca.append(torch.transpose(X_pca_rev,0,1))

    return torch.stack(X_pca)

for epoch in range(num_epochs):

    random_indices = np.random.choice(original_length-batch_size, size=1, replace=False)
    random_indices=random_indices[0]
    f = torch.from_numpy(f_train[random_indices:random_indices+batch_size]).float()
    a = torch.from_numpy(a_train[random_indices:random_indices+batch_size]).float()
    f = f.view(batch_size, len(f_train[0]))
    a = a.view(batch_size, len(a_train[0]))


    a_p,f_p = create_dataset(a,f, window_size)

    a_p=PCA(a_p,1).squeeze() #计算P内数据的平均情况，做成类似句子的二元结构
    a_p=a_p.unsqueeze(1)#创建(batch_size,seq_len,feature)结构
        #a_p, f_p =a,f

    if (torch.cuda.is_available()):
        f_p =f_p.cuda()
        a_p = a_p.cuda()
    total_teacher_loss = []
    total_student_loss = []
    total_total_loss = []
    if epoch == 20:
        print()
    with autocast():
        optimizer.zero_grad()
        pbar=tqdm(total=len(a_p))
        for i in range(len(a_p)):#一句句训练
            z, y, v, s = model(f_p[i], a_p[i])

            teacher_loss = criterion1(y, f_p[i])
            # 计算学生模型的损失
            student_loss = 0.8 * criterion1(v, z) + criterion1(s, f_p[i])

            total_loss = teacher_loss + student_loss

            # 计算梯度
            total_loss.backward()
            # 更新模型参数
            optimizer.step()
            total_teacher_loss.append(teacher_loss.item())
            total_student_loss.append(student_loss.item())
            total_total_loss.append(total_loss.item())
            pbar.update(1)
            # 打印训练信息
        loss_t = np.mean(total_teacher_loss)
        loss_s = np.mean(total_student_loss)
        loss = np.mean(total_total_loss)

        print(
            f"training:Epoch [{epoch + 1}/{num_epochs}], Teacher Loss: {loss_t.item():.4f}, Student Loss: {loss_s.item():.4f}, Total Loss: {loss.item():.4f}")
        pbar.close()

# loss_values = np.array(loss_values)   #把损失值变量保存为numpy数组
'''
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
'''
# 参数传递
student_model = StudentModel(csi_input_dim, input_dim, hidden_dim,embedding_dim).to(device)
student_model.seq2seq.load_state_dict(model.seq2seq.state_dict())
student_model.Ds.load_state_dict(model.Ds.state_dict())
# 在测试阶段只有学生模型的自编码器工作
#student_batch_size=10
#random_indices = np.random.choice(len(b), size=student_batch_size, replace=False)
#b= torch.from_numpy(b[random_indices, :]).float()
#g= torch.from_numpy(g[random_indices, :]).float()
with torch.no_grad():
    #b=b.view(student_batch_size, len(b[0]))
    b= torch.from_numpy(b).float()#a
    g= torch.from_numpy(g).float()#f

    b, g = create_dataset(b, g, window_size)

    b = PCA(b, 1).squeeze()  # 计算P内数据的平均情况，做成类似句子的二元结构
    b = b.unsqueeze(1)  # 创建(batch_size,seq_len,feature)结构

    b = b.to(device)
    g = g.to(device)
    r=[]
    #r = student_model(b)
    for i in range(len(b)):
        r_i = student_model(b[i])
        r.append(r_i)
    r=torch.stack(r)
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

# '''
