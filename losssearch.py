# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 17:38:21 2023

@author: Administrator
"""
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
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
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

        self.selayer = SELayer(ev_latent_dim).double()
    def forward(self, f):
        z = self.teacher_encoder_ev(f)
        z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z_atti)

        return y
class TeacherModel_D(nn.Module):
    def __init__(self, ev_input_dim):
        super(TeacherModel_D, self).__init__()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()
    def forward(self, input):
        output = self.teacher_discriminator_c(input)
        return output

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
    leg_index = leg_index[0:length_min*8]
    wave_index = wave_index[0:length_min*8]
    stand_index = stand_index[0:length_min]
    merged_index = leg_index + wave_index + stand_index
    return merged_index



def lossearch(numcase):
        
    
    ev_input_dim = 28
    ev_latent_dim = 64
    es_input_dim = 10
    es_hidden_dim = 300
    dv_output_dim = 28
    
    #points_in的准确率有60左右
    
    CSI_PATH = "./data/CSI_out_static.csv"
    Video_PATH = "./data/points_static.csv"
    CSI_OUTPUT_PATH = "./data/output/CSI_merged_output.csv"
    Video_OUTPUT_PATH = "./data/output/points_merged_output.csv"
    
    #aa = pd.read_csv(CSI_PATH, header=None,low_memory=False,encoding="utf-8-sig")
    with open(CSI_PATH, "r", encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile)
        data1 = list(csvreader)
    aa = pd.DataFrame(data1)                             #读取CSI数据到aa
    ff = pd.read_csv(Video_PATH, header=None)            #读取骨架关节点数据到ff
    print("data has loaded.")
    
    bb = reshape_and_average(aa)                #把多个CSI数据包平均为一个数据包，使一帧对应一个CSI数据包
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
    
    original_length = f_train.shape[0]
    
    # 剩余作为测试
    g = torch.from_numpy(data[train_data_length:data_length,0:28]).double()
    b = torch.from_numpy(data[train_data_length:data_length,28:78]).double()
    b = b.view(len(b),int(len(a_train[0])/10),10)#输入的维度可能不同，需要对输入大小进行动态调整
    
    #训练Teacher模型
    LR_G = 0.001
    LR_D = 0.001
    teacher_model_G=TeacherModel_G(ev_input_dim, ev_latent_dim, dv_output_dim).to(device)
    teacher_model_D=TeacherModel_D(ev_input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(teacher_model_G.parameters(), lr=LR_G)
    optimizer_D = torch.optim.Adam(teacher_model_D.parameters(), lr=LR_D)
    
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
    Teacher_num_epochs = 2000
    teacher_batch_size = 128
    for epoch in range(Teacher_num_epochs):
        random_indices = np.random.choice(original_length, size=teacher_batch_size, replace=False)
        f = torch.from_numpy(f_train[random_indices, :]).double()
        f = f.view(teacher_batch_size, 28, 1, 1)  # .shape(batch_size,28,1,1)
        y = teacher_model_G(f)
    
        # 进行对抗学习
        optimizer_D.zero_grad()
        
        real_target = teacher_model_D.teacher_discriminator_c(f)
        real_target = real_target.view(teacher_batch_size,1)
        
        fake_target = teacher_model_D.teacher_discriminator_c(y)
        fake_target = fake_target.view(teacher_batch_size,1)
      
        eps = 1e-8#平滑值，防止出现log0
        teacher_loss = torch.mean(torch.abs(real_target.mean(0) - fake_target.mean(0)))
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
    
    num_epochs =1500
    batch_size = 128
    
    for epoch in range(num_epochs):
        random_indices = np.random.choice(original_length, size=batch_size, replace=False)
        f = torch.from_numpy(f_train[random_indices, :]).double()
        a = torch.from_numpy(a_train[random_indices, :]).double()
        f = f.view(batch_size, 28, 1, 1)
        a = a.view(batch_size, int(len(a_train[0]) / 10), 10)
        optimizer.zero_grad()
        z, y, v, s = model(f, a)
        if numcase == 0:
            total_loss = criterion1(f,y) + criterion1(s, y)
        elif numcase == 1:
            total_loss = criterion1(s,y) + criterion1(s, f)
        elif numcase == 2:
            total_loss = criterion1(v,z) + criterion1(s, y)
        elif numcase == 3:
            total_loss = criterion1(f,y) + criterion1(v, z)
        elif numcase == 4:
            total_loss = criterion1(v,z) + criterion1(s, f)
        elif numcase == 5:
            total_loss = criterion1(v,z) + criterion1(s, y) + criterion1(f,y)
        elif numcase == 6:
            total_loss = criterion1(v,z) + criterion1(s, f) + criterion1(f,y)
        elif numcase == 7:
            total_loss = criterion1(v,z) + criterion1(s, f) + criterion1(s,y)
        elif numcase == 8:
            total_loss = criterion1(s,y) + criterion1(s, f) + criterion1(f,y)
        else:
            total_loss = criterion1(s,y) + criterion1(s, f) + criterion1(f,y) + criterion1(v,z)
        
        total_loss.backward()
        optimizer.step()
    
    student_model = StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim).to(device)
    student_model.student_encoder_es.load_state_dict(model.student_encoder_es.state_dict())
    student_model.student_decoder_ds.load_state_dict(model.teacher_decoder_dv.state_dict())
    
    with torch.no_grad():
        b = b.to(device)
        g = g.to(device)
        r = student_model(b)
        r = r.view(np.size(r, 0), np.size(r, 1))
        loss = criterion1(r, g)
        print("loss:", loss)
        g = g.cpu()
        r = r.cpu()
        gnp = g.numpy()
        rnp = r.numpy()
        np.savetxt(Video_OUTPUT_PATH, gnp, delimiter=',')
        np.savetxt(CSI_OUTPUT_PATH, rnp, delimiter=',')
        
    CSI_OUTPUT_PATH="./data/output/CSI_merged_output.csv"
    Video_OUTPUT_PATH="./data/output/points_merged_output.csv"
    CSI_OUTPUT = pd.read_csv(CSI_OUTPUT_PATH, header=None)
    Video_OUTPUT = pd.read_csv(Video_OUTPUT_PATH, header=None)
    CSI_OUTPUT=CSI_OUTPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
    Video_OUTPUT=Video_OUTPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
    CSI_OUTPUT = np.array(CSI_OUTPUT.tolist())
    Video_OUTPUT = np.array(Video_OUTPUT.tolist())
    def align_coordinates(coords1, coords2):
        # 计算两个坐标数组第二对坐标之间的差值
        x_diff = coords2[1][0] - coords1[1][0]
        y_diff = coords2[1][1] - coords1[1][1]
        # 根据差值平移第一个坐标数组中的所有坐标
        new_coords1 = [[x + x_diff, y + y_diff] for x, y in coords1]
        return new_coords1
    def align_all_coordinates(coords_list1,coords_list2):
        total_list_num=len(coords_list1)
        for i in range(total_list_num):
            #第一个坐标数组作为参考,改写第二组
            coords_list2[i]=align_coordinates(coords_list2[i],coords_list1[i])
        return coords_list2
    CSI_OUTPUT=align_all_coordinates(Video_OUTPUT,CSI_OUTPUT)

    def euclidean_distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    def PCS(CSI_OUTPUT,Video_OUTPUT,phi):
        total_num = len(CSI_OUTPUT)
        N = len(CSI_OUTPUT[0])
        pcs_total=0
        for i in range(total_num):
            sum=0
            for j in range(N):
                ed=euclidean_distance(CSI_OUTPUT[i][j][0],CSI_OUTPUT[i][j][1],Video_OUTPUT[i][j][0],Video_OUTPUT[i][j][1])
                if(ed<phi):
                    sum+=0
                else:
                    sum+=1
            pcs_total+=sum#统计本组数据的异常点个数
        pcs_avg=pcs_total/(total_num*N)#统计总体异常点个数占所有数据的多少
        return 1-pcs_avg#得到正确点的占比
        
    return PCS(CSI_OUTPUT,Video_OUTPUT,0.050)
 
resut = np.zeros((5,10))   
for i in range(50):
    if i%10 == 0:
        resut[i//10,i%10] = lossearch(0)
    elif i%10 == 1:
        resut[i//10,i%10] = lossearch(1)
    elif i%10 == 2:
        resut[i//10,i%10] = lossearch(2)
    elif i%10 == 3:
        resut[i//10,i%10] = lossearch(3)
    elif i%10 == 4:
        resut[i//10,i%10] = lossearch(4)
    elif i%10 == 5:
        resut[i//10,i%10] = lossearch(5)
    elif i%10 == 6:
        resut[i//10,i%10] = lossearch(6)
    elif i%10 == 7:
        resut[i//10,i%10] = lossearch(7)
    elif i%10 == 8:
        resut[i//10,i%10] = lossearch(8)
    else:
        resut[i//10,i%10] = lossearch(9)
# 结果是：
# 0.88879  0.92247  0.192659 0.916915 0.817436 0.658135	0.940699 0.907639 0.93001	0.923264
# 0.890476 0.914335 0.213914 0.536682 0.832366 0.902133	0.937723 0.909648 0.915104	0.866543
# 0.904439 0.909003 0.214286 0.90878	0.892783 0.906721	0.937922 0.917634 0.929514	0.897222
# 0.927331 0.90873  0.208606 0.786161	0.820213 0.84747	0.914683 0.925	  0.920833	0.900471
# 0.870263 0.910739 0.207068 0.836731	0.836905 0.889881	0.937574 0.894196 0.926687	0.931796


