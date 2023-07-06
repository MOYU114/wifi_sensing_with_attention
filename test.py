# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:43:47 2023

@author: Administrator
"""
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


'''
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

'''

class DecoderDv(nn.Module):
    def __init__(self, latent_dim, output_dim,pic_w,pic_h):
        super(DecoderDv, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(32, output_dim, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(size=(pic_h,pic_w), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.upsample(x)
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


# Student Model Components
class EncoderEs(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderEs, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.conv = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1,padding=1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
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
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1) #对应Excitation操作
        return x * y.expand_as(x)
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
        z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z_atti)

        return z, y
class TeacherModel_D(nn.Module):
    def __init__(self, ev_input_dim):
        super(TeacherModel_D, self).__init__()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()
    def forward(self, input):
        output = self.teacher_discriminator_c(input)
        return output


class TeacherStudentModel(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim,pic_w,pic_h):
        super(TeacherStudentModel, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim,pic_w,pic_h).double()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()

        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim,pic_w,pic_h).double()  # 分为了两个DS

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
        #v_atti = v
        s = self.student_decoder_ds(v_atti)

        return z, y, v, s


class StudentModel(nn.Module):
    def __init__(self, dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim,pic_w,pic_h):
        super(StudentModel, self).__init__()
        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim,pic_w,pic_h).double()
        self.CBAM = CBAM(ev_latent_dim).double()

    def forward(self, x):
        v = self.student_encoder_es(x)
        # v_atti=self.CBAM(v)
        v_atti = v
        s = self.student_decoder_ds(v_atti)
        return s



# Training configuration
learning_rate = 0.01
beta1 = 0.5
beta2 = 0.999
teacher_weights = {"wadv": 0.5, "wY": 1.0}
student_weights = {"wV": 0.5, "wS": 1.0}

# Initialize models
ev_input_dim = 3
ev_latent_dim = 64
es_input_dim = 10
es_hidden_dim = 300
dv_output_dim = 3

CSI_PATH = "./data/CSI_wave5.csv"
Video_PATH = "./data/image_wave5/"
CSI_OUTPUT_PATH = "./data/output/CSI_merged_output.csv"
Video_OUTPUT_PATH = "./data/output/points_merged_output.csv"

# 获取文件夹中的所有文件名
file_names = os.listdir(Video_PATH)
# 选择第一个文件
file_name = file_names[0]
# 构造文件的完整路径
file_path = os.path.join(Video_PATH, file_name)
# 打开图像文件
image = Image.open(file_path)
# 获取图像的宽度和高度
pic_w, pic_h = image.size

model = TeacherStudentModel(ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim,pic_w, pic_h).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
criterion1 = nn.MSELoss()
criterion2 = nn.BCELoss()  # 使用autocast
with open(CSI_PATH, "r") as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)  # 将读取的数据转换为列表
aa = pd.DataFrame(data1)

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

aa = aa.apply(fillna_with_previous_values, axis=1)

class CustomDataset(Dataset):
    def __init__(self, image_folder_path, input_vectors, pics_num):
        self.image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path)][:pics_num]
        self.input_vectors = input_vectors
        # 定义图像转换
        self.transform = transforms.Compose([
            transforms.Resize((300,300)),  # 缩小图像到300*300像素
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        input_vector = self.input_vectors[idx]
        image = Image.open(image_path)
        image = transforms.ToTensor()(image)
        return image, input_vector



image_folder_path = Video_PATH
CSI_train = aa.values.astype('float32')
CSI_train = CSI_train / np.max(CSI_train)
input_vectors=CSI_train
pics_num=len(CSI_train)#图片个数由CSI个数而定

dataset = CustomDataset(image_folder_path, input_vectors,pics_num)
num_samples = 100#选取的图像数，可看作batchsize
CSI_len=len(input_vectors[0])
batch_size=10#10个一组训练






# 训练TeacherStudent模型
num_epochs = 10
for epoch in range(num_epochs):
    sampler = RandomSampler(dataset, num_samples=num_samples, replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    pbar = tqdm(total=len(dataloader))
    for f,a in dataloader:
        f=f.double()
        a=a.double()
        a = a.view(batch_size, int(len(input_vectors[0])/ 10) , 10)

        if (torch.cuda.is_available()):
            f = f.cuda()
            a = a.cuda()
        #try:
        with autocast():
            z, y, v, s = model(f, a)

        #except RuntimeError as exception:
        #    if "out of memory" in str(exception):
        #        print('WARNING: out of memory')
        #        if hasattr(torch.cuda, 'empty_cache'):
        #            torch.cuda.empty_cache()
        #        else:
        #            raise exception
        # 计算教师模型的损失
        gc.collect()
        torch.cuda.empty_cache()
        real_target = model.teacher_discriminator_c(f)
        fake_target = model.teacher_discriminator_c(y)
        teacher_loss = criterion2(real_target, fake_target) + criterion1(y, f)
        # teacher_loss.backward()
        # optimizer.step()

        # 计算学生模型的损失
        student_loss = 0.5 * criterion1(v, z) + criterion1(s, y)

        total_loss = teacher_loss + student_loss
        optimizer.zero_grad()
        # 计算梯度
        total_loss.backward()
        # 更新模型参数
        optimizer.step()
        pbar.update(1)
        gc.collect()
        torch.cuda.empty_cache()
        # 打印训练信息
    print(f"TeacherStudentModel training:Epoch [{epoch + 1}/{num_epochs}], Teacher Loss: {teacher_loss.item():.4f}, Student Loss: {student_loss.item():.4f}")

# loss_values = np.array(loss_values)   #把损失值变量保存为numpy数组

#查看训练集效果
#y = y.cpu()
#s = s.cpu()
#ynp = y.detach().numpy()
#snp = s.detach().numpy()
#ynp=ynp.squeeze()
#snp=snp.squeeze()
#np.savetxt("./data/output/CSI_merged_output_training.csv", ynp, delimiter=',')
#np.savetxt("./data/output/points_merged_output_training.csv", snp, delimiter=',')





# 参数传递
student_model = StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim,pic_w, pic_h).to(device)
student_model.student_encoder_es.load_state_dict(model.student_encoder_es.state_dict())
student_model.student_decoder_ds.load_state_dict(model.student_decoder_ds.state_dict())

# 在测试阶段只有学生模型的自编码器工作
test_data_length=10
#student_model = StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim,pic_w, pic_h).to(device)


SAVE_PATH = "./data/output/photo/test/"
with torch.no_grad():
    sampler = RandomSampler(dataset, num_samples=test_data_length, replacement=True)
    dataloader = DataLoader(dataset, batch_size=test_data_length, sampler=sampler)
    for g,b  in dataloader:
        b = b.double()
        g = g.double()
        b = b.view(test_data_length, int(len(input_vectors[0]) / 10), 10)
        b = b.to(device)
        g = g.to(device)
        r = student_model(b)
        #r = r.view(np.size(r, 0), np.size(r, 1))

        loss = criterion1(r, g)
        # df = pd.DataFrame(r.numpy())
        # df.to_excel("result.xls", index=False)
        print("loss:", loss)
        #g = g.cpu()
        #r = r.cpu()
        #gnp = g.numpy()
        #rnp = r.numpy()
        #np.savetxt(Video_OUTPUT_PATH, gnp, delimiter=',')
        #np.savetxt(CSI_OUTPUT_PATH, rnp, delimiter=',')

        #保存图像
        # 定义转换
        to_pil_image = transforms.ToPILImage()

        # 选择前10张图像
        CSI_images = r[:test_data_length]
        Video_images = g[:test_data_length]

        # 遍历每张图像
        for i, image in enumerate(CSI_images):
            # 将张量转换为PIL图像
            pil_image = to_pil_image(image)
            # 构造文件名
            filename = os.path.join(SAVE_PATH, f'CSI_output_{i}.png')
            # 保存图像
            pil_image.save(filename)

        for i, image in enumerate(Video_images):
            # 将张量转换为PIL图像
            pil_image = to_pil_image(image)
            # 构造文件名
            filename = os.path.join(SAVE_PATH, f'Video_output_{i}.png')
            # 保存图像
            pil_image.save(filename)
