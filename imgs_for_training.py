import math
import csv
import os
from PIL import Image
from tqdm import tqdm,trange
import torch, gc
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast
from torch.utils.data import random_split, DataLoader,Dataset,RandomSampler
import torchvision.transforms as transforms
if (torch.cuda.is_available()):
    print("Using GPU for training.")
    device = torch.device("cuda:0")
else:
    print("Using CPU for training.")
    device = torch.device("cpu")


class EncoderEv(nn.Module):
    def __init__(self, input_dim,latent_dim):
        super(EncoderEv, self).__init__()
        self.gen1 = nn.Sequential(
            nn.Conv2d(input_dim, 8, kernel_size=3, stride=2,padding=1),
            #nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2, stride=2)
        )
        self.gen2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2,padding=1),
            #nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            #nn.MaxPool2d(2, 1),
        )
        self.gen3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2,padding=1),
            # nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # nn.MaxPool2d(2, 1),

        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 20 * 32, latent_dim)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid() #不再归一化，归一化对结果影响很大
    def forward(self, x):
        x = self.gen1(x)
        x = self.gen2(x)
        x = self.gen3(x)
        x = self.out(x)
        #x = self.avgpool(x)
        # print(x.shape)
        return x


class DecoderDv(nn.Module):
    def __init__(self, latent_dim, output_dim,pic_w,pic_h):
        super(DecoderDv, self).__init__()
        self.dec1= nn.Sequential(
            nn.Linear(latent_dim, 20 * 20 * 32),
            nn.Unflatten(1, (32, 20, 20)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU()
            #nn.BatchNorm2d(16),
        )
        self.dec2 = nn.Sequential(

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,padding=1),
            nn.LeakyReLU()
            #nn.BatchNorm2d(8),
        )
        self.dec3 = nn.Sequential(

            nn.ConvTranspose2d(8, output_dim, kernel_size=3, stride=2,padding=1),
            nn.Sigmoid()
            #nn.BatchNorm2d(output_dim),
        )

        self.upsample = nn.Sequential(

            nn.Upsample(size=(pic_h, pic_w), mode='bilinear', align_corners=True)
        # 先480后640
        )

    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        #x = self.dec3(x)
        x = self.upsample(x)
        return x


class DiscriminatorC(nn.Module):
    def __init__(self, input_dim):
        super(DiscriminatorC, self).__init__()
        self.f0 = nn.Sequential(
            nn.Conv2d(input_dim, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
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


# Student Model Components
class EncoderEs(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderEs, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1,batch_first=True)
        self.convTrans = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc=nn.Linear(hidden_dim,latent_dim)
    def forward(self, x):

        output, (h, _) = self.lstm(x) #取最后一步的数据
        output=output[:, -1, :]
        #output=output.squeeze()
        #output = output.unsqueeze(2).unsqueeze(3)
        h = h[-1]  # Get the hidden state of the last LSTM unit
        #h = self.relu(h)
        #h = h.unsqueeze(2).unsqueeze(3)  # Add dimensions for 2D convolution
        v = self.fc(h)
        #v = self.sigmoid(h)
        v = v.squeeze()
        # print(v.shape)
        return v




class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):#压缩比例
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
    def __init__(self, ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim,pic_w,pic_h):
        super(TeacherStudentModel, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim,ev_latent_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim,pic_w,pic_h).double()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()

        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = self.teacher_decoder_dv

        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, f, a):
        z = self.teacher_encoder_ev(f)
        #z_atti = self.selayer(z)
        y = self.teacher_decoder_dv(z)
        # test = self.teacher_discriminator_c(f)

        v = self.student_encoder_es(a)
        #v_atti = self.selayer(v.unsqueeze(2).unsqueeze(3))
        #v_atti = v_atti.squeeze()
        v_atti =v
        s = self.student_decoder_ds(v_atti)

        return z, y, v_atti, s
class TeacherModel(nn.Module):
    def __init__(self, ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim,pic_w,pic_h):
        super(TeacherModel, self).__init__()
        self.teacher_encoder_ev = EncoderEv(ev_input_dim,ev_latent_dim).double()
        self.teacher_decoder_dv = DecoderDv(ev_latent_dim, dv_output_dim,pic_w,pic_h).double()
        self.teacher_discriminator_c = DiscriminatorC(ev_input_dim).double()
    def forward(self, f):
        z = self.teacher_encoder_ev(f)
        y = self.teacher_decoder_dv(z)


        return z, y

class StudentModel(nn.Module):
    def __init__(self, dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim,pic_w,pic_h):
        super(StudentModel, self).__init__()
        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim,pic_w,pic_h).double()
        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, x):
        v = self.student_encoder_es(x)
        v_atti = self.selayer(v.unsqueeze(2).unsqueeze(3))
        v_atti = v_atti.squeeze()
        #v_atti = v
        s = self.student_decoder_ds(v_atti)
        return s


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


# Training configuration
beta1 = 0.5
beta2 = 0.999
teacher_weights = {"wadv": 0.5, "wY": 1.0}
student_weights = {"wV": 0.5, "wS": 1.0}

# Initialize models
ev_input_dim = 1
ev_latent_dim = 32
es_input_dim = 1 #取决于CSI的feature个数
es_hidden_dim = 64
dv_output_dim = 1
CSI_PATH = "./data/static data/CSI_new.csv"
Video_PATH = "./data/static data/photo/"
CSI_OUTPUT_PATH = "./data/output/CSI_merged_output.csv"
Video_OUTPUT_PATH = "./data/output/points_merged_output.csv"

#图像处理
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




# 25条子载波否则会报错ParserError: Error tokenizing data. C error: Expected 75 fields in line 20, saw 100
# aa = pd.read_csv(CSI_PATH, header=None,low_memory=False,encoding="utf-8-sig")
with open(CSI_PATH, "r", encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)  # 将读取的数据转换为列表
aa = pd.DataFrame(data1)

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
if (os.path.exists('./data/static data/CSI_avg.csv') != True):
    bb = reshape_and_average(aa)
    np.savetxt('./data/static data/CSI_avg.csv', bb, delimiter=',')
else:

    #with open('./data/static data/CSI_avg.csv', "r", encoding='utf-8-sig') as csvfile:
    #    csvreader = csv.reader(csvfile)
    #    data1 = list(csvreader)  # 将读取的数据转换为列表
    #bb = pd.DataFrame(data1)
    bb=pd.read_csv('./data/static data/CSI_avg.csv', header=None, encoding="utf-8-sig")


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
        image = image[0, :, :]
        return image, input_vector

image_folder_path = Video_PATH
CSI_train = bb.values.astype('float32')
CSI_train = bb.values*1
#CSI_train = CSI_train / np.max(CSI_train)#不再归一化
input_vectors=CSI_train
pics_num=len(CSI_train)#图片个数由CSI个数而定

dataset = CustomDataset(image_folder_path, input_vectors,pics_num)
num_samples = 1000#选取的图像数，可看作batchsize
CSI_len=len(input_vectors[0])
batch_size=20#20个一组训练

def picGen(images_array,length,name,SAVE_PATH):
    # 保存图像
    # 定义转换
    to_pil_image = transforms.ToPILImage()
    # 选择前10张图像
    images = images_array[:length]
    for i, image in enumerate(images):
        # 将张量转换为PIL图像
        pil_image = to_pil_image(image)
        # 构造文件名
        filename = os.path.join(SAVE_PATH, f'{name}_{i}.png')
        # 保存图像
        pil_image.save(filename)

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
num_samples=1000
batch_size=10
SAVE_PATH = "./data/output/photo/test/"
model = TeacherStudentModel(ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim,pic_w, pic_h).to(device)
teacher_model = TeacherModel(ev_input_dim, ev_latent_dim, es_input_dim, es_hidden_dim, dv_output_dim,pic_w, pic_h).to(device)

optimizer = torch.optim.Adam(model.student_encoder_es.parameters(), lr=0.01, betas=(beta1, beta2))#只训练学生es参数

optimizer_teacher = torch.optim.Adam(teacher_model.parameters(), lr=0.001, betas=(beta1, beta2))

criterion1 = nn.MSELoss()
criterion2 = nn.BCELoss()
teacher_epochs=2
#预训练自编码器
for i in range(teacher_epochs):
    optimizer_teacher.zero_grad()
    sampler = RandomSampler(dataset, num_samples=num_samples, replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    pbar = tqdm(total=len(dataloader))
    for f, a in dataloader:
        f = f.double()
        f = f.view(batch_size, 1, pic_h, pic_w)
        if (torch.cuda.is_available()):
            f = f.cuda()
        with autocast():
            z, y = teacher_model(f)
        '''
        to_pil_image = transforms.ToPILImage()
        # 选择前10张图像
        real_images = f[:10]
        fake_images = y[:10]
        SAVE_PATH = "./data/output/photo/test/"
        for i, image in enumerate(fake_images):
            # 将张量转换为PIL图像
            pil_image = to_pil_image(image)
            # 构造文件名
            filename = os.path.join(SAVE_PATH, f'real_output_{i}.png')
            # 保存图像
            pil_image.save(filename)
        '''
        real_prob = teacher_model.teacher_discriminator_c(f)
        fake_prob = teacher_model.teacher_discriminator_c(y)
        teacher_loss = 0.5 * criterion1(fake_prob, real_prob) + criterion1(y, f)#(result,target)
        optimizer_teacher.zero_grad()
        # 计算梯度
        teacher_loss.backward()
        # 更新模型参数
        optimizer_teacher.step()
        pbar.update(1)
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"training[{i+1}/{teacher_epochs}]:Teacher Loss: {teacher_loss.item():.4f}")
    pbar.close()
    name=f"training[{i+1}_{teacher_epochs}]"
    picGen(y,1,name,SAVE_PATH)
'''
# 在测试阶段只有学生模型的自编码器工作
test_data_length=10
#student_model = StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim,pic_w, pic_h).to(device)

with torch.no_grad():
    sampler = RandomSampler(dataset, num_samples=test_data_length, replacement=True)
    dataloader = DataLoader(dataset, batch_size=test_data_length, sampler=sampler)
    for g,b  in dataloader:
        g = g.double()
        g = g.view(test_data_length, 1, pic_h, pic_w)
        g = g.to(device)
        zz,r = teacher_model(g)
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
        picGen(g, test_data_length, "real_output", SAVE_PATH)
        picGen(r, test_data_length,"fake_output" , SAVE_PATH)
'''

# 参数传递

model.teacher_encoder_ev.load_state_dict(teacher_model.teacher_encoder_ev.state_dict())
model.teacher_decoder_dv.load_state_dict(teacher_model.teacher_decoder_dv.state_dict())
model.teacher_discriminator_c.load_state_dict(teacher_model.teacher_discriminator_c.state_dict())
model.student_decoder_ds.load_state_dict(teacher_model.teacher_decoder_dv.state_dict())

#student_model = StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim,pic_w, pic_h).to(device)
num_epochs = 1000
#原始数据为先宽后长
for epoch in range(num_epochs):
    optimizer.zero_grad()
    sampler = RandomSampler(dataset, num_samples=num_samples, replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    pbar = tqdm(total=len(dataloader))
    for f, a in dataloader:
        f = f.double()
        a = a.double()
        f = f.view(batch_size, 1, pic_h,pic_w)
        a = a.view(batch_size,len(input_vectors[0]),1)#batchsize,seq_len=len(input_vectors[0]),feature=1

        if (torch.cuda.is_available()):
            f = f.cuda()
            a = a.cuda()
        # try:
        with autocast():

            z, y, v, s = model(f, a)
        eps = 1e-8  # 平滑值，防止出现log0

        real_prob = model.teacher_discriminator_c(f)
        fake_prob = model.teacher_discriminator_c(y)
        '''
        d_loss = -torch.mean(torch.log(real_prob + eps) + torch.log(1 - fake_prob + eps))
        g_loss = -torch.mean(torch.log(fake_prob + eps))
        Ladv = d_loss + g_loss
        teacher_loss = 0.5 * Ladv + criterion1(f, y)
        '''
        teacher_loss = 0.5 * criterion1(fake_prob, real_prob) + criterion1(y, f)  # (result,target)
        # teacher_loss.backward()b
        # optimizer.step()

        # 计算学生模型的损失
        student_loss = 0.5 * criterion1(s, y) + criterion1(v, z) + 0.6 * criterion1(s, f)

        total_loss = teacher_loss + student_loss
        optimizer.zero_grad()
        # 计算梯度
        total_loss.backward()
        # 更新模型参数
        optimizer.step()
        pbar.update(1)
        gc.collect()
        torch.cuda.empty_cache()
        # 计算教师模型的损失

    # 打印训练信息
    print(
        f"training:Epoch [{epoch + 1}/{num_epochs}], "
        f"Teacher Loss: {teacher_loss.item():.4f}, "
        f"Student Loss: {student_loss.item():.4f}, "
        f"Total Loss: {total_loss.item():.4f}")

    pbar.close()
    name = f"training_real_[{epoch + 1}_{num_epochs}]"
    picGen(y, 1, name, SAVE_PATH)
    name = f"training_fake_[{epoch + 1}_{num_epochs}]"
    picGen(s, 1, name, SAVE_PATH)
# loss_values = np.array(loss_values)   #把损失值变量保存为numpy数组

# 参数传递
student_model = StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim,pic_w, pic_h).to(device)
student_model.student_encoder_es.load_state_dict(model.student_encoder_es.state_dict())
student_model.student_decoder_ds.load_state_dict(model.student_decoder_ds.state_dict())
student_model.selayer.load_state_dict(model.selayer.state_dict())
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
        g = g.view(test_data_length, 1, pic_h, pic_w)
        b = b.view(batch_size,len(input_vectors[0]),1)#batchsize,seq_len=len(input_vectors[0]),feature=1
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
        picGen(g, test_data_length, "real_output", SAVE_PATH)
        picGen(r, test_data_length, "fake_output", SAVE_PATH)
