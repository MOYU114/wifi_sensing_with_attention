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

class gopose(nn.Model):#输出为14个骨骼点
    def __int__(self,input_dim,output_dim):
        super(gopose,self).__int__()
        self.cnn = nn.Sequential(
            nn.Conv3d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout(0.1),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout(0.1),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout(0.1),
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout(0.1),
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout(0.1),
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
        )
        self.rnn = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, dropout=0.1),
        # Define the output layer
        self.fc = nn.Linear(256, output_dim)
    def forward(self,x):
        x=self.cnn(x)
        # Flatten the output of the CNN layers
        x = x.view(x.size(0), -1)
        # Reshape the output to a sequence of feature vectors
        x = x.view(-1, x.size(0), 256)
        # Pass the sequence through the LSTM layers
        x, _ = self.lstm1(x)
        # Take the last output of the LSTM as the final output
        x = x[-1]
        # Pass the output through the fully connected layer
        x = self.fc(x)
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


# Training configuration
beta1 = 0.5
beta2 = 0.999
teacher_weights = {"wadv": 0.5, "wY": 1.0}
student_weights = {"wV": 0.5, "wS": 1.0}

# Initialize models
input_dim = 4
output_dim = 14
#CSI为图像文件180*180*4，而对应的video变为14条关键点信息，下述路径信息未修改
CSI_PATH = "./data/static data/CSI_new.csv"
Video_PATH = "./data/static data/photo/"
CSI_OUTPUT_PATH = "./data/output/CSI_merged_output.csv"
Video_OUTPUT_PATH = "./data/output/points_merged_output.csv"
'''
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

'''
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
ff = pd.read_csv(Video_PATH, header=None)

image_folder_path = CSI_PATH
Video_train = ff.values.astype('float32')  # 共990行，每行28个数据，为关键点坐标，按照xi，yi排序
Video_train =ff.values*1

# csi_test = csi_test.values.astype('float32')
# video_test = video_test.values.astype('float32')
Video_train = Video_train.reshape(len(Video_train), 14, 2)  # 分成990组14*2(x,y)的向量
Video_train = Video_train / [800, 700]  # 根据输入的图像帧进行修改
Video_train = Video_train.reshape(len(Video_train), -1)
#CSI_train = CSI_train / np.max(CSI_train)#不再归一化
input_vectors=Video_train
pics_num=len(Video_train)#图片个数由CSI个数而定

dataset = CustomDataset(image_folder_path, input_vectors,pics_num)
num_samples = 1000#选取的图像数，可看作batchsize
Video_len=len(input_vectors[0])
batch_size=20#20个一组训练
dataset_length=len(dataset)
train_dataset=dataset[0:dataset_length*0.9]
test_dataset=dataset[dataset_length*0.9:]
print("data has loaded.")
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

num_samples=1000
batch_size=10
SAVE_PATH = "./data/output/photo/test/"
model = gopose(input_dim,output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(beta1, beta2))#只训练学生es参数
criterion1 = nn.MSELoss()
criterion2 = nn.SmoothL1Loss()#huberloss
teacher_epochs=2
num_epochs = 1000
Q_p=0.63
Q_h=0.37

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def Eu_loss(pred,true,batch_size):
    result=0.0
    for i in range(batch_size):
        sum=0.0
        for j in range(14):#14个关键点
            sum+=euclidean_distance(pred[i][j][0],pred[i][j][1],true[i][j][0],true[i][j][1])
        result+=sum/14
    result=result/batch_size
    return result
def Huber_loss(pred,true,batch_size):
    result=0.0
    for i in range(batch_size):
        sum=criterion2(pred[i],true[i])
        result+=sum/14
    result=result/batch_size
    return result

#原始数据为先宽后长
for epoch in range(num_epochs):
    optimizer.zero_grad()
    sampler = RandomSampler(train_dataset, num_samples=num_samples, replacement=True)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    pbar = tqdm(total=len(dataloader))
    for a,f in dataloader:
        f = f.double()
        a = a.double()
        a = a.view(batch_size, 4,1,180,180)#视频帧参数未说明
        f = f.view(batch_size,14,2)

        if (torch.cuda.is_available()):
            f = f.cuda()
            a = a.cuda()
        # try:
        with autocast():
            pred_points = model(a)
        eps = 1e-8  # 平滑值，防止出现log0
        pred_points=pred_points.view(batch_size,14,2)
        #loss计算
        #欧氏距离loss
        Eu_loss=Eu_loss(pred_points,f,batch_size)
        f = f.view(batch_size, 28)
        pred_points = pred_points.view(batch_size, 28)
        #Huberloss
        Huber_loss=Huber_loss(pred_points,f,batch_size)
        total_loss=Q_p*Eu_loss+Q_h*Huber_loss
        # 计算梯度
        total_loss.backward()
        # 更新模型参数
        optimizer.step()
        print(
            f"training:Epoch [{epoch + 1}/{num_epochs}], Eu_loss: {Eu_loss.item():.4f}, Huber_loss: {Huber_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

    # 打印训练信息

# loss_values = np.array(loss_values)   #把损失值变量保存为numpy数组
test_data_length=100
SAVE_PATH = "./data/output/photo/test/"
with torch.no_grad():
    sampler = RandomSampler(test_dataset, num_samples=test_data_length, replacement=True)
    dataloader = DataLoader(test_dataset, batch_size=test_data_length, sampler=sampler)
    for a, f in dataloader:
        f = f.double()
        a = a.double()
        a = a.view(batch_size, 4, 1, 180, 180)  # 视频帧参数未说明
        f = f.view(batch_size, 14, 2)

        if (torch.cuda.is_available()):
            f = f.cuda()
            a = a.cuda()
        # try:
        with autocast():
            pred_points = model(a)
        eps = 1e-8  # 平滑值，防止出现log0
        pred_points = pred_points.view(batch_size, 14, 2)
        # loss计算
        # 欧氏距离loss
        Eu_loss = Eu_loss(pred_points, f, batch_size)
        f = f.view(batch_size, 28)
        pred_points = pred_points.view(batch_size, 28)
        # Huberloss
        Huber_loss = Huber_loss(pred_points, f, batch_size)
        total_loss = Q_p * Eu_loss + Q_h * Huber_loss
        # 计算梯度
        total_loss.backward()
        # 更新模型参数
        optimizer.step()
    print(
        f"test: Eu_loss: {Eu_loss.item():.4f}, Huber_loss: {Huber_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")

