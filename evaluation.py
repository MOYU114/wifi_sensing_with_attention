# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:36:25 2023

@author: Administrator
"""
import csv
import torch, gc
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.cuda.amp import autocast
from tqdm import tqdm, trange

class DecoderDv0(nn.Module):
    def __init__(self, embedding_dim, output_dim=28):
        super(DecoderDv0, self).__init__()
        self.L2=nn.Sequential(
            nn.Linear(embedding_dim, 14),
            nn.LeakyReLU(),
            nn.Linear(14, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x=self.L2(x)
        return x
    
class EncoderEs0(nn.Module):
    def __init__(self, embedding_dim, csi_input_dim=50):
        super(EncoderEs0, self).__init__()
        self.L3 = nn.Sequential(
            nn.Linear(csi_input_dim, 25),
            nn.LeakyReLU(),
            nn.Linear(25, embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.L3(x)
        return x
    
class StudentModel0(nn.Module):
    def __init__(self,csi_input_dim, embedding_dim, output_dim):
        super(StudentModel0, self).__init__()
        self.Es = EncoderEs0(embedding_dim, csi_input_dim)
        self.Ds = DecoderDv0(embedding_dim, output_dim)
    def forward(self,a):
        v = self.Es(a)
        s = self.Ds(v)
        return s
    
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

class DecoderDv(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(DecoderDv, self).__init__()
        # self.deconv1 = nn.ConvTranspose2d(latent_dim, 16, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.relu = nn.LeakyReLU()
        # self.deconv2 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.relu = nn.LeakyReLU()
        # self.deconv3 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.relu = nn.LeakyReLU()
        # self.deconv4 = nn.ConvTranspose2d(64, output_dim, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(28)
        # self.relu = nn.LeakyReLU()
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
        # self.relu1 = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.relu(self.bn3(self.deconv3(x)))
        x = self.relu(self.bn4(self.deconv4(x)))
        return x

# Student Model Components
class EncoderEs(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderEs, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.conv = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.relu1 = nn.LeakyReLU()
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        # h=self.relu(h)
        h = h[-1]  # Get the hidden state of the last LSTM unit
        h = h.unsqueeze(2).unsqueeze(3)  # Add dimensions for 2D convolution
        v = self.conv(h)
        v = self.relu1(v)
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
    
class StudentModel(nn.Module):
    def __init__(self, dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim):
        super(StudentModel, self).__init__()
        self.student_encoder_es = EncoderEs(es_input_dim, es_hidden_dim, ev_latent_dim).double()
        self.student_decoder_ds = DecoderDv(ev_latent_dim, dv_output_dim).double()
        self.selayer = SELayer(ev_latent_dim).double()

    def forward(self, x):
        v = self.student_encoder_es(x)
        v_atti=self.selayer(v)
        s = self.student_decoder_ds(v_atti)
        return s
    
# CSI_test = "./data/CSI_mov_room.csv"
# Video_test = "./data/static/data/device/points_stand.csv"
# CSI_OUTPUT_PATH = "./data/output/CSI_merged_output.csv"
# Video_OUTPUT_PATH = "./data/output/points_merged_output.csv"
# CSI_test = "./data/static/data/loc_Tr/CSI_stand_6C_1.csv"
# Video_test = "./data/static/data/device/points_arm_left.csv"
criterion1 = nn.MSELoss()

# input_dim = 50
# embed_dim = 8
# output_dim = 28
# student_model = StudentModel0(input_dim, embed_dim, output_dim)
# student_model.load_state_dict(torch.load('./model/model_mov_linear.pth'))

ev_latent_dim = 64
es_input_dim = 10
es_hidden_dim = 300
dv_output_dim = 28
student_model = StudentModel(dv_output_dim, es_input_dim, es_hidden_dim, ev_latent_dim)
student_model.load_state_dict(torch.load('./model/student_model5_mov_6C.pth'))


for i in range(17,21):
    if i == 0:
        # Video_test = "./data/static/data/device/points_wave_right_6C.csv"
        # CSI_test = "./data/static/data/device/CSI_wave_right_6C_1.csv"
        Video_test = "./data/static/data/user/points_leg_left_u3.csv"
        CSI_test = "./data/static/data/user/CSI_leg_left_u1_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
    elif i == 1:
        # Video_test = "./data/static/data/device/points_leg_left_83do.csv"
        # CSI_test = "./data/static/data/device/CSI_leg_left_83do_1.csv"
        Video_test = "./data/static/data/user/points_leg_left_u2.csv"
        CSI_test = "./data/static/data/user/CSI_leg_left_u2_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output2.csv"
    elif i ==2:
        # Video_test = "./data/static/data/device/points_leg_left_D806.csv"
        # CSI_test = "./data/static/data/device/CSI_leg_left_D806_1.csv"
        Video_test = "./data/static/data/user/points_wave_left_u1.csv"
        CSI_test = "./data/static/data/user/CSI_wave_right_u1_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output3.csv"
    elif i == 3:
        Video_test = "./data/static/data/device/points_leg_left_H6C.csv"
        CSI_test = "./data/static/data/device/CSI_leg_left_H6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output4.csv"
    elif i == 4:
        Video_test = "./data/static/data/device/points_leg_left_MI.csv"
        CSI_test = "./data/static/data/device/CSI_leg_left_MI_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc5.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output5.csv"
    elif i == 5:
        Video_test = "./data/static/data/device/points_leg_left_TP.csv"
        CSI_test = "./data/static/data/device/CSI_leg_left_TP_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc6.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output6.csv"
    elif i == 6:
        Video_test = "./data/static/data/device/points_leg_left_YI.csv"
        CSI_test = "./data/static/data/device/CSI_leg_left_YI_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc7.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output7.csv"
    elif i == 7:
        Video_test = "./data/static/data/traindata/move/points_legright5.csv"
        CSI_test = "./data/static/data/traindata/move/CSI_leg_left5.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc8.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output8.csv"
    # tr_loc
    # elif i == 7:
    #     Video_test = "./data/static/data/loc_Tr/points_wave_right_6C_trloc1.csv"
    #     CSI_test = "./data/static/data/loc_Tr/CSI_wave_right_trloc1_6C_1.csv"
    #     CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
    #     Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
    elif i == 8:
        Video_test = "./data/static/data/loc_Tr/points_wave_right_6C_trloc2.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_wave_right_trloc2_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output2.csv"
    elif i ==9:
        Video_test = "./data/static/data/loc_Tr/points_wave_right_6C_trloc3.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_wave_right_trloc3_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output3.csv"
    elif i == 10:
        Video_test = "./data/static/data/loc_Tr/points_wave_right_6C_trloc4.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_wave_right_trloc4_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output4.csv"
    elif i == 11:
        Video_test = "./data/static/data/loc_Tr/points_wave_right_6C_trloc5.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_wave_right_trloc5_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc5.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output5.csv"
    elif i == 12:
        Video_test = "./data/static/data/loc_Tr/points_wave_right_6C_trloc6.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_wave_right_trloc6_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc6.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output6.csv"
    # user_loc
    elif i == 13:
        Video_test = "./data/static/data/loc_user/points_wave_right_6C_userloc1.csv"
        CSI_test = "./data/static/data/loc_user/CSI_wave_right_userloc1_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
    elif i == 14:
        Video_test = "./data/static/data/loc_user/points_userloc2_2.csv"
        CSI_test = "./data/static/data/loc_user/CSI_wave_right_userloc2_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output2.csv"
    elif i ==15:
        Video_test = "./data/static/data/loc_user/points_userloc3_2.csv"
        CSI_test = "./data/static/data/loc_user/CSI_wave_right_userloc3_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output3.csv"
    elif i == 16:
        Video_test = "./data/static/data/loc_user/points_userloc4_2.csv"
        CSI_test = "./data/static/data/loc_user/CSI_wave_right_userloc4_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output4.csv"
    # Room
    elif i == 17:
        # Video_test = "./data/static/data/device/points_wave_left_6c.csv"
        Video_test = "./data/static/data/room/points_wave_left_6c_2.csv"
        CSI_test = "./data/static/data/room/CSI_wave_left_6C_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
    elif i == 18:
        # Video_test = "./data/static/data/device/points_wave_right_6c.csv"
        Video_test = "./data/static/data/room/points_wave_right_6c_2.csv"
        CSI_test = "./data/static/data/room/CSI_wave_right_6C_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output2.csv"
    elif i ==19:
        # Video_test = "./data/static/data/device/points_leg_left_6c.csv"
        Video_test = "./data/static/data/room/points_leg_left_6c_2.csv"
        CSI_test = "./data/static/data/room/CSI_leg_left_6C_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output3.csv"
    elif i == 20:
        # Video_test = "./data/static/data/device/points_leg_right_6C.csv"
        Video_test = "./data/static/data/room/points_leg_right_6c_2.csv"
        CSI_test = "./data/static/data/room/CSI_leg_right_6C_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output4.csv"
    elif i == 21:
        Video_test = "./data/static/data/0114/points_wave_left_c205_5.csv"
        CSI_test = "./data/static/data/temp/CSI_wave_left_c205_5.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc5.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output5.csv"
    elif i == 22:
        Video_test = "./data/static/data/0114/points_wave_right_c205_5.csv"
        CSI_test = "./data/static/data/temp/CSI_wave_right_c205_5.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc6.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output6.csv"
    elif i ==23:
        Video_test = "./data/static/data/0114/points_leg_left_c205_5.csv"
        CSI_test = "./data/static/data/temp/CSI_leg_left_c205_5.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc7.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output7.csv"
    elif i == 24:
        Video_test = "./data/static/data/0114/points_leg_right_c205_5.csv"
        CSI_test = "./data/static/data/temp/CSI_leg_right_c205_5.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc8.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output8.csv"
    # static devices/user    
    elif i == 25:
        # Video_test = "./data/static/data/device/points_stand.csv"
        # CSI_test = "./data/static/data/device/CSI_stand_6C_1.csv"
        CSI_test = "./data/static/data/device/CSI_stand_6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
    elif i == 26:
        # Video_test = "./data/static/data/device/points_stand.csv"
        CSI_test = "./data/static/data/device/CSI_stand_83do_1.csv"
        # CSI_test = "./data/static/data/user/CSI_stand_U2_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output2.csv"
    elif i ==27:
        # Video_test = "./data/static/data/device/points_stand.csv"
        CSI_test = "./data/static/data/device/CSI_stand_D806_1.csv"
        # CSI_test = "./data/static/data/user/CSI_stand_U3_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output3.csv"
    elif i == 28:
        # Video_test = "./data/static/data/device/points_stand.csv"
        CSI_test = "./data/static/data/device/CSI_stand_H6C_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output4.csv"
    elif i == 29:
        # Video_test = "./data/static/data/device/points_stand.csv"
        CSI_test = "./data/static/data/device/CSI_stand_MI_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc5.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output5.csv"
    elif i == 30:
        # Video_test = "./data/static/data/device/points_stand.csv"
        CSI_test = "./data/static/data/device/CSI_stand_TP_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc6.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output6.csv"
    elif i == 31:
        # Video_test = "./data/static/data/device/points_stand.csv"
        CSI_test = "./data/static/data/device/CSI_stand_YI_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc7.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output7.csv"
        
    # loc TR
    elif i == 32:
        Video_test = "./data/static/data/loc_Tr/points_arm_right.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_arm_right_loc1_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
    elif i == 33:
        Video_test = "./data/static/data/loc_Tr/points_arm_right.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_arm_right_loc2_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output2.csv"
    elif i == 34:
        Video_test = "./data/static/data/loc_Tr/points_arm_right.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_arm_right_loc3_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output3.csv"
    elif i == 35:
        Video_test = "./data/static/data/loc_Tr/points_arm_right.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_arm_right_loc4_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output4.csv"
    elif i == 36:
        Video_test = "./data/static/data/loc_Tr/points_arm_right.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_arm_right_loc5_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc5.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output5.csv"
    elif i == 37:
        Video_test = "./data/static/data/loc_Tr/points_arm_right.csv"
        CSI_test = "./data/static/data/loc_Tr/CSI_arm_right_loc6_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc6.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output6.csv"
        
    # Loc User
    elif i == 38:
        Video_test = "./data/static/data/loc_user/points_arm_right_loc.csv"
        CSI_test = "./data/static/data/loc_user/CSI_arm_right_loc1_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
    elif i == 39:
        Video_test = "./data/static/data/loc_user/points_arm_right_loc.csv"
        CSI_test = "./data/static/data/loc_user/CSI_arm_right_loc2_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output2.csv"
    elif i ==40:
        Video_test = "./data/static/data/loc_user/points_arm_right_loc.csv"
        CSI_test = "./data/static/data/loc_user/CSI_arm_right_loc3_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output3.csv"
    elif i == 41:
        Video_test = "./data/static/data/loc_user/points_arm_right_loc.csv"
        CSI_test = "./data/static/data/loc_user/CSI_arm_right_loc4_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output4.csv"
        
    # Room
    elif i == 42:
        Video_test = "./data/static/data/device/points_arm_left.csv"
        # CSI_test = "./data/static/data/temp/CSI_arm_left_C109_6.csv"
        CSI_test = "./data/static/data/room/CSI_arm_left_B205_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc1.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output1.csv"
    elif i == 43:
        Video_test = "./data/static/data/device/points_arm_right.csv"
        # CSI_test = "./data/static/data/temp/CSI_arm_right_C109_6.csv"
        CSI_test = "./data/static/data/room/CSI_arm_right_B205_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output2.csv"
    elif i ==44:
        Video_test = "./data/static/data/device/points_leg.csv"
        # CSI_test = "./data/static/data/temp/CSI_leg_C109_6.csv"
        CSI_test = "./data/static/data/room/CSI_stand_B205_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output3.csv"
    elif i == 45:
        Video_test = "./data/static/data/device/points_stand.csv"
        # CSI_test = "./data/static/data/temp/CSI_stand_C109_6.csv"
        CSI_test = "./data/static/data/room/CSI_stand_B205_1.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output4.csv"
    elif i == 46:
        Video_test = "./data/static/data/room/points_arm_left.csv"
        CSI_test = "./data/static/data/room/CSI_arm_left_6C_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc5.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output5.csv"
    elif i == 47:
        Video_test = "./data/static/data/room/points_arm_right.csv"
        CSI_test = "./data/static/data/room/CSI_arm_right_6C_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc6.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output6.csv"
    elif i ==48:
        Video_test = "./data/static/data/room/points_leg.csv"
        CSI_test = "./data/static/data/room/CSI_leg_6C_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc7.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output7.csv"
    elif i == 49:
        Video_test = "./data/static/data/room/points_stand.csv"
        CSI_test = "./data/static/data/room/CSI_stand_6C_2.csv"
        CSI_OUTPUT_PATH = "./data/output/CSI_loc8.csv"
        Video_OUTPUT_PATH = "./data/output/points_merged_output8.csv"
    else:
        paras = 11
        CSI_OUTPUT_PATH = "./data/output/CSI_10.csv"

    with open(CSI_test, "r") as csvfilee:
        csvreadere = csv.reader(csvfilee)
        data2 = list(csvreadere)  # 将读取的数据转换为列表
    csi_test = pd.DataFrame(data2)
    video_test = pd.read_csv(Video_test, header=None)
    print("test data has loaded.")
    
    csi_tmp = reshape_and_average(csi_test)
    csi_tmp = csi_tmp.values.astype('float32')
    csi_tmp2 = csi_tmp / np.max(csi_tmp)
    # csi_tmp2 = csi_tmp2[60:1160,:]
    video_tmp = video_test.values.astype('float32')
    video_tmp1 = video_tmp.reshape(len(video_tmp), 14, 2)
    video_tmp1 = video_tmp1 / [1280, 720]
    video_tmp2 = video_tmp1.reshape(len(video_tmp1), -1)
    # video_tmp2 = video_tmp2[0:1100,:]
    
    # # Linear
    # g= torch.from_numpy(video_tmp2).float()
    # b = torch.from_numpy(csi_tmp2).float()
    
    # CNN
    g= torch.from_numpy(video_tmp2).double()
    b = torch.from_numpy(csi_tmp2).double()
    b = b.view(len(b),int(len(csi_tmp2[0])/10),10)
    with torch.no_grad():        
        r = student_model(b)
        r = r.view(np.size(r, 0), np.size(r, 1)) #CNN
        loss = criterion1(r, g)
        print("loss:", loss)
        g = g.cpu()
        r = r.cpu()
        gnp = g.numpy()
        rnp = r.numpy()
        np.savetxt(Video_OUTPUT_PATH, gnp, delimiter=',')
        np.savetxt(CSI_OUTPUT_PATH, rnp, delimiter=',')























