# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:06:18 2023

@author: Administrator
"""
import numpy as np
import pandas as pd
import csv

CSI_PATH="./data/CSI_in.csv"
Video_PATH="./data/points_in.csv"

# csi = pd.read_csv(CSI_PATH, header=None)
with open(CSI_PATH, "r", encoding='utf-8-sig') as csvfile:
    csvreader = csv.reader(csvfile)
    data1 = list(csvreader)
csi = pd.DataFrame(data1) 
frame = pd.read_csv(Video_PATH, header=None)

csi_value = csi.values.astype('float32')
frame_value = frame.values.astype('float32')

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

csi_wave = csi_value[wave_index,:]
frame_wave = frame_value[wave_index,:]
csi_leg = csi_value[leg_index,:]
frame_leg = frame_value[leg_index,:]
csi_stand = csi_value[stand_index,:]
frame_stand = frame_value[stand_index,:]


