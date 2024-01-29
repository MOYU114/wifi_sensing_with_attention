import pandas as pd
import numpy as np
import math

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
#phi为阈值，其的设置是用来规定两个坐标点的欧拉距离应该要低于多少(一般点距离都在0.1以下)
#阈值应该根据具体情况设置
# Real_Image_PATH="./data/output/real_output_training.csv"
# Real_Image=pd.read_csv(Real_Image_PATH, header=None)
# Real_Image=Real_Image.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
# Real_Image = np.array(Real_Image.tolist())
    
# Video_OUTPUT_PATH="./data/output/points_merged_output.csv"
# Video_OUTPUT = pd.read_csv(Video_OUTPUT_PATH, header=None)
# Video_OUTPUT=Video_OUTPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
# Video_OUTPUT = np.array(Video_OUTPUT.tolist())

for i in range(1):
    if i == 0:
        CSI_OUTPUT_PATH="./data/output/CSI_merged_output_training.csv"
        Video_OUTPUT_PATH="./data/output/real_output_training.csv"
    elif i == 1:
        CSI_OUTPUT_PATH="./data/output/CSI_loc2.csv"
        Video_OUTPUT_PATH="./data/output/points_merged_output2.csv"
    elif i == 2:
        CSI_OUTPUT_PATH="./data/output/CSI_loc3.csv"
        Video_OUTPUT_PATH="./data/output/points_merged_output3.csv"
    elif i == 3:
        CSI_OUTPUT_PATH="./data/output/CSI_loc4.csv"
        Video_OUTPUT_PATH="./data/output/points_merged_output4.csv"
    elif i == 4:
        CSI_OUTPUT_PATH="./data/output/CSI_loc5.csv"
        Video_OUTPUT_PATH="./data/output/points_merged_output5.csv"
    elif i == 5:
        CSI_OUTPUT_PATH="./data/output/CSI_loc6.csv"
        Video_OUTPUT_PATH="./data/output/points_merged_output6.csv"
    elif i == 6:
        CSI_OUTPUT_PATH="./data/output/CSI_loc7.csv"
        Video_OUTPUT_PATH="./data/output/points_merged_output7.csv"
    elif i == 7:
        CSI_OUTPUT_PATH="./data/output/CSI_loc8.csv"
        Video_OUTPUT_PATH="./data/output/points_merged_output8.csv"
    else:
        CSI_OUTPUT_PATH="./data/output/CSI_merged_output.csv"
        Video_OUTPUT_PATH="./data/output/points_merged_output.csv"
    Video_OUTPUT = pd.read_csv(Video_OUTPUT_PATH, header=None)
    Video_OUTPUT=Video_OUTPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
    Video_OUTPUT = np.array(Video_OUTPUT.tolist())
    CSI_OUTPUT = pd.read_csv(CSI_OUTPUT_PATH, header=None)
    CSI_OUTPUT=CSI_OUTPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
    CSI_OUTPUT = np.array(CSI_OUTPUT.tolist())
    CSI_OUTPUT=align_all_coordinates(Video_OUTPUT,CSI_OUTPUT)
    
    a1=PCS(CSI_OUTPUT,Video_OUTPUT,0.025)
    a2=PCS(CSI_OUTPUT,Video_OUTPUT,0.030)
    a3=PCS(CSI_OUTPUT,Video_OUTPUT,0.040)
    a4=PCS(CSI_OUTPUT,Video_OUTPUT,0.050)
    print(f"{a1:.4f}")
    print(f"{a2:.4f}")
    print(f"{a3:.4f}")
    print(f"{a4:.4f}")

# temp(线性层)训练静态数据
# PCS 25=0.7547619047619047
# PCS 30=0.7942857142857143
# PCS 40=0.8329166666666666
# PCS 50=0.8417261904761905
# temp(线性层)训练动态数据
# PCS 25=0.7493601190476191
# PCS 30=0.8119196428571429
# PCS 40=0.9017410714285714
# PCS 50=0.933125

# pft(卷积层)训练静态数据
# PCS 25=0.7327976190476191
# PCS 30=0.7491071428571429
# PCS 40=0.7736309523809524
# PCS 50=0.7948214285714286
# pft(卷积层)训练动态数据
# PCS 25=0.375
# PCS 30=0.4503571428571429
# PCS 40=0.569404761904762
# PCS 50=0.6481845238095238
# pft(卷积层)训练动态数据 卷积大小大
# PCS 25=0.321577380952381
# PCS 30=0.34958333333333336
# PCS 40=0.4152976190476191
# PCS 50=0.457723214285714
# pft(卷积层)训练静态数据 卷积大小大
# PCS 25=0.7561309523809524
# PCS 30=0.7624404761904762
# PCS 40=0.7755952380952381
# PCS 50=0.7933333333333333