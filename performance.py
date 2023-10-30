import pandas as pd
import numpy as np
import math


CSI_OUTPUT_PATH="./data/output/CSI_merged_output.csv"
Video_OUTPUT_PATH="./data/output/points_merged_output.csv"
Real_Image_PATH="./data/output/real_output_training.csv"
CSI_OUTPUT = pd.read_csv(CSI_OUTPUT_PATH, header=None)
Video_OUTPUT = pd.read_csv(Video_OUTPUT_PATH, header=None)
Real_Image=pd.read_csv(Real_Image_PATH, header=None)
CSI_OUTPUT=CSI_OUTPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
Video_OUTPUT=Video_OUTPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
Real_Image=Real_Image.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
CSI_OUTPUT = np.array(CSI_OUTPUT.tolist())
Video_OUTPUT = np.array(Video_OUTPUT.tolist())
Real_Image = np.array(Real_Image.tolist())
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

# print(CSI_OUTPUT)
# print(Video_OUTPUT)
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
print(f"PCS 25={PCS(CSI_OUTPUT,Video_OUTPUT,0.025)}")
print(f"PCS 30={PCS(CSI_OUTPUT,Video_OUTPUT,0.030)}")
print(f"PCS 40={PCS(CSI_OUTPUT,Video_OUTPUT,0.040)}")
print(f"PCS 50={PCS(CSI_OUTPUT,Video_OUTPUT,0.050)}")