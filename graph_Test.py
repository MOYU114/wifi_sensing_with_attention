import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
training = False
#读取数据，并对数据进行处理，准备绘制图像
CSI_OUTPUT_PATH="./data/output/CSI_merged_output_training.csv"
Video_OUTPUT_PATH="./data/output/real_output_training.csv"
# CSI_OUTPUT_PATH="./data/output/CSI_1.csv"
# Video_OUTPUT_PATH="./data/static/data/device/points_right_left_leg_stand_sit.csv"
CSI_OUTPUT_TRAINING_PATH="./data/output/real_output_training.csv"
Video_OUTPUT_TRAINING_PATH="./data/output/points_merged_output_training.csv"
if training:
    CSI_OUTPUT = pd.read_csv(CSI_OUTPUT_TRAINING_PATH, header=None)
    Video_OUTPUT = pd.read_csv(Video_OUTPUT_TRAINING_PATH, header=None)
    SAVE_PATH = "./data/output/photo/test/"
else:
    CSI_OUTPUT = pd.read_csv(CSI_OUTPUT_PATH, header=None)
    Video_OUTPUT = pd.read_csv(Video_OUTPUT_PATH, header=None)
    SAVE_PATH = "./data/output/photo/test/"

CSI_OUTPUT=CSI_OUTPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
Video_OUTPUT=Video_OUTPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
CSI_OUTPUT = np.array(CSI_OUTPUT.tolist())
Video_OUTPUT = np.array(Video_OUTPUT.tolist())

def draw_single_pic(i,arrary,pic_name):
    points_num = len(arrary[0])
    x=[]
    y=[]
    colors = np.array(["red","red", "green","green","green","orange","orange","orange", "purple", "purple","purple","cyan","cyan","cyan" ])
    for j in range(points_num):
        x.append(arrary[i][j][0])
        y.append(arrary[i][j][1])
    #绘制点
    plt.scatter(x, y,c = colors)
    plt.gca().invert_yaxis()
    #链接边
    plt.plot([x[0], x[1]], [y[0], y[1]])
    plt.plot([x[1], x[2]], [y[1], y[2]])
    plt.plot([x[2], x[3]], [y[2], y[3]])
    plt.plot([x[3], x[4]], [y[3], y[4]])
    plt.plot([x[1], x[5]], [y[1], y[5]])
    plt.plot([x[5], x[6]], [y[5], y[6]])
    plt.plot([x[6], x[7]], [y[6], y[7]])
    plt.plot([x[1], x[8]], [y[1], y[8]])
    plt.plot([x[8], x[9]], [y[8], y[9]])
    plt.plot([x[9], x[10]], [y[9], y[10]])
    plt.plot([x[1], x[11]], [y[1], y[11]])
    plt.plot([x[11], x[12]], [y[11], y[12]])
    plt.plot([x[12], x[13]], [y[12], y[13]])
    # 去除坐标轴上的数字
    plt.xticks([])
    plt.yticks([])
    plt.axis('tight')
    plt.show()
    # plt.savefig(SAVE_PATH+pic_name,dpi=600)
    plt.clf()
pics_num = 50
base=20
for i in range(base,base+pics_num):
    num=i+1
    draw_single_pic(i,CSI_OUTPUT,"CSI_OUTPUT_"+str(num)+".png")
    draw_single_pic(i,Video_OUTPUT,"Video_OUTPUT_"+str(num)+".png")
    #draw_single_pic(i,CSI_OUTPUT,"CSI_OUTPUT_"+str(num)+".png")
    #draw_single_pic(i,Video_OUTPUT,"Video_OUTPUT_"+str(num)+".png")

