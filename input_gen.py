import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
training=True
#读取数据，并对数据进行处理，准备绘制图像

Video_INPUT_PATH="./data/points_test_legwave.csv"

Video_INPUT_TRAINING_PATH="./data/points_train_legwave.csv"
if training:
    Video_INPUT = pd.read_csv(Video_INPUT_TRAINING_PATH, header=None)
    SAVE_PATH = "./data/input/training/"
else:
    Video_INPUT = pd.read_csv(Video_INPUT_PATH, header=None)
    #SAVE_PATH = "./data/input/test/"
#width=160
#height=120
Video_INPUT=Video_INPUT.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
Video_INPUT = np.array(Video_INPUT.tolist())
pics_num = len(Video_INPUT)
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
    # 隐藏坐标轴
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    #plt.show()

    plt.savefig(SAVE_PATH+pic_name)
    plt.clf()


for i in range(4000):
    num=i+1
    draw_single_pic(i,Video_INPUT,"Video_OUTPUT_"+str(num)+".png")
