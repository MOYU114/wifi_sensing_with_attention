import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
training = False
'''
# 读取前八个csv文件
files = ['CSI_wave_left1.csv', 'CSI_wave_left2.csv', 'CSI_wave_right1.csv', 'CSI_wave_right2.csv','CSI_leg1.csv', 'CSI_leg2.csv', 'CSI_stand1.csv', 'CSI_stand2.csv' ]
data = [pd.read_csv(f'./data/static data/{file}', header=None) for file in files]

# 读取point_left_right_leg_stand.csv文件中的四行信息
point_data = pd.read_csv('./data/static data/point_left_right_leg_stand.csv', header=None)

# 创建新的csv文件point_new.csv
with open('./data/static data/point_new.csv', 'w') as f:
    for i in range(4):
        row = point_data.iloc[i]
        row_str = ','.join(map(lambda x: f'{x:.2e}', row)) + '\n'
        f.write(row_str * (len(data[i*2]) + len(data[i*2+1])))

# 合并前八个csv文件为CSI_new.csv
pd.concat(data).to_csv('./data/static data/CSI_new.csv', index=False)

#图像处理
'''
#读取数据，并对数据进行处理，准备绘制图像
data_path="./data/static data/point_left_right_leg_stand.csv"

data_img=pd.read_csv(data_path, header=None)
SAVE_PATH = "./data/static data/photo/"
data_img=data_img.apply(lambda x: [x[i:i+2] for i in range(0, len(x), 2)], axis=1)
data_img = np.array(data_img.tolist())

def draw_single_pic(i,arrary,pic_name):
    points_num = len(arrary[0])
    x=[]
    y=[]
    colors = np.array(["red","red", "green","green","green","orange","orange","orange", "purple", "purple","purple","cyan","cyan","cyan" ])
    plt.figure(figsize=(1.6, 1.6))
    for j in range(points_num):
        x.append(arrary[i][j][0])
        y.append(arrary[i][j][1])
    #绘制点
    plt.scatter(x, y,c = "black")
    plt.gca().invert_yaxis()
    #链接边
    plt.plot([x[0], x[1]], [y[0], y[1]],c = "black")
    plt.plot([x[1], x[2]], [y[1], y[2]],c = "black")
    plt.plot([x[2], x[3]], [y[2], y[3]],c = "black")
    plt.plot([x[3], x[4]], [y[3], y[4]],c = "black")
    plt.plot([x[1], x[5]], [y[1], y[5]],c = "black")
    plt.plot([x[5], x[6]], [y[5], y[6]],c = "black")
    plt.plot([x[6], x[7]], [y[6], y[7]],c = "black")
    plt.plot([x[1], x[8]], [y[1], y[8]],c = "black")
    plt.plot([x[8], x[9]], [y[8], y[9]],c = "black")
    plt.plot([x[9], x[10]], [y[9], y[10]],c = "black")
    plt.plot([x[1], x[11]], [y[1], y[11]],c = "black")
    plt.plot([x[11], x[12]], [y[11], y[12]],c = "black")
    plt.plot([x[12], x[13]], [y[12], y[13]],c = "black")
    #plt.show()
    plt.axis('off')

    plt.savefig(SAVE_PATH+pic_name,transparent=True)
    plt.close()


# 读取前八个csv文件
files = ['CSI_leg1.csv', 'CSI_leg2.csv', 'CSI_stand1.csv', 'CSI_stand2.csv', 'CSI_wave_left1.csv',
        'CSI_wave_left2.csv', 'CSI_wave_right1.csv', 'CSI_wave_right2.csv']
data = [pd.read_csv(f'./data/static data/{file}', header=None) for file in files]

for i in range(4):
    for j in range(len(data[i * 2]) + len(data[i * 2 + 1])):
        draw_single_pic(i,data_img,"img"+str(i)+"_"+str(j)+".png")



