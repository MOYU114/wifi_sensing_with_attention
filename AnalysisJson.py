# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:35:31 2023

@author: Administrator
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def AnalysisJson(path):
    #一个视频经过OpenPose处理后，每一帧的人体骨架点JSON数据所在文件夹
    file_path = "data\\static\\datawall\\" + path + "\\"
    files = os.listdir(file_path)  # 遍历file_path下所有的子目录及文件
    points = np.zeros((len(files),75))
    i = 0
    for file in files:  #遍历当前路径下所有非目录子文件
            InputPath = open(file_path + file, encoding="utf-8")
            temp = json.load(InputPath)      #json格式数据转换为python字典类型
            points[i,:] = (temp.get('people')[0]).get('pose_keypoints_2d')
            if points[i,0] == 0:
                if points[i,51] == 0 and points[i,54] != 0:
                    points[i,0] = points[i,3]
                    points[i,1] = points[i,55]
                elif points[i,51] != 0 and points[i,54] == 0:
                    points[i,0] = points[i,3]
                    points[i,1] = points[i,52]
                elif points[i,51] != 0 and points[i,54] != 0:
                    points[i,0] = (points[i,51]+points[i,54])/2
                    points[i,1] = points[i,52]
                else:
                    points[i,0] = points[i,3]
                    points[i,1] = points[i,4] + abs(points[i,3]-points[i,6])
            i = i+1
            #temp["data"]     # 因为此时已经转换为了字典类型，对key(data) 取值后可以得到每天的具体空气数据的value值
    points = points.reshape(len(points),25,3)   
    k14_p1 = points[40:1240,0:8,0:2]
    k14_p2 = points[40:1240,9:15,0:2]
    k14 = np.concatenate((k14_p1, k14_p2), axis=1)
    
    # return points, k14
    
# 本方案中14个点分别表示的内容，跟OpenPose的稍微不一样，我们取的是它25个点中的第0,1,2,3,4,5,6,7,9,10,11,12,13,14这14个点。    
#     {0,  "Nose"}
#     {1,  "Neck"}
#     {2,  "RShoulder"}
#     {3,  "RElbow"}
#     {4,  "RWrist"}
#     {5,  "LShoulder"}
#     {6,  "LElbow"}
#     {7,  "LWrist"}
#     {8,  "RHip"}
#     {9, "RKnee"}
#     {10, "RAnkle"}
#     {11, "LHip"}
#     {12, "LKnee"}
#     {13, "LAnkle"}

# k14(28)-k14(22)>20表示左腿或者右腿在抬起。k14(10)-k14(6)>180表示双臂在抬起。


    
    OutTextPath = "data\\static\\datawall\\" + path + ".csv"
    np.savetxt(OutTextPath, k14.reshape(len(k14),-1),delimiter=',')

def draw_single_pic25(i,arrary,pic_name):
    points_num = len(arrary[0])
    x=[]
    y=[]
    colors = np.array(["red","red","orange","yellow","yellow", "green","green","green","red","cyan","cyan", "cyan", "blue","blue","blue","pink","purple","pink","purple","blue","blue","blue","cyan","cyan","cyan" ])
    for j in range(points_num):
        x.append(arrary[i][j][0])
        y.append(arrary[i][j][1])
    #绘制点
    plt.scatter(x, y,c = colors)
    plt.gca().invert_yaxis()
    #链接边
    plt.plot([x[0], x[1]], [y[0], y[1]])
    plt.plot([x[0], x[15]], [y[0], y[15]])
    plt.plot([x[15], x[17]], [y[15], y[17]])
    plt.plot([x[0], x[16]], [y[0], y[16]])
    plt.plot([x[16], x[18]], [y[16], y[18]])
    plt.plot([x[1], x[2]], [y[1], y[2]])
    plt.plot([x[2], x[3]], [y[2], y[3]])
    plt.plot([x[3], x[4]], [y[3], y[4]])
    plt.plot([x[1], x[5]], [y[1], y[5]])
    plt.plot([x[5], x[6]], [y[5], y[6]])
    plt.plot([x[6], x[7]], [y[6], y[7]])
    plt.plot([x[1], x[8]], [y[1], y[8]])
    plt.plot([x[8], x[9]], [y[8], y[9]])
    plt.plot([x[9], x[10]], [y[9], y[10]])
    plt.plot([x[10], x[11]], [y[10], y[11]])
    plt.plot([x[11], x[22]], [y[11], y[22]])
    plt.plot([x[11], x[24]], [y[11], y[24]])
    plt.plot([x[22], x[23]], [y[22], y[23]])
    plt.plot([x[8], x[12]], [y[8], y[12]])
    plt.plot([x[13], x[14]], [y[13], y[14]])
    plt.plot([x[12], x[13]], [y[12], y[13]])
    plt.plot([x[14], x[21]], [y[14], y[21]])
    plt.plot([x[14], x[19]], [y[14], y[19]])
    plt.plot([x[19], x[20]], [y[19], y[20]])
    plt.savefig("./data/output/photo/point25.pdf")
    plt.show()
    plt.clf()
    
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
    plt.savefig("./data/output/photo/point14.pdf", format="pdf")
    plt.show()
    plt.clf()
    
if __name__ == '__main__':
    AnalysisJson("wave_right_6C")
    # for i in range(15):
    #     if i == 0:
    #         AnalysisJson("arm_left_c109")
    #     elif i == 1:
    #         AnalysisJson("arm_right")
    #     elif i == 2:
    #         AnalysisJson("arm_right_c109")
    #     elif i == 3:
    #         AnalysisJson("leg")
    #     elif i == 4:
    #         AnalysisJson("leg_c109")
    #     elif i == 5:
    #         AnalysisJson("leg_left")
    #     elif i == 6:
    #         AnalysisJson("leg_right")
    #     elif i == 7:
    #         AnalysisJson("sit")
    #     elif i == 8:
    #         AnalysisJson("arm_left")
    #     elif i == 9:
    #         AnalysisJson("stand")
    #     elif i == 10:
    #         AnalysisJson("stand_c109")
    #     elif i == 11:
    #         AnalysisJson("wave_left")
    #     elif i == 12:
    #         AnalysisJson("wave_right")
    #     elif i == 13:
    #         AnalysisJson("wave_right_MI")
    #     elif i == 14:
    #         AnalysisJson("wave_right_H6C")
    #     # elif i == 15:
    #     #     AnalysisJson("arm_left")
    #     else:
    #         AnalysisJson("wave_right_6C")
    # po25,po14 = AnalysisJson()
    # draw_single_pic25(247,po25,"CSI_OUTPUT")
    # draw_single_pic(247,po14,"CSI_OUTPUT")
    


            # for i in base.values():
            #     AQI_str = ",".join(i)   #将列表转换为csv的形式(以逗号分隔)
            #     fileOut = open(OutTextPath, "a", encoding='utf8')
            #     fileOut.write(AQI_str + '\n')
            #     fileOut.close()
            
# def trans25to18():
#     json18_path = 'E://openpose//openpose1//output//18.json'
#     json25_path = 'E://openpose//openpose1//output//test2//test2_000000000008_keypoints.json'
#     # dict={}

#     def joint_map(k25):
#         k18 = []
#         joint_index = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
#         for k in joint_index:
#             k18.append(k25[3*k])
#             k18.append(k25[3*k+1])
#             k18.append(k25[3*k+2])
#         assert len(k18) == 18*3
#         return k18
        
#     def get_json_data(json_path):
#         with open(json_path,'rb') as f:
#             params = json.load(f)
#             for i in range(len(params['people'])):
#                 params['people'][i]['pose_keypoints_2d'] = joint_map(params['people'][i]['pose_keypoints_2d'])
#             dict = params
#         f.close()
#         return dict
    
#     def write_json_data(dict):
#         with open(json18_path,'w') as r:
#             json.dump(dict,r)
#         r.close()
    
#     the_revised_dict = get_json_data(json25_path)
#     write_json_data(the_revised_dict)

