# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:35:31 2023

@author: Administrator
"""
import os
import json
import numpy as np

def AnalysisJson():
    file_path = "E:\\openpose\\openpose1\\output\\test\\" #一个视频经过OpenPose处理后，每一帧的人体骨架点JSON数据所在文件夹
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
    k14_p1 = points[0:800,0:8,0:2]
    k14_p2 = points[0:800,9:15,0:2]
    k14 = np.concatenate((k14_p1, k14_p2), axis=1)
    
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


    
    OutTextPath = "E:\\openpose\\openpose1\\output\\test\\points_test.csv"
    np.savetxt(OutTextPath, k14.reshape(len(k14),-1),delimiter=',')
    
if __name__ == '__main__':
    AnalysisJson()


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

