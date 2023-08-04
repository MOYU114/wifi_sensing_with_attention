import os
import csv

# 获取当前文件夹路径
folder_path = os.getcwd()

# 指定输出文件名
output_file = "points_in.csv"

# 获取文件夹中所有的 CSV 文件名
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# 合并文件
with open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)

    # 遍历每个 CSV 文件
    for file in csv_files:
        file_path = os.path.join(folder_path, file)

        # 读取 CSV 文件并将内容写入输出文件
        with open(file_path, "r") as infile:
            reader = csv.reader(infile)
            writer.writerows(reader)

print("CSV 文件合并完成！")

# import glob
# import pandas as pd

# # 获取当前文件夹下所有的CSV文件
# csv_files = glob.glob('*.csv')

# # 创建一个空的DataFrame来存储合并后的数据
# merged_data = pd.DataFrame()

# # 遍历每个CSV文件
# for file in csv_files:
#     if len(merged_data) == 0:
#         merged_data = pd.read_csv(file, nrows=3000)
#         print(merged_data.shape)
#     else:
#         # 读取CSV文件的前300行数据
#         data = pd.read_csv(file, nrows=3000)
#         merged_data.to_csv('merged.csv', index=False)
#         print(data.shape)
    
#         # 将当前文件的数据合并到总的数据中
#         merged_data = merged_data.append(data, ignore_index=True)
#         print(merged_data.shape)
    

# # 将合并后的数据写入新的CSV文件
# merged_data.to_csv('merged.csv', index=False)
# print("CSV 文件合并完成！")