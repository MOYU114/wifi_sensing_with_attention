import os
import pandas as pd

# 合并当前文件夹下的所有CSV文件到一个CSV文件
# 获取当前文件夹路径
folder_path = os.getcwd()

# 获取文件夹中所有CSV文件的路径
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".csv")]

# 创建一个空的DataFrame来保存合并后的数据
data_frames = []

# 遍历每个CSV文件
for file_path in file_paths:
    # Read the CSV file into a DataFrame and append it to the list
    data = pd.read_csv(file_path, header=None)
    data_frames.append(data)

# 将合并后的数据保存为新的CSV文件
merged_data = pd.concat(data_frames, ignore_index=True)

# Save the merged data as a new CSV file
merged_data.to_csv("merged.csv", index=False)