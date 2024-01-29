import os

# 获取当前文件夹路径
folder_path = os.getcwd()

# 列出当前文件夹下的所有文件
files = os.listdir(folder_path)

# 循环遍历文件并重命名
for filename in files:
    if os.path.isfile(filename):
        new_filename = "CSI_" + filename
        os.rename(filename, new_filename)
