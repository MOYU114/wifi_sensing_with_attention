import os
import csv

# 获取当前文件夹路径
folder_path = os.getcwd()

# 指定输出文件名
output_file = "CSI_test_legwave_25.csv"

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