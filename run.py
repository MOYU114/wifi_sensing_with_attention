import os

for i in range(10):
    PATH="./test"+str(i+1)+".txt"
    os.system("python temp.py")
    os.system("python performance.py > " + PATH)

