import os

for i in range(10):
    PATH="./result/test"+str(i+1)+".txt"
    os.system("python points_for_training.py")
    os.system("python performance.py > " + PATH)

