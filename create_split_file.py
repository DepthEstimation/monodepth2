import os
import glob
import random

# parent directory that contains all folders
# sub folders should directly contain image files
path = input("path: ")
split_ratio = input("split ratio [train:val] (ex 7:3): ")
a, b = split_ratio.split(":")
a = int(a)
b = int(b)
sum = a+b


if not(os.path.exists(path) and os.path.isdir(path)):
    print("invalid directory path")
    exit()

path = os.path.normpath(path)
os.chdir(path)

odom_test_file_path = "odom_test_{}.txt"
train_file_path = os.path.join("train_files.txt")
val_file_path = os.path.join("val_files.txt")


all_file_list = []

for folder in os.listdir(path):
    if os.path.isdir(folder):
        file_list = sorted(glob.glob(os.path.join(folder, "*.jpg")))
        print(len(file_list))
        # f.writelines(file_list)
        # format is <folder> <number> <r | l>

        # for individual file
        with open(odom_test_file_path.format(folder), 'w') as f:
            for item in file_list:
                f.write(folder + " " + str(int("".join(os.path.splitext(os.path.basename(item))[:-1]))) + " " + "l" + "\n")
        
        # for train and val files
        # need to remove first and last one because of frame_index [0, -1, 1]
        file_list = file_list[1:-1]
        for item in file_list:
            all_file_list.append(folder + " " + str(int("".join(os.path.splitext(os.path.basename(item))[:-1]))) + " " + "l" + "\n")


random.shuffle(all_file_list)
file_len = len(all_file_list)

split_index = int(file_len*(a/sum))

with open(train_file_path, 'w') as f:
    for item in all_file_list[:split_index]:
        f.write(item)

with open(val_file_path, 'w') as f:
    for item in all_file_list[split_index:]:
        f.write(item)


# D:\CGVL\DepthEstimation\HandongDataset\camera_eun\images
