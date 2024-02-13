import os
import glob
import random

import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--data_path', type=str,
                    help='data path', required=True)
parser.add_argument('--save_path', type=str, help='save path', required=True)

args = parser.parse_args()


random.seed(999) # fix seed

# parent directory that contains all folders
# sub folders should directly contain image files
# path = input("path: ")
# path = "assets/frames"

path = args.data_path

# split_ratio = input("split ratio [train:val] (ex 7:3): ")
split_ratio = "9:1"
a, b = split_ratio.split(":")
a = int(a)
b = int(b)
sum = a+b


if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if not(os.path.exists(path) and os.path.isdir(path)):
    print("invalid directory path")
    exit()

path = os.path.normpath(path)
# print(os.listdir(path))
# print(os.listdir(path))


# odom_test_file_path = "odom_test_{}.txt"
train_file_path = os.path.join(args.save_path, "train_files.txt")
val_file_path = os.path.join(args.save_path, "val_files.txt")

print(train_file_path)

all_file_list = []

'''
folders = os.listdir(path)
os.chdir(path)
for folder in folders:
    if os.path.isdir(folder):
        # print('yes')
        # file_list = sorted(glob.glob(os.path.join(folder, 'sensor_raw_data', 'camera', "*.jpg")))
        file_list = sorted(glob.glob(os.path.join(folder, '**', '**', 'image_00', 'data', "*.jpg")))
        # print(len(file_list))
        # f.writelines(file_list)
        # format is <folder> <number> <r | l>

        # # for individual file
        # with open(odom_test_file_path.format(folder), 'w') as f:
        #     for item in file_list:
        #         f.write(folder + " " + str(int("".join(os.path.splitext(os.path.basename(item))[:-1]))) + " " + "l" + "\n")
        
        # for train and val files
        # need to remove first and last one because of frame_index [0, -1, 1]
        file_list = file_list[1:-1]
        for item in file_list:
            f_name = os.path.splitext(os.path.basename(item))
            f_num = "".join(f_name[:-1]).split('_')[-1]

            all_file_list.append(folder + " " + str(int(f_num)) + " " + "l" + "\n")
'''

os.chdir(path)

folders = os.listdir(path)
# print(folders)

for folder in folders:
    if os.path.isdir(folder):


        sub_folders = os.listdir(os.path.join(path, folder))
        # os.chdir(os.path.join(path, folder))
        # print(sub_folders)
        for sf in sub_folders:
            if os.path.isdir(os.path.join(folder, sf)):

                # print('yes')
                # file_list = sorted(glob.glob(os.path.join(folder, 'sensor_raw_data', 'camera', "*.jpg")))
                file_list = sorted(glob.glob(os.path.join(folder, sf, 'image_00', 'data', "*.jpg")))
                # print(len(file_list))
                # f.writelines(file_list)
                # format is <folder> <number> <r | l>

                # # for individual file
                # with open(odom_test_file_path.format(folder), 'w') as f:
                #     for item in file_list:
                #         f.write(folder + " " + str(int("".join(os.path.splitext(os.path.basename(item))[:-1]))) + " " + "l" + "\n")
            
                # for train and val files
                # need to remove first and last one because of frame_index [0, -1, 1]
                file_list = file_list[1:-1]
                for item in file_list:
                    f_name = os.path.splitext(os.path.basename(item))
                    f_num = "".join(f_name[:-1]).split('_')[-1]

                    all_file_list.append(os.path.join(folder, sf) + " " + str(int(f_num)) + " " + "l" + "\n")


random.shuffle(all_file_list)
file_len = len(all_file_list)

split_index = int(file_len*(a/sum))

with open(train_file_path, 'w') as f:
    for item in all_file_list[:split_index]:
        f.write(item)

with open(val_file_path, 'w') as f:
    for item in all_file_list[split_index:]:
        f.write(item)

print("split file created in data path")

# D:\CGVL\DepthEstimation\HandongDataset\camera_eun\images