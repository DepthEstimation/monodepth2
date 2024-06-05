import os
import glob
import random
import argparse
from itertools import islice


## by GPT
# used for filtering aihub dataset
def check_word_in_line(file_path, word, line_number) -> bool:
    try:
        with open(file_path, 'r') as text_file:
            target_line = next(islice(text_file, line_number - 1, line_number), None)

            if target_line is not None:
                if word in target_line:
                    print(f"The word '{word}' exists in line {line_number}.")
                    return True
                else:
                    print(f"The word '{word}' does not exist in line {line_number}.")
                    return False
            else:
                print(f"Error: Line {line_number} does not exist in the file.")
                return False
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False



parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', type=str,
                    help='data path', required=True)
parser.add_argument('--save_path', type=str, help='save path', required=True)
parser.add_argument('--dataset', type=str, help='dataset name', required=True)
args = parser.parse_args()


random.seed(999) # fix seed

# parent directory that contains all folders
# sub folders should directly contain image files
# path = input("path: ")
# path = "assets/frames"



# split_ratio = input("split ratio [train:val] (ex 7:3): ")
split_ratio = "9:1"
a, b = split_ratio.split(":")
a = int(a)
b = int(b)
sum = a+b

path = args.data_path
path = os.path.normpath(path)

print(path)
print(os.path.exists(path))
print(os.path.isdir(path))
if not(os.path.exists(path) and os.path.isdir(path)):
    print("invalid directory path")
    exit()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# print(os.listdir(path))


# odom_test_file_path = "odom_test_{}.txt"
train_file_path = os.path.join(args.save_path, "train_files.txt")
val_file_path = os.path.join(args.save_path, "val_files.txt")

print(train_file_path)
all_file_list = []

dataset = args.dataset

if dataset in ['kitti', 'kitti_resize']:
    os.chdir(path)

    date_folders = os.listdir(path)

    # temporary setting to create small KITTI split
    count = 0

    for df in date_folders:
        if os.path.isdir(df):

            sub_folders = os.listdir(os.path.join(path, df))
            # os.chdir(os.path.join(path, folder))
            # print(sub_folders)
            for sf in sub_folders:
                if os.path.isdir(os.path.join(df, sf)):

                    # temporary setting to create small KITTI split
                    if count > 0:
                        break
                    count += 1

                    # print('yes')
                    # file_list = sorted(glob.glob(os.path.join(folder, 'sensor_raw_data', 'camera', "*.jpg")))
                    file_list = sorted(glob.glob(os.path.join(df, sf, 'image_02', 'data', "*.jpg")))
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

                        all_file_list.append(os.path.join(df, sf) + " " + str(f_num) + " " + "l" + "\n") # str(int(f_num)) ???

elif dataset == 'kitti_campus':
    os.chdir(path)

    date_folders = os.listdir(path)
    # print(folders)

    for df in date_folders:
        if os.path.isdir(df):


            sub_folders = os.listdir(os.path.join(path, df))
            # os.chdir(os.path.join(path, folder))
            # print(sub_folders)
            for sf in sub_folders:
                if os.path.isdir(os.path.join(df, sf)):

                    # print('yes')
                    # file_list = sorted(glob.glob(os.path.join(folder, 'sensor_raw_data', 'camera', "*.jpg")))
                    file_list = sorted(glob.glob(os.path.join(df, sf, 'image_00', 'data', "*.jpg")))
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

                        all_file_list.append(os.path.join(df, sf) + " " + str(int(f_num)) + " " + "l" + "\n")

elif dataset in ['aihub88', 'aihub88_resize']:
    # for AI Hub dataset
    date_folders = os.listdir(path)
    os.chdir(path)
    for df in date_folders:
        if os.path.isdir(df):
            # print('yes')

            # json파일 열어서 check 맑음
            # if 맑음 있음 -> 하고 아니면 continue

            # if not check_word_in_line(f'{folder}/{folder}_meta_data.json', '맑음', 7):
            #     continue

            if not check_word_in_line(f'{df}/{df}_meta_data.json', 'DN8', 2):
                continue

            file_list = sorted(glob.glob(os.path.join(df, 'sensor_raw_data', 'camera', "*.jpg")))
            # file_list = sorted(glob.glob(os.path.join(folder, '**', '**', 'image_00', 'data', "*.jpg")))
            # print(len(file_list))
            # f.writelines(file_list)
            # format is <folder> <number> <r | l>

            # # for individual file
            # with open(odom_test_file_path.format(folder), 'w') as f:
            #     for item in file_list:
            #         f.write(folder + " " + str(int("".join(os.path.splitext(os.path.basename(item))[:-1]))) + " " + "l" + "\n")
            
            # for train and val files
            
            # need to remove first and last one because of frame_index [0, -1, 1]
            file_list = file_list[2:-2]
            for item in file_list:
                f_name = os.path.splitext(os.path.basename(item))
                f_num = "".join(f_name[:-1]).split('_')[-1]

                all_file_list.append(df + " " + str(int(f_num)) + " " + "l" + "\n")
elif dataset == 'boreas':
    print(path)
    date_folders = os.listdir(path)
    os.chdir(path)
    for df in date_folders:
        if os.path.isdir(df):
            print(df)
            file_list = sorted(glob.glob(os.path.join(df, 'camera_renamed', "*.png")))
            file_list = file_list[1:-1]
            for item in file_list:
                f_name = os.path.splitext(os.path.basename(item))
                f_num = f_name[0]
                print(f_num)
                all_file_list.append(df + '/camera_renamed' + " " + str(f_num) + " " + "l" + "\n")


elif dataset == 'kitti360':
    print(path)
    date_folders = os.listdir(path)
    os.chdir(path)
    for df in date_folders:
        if os.path.isdir(df):
            print(df)
            file_list = sorted(glob.glob(os.path.join(df, 'image_02', 'data_rgb', "*.png")))
            file_list = file_list[1:-1] # 첫번째와 마지막은 제외 시킨다
            for item in file_list:
                f_name = os.path.splitext(os.path.basename(item))
                f_num = f_name[0]
                print(f_num)
                all_file_list.append(df + '/image_02/data_rgb' + " " + str(f_num) + " " + "l" + "\n")

elif dataset == 'ncn_ch0':
    print(path)
    date_folders = os.listdir(path)
    os.chdir(path)
    for df in date_folders:
        if os.path.isdir(df):
            print(df)
            file_list = sorted(glob.glob(os.path.join(df, "*.jpg")))
            file_list = file_list[1:-1]
            for item in file_list:
                f_name = os.path.splitext(os.path.basename(item))
                f_num = f_name[0]
                print(f_num)
                all_file_list.append(df + " " + str(f_num) + " " + "l" + "\n")

elif dataset == 'hgu2023':
    print(path)
    date_folders = os.listdir(path)
    os.chdir(path)
    for df in date_folders:
        if os.path.isdir(df):
            print(df)
            file_list = sorted(glob.glob(os.path.join(df, "*.jpg")))
            file_list = file_list[1:-1]
            for item in file_list:
                f_name = os.path.splitext(os.path.basename(item))
                f_num = f_name[0]
                print(f_num)
                all_file_list.append(df + " " + str(f_num) + " " + "l" + "\n")


random.shuffle(all_file_list)
file_len = len(all_file_list)
# print(all_file_list)

split_index = int(file_len*(a/sum))

with open(train_file_path, 'w') as f:
    for item in all_file_list[:split_index]:
        f.write(item)

with open(val_file_path, 'w') as f:
    for item in all_file_list[split_index:]:
        f.write(item)

print("split file created in data path")

# D:\CGVL\DepthEstimation\HandongDataset\camera_eun\images