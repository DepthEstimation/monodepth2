# Video preprocessor
# Extracts frames from videos and stores it in .jpg format

##
# parent_folder
#   - videos
#   - frames
##

## Usage
# provide videos folder and it will create images in the frames folder
# images will be stored in folder with video names
# will skip if directory exists

import os
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Get working directory")

# Add a named parameter
parser.add_argument("--wd", type=str, required=True, help="path to working directory (directory that contains videos)")

# Parse the command-line arguments
args = parser.parse_args()

# Access the value of the named parameter
wd = args.wd

if not (os.path.exists(wd) and os.path.isdir(wd)):
    print("provided wd does not exits: {}".format(wd))
    exit()

# scale값 변경으로 output imagesize 조절 가능
# -vf scale=1280:-1
# crop 하기:
#   -filter:v crop=...
# cmd = 'ffmpeg -i {} -qscale:v 2 -vf "scale=640:-1,crop=640:352:0:0" -start_number 0 {}/%10d.jpg'
cmd = 'ffmpeg -i {} -qscale:v 2 -start_number {} {}/%10d.jpg'

os.chdir(wd)
files = os.listdir(".")
# print(files)

for file in files:
    if os.path.isfile(file):
        file_extension = os.path.splitext(file)[-1]
        if file_extension in [".avi"]:
            # 아래 directory 부분 수정 필요
            new_dir = os.path.join(f"../../NCN_frames_together/{wd.split('/')[-1]}", "".join(os.path.splitext(os.path.basename(file))[:-1]))
            if os.path.exists(new_dir) and os.path.isdir(new_dir):
                continue
            else:
                os.mkdir(new_dir)
            start_num = len([name for name in os.listdir(new_dir) if os.path.isfile(name)])
            print(start_num)
            os.system(cmd.format(start_num, file, new_dir))
        else:
            print("not .avi")
    else:
        print(file)
        print("not file")

# working


# "".join(os.path.splitext(os.path.basename(test))[:-1])

# ffmpeg -i newton.mp4 -qscale:v 2 newton/%10d.jpg

# ffmpeg -i newton.mp4 -qscale:v 2 -vf scale=1280:-1 newton/%10d.jpg