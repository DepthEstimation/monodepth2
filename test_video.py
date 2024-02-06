# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

# Code: test_video.py
# In this code, there are four command
# 1. split a video into images
# 2. run test_simple.py
# 3. creating a depth map video
# 4. creating a concatenated video and log file
# User should specify the --model_name, --file_name

import os
import cv2
import time
import datetime
import argparse
import glob


# Input video file should be placed under `assets/original_video` folder
assets_path = "assets"
original_video_path = f"{assets_path}/original_video"
# splits_path = f"{assets_path}/splits"

def print_message(msg):
    print(msg)
    print("..")

def store_video_info(file_name, total_time, log_path):
    # creating a log file and store it
    print("storing video information...")
    cap = cv2.VideoCapture(f"{original_video_path}/{file_name}")

    if not cap.isOpened():
        print("could not open :", f"{original_video_path}/{file_name}")
        exit(0)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    video_info = f"Video Info:\n> file path : {assets_path}/original_video/{file_name}\n> length : {length}\n> width : {width}\n> height : {height}\n> fps : {round(fps)}\n> reflective : {args.reflective}\n"

    evaluation_info = f"\nEvaluation:\n> Average of execution time per image: {total_time/length:.5f} sec\n"

    current_time = datetime.datetime.now()
    output_file_path = f"{log_path}/log_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    with open(output_file_path, "w") as f:
        f.write(video_info)
        f.write(evaluation_info)

    print_message(video_info + evaluation_info)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use')
    parser.add_argument('--reflective', type=bool,
                        help='using reflective padding')
    parser.add_argument('--file_name', type=str,
                        help='using reflective padding')

    return parser.parse_args()

def test(args):
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    # Specify the name of model and video file
    model = args.model_name
    file_name = args.file_name
    total_time = 0

    # Paths
    frames_path = f"{assets_path}/frames"
    depth_path = f"assets/models/{model}/depth"
    depth_video_path = f"assets/models/{model}/depth_video"
    comparison_video_path = f"assets/models/{model}/comparison_video"
    pose_path = f"assets/models/{model}/pose"
    log_path = f"assets/models/{model}/log"

    model_paths = [depth_path, depth_video_path, comparison_video_path, pose_path, log_path]

    # check if given video file exists in "assets/original_video"
    if not os.path.isfile(os.path.join(original_video_path, file_name)):
        print(f"--file_name {file_name}\nNo such file in the {original_video_path}")
        exit()

    base_name = os.path.splitext(file_name)[0]
    frames_base = base_name
    if args.reflective:
        base_name = base_name + '_reflective'
    base_paths = [path + f'/{base_name}' for path in model_paths]

    # Concatenated video file path
    output_file = f'{comparison_video_path}/{base_name}/comp_{base_name}.mp4'
    
    # Check if the concatenated video file path exists
    if not os.path.isfile(output_file):
        for path in base_paths:
            # Command 0s
            # create folders to make a workspace
            command0 = f'mkdir -p {path}'
            os.system(command0)
        command0 = f'mkdir -p {frames_path}/{frames_base}'
        os.system(command0)
        print_message("Create folders (command 0 executed)!")

        # Command 1
        # split the video into images and place them in the 'frames' folder"
        # must be "jpg" NOT "jpeg"
        command1 = f"ffmpeg -i {original_video_path}/{file_name} -qscale:v 2 -start_number 0 {frames_path}/{frames_base}/%010d.jpg"
        if len(os.listdir(os.path.join(frames_path, frames_base))) == 0:
            os.system(command1)
            print_message("Extract frames (command 1 executed)!")
        else:
            print_message('skip extracting frames (command 1 skipped)!')
    else:
        print_message('skip creating folders (command 0 skipped)!')
        print_message('skip extracting frames (command 1 skipped)!')

    # Command 2
    # run the test_simple.py and calculate the execution time
    # depth map image is created and saved into the 'test_out_frames'
    if len(os.listdir(os.path.join(depth_path, base_name))) == 0:
        command2 = f"python test_simple.py --image_path {frames_path}/{frames_base} --out {depth_path}/{base_name} --model_name {model}"
        if args.reflective:
            command2 += ' --reflective true'
        start = time.time()
        os.system(command2)
        end = time.time()
        total_time += (end - start)
        print_message("run test.simple.py (command 2 executed)!")
        store_video_info(file_name, total_time, f'{log_path}/{base_name}')
    else:
        print_message('skip depth estimation (command 2 skipped)!')

    # Command 3
    # create a video using the depth map image
    video_file = file_name
    if args.reflective:
        video_file = video_file.replace('.mp4', '_reflective.mp4')
    # command3 = f"ffmpeg -framerate 30 -pattern_type glob -i '{depth_path}/{base_name}/*.jpeg' -c:v h264_nvenc -pix_fmt yuv420p {depth_video_path}/{base_name}/{video_file}"
    command3 = f"ffmpeg -framerate 30 -pattern_type glob -i '{depth_path}/{base_name}/*.jpeg' -c:v copy -pix_fmt yuv420p {depth_video_path}/{base_name}/{video_file}"
    command3_1 = f'ffmpeg -i {depth_video_path}/{base_name}/{video_file} -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 {depth_video_path}/{base_name}/{base_name}.gif'

    if not os.path.exists(os.path.join(f'{depth_video_path}/{base_name}', video_file)):
        os.system(command3)
        print_message('creating depth video (command 3 executed)!')

        os.system(command3_1)
    else:
        print_message('skip creating depth video (command 3 skipped)!')

        if not os.path.exists(os.path.join(f'{depth_video_path}/{base_name}', f'{base_name}.gif')):
            os.system(command3_1)
            print_message('creating depth gif (command 3_1)!')
        else:
            print_message('skip creating depth gif (command 3_1)!')



    # Command 4
    # create a concatenated video with orignal video and depth map video
    # input video file path: input_file1, input_file2
    input_file1 = f'{original_video_path}/{file_name}' # original
    input_file2 = f'{depth_video_path}/{base_name}/{video_file}' # depth video
    command4 = f'ffmpeg -i {input_file1} -i {input_file2} -filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" {output_file}'
    if not os.path.exists(os.path.join(f'{comparison_video_path}/{base_name}', f"comp_{video_file}")):
        os.system(command4)
        print_message('concatenating 2 videos (command 4 executed)!')
    else:
        print_message('skip concatenating 2 videos (command 4 skipped)!')

    # Command 5
    # evaluate pose
    # split file
    # ... 

    # need to create split file before doing this
    # odom_test_file_path = "pose_{}.txt"
    # file_list = sorted(glob.glob(os.path.join(frames_path, base_name, "*.jpg")))
    # file_list = file_list[:-1]
    # with open(os.path.join(splits_path, odom_test_file_path.format(base_name)), 'w') as f:
    #     for item in file_list:
    #         f.write(os.path.join(base_name) + " " + str(int("".join(os.path.splitext(os.path.basename(item))[:-1]))) + " " + "l" + "\n")
    command5 = f"python test_pose.py --eval_split {base_name} --load_weights_folder ./models/{model} --data_path {assets_path}"
    if not os.path.exists(f"{pose_path}/{base_name}/poses.npy"):
        os.system(command5)
        print_message('evaluating pose (command 5 executed)!')
    else: 
        print('poses.npy already exists. skip generating new one.')
    
    # Command 6
    # convert pose to trajectory
    command6 = f"python convert_np_pose_to_traj_txt.py --file {pose_path}/{base_name}/poses.npy --output_path {pose_path}/{base_name}"
    os.system(command6)

    # # Command 7
    # # draw the trajectory
    command7 = f"python draw_pose_trajectory.py --file_path {pose_path}/{base_name} --file_name poses"
    os.system(command7)
    # print_message('converting pIf you are familiar with Alan Turing, then you might be surprised that a famous computer scientist would appear in the first sentence of a book on biological modeling. After all, he is best known for two achievements that have nothing to do with biology. In 1936, Turing theorized a primitive computer consisting of a tape of cells along with a machine that writes symbols on the tape according to a set of predetermined set of rules; this \u201cTuring machine\u201d 1 is simple but nevertheless has been proven capable of solving any problem that a modern computer can solve. Then, during World War II, Turing worked with Allied cryptographers at Bletchley Park to devise machines that broke several German ciphers.ose to trajectory and drawing it (command 6 & 7 executed)!')
    
if __name__ == '__main__':
    args = parse_args()
    test(args)