import os
import cv2
import time

folder_path = "assets/test_videos"  # Update this with the actual path
model = "mono_1024x320"

# List all files in the folder
file_list = os.listdir(folder_path)
total_file = len(file_list)
total_time = 0

# Iterate through the files and execute shell commands
for file_name in file_list:
    if os.path.isfile(os.path.join(folder_path, file_name)):

        base_name = os.path.splitext(file_name)[0]

        # create folders
        command0 = f"mkdir -p assets/{model}/test_frames/{base_name} && mkdir -p assets/{model}/test_out/{base_name} && mkdir -p assets/{model}/test_results && mkdir -p assets/{model}/test_concatenated"
        os.system(command0)
        print("done!")

        # Command 1
        # must be "jpg" NOT "jpeg"
        command1 = f"ffmpeg -i assets/test_videos/{file_name} assets/{model}/test_frames/{base_name}/%04d.jpg"
        os.system(command1)
        
        # Command 2
        command2 = f"python test_simple.py --image_path assets/{model}/test_frames/{base_name} --out assets/{model}/test_out/{base_name} --model_name {model}"
        start = time.time()
        os.system(command2)
        end = time.time()
        total_time += end - start
        
        # Command 3
        command3 = f"ffmpeg -framerate 30 -pattern_type glob -i 'assets/{model}/test_out/{base_name}/*.jpeg' -c:v h264_nvenc -pix_fmt yuv420p assets/{model}/test_results/{base_name}.mp4"
        os.system(command3)

	    # Command 4

	    # 입력 영상 파일 경로
        input_file1 = f'assets/test_videos/{file_name}'			# original
        input_file2 = f'assets/{model}/test_results/{base_name}.mp4'	# output of monodepth2

        # output 파일 저장 경로
        output_file = f'assets/{model}/test_concatenated/output_concatenated_{base_name}.mp4'

        #command4 = f"ffmpeg -i {input_file1} -i {input_file2} -filter_complex vstack=inputs=2 {output_file}"
        command4 = f'ffmpeg -i {input_file1} -i {input_file2} -filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" {output_file}'
        os.system(command4)
