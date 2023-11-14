import os
import cv2
import time
import glob

assets_path = "assets"
folder_path = f"{assets_path}/original_video"  # Update this with the actual path
# model = "finetuned_mono_1024x320"
model = "eun_mono_640x352"

# paths
original_video_path = f"assets/original_video"
frames_path = f"assets/frames"
splits_path = f"assets/splits"
depth_path = f"assets/{model}/depth"
depth_video_path = f"assets/{model}/depth_video"
comparison_video_path = f"assets/{model}/comparison_video"
pose_path = f"assets/{model}/pose"
log_path = f"assets/{model}/log"


# List all files in the folder
# 비디오 파일이어야 한다
file_list = os.listdir(folder_path)
total_file = len(file_list)
total_time = 0


def store_video_info(file_name, total_time):
    print("storing video information...")
    # 영상 정보와 depth로 변환하는데 걸린 시간을 depth path에 txt파일 형식으로 저장한다
    cap = cv2.VideoCapture(f"{original_video_path}/{file_name}")

    if not cap.isOpened():
        print("could not open :", f"{original_video_path}/{file_name}")
        exit(0)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("\nVideo Info:")
    print('> file path :', f"{original_video_path}/{file_name}")
    print('> length :', length)
    print('> width :', width)
    print('> height :', height)
    print(f'> fps : {round(fps)}\n')
    print("Evaluation:")
    print(f"> Average of execution time per image: {total_time/length:.5f} sec")

    f = open(f"{log_path}/{base_name}_log.txt", "w")
    f.write("Video Info:\n")
    f.write(f"> file path : assets/original_video/{file_name}\n")
    f.write(f"> length : {length}\n")
    f.write(f"> width : {width}\n")
    f.write(f"> height : {height}\n")
    f.write(f"> fps : {round(fps)}\n")
    f.write("\nEvaluation:\n")
    f.write(f"> Average of execution time per image: {total_time/length:.5f} sec\n")
    f.close()



# Iterate through the files and execute shell commands
for file_name in file_list:
    if os.path.isfile(os.path.join(folder_path, file_name)):
        
        # 영상 파일 이름을 base_name에 저장해 둔다 (뒤에 오는 .mp4와 같은 확장명은 제외한다)
        base_name = "".join(os.path.splitext(file_name)[:-1])

        # output 파일 저장 경로
        output_file = f'{comparison_video_path}/comp_{base_name}.mp4'

        # 이미 결과물이 생성된 비디오 파일이면 건너뛴다
        # if (os.path.isfile(output_file)):
        #     continue

        # create folders if needed
        command0 = f"mkdir -p {frames_path}/{base_name} && \
                        mkdir -p {depth_path}/{base_name} && \
                        mkdir -p {depth_video_path} && \
                        mkdir -p {comparison_video_path} && \
                        mkdir -p {pose_path} && \
                        mkdir -p {log_path}"
        os.system(command0)
        print("-------------------")
        print("command 0 executed!")
        print("-------------------")

        # Command 1: Extract frames from video
        # must be "jpg" NOT "jpeg"
        command1 = f"ffmpeg -i {original_video_path}/{file_name} -qscale:v 2 -start_number 0 {frames_path}/{base_name}/%010d.jpg"
        # -vf scale=1280:-1
        # 위에 옵션을 주면 해상도를 바꾼다
        # os.path.file_list
        if len(os.listdir(os.path.join(frames_path, base_name))) == 0:  # 폴더가 비어 있으면 command1 돌리고 뭔가 있으면 스킵
            os.system(command1)
        
        # Command 2
        command2 = f"python test_simple.py --image_path {frames_path}/{base_name} --out {depth_path}/{base_name} --model_name {model}"
        
        if len(os.listdir(os.path.join(depth_path, base_name))) == 0:   # 폴더가 비어 있으면 command2 돌리고 뭔가 있으면 스킵
            # print("wow "*100)
            start = time.time()
            os.system(command2)
            end = time.time()
            total_time += (end - start)
            store_video_info(file_name, total_time)
        else:
            print("&&&&&&&&&&&&&")
            print("\t skip depth estimation")
            print("&&&&&&&&&&&&&")
        
        # Command 3
        # depth 이미지들을 모아서 영상으로 만들어 저장한다
        command3 = f"ffmpeg -pattern_type glob -i '{depth_path}/{base_name}/*.jpeg' -c:v h264_nvenc -pix_fmt yuv420p {depth_video_path}/{base_name}.mp4"
        if not os.path.exists(os.path.join(depth_video_path, f"{base_name}.mp4")):
            os.system(command3)

	    # Command 4
        # 원본과 depth 영상을 위 아래로 합친 영상을 만들어 저장한다. 비교를 위해 사용.

	    # 입력 영상 파일 경로
        input_file1 = f'{original_video_path}/{file_name}'			# original
        input_file2 = f'{depth_video_path}/{base_name}.mp4'	        # output of monodepth2

        print("concatenating 2 videos...")
        #command4 = f"ffmpeg -i {input_file1} -i {input_file2} -filter_complex vstack=inputs=2 {output_file}"
        command4 = f'ffmpeg -i {input_file1} -i {input_file2} -filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" {output_file}'
        # 가로 스택이 보기 더 편한 것 같다
        if not os.path.exists(os.path.join(comparison_video_path, f"comp_{base_name}.mp4")):
            os.system(command4)


        # Need to create split file before doing this
        odom_test_file_path = "odom_test_{}.txt"
        file_list = sorted(glob.glob(os.path.join(frames_path, base_name, "*.jpg")))
        file_list = file_list[:-1] # 마지막 없애야 됨
        with open(os.path.join(splits_path, odom_test_file_path.format(base_name)), 'w') as f:
            for item in file_list:
                f.write(os.path.join(base_name) + " " + str(int("".join(os.path.splitext(os.path.basename(item))[:-1]))) + " " + "l" + "\n")


        print("evaluating pose...")
        # Command 5 & 6
        command5 = f"python evaluate_pose.py --eval_split {base_name} --load_weights_folder ./models/{model} --data_path {assets_path}"
        if not os.path.exists(f"{pose_path}/{base_name}.npy"):
            os.system(command5)
        command6 = f"python convert-np-pose-to-traj-txt.py --file {pose_path}/{base_name}.npy --output_path {pose_path}"
        os.system(command6)
        command7 = f"python draw-pose-trajectory.py --file_path {pose_path} --file_name {base_name}"
        os.system(command7)