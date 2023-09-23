import os
import cv2
import time

folder_path = "assets/original_video"  # Update this with the actual path
model = "mono_1024x320"

# paths
original_vieo_path = f"assets/original_video"
frames_path = f"assets/{model}/frames"
depth_path = f"assets/{model}/depth"
depth_video_path = f"assets/{model}/depth_video"
comparison_video_path = f"assets/{model}/comparison_video"
trajectory_path = f"assets/{model}/trajectory"


# List all files in the folder
# 비디오 파일이어야 한다
file_list = os.listdir(folder_path)
total_file = len(file_list)
total_time = 0


# Iterate through the files and execute shell commands
for file_name in file_list:
    if os.path.isfile(os.path.join(folder_path, file_name)):
        
        # 영상 파일 이름을 base_name에 저장해 둔다 (뒤에 오는 .mp4와 같은 확장명은 제외한다)
        base_name = "".join(os.path.splitext(file_name)[:-1])

        # output 파일 저장 경로
        output_file = f'{comparison_video_path}/comp_{base_name}.mp4'

        # 이미 결과물이 생성된 비디오 파일이면 건너뛴다
        if (os.path.isfile(output_file)):
            continue

        # create folders if needed
        command0 = f"mkdir -p {frames_path}/{base_name} && \
                        mkdir -p {depth_path}/{base_name} && \
                        mkdir -p {depth_video_path} && \
                        mkdir -p {comparison_video_path}"
        os.system(command0)


        # Command 1: Extract frames from video
        # must be "jpg" NOT "jpeg"
        # 
        command1 = f"ffmpeg -i {original_vieo_path}/{file_name} -qscale:v 2 -vf scale=1280:-1 {frames_path}/{base_name}/%04d.jpg"
        os.system(command1)
        
        # Command 2
        command2 = f"python test_simple.py --image_path {frames_path}/{base_name} --out {depth_path}/{base_name} --model_name {model}"
        start = time.time()
        os.system(command2)
        end = time.time()
        total_time += (end - start)
        
        # Command 3
        # depth 이미지들을 모아서 영상으로 만들어 저장한다
        command3 = f"ffmpeg -framerate 30 -pattern_type glob -i '{depth_path}/{base_name}/*.jpeg' -c:v h264_nvenc -pix_fmt yuv420p {depth_video_path}/{base_name}.mp4"
        os.system(command3)

	    # Command 4
        # 원본과 depth 영상을 위 아래로 합친 영상을 만들어 저장한다. 비교를 위해 사용.

	    # 입력 영상 파일 경로
        input_file1 = f'{original_vieo_path}/{file_name}'			# original
        input_file2 = f'{depth_video_path}/{base_name}.mp4'	        # output of monodepth2

        #command4 = f"ffmpeg -i {input_file1} -i {input_file2} -filter_complex vstack=inputs=2 {output_file}"
        command4 = f'ffmpeg -i {input_file1} -i {input_file2} -filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" {output_file}'

        # 가로 스택이 보기 더 편한 것 같다
        os.system(command4)


        # Command 5
        command5 = f"python evaluate_pose.py --eval_split "


        # 영상 정보와 depth로 변환하는데 걸린 시간을 depth path에 txt파일 형식으로 저장한다
        cap = cv2.VideoCapture(f"{original_vieo_path}/{file_name}")
 
        if not cap.isOpened():
            print("could not open :", f"{original_vieo_path}/{file_name}")
            exit(0)
    
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print("\nVideo Info:")
        print('> file path :', f"{original_vieo_path}/{file_name}")
        print('> length :', length)
        print('> width :', width)
        print('> height :', height)
        print(f'> fps : {round(fps)}\n')
        print("Evaluation:")
        print(f"> Average of execution time per image: {total_time/length:.5f} sec")

        f = open(f"{depth_video_path}/{base_name}_log.txt", "w")
        f.write("Video Info:\n")
        f.write(f"> file path : assets/original_video/{file_name}\n")
        f.write(f"> length : {length}\n")
        f.write(f"> width : {width}\n")
        f.write(f"> height : {height}\n")
        f.write(f"> fps : {round(fps)}\n")
        f.write("\nEvaluation:\n")
        f.write(f"> Average of execution time per image: {total_time/length:.5f} sec\n")
        f.close()
