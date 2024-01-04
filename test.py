import os
import cv2
import time
import datetime
import argparse

folder_path = "assets/test_videos"  # Update this with the actual path

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

    if os.path.isfile(os.path.join(folder_path, file_name)):

        base_name = os.path.splitext(file_name)[0]

        # output 파일 저장 경로
        if (args.reflective):
            output_file = f'assets/{model}/{base_name}/reflective/test_concatenated_video/output_concatenated_{base_name}.mp4'
        else:
            output_file = f'assets/{model}/{base_name}/test_concatenated_video/output_concatenated_{base_name}.mp4'
        
        if not os.path.isfile(output_file):
            # create folders
            if (args.reflective): # using reflective padding
                command0 = f"mkdir -p assets/{model}/{base_name}/test_frames && mkdir -p assets/{model}/{base_name}/reflective/test_out_frames && mkdir -p assets/{model}/{base_name}/reflective/test_result_video && mkdir -p assets/{model}/{base_name}/reflective/test_concatenated_video"
            else: # not using
                command0 = f"mkdir -p assets/{model}/{base_name}/test_frames && mkdir -p assets/{model}/{base_name}/test_out_frames && mkdir -p assets/{model}/{base_name}/test_result_video && mkdir -p assets/{model}/{base_name}/test_concatenated_video"
            os.system(command0)
            print("done!")

            # Command 1
            # must be "jpg" NOT "jpeg"
            command1 = f"ffmpeg -i assets/test_videos/{file_name} assets/{model}/{base_name}/test_frames/%04d.jpg"
            os.system(command1)


        # Command 2
        if (args.reflective): # using reflective padding
            command2 = f"python test_simple.py --reflective true --image_path assets/{model}/{base_name}/test_frames --out assets/{model}/{base_name}/reflective/test_out_frames --model_name {model}"
        else:
            command2 = f"python test_simple.py --image_path assets/{model}/{base_name}/test_frames --out assets/{model}/{base_name}/test_out_frames --model_name {model}"
        start = time.time()
        os.system(command2)
        end = time.time()
        total_time += (end - start)
        
        # Command 3
        if (args.reflective): # using reflective padding 
            command3 = f"ffmpeg -framerate 30 -pattern_type glob -i 'assets/{model}/{base_name}/reflective/test_out_frames/*.jpeg' -c:v h264_nvenc -pix_fmt yuv420p assets/{model}/{base_name}/reflective/test_result_video/{base_name}_reflective.mp4"
        else: # not using
            command3 = f"ffmpeg -framerate 30 -pattern_type glob -i 'assets/{model}/{base_name}/test_out_frames/*.jpeg' -c:v h264_nvenc -pix_fmt yuv420p assets/{model}/{base_name}/test_result_video/{base_name}.mp4"
        os.system(command3)

        # Command 4

        # 입력 영상 파일 경로
        input_file1 = f'assets/test_videos/{file_name}'			# original
        if (args.reflective): # using reflective padding
            input_file2 = f'assets/{model}/{base_name}/reflective/test_result_video/{base_name}_reflective.mp4'	# output of monodepth2
        else: # not using
            input_file2 = f'assets/{model}/{base_name}/test_result_video/{base_name}.mp4'	# output of monodepth2
        command4 = f'ffmpeg -i {input_file1} -i {input_file2} -filter_complex "[0:v][1:v]vstack=inputs=2[v]" -map "[v]" {output_file}'
        os.system(command4)

        cap = cv2.VideoCapture(f"assets/test_videos/{file_name}")

        if not cap.isOpened():
            print("could not open :", f"assets/test_videos/{file_name}")
            exit(0)
    
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        video_info = f"Video Info:\n> file path : assets/test_videos/{file_name}\n> length : {length}\n> width : {width}\n> height : {height}\n> fps : {round(fps)}\n> reflective : {args.reflective}"

        evaluation_info = f"\nEvaluation:\n> Average of execution time per image: {total_time/length:.5f} sec\n"

        current_time = datetime.datetime.now()
        output_file_path = f"assets/{model}/{base_name}/log_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"

        with open(output_file_path, "w") as f:
            f.write(video_info)
            f.write(evaluation_info)

        print(video_info)
        print(evaluation_info)

if __name__ == '__main__':
    args = parse_args()
    test(args)