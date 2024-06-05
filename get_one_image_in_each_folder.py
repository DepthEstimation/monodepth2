import os
import glob
import shutil

def copy_file(src, dst):
    try:
        shutil.copy2(src, dst)
        print(f"File '{src}' successfully copied to '{dst}'.")
    except FileNotFoundError:
        print(f"Error: File '{src}' not found.")
        exit()
    except PermissionError:
        print(f"Error: Permission denied. Unable to copy file '{src}' to '{dst}'.")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()


path = '/media/cgv/f9dacde2-bd46-4e02-858d-ef0977ddf2a2/088.승용_자율주행차_주간_도심도로_데이터/01-1.정식개방데이터/Training/01.원천데이터/TS'


os.chdir(path)
folders = os.listdir('.')

for folder in folders:
    if os.path.isdir(folder):
        file_list = sorted(glob.glob(os.path.join(folder, 'sensor_raw_data', 'camera', "*.jpg")))
        file_list[0]
        # print(file_list[0])
        if os.path.isfile(file_list[0]):
            copy_file(file_list[0], "/home/cgv/DepthEstimation/monodepth2/assets/frames/aihub_samples")
            # src = path + "/" + file_list[0]
            # dst = "/home/cgv/DepthEstimation/monodepth2/assets/frames/aihub_samples"
            # os.system(f'cp {src} {dst}')
        else:
            print('not a file')