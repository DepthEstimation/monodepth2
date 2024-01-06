import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file")
parser.add_argument("--output_path")
args = parser.parse_args()

file = args.file
output_path = args.output_path
base_name = "".join(os.path.splitext(os.path.basename(file))[:-1])


# 파일 경로 지정
# file_path = 'my_poses_with_default_calib.npy'


# poses.npy 파일을 NumPy 배열로 읽어오기
try:
    data_array = np.load(file)
    # print("Loaded data:\n", data_array)
except Exception as e:
    print("Error when loading:", e)
    exit()


# create trajectory from transformation matrices

traj = [np.identity(4)]     # homogeneous matrix

for pose in data_array:
    traj.append(np.matmul(traj[-1], pose))

traj = np.asarray(traj)

data_array = traj

# for printing to txt file
data_array = data_array[:, :3, :4]

rows = int(np.prod(data_array.shape) / 12)
data_array = data_array.reshape(( rows , 12))

np.savetxt(f"{output_path}/{base_name}.txt", data_array, delimiter=" ", fmt='%10.6e')