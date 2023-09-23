# https://medium.com/@jaimin-k/exploring-kitti-visual-ododmetry-dataset-8ac588246cdc

from fileinput import filename
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--file_path")
parser.add_argument("--file_name")
args = parser.parse_args()

file_path = args.file_path
file_name = args.file_name

show_flat = True


# file_name = "eval_poses.txt"
# file_name = "my_poses.txt"
# file_name = "my_poses_with_default_calib.txt"
# file_name = "00.txt"
# file_name = "09.txt"


poses = pd.read_csv(os.path.join(file_path, f"{file_name}.txt"), delimiter=' ', header=None)
print('Size of pose dataframe:', poses.shape)
poses.head()

ground_truth = np.zeros((len(poses), 3, 4))
for i in range(len(poses)):
    ground_truth[i] = np.array(poses.iloc[i]).reshape((3, 4))

# print(ground_truth[:, :, 3])

# %matplotlib widget
fig = plt.figure(figsize=(7,6))
if show_flat:
    traj = fig.add_subplot(111)
else:
    traj = fig.add_subplot(111, projection='3d')


# a
# [[[0, 1, 2, 3],
#   [4, 5, 6, 7],
#   [8, 9, a, b]]]
# a[:]

# traj.plot(x, y, z)

if show_flat:
    # 일단 z(?) 값에 -를 곱해줬다
    traj.plot(ground_truth[:,:,3][:,0], -ground_truth[:,:,3][:,2]) # x, z(?)
    # traj.plot(ground_truth[:,:,3][:,0], ground_truth[:,:,3][:,1]) # x, y
else:
    traj.plot(ground_truth[:,:,3][:,0], ground_truth[:,:,3][:,1], ground_truth[:,:,3][:,2])

traj.set_xlabel('x')
traj.set_ylabel('z')

# plt.show()
plt.savefig(os.path.join(file_path, f"{file_name}.png"))