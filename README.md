# Custom Monodepth2(Ours)

We developed customed open source for our project using original monodepth2 open source linked in:

> [Monodepth2(Original)](https://github.com/nianticlabs/monodepth2)

## Setup

Follow the instruction in the original repository for environment setup.

If code gets stuck executing `torch.load()`, then try using higher version of pytorch.

```
conda create -n monodepth2 python=3.6.6
conda activate monodepth2
conda install pytorch torchvision=0.2.1 -c pytorch
conda install opencv=3.3.1

pip install tensorboardX==1.4 scikit-image ipython
```
> need to use pytorch version 1.1.0> and 0.4.1< (otherwise warning...)

-> or might want to try `conda install pytorch=0.4.1 cuda90 -c pytorch` [reference](https://pytorch.org/get-started/previous-versions/#commands-for-versions--100-1)

### Used 
```
FFMPEG            3.4
torch             1.10.2
torchvision       0.2.1
scikit-learn      0.24.2
scipy             1.5.4
numpy             1.19.2
ipython           7.16.3
```

## Test

We added a python sript `test_video.py` that will create a depth map video of a specified video file. It has 2 required arguments `--model` and `--file.`--reflective` is an optional argument that slightly increases the quality of the created depth map.
* `--model` pre-trained model to use
* `--file` video to create depth map
* `--reflective` boolean string value: 'True' or 'False'(default)

This script executes system command `ffmpeg` internally. Therefore, `ffmpeg` should be installed in the running device. 

The script not only creates a single depth map video, but the intermediate results are also stored in the system. They are organized in folder structure like shown below:

```
assets
├── original_video
│   ├── <file_name>.mp4
├── frames
│   ├── <file_name>
│   │   ├── 000000000.jpg
│   │   ├── ...
│   │   └── #########.jpg
└── models
    └── <model_name>
        ├── comparison_video
        │   └── <file_name>
        │       └── comp_oseok.mp4
        ├── depth
        │   └── <file_name>
        │       ├── 000000000_disp.jpeg
        │       ├── 000000000_disp.npy
        │       ├── ...
        │       ├── #########_disp.jpeg
        │       └── #########_disp.npy
        ├── depth_video
        │   └── <file_name>
        │       └── <file_name>.mp4
        ├── log
        │   └── <file_name>
        │       └── log_<date>.txt
        └── pose
            └── <file_name>
                └── log_<date>.txt
```

* original_video : place video to test here
* frames : stores splitted video frames
* comparison_video : stores vertically stacked original and depth map video 
* depth : stores depth map for each frames
* depth_video : stores depth map video
* log : stores video data
* pose : stores pose trajectory image of the video

when reflective option is set to True, `<file_name>` becomes `<file_name>_reflective`

example:

* `python test_video.py --model_name mono_640x192 --file_name oseok.mp4`


## Training

To train using kitti dataset, follow the instruction in the original readme file.

example)
* `python train.py --model_name kitti_mono_model --data_path /path/to/kitti_data --log_dir models`

To train using custom dataset, training options must be specified
* `--dataset` : set the value to `my`
* `--width` : width of the training image (must be multiple of 32)
* `--height` : height of the training image (must be multiple of 32)
* `--split` : set this to `auto`

batch size can be reduced if the training dataset is too large
* `--batch_size`

The folder that contains the training dataset should have the following directory structure:
```
/train/data/path
├── A
│   ├── 000000000.jpg
│   ├── ...
│   └── #########.jpg
├── ...
└── Z
    ├── 000000000.jpg
    ├── ...
    └── #########.jpg

```

Each folder contains frames from a distinct video shot. However, the camera used to take the video should be the same.

Furthermore, before traing on the dataset, camera should be calibrated and the K value in the `datasets/my_dataset.py` should be set to the camera intrinsic value.

example)
* `python train.py --model_name my_mono_model --data_path path/to/my_data --log_dir models --dataset my --width 1280 --height 704 --split my --batch_size 8`