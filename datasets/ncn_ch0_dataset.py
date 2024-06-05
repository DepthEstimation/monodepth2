from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

class NCN_CH0_Dataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NCN_CH0_Dataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        # fx -- cx --
        # -- fy cy --
        # -- -- -- --
        # -- -- -- --
        # AI Hub camera front intrinsic 1920 * 1080
        self.K = np.array([
            [1,       0,                  1920/2,     0],
            [0,                     1,    1080/2,     0],
            [0,                     0,                  1,                  0],
            [0,                     0,                  0,                  1]], dtype=np.float32)
        self.K[0] /= 1920   # normalization process
        self.K[1] /= 1080

        # self.full_res_shape = (1242, 375)
        # self.full_res_shape = (1920, 1080) # where is this used?

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        # image crop
        color = color.crop((0, 504, 1920, 1080)) # sky crop
        
        # color.show()
        # input()

        return color
    
    # since we don't have side, it is not used
    def get_image_path(self, folder, frame_index, side):
        # f_str = "{:010d}{}".format(frame_index, self.img_ext)
        f_str = "{:010}{}".format(frame_index, self.img_ext) # for ai-hub data


        # image_path = os.path.join(
        #     # 폴더 경로 == <data_path>/<folder>/{:010d}.jpg
        #     # ex) ./train_data/frames/lozan/00000000001.jpg
        #     self.data_path, folder, 'image_00', 'data', f_str)
        # return image_path

        image_path = os.path.join(
            # 폴더 경로 == <data_path>/<folder>/{:010d}.jpg
            # ex) ./train_data/frames/lozan/00000000001.jpg
            self.data_path, folder, f_str) # for ai-hub data
        return image_path
    
    def check_depth(self):
        # We don't have depth data
        return False
    
    def get_depth(self, folder, frame_index, side, do_flip):
        print("get_depth: This function should not be called!") # because we don't have depth data
        return None