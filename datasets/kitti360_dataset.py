from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTI360Dataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTI360Dataset, self).__init__(*args, **kwargs)

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

        # fx = gamma1
        # fy = gamma2
        # cx = u0
        # cy = v0
        # s = 0

        # K = np.array([
        #     [fx, s, cx],
        #     [0, fy, cy],
        #     [0, 0, 1]
        # ])

        self.K = np.array([
            [1.3363220825849971e+03,       0,                              7.1694323510126321e+02,     0],
            [0,                                 1.3357883350012958e+03 ,    7.0576498308221585e+02,     0],
            [0,                                 0,                              1,                                        0],
            [0,                                 0,                              0,                                        1]], dtype=np.float32)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    # since we don't have side, it is not used
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)

        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path
    
    def check_depth(self):
        # We don't have depth data
        return False
    
    def get_depth(self, folder, frame_index, side, do_flip):
        print("get_depth: This function should not be called!") # because we don't have depth data
        return None