from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class BoreasDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(BoreasDataset, self).__init__(*args, **kwargs)

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
        # boreas camera front intrinsic 2448 * 2048
        # 2304 x 1920 -> 384 x 320

        # 원본 이미지 크기를 알 수 없다
        # 
        original_width = 2448
        # height 2048
    
        # 위 아래를 잘라서 kiiti dataset의 비율과 같이 만든다 -> 620/192 = 3.2291666667
        # 비율을 맞춰보면 height의 값이 758이 나와야 한다 -> 2448/3.229.. = 758.0903225728
        # 아래 차를 가리려면 1/4은 잘라야 해서 2048/4 = 512만큼 아래서 자르고
        # 2048-(758+512) = 778 만큼 하늘에서 자를거다 그러면
        # 2448x758 크기의 이미지가 될 것이다

        self.K = np.array([
            [1.4401585693,       0,                              1.2182277290,     0],
            [0,                                 1.4469215087 ,    (1.0452721531 * 2048 - 778) / 2048,     0],
            [0,                                 0,                              1,                                        0],
            [0,                                 0,                              0,                                        1]], dtype=np.float32)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        color = color.crop((0, 778, 2448, 2048 - 512)) # crop top and bottom
        color.show()

        return color
    
    # since we don't have side, it is not used
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:05d}{}".format(frame_index, self.img_ext)

        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path
    
    def check_depth(self):
        # We don't have depth data
        return False
    
    def get_depth(self, folder, frame_index, side, do_flip):
        print("get_depth: This function should not be called!") # because we don't have depth data
        return None