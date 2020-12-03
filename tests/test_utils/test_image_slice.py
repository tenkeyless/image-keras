from image_keras.utils.image_slice import apply_each_slice
import os
from unittest import TestCase

import cv2
import numpy as np
from image_keras.utils import image_slice


class TestImageSlice(TestCase):
    current_path: str = os.path.dirname(os.path.abspath(__file__))
    working_path: str = os.getcwd()
    test_path: str = os.path.join(working_path, "tests")
    test_resource_folder_name: str = "test_resources"

    def __get_lenna_image(self) -> np.ndarray:
        test_file_name: str = "lenna.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        return cv2.imread(test_image_fullpath)

    def test_image_slice(self):
        # Prerequisite
        image: np.ndarray = self.__get_lenna_image()
        tile_height_width = (128, 128)
        inbox_height_width = (64, 64)
        stride_height_width = (64, 64)

        # Processing
        sliced_images = image_slice.slice_by_pixel_size(
            image,
            tile_height_width=tile_height_width,
            inbox_height_width=inbox_height_width,
            stride_height_width=stride_height_width,
        )

        # Result
        test_result_sliced_images_row_num = 8
        test_result_sliced_images_col_num = 8

        def _check_image(row_num, col_num, sliced_img):
            print("{:02d}_{:02d}".format(row_num, col_num))

        i, j, _ = apply_each_slice(sliced_images, _check_image)

        # Check
        self.assertEquals(i, test_result_sliced_images_row_num)
        self.assertEquals(j, test_result_sliced_images_col_num)
