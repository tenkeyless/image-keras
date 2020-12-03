import os
from typing import Optional
from unittest import TestCase

from image_keras.supports import path


class TestPath(TestCase):
    current_path: str = os.path.dirname(os.path.abspath(__file__))
    working_path: str = os.getcwd()
    test_path: str = os.path.join(working_path, "tests")
    test_resource_folder_name: str = "test_resources"

    def test_split_fullpath_1(self):
        test_resource_path = path.split_fullpath(
            os.path.join(self.test_path, self.test_resource_folder_name)
        )
        print(test_resource_path)
        self.assertEqual(test_resource_path[1], None)
        self.assertEqual(test_resource_path[2], None)

    def test_split_fullpath_2(self):
        test_resource_sample_img_path = path.split_fullpath(
            os.path.join(self.test_path, self.test_resource_folder_name, "sample.png")
        )
        print(test_resource_sample_img_path)
        self.assertEqual(test_resource_sample_img_path[1], "sample")
        self.assertEqual(test_resource_sample_img_path[2], ".png")

    def test_split_fullpath_3(self):
        test_filename_path = path.split_fullpath(
            os.path.join(self.working_path, "LICENSE")
        )
        print(test_filename_path)
        self.assertEqual(test_filename_path[1], "LICENSE")
        self.assertEqual(test_filename_path[2], "")

    def test_get_image_filenames(self):
        test_image_filenames = path.get_image_filenames(
            os.path.join(self.test_path, self.test_resource_folder_name)
        )
        self.assertEqual(len(test_image_filenames), 4)
