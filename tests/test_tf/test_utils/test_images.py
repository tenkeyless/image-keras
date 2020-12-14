import os
from unittest import TestCase

import numpy as np
import tensorflow as tf
from image_keras.tf.utils import images as tf_images
from tensorflow.python.ops import gen_array_ops


class TestTFImage(TestCase):
    current_path: str = os.path.dirname(os.path.abspath(__file__))
    working_path: str = os.getcwd()
    test_path: str = os.path.join(working_path, "tests")
    test_resource_folder_name: str = "test_resources"

    def test_decode_image(self):
        # Expected Result
        sample_image_shape = [180, 180, 3]
        result = tf.constant(sample_image_shape)

        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        # 2) Calculate shape
        self.assertTrue(tf.math.reduce_all(tf.math.equal(tf.shape(tf_image), result)))

    def test_tf_img_to_minmax(self):
        # Expected Result
        threshold = 127
        min_maxed_unique_values = (255, 0)
        number_of_black_color = 31760

        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        grayscale_tf_image = tf_images.decode_png(test_image_fullpath, 1)

        # 2) Calculate min max
        min_maxed_grayscale_tf_image = tf_images.tf_img_to_minmax(
            grayscale_tf_image, threshold, (0, 255)
        )

        # check unique
        reshaped_min_maxed_grayscale_tf_image = tf.reshape(
            min_maxed_grayscale_tf_image, (-1, 1)
        )
        unique_min_max_values = gen_array_ops.unique_v2(
            reshaped_min_maxed_grayscale_tf_image, axis=[-2]
        )[0]

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    unique_min_max_values,
                    tf.cast(
                        tf.expand_dims(tf.constant(min_maxed_unique_values), axis=-1),
                        tf.float32,
                    ),
                )
            )
        )
        self.assertTrue(
            tf.math.equal(
                tf.math.count_nonzero(min_maxed_grayscale_tf_image),
                number_of_black_color,
            )
        )

    def test_tf_get_sorted_unique_color(self):
        sample_image_colors = [
            [88.0, 18.0, 60.0],
            [176.0, 67.0, 79.0],
            [205.0, 96.0, 97.0],
            [178.0, 69.0, 80.0],
            [206.0, 100.0, 94.0],
        ]
        result = tf.constant(sample_image_colors)

        # 1) Get sample image as tf
        test_file_name: str = "lenna.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_get_sorted_unique_color(tf_image)[:5], result
                )
            )
        )

    def test_tf_get_all_colors(self):
        # Expected Result
        sample_image_colors = [
            [0.0, 0.0, 0.0],
            [245.0, 245.0, 245.0],
            [255.0, 145.0, 77.0],
            [71.0, 71.0, 71.0],
            [72.0, 72.0, 72.0],
        ]
        result = tf.constant(sample_image_colors)

        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        # 2) Calculate colors in image
        tf_image_colors = tf_images.tf_get_all_colors(tf_image)
        print(tf_image_colors)
        self.assertTrue(tf.math.reduce_all(tf.math.equal(tf_image_colors, result)))

    def test_tf_get_all_colors_without_black(self):
        # 1) Get sample image as tf
        test_file_name: str = "lenna.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        # 2) Calculate colors in image
        tf_image_colors = tf_images.tf_get_all_colors(tf_image, black_first=False)
        self.assertTrue(
            tf.math.reduce_all(tf.math.not_equal(tf_image_colors[0], [0, 0, 0]))
        )

    def test_tf_generate_color_map(self):
        sample_indexes = [0.0, 1.0, 2.0, 3.0, 4.0]
        sample_image_colors = [
            [0.0, 0.0, 0.0],
            [245.0, 245.0, 245.0],
            [255.0, 145.0, 77.0],
            [71.0, 71.0, 71.0],
            [72.0, 72.0, 72.0],
        ]
        result = tf.constant(sample_image_colors)

        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        color_map = tf_images.tf_generate_color_map(tf_image)
        print(color_map)
        self.assertTrue(tf.math.reduce_all(tf.math.equal(color_map[0], sample_indexes)))
        self.assertTrue(tf.math.reduce_all(tf.math.equal(color_map[1], result)))

    def test_tf_generate_random_color_map(self):
        sample_indexes = [0.0, 4.0, 1.0, 2.0, 3.0]
        sample_image_colors = [
            [0.0, 0.0, 0.0],
            [245.0, 245.0, 245.0],
            [255.0, 145.0, 77.0],
            [71.0, 71.0, 71.0],
            [72.0, 72.0, 72.0],
        ]
        result = tf.constant(sample_image_colors)

        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        color_map = tf_images.tf_generate_random_color_map(
            tf_image, shuffle_exclude_first=1, bin_size=5, seed=42
        )
        self.assertTrue(tf.math.reduce_all(tf.math.equal(color_map[0], sample_indexes)))
        self.assertTrue(tf.math.reduce_all(tf.math.equal(color_map[1], result)))

    def test_tf_change_channel_for_batch(self):
        batch_sample_image_colors = [
            [
                [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
                [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
            ],
            [
                [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
                [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
            ],
        ]
        new_order = [[2, 0, 1], [0, 2, 1]]

        result = [
            [
                [[20.0, 0.0, 10.0], [245.0, 225.0, 235.0]],
                [[31.0, 71.0, 41.0], [77.0, 255.0, 145.0]],
            ],
            [
                [[0.0, 20.0, 10.0], [225.0, 245.0, 235.0]],
                [[71.0, 31.0, 41.0], [255.0, 77.0, 145.0]],
            ],
        ]

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_channel_for_batch(
                        batch_sample_image_colors, new_order, axis=3
                    ),
                    result,
                )
            )
        )

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_channel_for_batch(
                        batch_sample_image_colors, new_order, axis=-1
                    ),
                    result,
                )
            )
        )

    def test_tf_change_channel_for_batch2(self):
        batch_sample_image_colors = [
            [
                [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
                [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
            ],
            [
                [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
                [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
            ],
        ]
        new_order = [[0, 1], [1, 0]]

        result = [
            [
                [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
                [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
            ],
            [
                [[225.0, 235.0, 245.0], [0.0, 10.0, 20.0]],
                [[255.0, 145.0, 77.0], [71.0, 41.0, 31.0]],
            ],
        ]

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_channel_for_batch(
                        batch_sample_image_colors, new_order, axis=2
                    ),
                    result,
                )
            )
        )

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_channel_for_batch(
                        batch_sample_image_colors, new_order, axis=-2
                    ),
                    result,
                )
            )
        )

    def test_tf_change_channel_for_batch3(self):
        batch_sample_image_colors = [
            [
                [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
                [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
            ],
            [
                [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
                [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
            ],
        ]
        new_order = [[0, 1], [1, 0]]

        result = [
            [
                [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
                [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
            ],
            [
                [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
                [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
            ],
        ]

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_channel_for_batch(
                        batch_sample_image_colors, new_order, axis=1
                    ),
                    result,
                )
            )
        )

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_channel_for_batch(
                        batch_sample_image_colors, new_order, axis=-3
                    ),
                    result,
                )
            )
        )

    def test_tf_change_channel(self):
        batch_sample_image_colors = [
            [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
            [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
        ]
        new_order = [[2, 0, 1]]

        result = [
            [[20.0, 0.0, 10.0], [245.0, 225.0, 235.0]],
            [[31.0, 71.0, 41.0], [77.0, 255.0, 145.0]],
        ]

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_channel(
                        batch_sample_image_colors, new_order, axis=2
                    ),
                    result,
                )
            )
        )

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_channel(
                        batch_sample_image_colors, new_order, axis=-1
                    ),
                    result,
                )
            )
        )

    def test_tf_change_order(self):
        batch_sample_image_colors = [
            [[0.0, 10.0, 20.0], [225.0, 235.0, 245.0]],
            [[71.0, 41.0, 31.0], [255.0, 145.0, 77.0]],
        ]
        new_order = [[2, 0, 1]]

        result = [
            [[20.0, 0.0, 10.0], [245.0, 225.0, 235.0]],
            [[31.0, 71.0, 41.0], [77.0, 255.0, 145.0]],
        ]

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_order(batch_sample_image_colors, new_order),
                    tf.expand_dims(result, -1),
                )
            )
        )

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    tf_images.tf_change_order(batch_sample_image_colors, new_order),
                    tf.expand_dims(
                        tf_images.tf_change_channel(
                            batch_sample_image_colors, new_order, axis=2
                        ),
                        -1,
                    ),
                )
            )
        )

    def test_tf_shrink3D(self):
        # Expected Result
        result1 = tf.constant([[[16, 20]], [[48, 52]]], dtype=tf.int64)
        result2 = tf.constant([[[36]], [[100]]], dtype=tf.int64)

        a = tf.constant(
            np.array(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                    [[13, 14], [15, 16]],
                ]
            )
        )

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(tf_images.tf_shrink3D(a, 2, 1, 2), result1)
            )
        )
        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(tf_images.tf_shrink3D(a, 2, 1, 1), result2)
            )
        )

    def test_tf_extract_patches(self):
        batch_size = 1
        img_wh = 5
        channel = 1
        sample_image = tf.constant(
            np.random.randint(10, size=(batch_size, img_wh, img_wh, channel))
        )

        ksize = 3
        k_size2 = 5

        self.assertTrue(
            tf.math.reduce_all(
                tf.image.extract_patches(
                    sample_image,
                    sizes=[1, ksize, ksize, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="SAME",
                )
                == tf_images.tf_extract_patches(sample_image, ksize, img_wh, channel)
            )
        )

        self.assertTrue(
            tf.math.reduce_all(
                tf.image.extract_patches(
                    sample_image,
                    sizes=[1, k_size2, k_size2, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="SAME",
                )
                == tf_images.tf_extract_patches(sample_image, k_size2, img_wh, channel)
            )
        )

        batch_size2 = 1
        img_wh2 = 32
        channel2 = 1
        sample_image2 = tf.constant(
            np.random.randint(10, size=(batch_size2, img_wh2, img_wh2, channel2))
        )
        self.assertTrue(
            tf.math.reduce_all(
                tf.image.extract_patches(
                    sample_image2,
                    sizes=[1, k_size2, k_size2, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="SAME",
                )
                == tf_images.tf_extract_patches(
                    sample_image2, k_size2, img_wh2, channel2
                )
            )
        )

        batch_size3 = 2
        img_wh3 = 32
        channel3 = 30
        sample_image3 = tf.constant(
            np.random.randint(10, size=(batch_size3, img_wh3, img_wh3, channel3))
        )
        self.assertTrue(
            tf.math.reduce_all(
                tf.image.extract_patches(
                    sample_image3,
                    sizes=[1, k_size2, k_size2, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="SAME",
                )
                == tf_images.tf_extract_patches(
                    sample_image3, k_size2, img_wh3, channel3
                )
            )
        )

    def test_remove_element(self):
        tensors = tf.constant([[155, 132, 10], [33, 22, 17], [42, 44, 4]])
        removed_tensors = tf_images.remove_element(tensors, [33, 22, 17])
        result_tensors = tf.constant([[155, 132, 10], [42, 44, 4]])

        self.assertTrue(
            tf.math.reduce_all(tf.math.equal(removed_tensors, result_tensors))
        )

    def test_tf_set_random_bucket(self):
        result_tensors = tf.constant(
            [
                0,
                1,
                2,
                3,
                4,
                17,
                23,
                11,
                22,
                10,
                14,
                21,
                19,
                16,
                7,
                8,
                15,
                20,
                27,
                26,
                28,
                25,
                13,
                12,
                29,
                6,
                9,
                5,
                18,
                24,
            ]
        )
        seed = 42
        bin_size = tf.shape(result_tensors)[-1]
        random_exclude_first = 5
        ordered_index = tf.range(bin_size, dtype=tf.int32)

        random_bucked_tensors = tf_images.tf_set_random_bucket(
            ordered_index,
            bin_size=30,
            shuffle_exclude_first=random_exclude_first,
            seed=seed,
        )

        self.assertTrue(
            tf.math.reduce_all(tf.math.equal(random_bucked_tensors, result_tensors))
        )

    def test_tf_image_detach_with_id_color_list(self):
        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        # 2) ID Color list
        # id_color_list = tf_images.tf_generate_random_color_map(
        #     tf_image, shuffle_exclude_first=1, bin_size=30
        # )
        id_color_list = tf_images.tf_generate_color_map(tf_image)

        print(tf.shape(tf_image))
        print(id_color_list)

        print(
            tf.shape(
                tf_images.tf_image_detach_with_id_color_list(
                    tf_image, id_color_list=id_color_list, bin_num=30
                )
            )
        )
