import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import cv2
from image_keras.supports.path import get_image_filenames, split_fullpath
from image_keras.utils.image_slice import apply_each_slice, slice_by_pixel_size
from natsort import natsorted


def slice_and_write_single_image(
    full_image_path,
    as_gray,
    tile_size,
    inbox_size,
    stride_size,
    add_same_padding,
    not_discard_rest_horizontal_tile,
    not_discard_rest_vertical_tile,
    target_folder,
):
    _, _base_filename, _base_ext = split_fullpath(full_image_path)
    _image = cv2.imread(full_image_path)
    if as_gray:
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    _sliced_images = slice_by_pixel_size(
        _image,
        tile_height_width=(tile_size, tile_size),
        inbox_height_width=(inbox_size, inbox_size),
        stride_height_width=(stride_size, stride_size),
        add_same_padding=add_same_padding,
        discard_horizontal_overflow=not not_discard_rest_horizontal_tile,
        discard_vertical_overflow=not not_discard_rest_vertical_tile,
    )

    def _write_image(_row_num, _col_num, sliced_img):
        result_path = os.path.join(
            target_folder,
            "{}_{:02d}_{:02d}{}".format(_base_filename, _row_num, _col_num, _base_ext),
        )
        cv2.imwrite(result_path, sliced_img)

    apply_each_slice(_sliced_images, _write_image)


def slice_and_write_for_folder(
    full_image_path,
    as_gray,
    tile_size,
    inbox_size,
    stride_size,
    add_same_padding,
    not_discard_rest_horizontal_tile,
    not_discard_rest_vertical_tile,
    target_folder,
):
    files = get_image_filenames(full_image_path)
    files = natsorted(files)
    num_files = len(files)
    for index, file in enumerate(files):
        print("Processing {} ({}/{})...".format(file, (index + 1), num_files))
        slice_and_write_single_image(
            os.path.join(full_image_path, file),
            as_gray=as_gray,
            tile_size=tile_size,
            inbox_size=inbox_size,
            stride_size=stride_size,
            add_same_padding=add_same_padding,
            not_discard_rest_horizontal_tile=not_discard_rest_horizontal_tile,
            not_discard_rest_vertical_tile=not_discard_rest_vertical_tile,
            target_folder=target_folder,
        )


if __name__ == "__main__":
    # Parameters
    parser: ArgumentParser = ArgumentParser(description="Arguments for tiling slice")
    parser.add_argument("--full_image_path", required=True, type=str, help="Full path.")
    parser.add_argument(
        "--not_discard_rest_vertical_tile",
        action="store_true",
        help="With this option, we keep the last vertical tile, even if it doesn't fit the tile size.",
    )
    parser.add_argument(
        "--not_discard_rest_horizontal_tile",
        action="store_true",
        help="With this option, we keep the last horizontal tile, even if it doesn't fit the tile size.",
    )
    parser.add_argument(
        "--as_gray",
        action="store_true",
        help="With this option, sliced ​​results will be grayscale images.",
    )
    parser.add_argument(
        "--add_same_padding",
        action="store_true",
        help="With this option, zero padding add automatically for inbox, tile, stride size.",
    )
    parser.add_argument("--tile_size", default=256, type=int, help="Tile size")
    parser.add_argument("--inbox_size", default=128, type=int, help="Inbox size")
    parser.add_argument("--stride_size", default=64, type=int, help="Stride size")
    parser.add_argument(
        "--target_folder",
        required=True,
        type=str,
        help="Target folder.",
    )
    args = parser.parse_args()

    # parameter
    full_image_path: str = args.full_image_path
    not_discard_rest_vertical_tile: bool = args.not_discard_rest_vertical_tile
    not_discard_rest_horizontal_tile: bool = args.not_discard_rest_horizontal_tile
    as_gray: bool = args.as_gray
    add_same_padding: bool = args.add_same_padding
    tile_size: int = args.tile_size
    inbox_size: int = args.inbox_size
    stride_size: int = args.stride_size
    target_folder: str = os.path.join(args.target_folder)

    os.makedirs(target_folder) if not os.path.exists(target_folder) else None

    # If file,
    if os.path.isfile(full_image_path):
        slice_and_write_single_image(
            full_image_path,
            as_gray=as_gray,
            tile_size=tile_size,
            inbox_size=inbox_size,
            stride_size=stride_size,
            add_same_padding=add_same_padding,
            not_discard_rest_horizontal_tile=not_discard_rest_horizontal_tile,
            not_discard_rest_vertical_tile=not_discard_rest_vertical_tile,
            target_folder=target_folder,
        )
    # else if Path,
    else:
        slice_and_write_for_folder(
            full_image_path,
            as_gray=as_gray,
            tile_size=tile_size,
            inbox_size=inbox_size,
            stride_size=stride_size,
            add_same_padding=add_same_padding,
            not_discard_rest_horizontal_tile=not_discard_rest_horizontal_tile,
            not_discard_rest_vertical_tile=not_discard_rest_vertical_tile,
            target_folder=target_folder,
        )
