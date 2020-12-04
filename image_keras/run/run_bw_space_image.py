import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import cv2
from image_keras.supports.folder import create_folder_if_not_exist
from image_keras.supports.path import get_image_filenames, split_fullpath
from image_keras.utils.image_transform import draw__mask_with_edge
from natsort import natsorted


def bw_space_single_image(
    full_image_path,
    edge_size,
    target_folder,
):
    _, _base_filename, _base_ext = split_fullpath(full_image_path)
    image = cv2.imread(full_image_path)
    bw_spaced_image = draw__mask_with_edge(image, edge_size)
    result_path = os.path.join(target_folder, "{}{}".format(_base_filename, _base_ext))
    cv2.imwrite(result_path, bw_spaced_image)


def bw_space_image_for_folder(
    full_image_path,
    edge_size,
    target_folder,
):
    files = get_image_filenames(full_image_path)
    files = natsorted(files)
    num_files = len(files)
    for index, file in enumerate(files):
        print("Processing {} ({}/{})...".format(file, (index + 1), num_files))
        bw_space_single_image(
            os.path.join(full_image_path, file),
            edge_size=edge_size,
            target_folder=target_folder,
        )


if __name__ == "__main__":
    # Parameters
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for image black white spaced edge."
    )
    parser.add_argument("--full_image_path", required=True, type=str, help="Full path.")
    parser.add_argument("--edge_size", default=256, type=int, help="Edge size")
    parser.add_argument(
        "--target_folder",
        required=True,
        type=str,
        help="Target folder.",
    )
    args = parser.parse_args()

    # parameter
    full_image_path: str = args.full_image_path
    edge_size: int = args.edge_size
    target_folder: str = os.path.join(args.target_folder)

    create_folder_if_not_exist(target_folder)

    # If file,
    if os.path.isfile(full_image_path):
        bw_space_single_image(
            full_image_path,
            edge_size=edge_size,
            target_folder=target_folder,
        )
    # else if Path,
    else:
        bw_space_image_for_folder(
            full_image_path,
            edge_size=edge_size,
            target_folder=target_folder,
        )
