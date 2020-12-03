import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import cv2
from image_keras.supports.folder import create_folder_if_not_exist
from image_keras.supports.path import get_image_filenames, split_fullpath
from image_keras.utils.image_transform import mask_image
from natsort import natsorted


def mask_single_image(
    full_image_path,
    mask_bw_image_path,
    target_folder,
):
    _, _base_filename, _base_ext = split_fullpath(full_image_path)
    image = cv2.imread(full_image_path)
    masking_image = cv2.imread(mask_bw_image_path, cv2.IMREAD_GRAYSCALE)
    masked_image = mask_image(image, masking_image)
    result_path = os.path.join(target_folder, "{}{}".format(_base_filename, _base_ext))
    cv2.imwrite(result_path, masked_image)


def mask_image_for_folder(
    full_image_path,
    mask_bw_image_path,
    target_folder,
):
    files = get_image_filenames(full_image_path)
    files = natsorted(files)
    num_files = len(files)
    for index, file in enumerate(files):
        print("Processing {} ({}/{})...".format(file, (index + 1), num_files))
        mask_single_image(
            os.path.join(full_image_path, file),
            mask_bw_image_path=os.path.join(mask_bw_image_path, file),
            target_folder=target_folder,
        )


if __name__ == "__main__":
    # Parameters
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for image black white spaced edge."
    )
    parser.add_argument("--full_image_path", required=True, type=str, help="Full path.")
    parser.add_argument(
        "--mask_bw_image_path", required=True, type=str, help="Full path."
    )
    parser.add_argument(
        "--target_folder",
        required=True,
        type=str,
        help="Target folder.",
    )
    args = parser.parse_args()

    # parameter
    full_image_path: str = args.full_image_path
    mask_bw_image_path: str = args.mask_bw_image_path
    target_folder: str = os.path.join(args.target_folder)

    create_folder_if_not_exist(target_folder)

    # If file,
    if os.path.isfile(full_image_path):
        mask_single_image(
            full_image_path,
            mask_bw_image_path=mask_bw_image_path,
            target_folder=target_folder,
        )
    # else if Path,
    else:
        mask_image_for_folder(
            full_image_path,
            mask_bw_image_path=mask_bw_image_path,
            target_folder=target_folder,
        )


# img = cv2.imread("000.png")
# bw_img = img_color_to_bw(img)
# cv2.imwrite("000_bw_2.png", bw_img)

# # Edge draw
# img_edge = draw__edge_only(img, 10)
# cv2.imwrite("000_edge_2.png", img_edge)

# # Mask with Edge
# mask_with_image = draw__mask_with_edge(img, 10)
# cv2.imwrite("000_mask_with_image_2.png", mask_with_image)
