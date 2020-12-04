import os
import sys

sys.path.append(os.getcwd())

import shutil
from argparse import ArgumentParser
from os.path import basename
from typing import Optional

from image_keras.supports.folder import create_folder_if_not_exist, files_in_folder
from image_keras.supports.list import list_diff, list_intersection
from image_keras.supports.path import get_image_filenames, split_fullpath
from natsort import natsorted


def create_image_frame(
    full_image_path,
    zero_folder,
    prev_folders,
    next_folders,
    num_fillzero,
    target_folder,
    separator: str = "_",
):
    _, _base_filename, _base_ext = split_fullpath(full_image_path)

    zero_folder_2 = os.path.join(target_folder, zero_folder)
    zero_file = os.path.join(zero_folder_2, "{}{}".format(_base_filename, _base_ext))
    shutil.copyfile(full_image_path, zero_file)

    zero_file_number: str = _base_filename[: _base_filename.find(separator)]
    zero_file_after: str = _base_filename[_base_filename.find(separator) :]

    for index, prev_folder in enumerate(prev_folders):
        new_number: int = int(zero_file_number) + (index + 1)
        new_name: str = "{}{}".format(
            str(new_number).zfill(num_fillzero), zero_file_after
        )
        prev_folder_2 = os.path.join(target_folder, prev_folder)
        prev_file = os.path.join(prev_folder_2, "{}{}".format(new_name, _base_ext))
        shutil.copyfile(full_image_path, prev_file)

    for index, next_folder in enumerate(next_folders):
        new_number: int = int(zero_file_number) - (index + 1)
        new_name: str = "{}{}".format(
            str(new_number).zfill(num_fillzero), zero_file_after
        )
        next_folder_2 = os.path.join(target_folder, next_folder)
        next_file = os.path.join(next_folder_2, "{}{}".format(new_name, _base_ext))
        shutil.copyfile(full_image_path, next_file)


def create_image_frame_for_folder(
    full_image_path,
    zero_folder,
    prev_folders,
    next_folders,
    num_fillzero,
    target_folder,
):
    files = get_image_filenames(full_image_path)
    files = natsorted(files)
    num_files = len(files)
    for index, file in enumerate(files):
        print("Processing {} ({}/{})...".format(file, (index + 1), num_files))
        create_image_frame(
            os.path.join(full_image_path, file),
            zero_folder=zero_folder,
            prev_folders=prev_folders,
            next_folders=next_folders,
            num_fillzero=num_fillzero,
            target_folder=target_folder,
        )


def keep_only_common_files(zero_folder, prev_folders, next_folders, target_folder):
    # get common files
    common_files = files_in_folder(os.path.join(target_folder, zero_folder))

    for prev_folder in prev_folders:
        prev_files = files_in_folder(os.path.join(target_folder, prev_folder))
        common_files = list_intersection(common_files, prev_files)

    for next_folder in next_folders:
        next_files = files_in_folder(os.path.join(target_folder, next_folder))
        common_files = list_intersection(common_files, next_files)

    # remove other files
    zero_files = files_in_folder(os.path.join(target_folder, zero_folder))
    zero_other_files = list_diff(zero_files, common_files)
    for zero_other_file in zero_other_files:
        os.remove(os.path.join(target_folder, zero_folder, zero_other_file))

    for prev_folder in prev_folders:
        prev_files = files_in_folder(os.path.join(target_folder, prev_folder))
        prev_other_files = list_diff(prev_files, common_files)
        for prev_other_file in prev_other_files:
            os.remove(os.path.join(target_folder, prev_folder, prev_other_file))

    for next_folder in next_folders:
        next_files = files_in_folder(os.path.join(target_folder, next_folder))
        next_other_files = list_diff(next_files, common_files)
        for next_other_file in next_other_files:
            os.remove(os.path.join(target_folder, next_folder, next_other_file))


if __name__ == "__main__":
    # Parameters
    parser: ArgumentParser = ArgumentParser(description="Arguments for image resize")
    parser.add_argument(
        "--full_image_abspath", required=True, type=str, help="Full path."
    )
    parser.add_argument(
        "--num_prev_frame", default=0, type=int, help="Prev frame number."
    )
    parser.add_argument(
        "--num_next_frame", default=0, type=int, help="Next frame number."
    )
    parser.add_argument(
        "--num_fillzero", default=3, type=int, help="Number of fill zero."
    )
    parser.add_argument(
        "--common_folder_name",
        type=str,
        help="Common folder name.",
    )
    parser.add_argument(
        "--target_folder",
        required=True,
        type=str,
        help="Target folder.",
    )
    args = parser.parse_args()

    # parameter
    full_image_abspath: str = args.full_image_abspath
    num_prev_frame: int = args.num_prev_frame
    num_next_frame: int = args.num_next_frame
    num_fillzero: int = args.num_fillzero
    common_folder_name: Optional[str] = args.common_folder_name
    target_folder: str = os.path.join(args.target_folder)

    if not os.path.exists(full_image_abspath):
        raise Exception("'{}' does not exist.".format(full_image_abspath))

    create_folder_if_not_exist(target_folder)

    fp, _, _ = split_fullpath(full_image_abspath)
    if common_folder_name is None:
        last_path = basename(fp)
    else:
        last_path = common_folder_name

    zero_folder = "{}zero".format(last_path)
    prev_folders = []
    next_folders = []

    path = os.path.join(target_folder, zero_folder)
    create_folder_if_not_exist(path)

    for prev_i in range(num_prev_frame):
        prev_folder = "{}p{}".format(last_path, prev_i + 1)
        path = os.path.join(target_folder, prev_folder)
        create_folder_if_not_exist(path)
        prev_folders.append(prev_folder)

    for next_i in range(num_next_frame):
        next_folder = "{}n{}".format(last_path, next_i + 1)
        path = os.path.join(target_folder, next_folder)
        create_folder_if_not_exist(path)
        next_folders.append(next_folder)

    # If file,
    if os.path.isfile(full_image_abspath):
        create_image_frame(
            full_image_abspath,
            zero_folder=zero_folder,
            prev_folders=prev_folders,
            next_folders=next_folders,
            num_fillzero=num_fillzero,
            target_folder=target_folder,
        )
    # else if Path,
    else:
        create_image_frame_for_folder(
            full_image_abspath,
            zero_folder=zero_folder,
            prev_folders=prev_folders,
            next_folders=next_folders,
            num_fillzero=num_fillzero,
            target_folder=target_folder,
        )
        keep_only_common_files(
            zero_folder=zero_folder,
            prev_folders=prev_folders,
            next_folders=next_folders,
            target_folder=target_folder,
        )
