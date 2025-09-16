"""
    @Project: D2AFM
    @Author: Panke
    @FileName: split_data.py
    @Time: 2025/9/17 00:13
    @Email: None
"""

import os
import shutil
from sklearn.model_selection import train_test_split

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

def split_and_copy_dataset(src_dir, input_subdir, target_subdir, train_dir, val_dir, test_size=0.1, random_state=42):



    input_files = sorted(os.listdir(os.path.join(src_dir, input_subdir)))
    target_files = sorted(os.listdir(os.path.join(src_dir, target_subdir)))

    input_paths = [os.path.join(src_dir, input_subdir, x) for x in input_files if is_image_file(x)]
    target_paths = [os.path.join(src_dir, target_subdir, x) for x in target_files if is_image_file(x)]


    assert len(input_paths) == len(target_paths), "The number of input and target images does not match!"


    train_inp, val_inp, train_tar, val_tar = train_test_split(
        input_paths, target_paths, test_size=test_size, random_state=random_state)


    copy_files(train_inp, train_dir, 'input')
    copy_files(train_tar, train_dir, 'GT')
    copy_files(val_inp, val_dir, 'input')
    copy_files(val_tar, val_dir, 'GT')


def copy_files(file_list, dest_dir, subfolder):
    if not os.path.exists(os.path.join(dest_dir, subfolder)):
        os.makedirs(os.path.join(dest_dir, subfolder))

    for file in file_list:
        shutil.copy(file, os.path.join(dest_dir, subfolder, os.path.basename(file)))


if __name__ == '__main__':
    src_dir = r'' # root dir
    input_subdir = 'input' # input img
    target_subdir = 'GT'    # Ground Truth

    train_dir = r''# train dataset dir
    val_dir = r'' # val dataset dir

    split_and_copy_dataset(src_dir, input_subdir, target_subdir, train_dir, val_dir, test_size=0.1)


