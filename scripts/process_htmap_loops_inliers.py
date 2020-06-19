#!/usr/bin/env python
# coding: utf-8

from multiprocessing import Pool
from tqdm.auto import tqdm
import numpy as np
import argparse
import os


def get_filepath(results_dir):
    if not os.path.isdir(results_dir):
        raise IOError(f"Results dir {results_dir} does not exist")

    filename_prefix = 'htmap_loops_'
    possible_filenames = list(filter(lambda x: x.startswith(filename_prefix), os.listdir(results_dir)))
    if len(possible_filenames) < 1:
        raise IOError(f"No file with prefix {filename_prefix} found")
    elif len(possible_filenames) > 1:
        raise IOError(f"Multiple files with prefix {filename_prefix} found: {possible_filenames}")

    filename = possible_filenames[0]
    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        raise IOError(f"File {filepath} not found")
    return filepath


def get_match_indices(line):
    bool_values_list = line.split('\t')[:-1]  # remove final element \n at the end of each line
    bool_values_array = np.asarray(bool_values_list).astype('bool')  # convert string arr to bool arr
    return np.where(bool_values_array == True)[0]  # np.where returns (index_arr, ) for 1 dimension, return index_arr


def write_image_match_list_to_file(image_match_list, filepath):
    with open(filepath, 'w') as f:
        for match_list in tqdm(image_match_list):
            if len(match_list) == 0:
                print('', file=f)
            else:
                print('\t'.join(map(str, match_list)), file=f)


def main(results_dir):

    image_match_list = []
    print(f"Processing folder: {results_dir}")

    filepath = get_filepath(results_dir)
    print(f"  Processing file: {filepath}")

    with open(filepath, 'r') as loops_file:
        lines = loops_file.readlines()
        with Pool() as pool:
            image_match_list = list(tqdm(pool.imap(get_match_indices, lines), total=len(lines)))
    print(f"    Processed lines: {len(image_match_list)}")

    new_filename = os.path.basename(filepath)
    split_index = new_filename.rindex('_')
    new_filename = new_filename[:split_index] + "_processed" + new_filename[split_index:]
    image_matches_filepath = os.path.join(results_dir, new_filename)
    print(f"    Writing matches to file: {image_matches_filepath}")
    write_image_match_list_to_file(image_match_list, image_matches_filepath)

    print(f"Process completed successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory "results" in htmap output folder')
    args = parser.parse_args()
    main(results_dir=args.results_dir)
