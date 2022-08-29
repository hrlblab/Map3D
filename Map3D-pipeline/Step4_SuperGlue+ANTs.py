import glob
import os
import argparse
import cv2
import nibabel as nib
import numpy as np
import pandas as pd

from superglue_ants_registration_onpair_new_RGB import register_a_pair, register_3D


# based on https://github.com/ANTsX/ANTs/wiki/Forward-and-inverse-warps-for-warping-images,-pointsets-and-Jacobians
# and
import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_df(fixed_bbox):
    df = pd.DataFrame(columns=['x', 'y', 't', 'label'])
    row = 0
    for fi in range(len(fixed_bbox)):
        bbox = fixed_bbox[fi]
        assert len(bbox) == 4

        df.loc[row] = [bbox[0], bbox[1], fi, 0]
        row = row + 1
        df.loc[row] = [bbox[2], bbox[1], fi, 0]
        row = row + 1
        df.loc[row] = [bbox[2], bbox[3], fi, 0]
        row = row + 1
        df.loc[row] = [bbox[0], bbox[3], fi, 0]
        row = row + 1

    return df


def mask_to_box(fixed_mask_jpg):
    img = cv2.imread(fixed_mask_jpg)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    resize_box = [cmin, rmin, (cmax - cmin), (rmax - rmin)]

    return resize_box


if __name__ == "__main__":

    png_input = 'input_png'
    folder = os.path.join(os.getcwd(), 'data')
    with_sg = True

    parser = argparse.ArgumentParser(description="Map3D Registration")
    parser.add_argument("--middle_images", type=str, default='2,2')
    args = parser.parse_args()
    middle_image_list = args.middle_images.strip().split(',')
    middle_idx_list = [int(x) for x in middle_image_list]

    cases = glob.glob(os.path.join(folder, '*'))
    cases.sort(key=natural_keys)

    for ci in range(0, len(cases)):
        # for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)

        image_input_dir = os.path.join(png_input, now_case)
        if with_sg:
            image_output_root_dir = os.path.join(case, 'ANTs_affine')
        else:
            image_output_root_dir = os.path.join(case, 'ANTs_only_affine')
        #
        # image_input_dir = '/Data/fromHaichun/tracking_pairwise/slices_all'
        # image_mask_dir = '/Data/fromHaichun/tracking_pairwise/mask_all
        # image_output_root_dir = '/Data/fromHaichun/major_review/registration_all_superglue'

        overall_results_dir = os.path.join(image_output_root_dir, 'all_results')

        slice_files = glob.glob(os.path.join(image_input_dir, '*'))
        slice_files.sort(key=natural_keys)
        print(image_input_dir)

        roi_list = []
        for mi in range(len(slice_files)):
            roi_list.append(int(os.path.basename(slice_files[mi]).split('-')[0].replace('.png', '')))
        print(roi_list)

        image_range_min = roi_list[0]
        image_range_max = roi_list[-1]
        middle_idx = int(middle_idx_list[ci])
        print('Three number')
        print(image_range_min)
        print(image_range_max)
        print(middle_idx)

        image_output_dir = image_output_root_dir
        overall_results_sub_dir = overall_results_dir

        for i in range(1, len(slice_files)):
            # for i in range(17, 18):

            moving_roi_curr = i + 1
            fixed_roi_curr = i

            print('now is %d to %d' % (moving_roi_curr, fixed_roi_curr))
            moving_jpg = slice_files[i]
            fixed_jpg = slice_files[i - 1]

            working_dir = os.path.join(image_output_dir, '%d-to-%d' % (moving_roi_curr, fixed_roi_curr))
            register_a_pair(with_sg, moving_jpg, fixed_jpg, working_dir)
