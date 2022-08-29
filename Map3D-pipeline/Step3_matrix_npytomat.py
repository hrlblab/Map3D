import cv2 as cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import SimpleITK as sitk
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import glob

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

if __name__ == "__main__":

    png_input = 'input_png'
    folder = 'data'

    cases = glob.glob(os.path.join(folder, '*'))
    cases.sort(key=natural_keys)

    print("Step 4 is running.")

    for ci in range(len(cases)):
        case = cases[ci]
        now_case = os.path.basename(case)

        affine_folder = os.path.join(case, 'sg_affine')
        output_folder = os.path.join(case, 'ANTs_affine')
        images_folder = os.path.join(png_input, now_case)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images = glob.glob(os.path.join(images_folder, '*'))

        images.sort(key=natural_keys)

        for ki in range(1, len(images)):
            slice = images[ki - 1]
            now_slice = os.path.basename(slice)
            now_idx = int(now_slice.split('-')[0].replace('.png', ''))
            now_idx = ki
            moving_idx = now_idx + 1
            fixed_idx = now_idx
            img_highres = plt.imread(slice)[:, :, :3]

            print('now is %d to %d' % (moving_idx, fixed_idx))
            affine_root = os.path.join(affine_folder, '%d-to-%d' % (moving_idx, fixed_idx), 'sg_affine_init.npy')
            affine = np.load(affine_root)
            M_inv = cv2.invertAffineTransform(affine)

            M_vector = np.zeros([6])

            # world coordinate to itk
            FixParameters = [img_highres.shape[1] / 2.0, img_highres.shape[0] / 2.0]
            M_vector[0] = M_inv[0, 0]
            M_vector[1] = M_inv[0, 1]
            M_vector[2] = M_inv[1, 0]
            M_vector[3] = M_inv[1, 1]
            M_vector[4] = (FixParameters[0] * M_inv[0, 0] + FixParameters[1] * M_inv[0, 1] - FixParameters[0]) + M_inv[
                0, 2]
            M_vector[5] = (FixParameters[0] * M_inv[1, 0] + FixParameters[1] * M_inv[1, 1] - FixParameters[1]) + M_inv[
                1, 2]

            m = sitk.ReadTransform('test.mat')

            m.SetParameters(M_vector)
            m.SetFixedParameters(FixParameters)

            save_root = os.path.dirname(affine_root.replace(affine_folder, output_folder))
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            sitk.WriteTransform(m, os.path.join(save_root, 'sg_affine_init.mat'))




