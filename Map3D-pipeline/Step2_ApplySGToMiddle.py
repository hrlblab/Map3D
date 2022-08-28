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



def affine_backward(slice, middle_image, now_idx, now_middle_idx, images_folder, affine_folder, output_folder):
    now_image = plt.imread(slice)[:,:,:3]

    affine_matrix = np.zeros((3,3))
    affine_matrix[2,2] = 1.

    for sgi in range(now_idx, now_middle_idx):
        if sgi == now_idx:
            matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 1, sgi), 'sg_affine_init.npy')
            affine_matrix[:2,:] = cv2.invertAffineTransform(np.load(matrix_root))
            print(matrix_root)
        else:
            matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 1, sgi), 'sg_affine_init.npy')
            new_affine = np.zeros((3,3))
            new_affine[2,2] = 1.
            new_affine[:2, :] = cv2.invertAffineTransform(np.load(matrix_root))
            affine_matrix = affine_matrix.dot(new_affine)
            print(matrix_root)

    affine_matrix_inv = cv2.invertAffineTransform(affine_matrix[:2,:])

    img1_affine = cv2.warpAffine(now_image, affine_matrix[:2,:], (middle_image.shape[1], middle_image.shape[0]))
    new_root = slice.replace(images_folder, output_folder)
    plt.imsave(new_root, img1_affine)


def affine_forward(slice, middle_image, now_idx, now_middle_idx, images_folder, affine_folder, output_folder):
    now_image = plt.imread(slice)[:, :, :3]

    affine_matrix = np.zeros((3,3))
    affine_matrix[2,2] = 1.

    for sgi in range(now_middle_idx, now_idx):
        if sgi == now_middle_idx:
            matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 1, sgi), 'sg_affine_init.npy')
            affine_matrix[:2,:] = np.load(matrix_root)
            print(matrix_root)
        else:
            new_affine = np.zeros((3, 3))
            new_affine[2, 2] = 1.
            matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 1, sgi), 'sg_affine_init.npy')
            new_affine[:2, :] = np.load(matrix_root)
            affine_matrix = affine_matrix.dot(new_affine)
            print(matrix_root)

    img1_affine = cv2.warpAffine(now_image, affine_matrix[:2,:], (middle_image.shape[1], middle_image.shape[0]))
    new_root = slice.replace(images_folder, output_folder)
    plt.imsave(new_root, img1_affine)

if __name__ == "__main__":

    folder = 'input_png'

    parser = argparse.ArgumentParser(description="Map3D Registration")
    parser.add_argument("--middle_images", type=str, default='1')
    args = parser.parse_args()
    middle_image_list = args.middle_images.strip().split(',')
    middle_idx = [int(x) for x in middle_image_list]

    print("list")
    print(middle_idx)

    cases = glob.glob(os.path.join(folder, '*'))
    cases.sort(key=natural_keys)

    print("Step 2 is running.")

    for ci in range(len(cases)):
    #for ci in range(0,1):
        print(cases[ci])
        case = cases[ci]
        now_case = os.path.basename(case)

        images_folder = os.path.join(case, '10X')
        affine_folder = os.path.join(case, 'sg_affine')
        output_folder = os.path.join(case, 'affine_10X')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images = glob.glob(os.path.join(images_folder, '*'))

        images.sort(key=natural_keys)
        now_middle_idx = middle_idx[ci]

        middle_image = plt.imread(images[now_middle_idx - 1])
        print(len(images))

        for ki in range(1,len(images)+1):
            slice = images[ki-1]
            now_slice = os.path.basename(slice)
            now_idx = int(now_slice.split('-')[0].replace('.png',''))

            if ki < now_middle_idx:
                print('%s to %s' % (ki, now_middle_idx) )
                affine_backward(slice, middle_image, ki, now_middle_idx, images_folder, affine_folder, output_folder)
            elif ki > now_middle_idx:
                print('%s to %s' % (ki, now_middle_idx))
                affine_forward(slice, middle_image,ki, now_middle_idx, images_folder, affine_folder, output_folder)
            else:
                now_image = middle_image
                new_root = slice.replace(images_folder, output_folder)
                plt.imsave(new_root, now_image)





