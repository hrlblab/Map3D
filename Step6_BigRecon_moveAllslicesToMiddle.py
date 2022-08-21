import cv2 as cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')
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

    folder = '/Data3/Peize/Leeds_png_5'
    with_sg = False

    sections = glob.glob(os.path.join(folder, '*'))
    sections.sort()
    middle_idx_list = [3, 3, 3]
    for si in range(len(sections)):
        cur_section = sections[si]
        cases = glob.glob(os.path.join(cur_section, '*'))
        cases.sort(key=natural_keys)

        #for ci in range(0,1):
        for ci in range(0, len(cases)):

            case = cases[ci]
            now_case = os.path.basename(case)

            image_input_dir = os.path.join(case, '10X')
            if with_sg:
                ANTs_root_dir = os.path.join(case, 'ANTs_affine')
                output_folder = os.path.join(case, 'final_image_10X')
                output_mat_folder = os.path.join(case, 'final_affine')
            else:
                ANTs_root_dir = os.path.join(case, 'ANTs_only_affine')
                output_folder = os.path.join(case, 'ANTs_image_10X')
                output_mat_folder = os.path.join(case, 'ANTs_final_affine')

            slice_files = glob.glob(os.path.join(image_input_dir, '*'))
            slice_files.sort(key=natural_keys)

            roi_list = []
            for mi in range(len(slice_files)):
                roi_list.append(int(os.path.basename(slice_files[mi]).split('-')[0].replace('.png','')))
            print(roi_list)

            image_range_min = roi_list[0]
            image_range_max = roi_list[-1]
            middle_idx = int(middle_idx_list[ci])
            print('Three number')
            print(image_range_min)
            print(image_range_max)
            print(middle_idx)

            images = glob.glob(os.path.join(image_input_dir, '*'))
            images.sort(key=natural_keys)

            if not os.path.exists(output_mat_folder):
                os.makedirs(output_mat_folder)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for ki in range(1, len(slice_files)):
                print('Now copy affine mat %s' % (str(ki)))

                moving_index = ki + 1
                fixed_index = ki
                #print('Now combining the mtx %d-to-%d' % (moving_index, fixed_index))

                registration_dir = os.path.join(ANTs_root_dir, '%d-to-%d' % (moving_index, fixed_index))
                affine_mtx_file = os.path.join(registration_dir, 'step2_run_ants_reg', 'output0GenericAffine.mat')

                copy_mtx = os.path.join(output_mat_folder, '%s.mat' % (ki))
                print(affine_mtx_file)
                print(copy_mtx)
                os.system("cp %s %s" % (affine_mtx_file, copy_mtx))


            'affine in cv2'
            middle_image = plt.imread(images[middle_idx - 1])

            for ki in range(len(images)):
                slice = images[ki]
                now_slice = os.path.basename(slice)
                now_idx = int(now_slice.split('-')[0].replace('.png',''))
                now_idx=ki+1
                moving_idx = now_idx + 1
                fixed_idx = now_idx
                img_highres = plt.imread(slice)[:,:,:3]

                print('now is %d to %d' % (moving_idx, fixed_idx))

                if now_idx < middle_idx:
                    M_new = np.zeros((3, 3))
                    M_new[2, 2] = 1.
                    for ri in range(now_idx, middle_idx):
                        affine_root = os.path.join(output_mat_folder,'%d.mat' % (ri))
                        print(affine_root)
                        m = sitk.ReadTransform(affine_root)
                        FixP = m.GetFixedParameters()
                        M_vec = m.GetParameters()
                        M_inv = np.zeros((2, 3))

                        M_inv[0, 0] = M_vec[0]
                        M_inv[0, 1] = M_vec[1]
                        M_inv[1, 0] = M_vec[2]
                        M_inv[1, 1] = M_vec[3]
                        M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
                        M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])

                        M1 = np.zeros((3,3))
                        M1[2,2] = 1.
                        M1[:2, :] = cv2.invertAffineTransform(M_inv)

                        if ri == now_idx:
                            M_new = M1.copy()
                        else:
                            M_new = M_new.dot(M1)

                    now_image = plt.imread(images[ki])[:,:,:3]

                    affine_matrix_inv = cv2.invertAffineTransform(M_new[:2, :])
                    img1_affine = cv2.warpAffine(now_image, affine_matrix_inv[:2,:], (middle_image.shape[1], middle_image.shape[0]))
                    new_root = images[ki].replace(image_input_dir, output_folder)
                    plt.imsave(new_root, img1_affine)

                elif now_idx > middle_idx:
                    M_new = np.zeros((3, 3))
                    M_new[2, 2] = 1.
                    for ri in range(middle_idx, now_idx):
                        affine_root = os.path.join(output_mat_folder, '%d.mat' % (ri))
                        print(affine_root)
                        m = sitk.ReadTransform(affine_root)
                        FixP = m.GetFixedParameters()
                        M_vec = m.GetParameters()
                        M_inv = np.zeros((2, 3))

                        M_inv[0, 0] = M_vec[0]
                        M_inv[0, 1] = M_vec[1]
                        M_inv[1, 0] = M_vec[2]
                        M_inv[1, 1] = M_vec[3]
                        M_inv[0, 2] = M_vec[4] - (FixP[0] * M_inv[0, 0] + FixP[1] * M_inv[0, 1] - FixP[0])
                        M_inv[1, 2] = M_vec[5] - (FixP[0] * M_inv[1, 0] + FixP[1] * M_inv[1, 1] - FixP[1])

                        M1 = np.zeros((3, 3))
                        M1[2, 2] = 1.
                        M1[:2, :] = cv2.invertAffineTransform(M_inv)

                        if ri == middle_idx:
                            M_new = M1.copy()
                        else:
                            M_new = M_new.dot(M1)

                    now_image = plt.imread(images[ki])[:, :, :3]

                    # affine_matrix_inv = cv2.invertAffineTransform(M_new[:2, :])
                    img1_affine = cv2.warpAffine(now_image, M_new[:2, :], (middle_image.shape[1], middle_image.shape[0]))
                    new_root = images[ki].replace(image_input_dir, output_folder)
                    plt.imsave(new_root, img1_affine)
                else:
                    new_root = images[middle_idx  - 1].replace(image_input_dir, output_folder)
                    plt.imsave(new_root, middle_image)
