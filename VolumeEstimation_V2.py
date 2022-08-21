from random import randint
from MOTSequence import *

import matplotlib
import pandas as pd
from shapely.geometry import Polygon, MultiPoint

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import cv2 as cv2
import os
import math

import glob

import argparse
import numpy as np
#import torchvision
import matplotlib.cm as cm
import SimpleITK as sitk
import nibabel as nib

from math import acos
from math import sqrt
from math import pi
import cv2

from numpy import mean
from numpy import std
from numpy import median
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
from numpy import cov

# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr, spearmanr


import matplotlib.pyplot as plt
import numpy as np

def bland_altman_plot(ax, data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    ax.scatter(mean, diff, *args, edgecolor="black", **kwargs)
    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    #ax.fill_between(diff, md - 1.96*sd, md + 1.96*sd, color='gray')


def GetbboxFromMask(pred):
    # need to use cnt to transfer 512 coordinator to original image

    grey = cv2.cvtColor(np.stack([pred,pred,pred],2), cv2.COLOR_BGR2GRAY) * 255
    ret, thresh = cv2.threshold(grey, 127, 255, 0)

    im, contours, hierarchy = cv2.findContours(thresh.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # not copying here will throw an error
    now_x = 0
    now_y = 0
    now_w = 0
    now_h = 0
    now_area = 0

    output_bbox = np.zeros((4,1,2))
    for i in range(len(contours)):
        c = contours[i]
        x, y, w, h = cv2.boundingRect(c)
        #cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
        new_cnt = np.zeros((4,1,2))
        new_cnt[0,0,0] = x
        new_cnt[0,0,1] = y
        new_cnt[1,0,0] = x + w
        new_cnt[1,0,1] = y
        new_cnt[2,0,0] = x + w
        new_cnt[2,0,1] = y + h
        new_cnt[3,0,0] = x
        new_cnt[3,0,1] = y + h

        now_x = x
        now_y = y
        now_w = w
        now_h = h
        now_area = cv2.contourArea(c)

    r = (now_w + now_h) / 4

    return now_area, r

def CheckBboxAdd(pred, img, cnt, ratio):
    pixel_sum = pred[:,:].sum()

    if 1:#pixel_sum > 255 * (pred.shape[0] * pred.shape[1]) / 10:
        new_bbox, im = GetbboxFromMask(pred, img, cnt, ratio)
        return new_bbox, im
    else:
        return np.zeros((4,1,2)), None

def Box_IOU(new_bbox, cnt):

    x_list1 = [new_bbox[0, 0, 0], new_bbox[1, 0, 0],new_bbox[2, 0, 0], new_bbox[3, 0, 0]]
    y_list1 = [new_bbox[0, 0, 1], new_bbox[1, 0, 1], new_bbox[2, 0, 1], new_bbox[3, 0, 1]]
    x_list2 = [cnt[0, 0, 0], cnt[1, 0, 0], cnt[2, 0, 0], cnt[3, 0, 0]]
    y_list2 = [cnt[0, 0, 1], cnt[1, 0, 1], cnt[2, 0, 1], cnt[3, 0, 1]]

    line1 = [int(x_list1[0]), int(y_list1[0]), int(x_list1[1]), int(y_list1[1]),
             int(x_list1[2]), int(y_list1[2]), int(x_list1[3]), int(y_list1[3])]
    a = np.array(line1).reshape(4, 2)
    poly1 = Polygon(a).convex_hull

    line2 = [int(x_list2[0]), int(y_list2[0]), int(x_list2[1]), int(y_list2[1]),
             int(x_list2[2]), int(y_list2[2]), int(x_list2[3]), int(y_list2[3])]
    b = np.array(line2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull

    union_poly = np.concatenate((a, b))

    if not poly1.intersects(poly2):
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)
            else:
                iou = float(inter_area) / union_area

        except shapely.geos.TopologicalError:
            # print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0

    return iou

def patch_displacement(cnt, id, start, r_root, ratio):
    start = int(start)

    if start == 8:
        new_cnt = np.zeros(cnt.shape)
        for ci in range(len(cnt)):
            # print('aaa')
            new_cnt[ci, :, 0] = cnt[ci, :, 0] * ratio
            new_cnt[ci, :, 1] = cnt[ci, :, 0] * ratio
        return new_cnt

    elif start < 8:
        folder = os.path.join(r_root, 'Label_%05d_aff' % (id), 'displacement')
        fixed_roi_num = 8

        for ri in range(start, fixed_roi_num):
            if ri == start:
                displacement_root = os.path.join(folder, '%s.npy' % (ri))
                displacement = - np.load(displacement_root)
            else:
                displacement_root = os.path.join(folder, '%s.npy' % (ri))
                new_displacement = - np.load(displacement_root)
                displacement += new_displacement

    else:
        folder = os.path.join(r_root, 'Label_%05d_aff' % (id), 'displacement')
        fixed_roi_num = 8
        for ri in range(fixed_roi_num, start):
            if ri == fixed_roi_num:
                displacement_root = os.path.join(folder, '%s.npy' % (ri))
                displacement = np.load(displacement_root)
            else:
                displacement_root = os.path.join(folder, '%s.npy' % (ri))
                new_displacement = np.load(displacement_root)
                displacement += new_displacement

    # small_coor = pd.read_csv(os.path.join(r_root, 'Label_%05d_aff' % (id), 'Time_%05d_Label_%05d_Cropped_highres_aff.csv' % (start, id)))
    # x_list = small_coor['x'].tolist()
    # y_list = small_coor['y'].tolist()
    # small_cnt = np.zeros(cnt.shape)


    #small_coor
    new_cnt = np.zeros(cnt.shape)
    for ci in range(len(cnt)):
        #new_cnt[ci,:,0] = cnt[ci,:,0] * ratio + displacement[500,500,0] * 500
        #new_cnt[ci,:,1] = cnt[ci,:,0] * ratio + displacement[500,500,1] * 500

        new_cnt[ci, :, 0] = cnt[ci, :, 0] * ratio + np.mean(displacement[:,:,0]) * 500
        new_cnt[ci, :, 1] = cnt[ci, :, 0] * ratio + np.mean(displacement[:,:,1]) * 500

    return new_cnt

if __name__ == "__main__":

    case = ['WD-35686','WD-47744','WD-48350','WD-55197','WD-64503']
    # case = ['WD-64503']
    # case = ['WD-47744']
    Failed_list = []

    evaluation = 0

    image_output_root_dir = '/Data2/HumanKidney/2profile/R24_png_big/'

    registration_root = '/Data2/HumanKidney/2profile/R24_png/'
    output_folder = '/Data2/HumanKidney/2profile/R24_png_big_ss'
    patch_folder = '/Data2/HumanKidney/2profile/R24_Seq_Voxel'


    test_mat = '/Data2/HumanKidney/2profile/test.mat'


    for si in range(0, len(case)):
        subName = case[si]
        if subName in Failed_list:
            continue

        ssfolder = os.path.join(output_folder, subName)
        disfolder = os.path.join(registration_root, subName)

        sequences = os.path.join(patch_folder, subName)

        if evaluation:
            MiddleMap_csv = os.path.join(ssfolder, 'MiddleMapping_SG_FP_merge.csv')
            MappingList = pd.read_csv(MiddleMap_csv)

            x_list = MappingList['x'].tolist()
            y_list = MappingList['y'].tolist()
            t_list = MappingList['t'].tolist()
            l_list = MappingList['l'].tolist()
            std_list = MappingList['std'].tolist()
            end_list = MappingList['end'].tolist()

            registrationList = pd.read_csv(os.path.join(output_folder, subName, 'SelectPatch.csv'))
            second_registration_dir = os.path.join(patch_folder, subName)

            ratio = 4

            use_contour = 1
            if use_contour:
                mask_folder = 'maskselection_use_contour_png'
            else:
                mask_folder = 'maskselection'

            annotation_folder = 'manual'


            volume_size = 0.5
            volume_thick = 8

            each_voxel = volume_size * volume_size * volume_thick

            df = pd.DataFrame(columns = ['id','middle_max','two_profile'])

            ### check single patch
            for i in range(int(len(MappingList)/4)):
                row1 = i * 4
                id1 = l_list[row1]
                start1 = std_list[row1]
                end1 = end_list[row1]
                middle = int((end1 + start1) / 2)
                print(id1)

                annotation_volume = 0
                slice_volume = 0
                middle_max_volume = 0
                mean_area_volume = 0
                two_profile_volume = 0

                # now_folder = glob.glob(os.path.join(second_registration_dir, "Label_%05d_aff" % (id1), annotation_folder))
                #
                # if len(now_folder) == 0:
                #     continue

                '''No annotation'''
                # annotation_file = glob.glob(os.path.join(second_registration_dir, "Label_%05d_aff" % (id1), annotation_folder,"*.png" ))
                #
                # for ai in range(len(annotation_file)):
                #     annotation = plt.imread(annotation_file[ai])[:, :, 0]
                #     annotation[annotation > 0.5] = 1.
                #     annotation[annotation != 1.] = 0.
                #     annotation_volume += annotation.sum() * each_voxel

                if end1 - start1 == 0:
                    time = start1

                    mask_root = os.path.join(second_registration_dir, "Label_%05d_aff" % (id1), mask_folder,
                                             "Time_%05d_Label_%05d_Cropped_highres_aff.png" % (time, id1))

                    #annotation = plt.imread(annotation_root)[:,:,0]
                    mask = plt.imread(mask_root)[:,:,0]

                    mask[mask > 0.5] = 1.
                    mask[mask != 1.] = 0.

                    # annotation volume
                    #annotation_volume = annotation.sum() * each_voxel

                    #slice-based volume
                    # slice_volume = mask.sum() * each_voxel

                    # middle max volume with a radius
                    nowArea, r = GetbboxFromMask(mask)
                    r_space = r * volume_size
                    middle_max_volume = 4/3 * math.pi * math.pow(r_space,3)

                    # mean area volume
                    # area_space = mask.sum() * volume_size * volume_size
                    # mean_area_volume = math.sqrt(math.pow(area_space,3)) * 1.38 / 1.01

                    #two_profile
                    two_profile_volume_final = 4/3 * math.pi * math.pow(r_space,3)

                else:
                    maxArea = 0
                    maxR = 0

                    R1 = 0
                    R2 = 0

                    meanArea_sum = 0
                    slice_cnt = 0
                    middle = start1

                    '''MiddleMax calculation only'''

                    for ii in range(int(start1),int(end1) + 1):
                        time = ii
                        slice_cnt += 1
                        #annotation_root = os.path.join(second_registration_dir,"Label_%05d_aff" % (id1), annotation_folder, "Time_%05d_Label_%05d_Cropped_highres_aff.png" % (time, id1))
                        mask_root = os.path.join(second_registration_dir,"Label_%05d_aff" % (id1), mask_folder, "Time_%05d_Label_%05d_Cropped_highres_aff.png" % (time, id1))

                        #annotation = plt.imread(annotation_root)[:, :, 0]
                        mask = plt.imread(mask_root)[:, :, 0]

                        mask[mask > 0.5] = 1.
                        mask[mask != 1.] = 0.

                        # annotation volume
                        #annotation_volume += annotation.sum() * each_voxel

                        # slice-based volume
                        # slice_volume += mask.sum() * each_voxel

                        # middle max volume with a radius
                        nowArea, r = GetbboxFromMask(mask)

                        if nowArea >= maxArea:
                            maxArea = nowArea
                            maxR = r


                    # middle max volume with a radius
                    r_space = maxR * volume_size
                    middle_max_volume = 4 / 3 * math.pi * math.pow(r_space,3)


                    '''two profile calculation only'''

                    length = end1 - start1 + 1
                    two_profile_volume = np.zeros((int(length)))
                    for ii in range(int(start1), int(end1)):

                        time = ii
                        mask_root = os.path.join(second_registration_dir,"Label_%05d_aff" % (id1), mask_folder, "Time_%05d_Label_%05d_Cropped_highres_aff.png" % (time, id1))

                        #annotation = plt.imread(annotation_root)[:, :, 0]
                        mask = plt.imread(mask_root)[:, :, 0]

                        mask[mask > 0.5] = 1.
                        mask[mask != 1.] = 0.

                        nowArea, R1 = GetbboxFromMask(mask)

                        time = ii + 1
                        mask_root = os.path.join(second_registration_dir, "Label_%05d_aff" % (id1), mask_folder,
                                                 "Time_%05d_Label_%05d_Cropped_highres_aff.png" % (time, id1))

                        # annotation = plt.imread(annotation_root)[:, :, 0]
                        mask = plt.imread(mask_root)[:, :, 0]

                        mask[mask > 0.5] = 1.
                        mask[mask != 1.] = 0.

                        nowArea, R2 = GetbboxFromMask(mask)
                        if (R1 == 0) and (R2 == 0):
                            two_profile_volume[int(ii-start1)] = 0

                        elif (R1 == 0) and (R2 != 0):
                            r_space = R2 * volume_size
                            two_profile_volume[int(ii - start1)] = 4 / 3 * math.pi * math.pow(r_space, 3)

                        elif (R1 != 0) and (R2 == 0):
                            r_space = R1 * volume_size
                            two_profile_volume[int(ii - start1)] = 4 / 3 * math.pi * math.pow(r_space, 3)

                        else:
                            # two_profile
                            R2_space = R2 * volume_size
                            R1_space = R1 * volume_size
                            R_two = math.sqrt(math.pow(R2_space, 2) + math.pow(
                                (math.pow(R2_space, 2) - math.pow(R1_space, 2) - math.pow(volume_thick, 2)) / (
                                            2 * volume_thick), 2))

                            two_profile_volume[int(ii-start1)] = 4 / 3 * math.pi * math.pow(R_two, 3)


                    if two_profile_volume.sum() == 0:
                        two_profile_volume_final = 0
                    else:
                        two_profile_volume_final = np.mean(two_profile_volume[two_profile_volume != 0])


                now_row = len(df)
                df.loc[now_row] = [id1,middle_max_volume,two_profile_volume_final]

    #            np.save(os.path.join(second_registration_dir, "Label_%05d_aff" % (id1), "slices_mask.npy",),volume)
    #             np.save(os.path.join(second_registration_dir, "Label_%05d_aff" % (id1), "middle_mask.npy",),volume)

            df.to_csv(os.path.join(ssfolder, 'VolumeEstimation.csv'), index = False)

        else:
            df = pd.read_csv(os.path.join(ssfolder, 'VolumeEstimation.csv'))

        # Draw the plot
        # df = pd.DataFrame(columns = ['id','annotation','slice','middle_max','mean_area','two_profile'])
        # data1 = df['annotation']
        # data2 = df['slice']
        data3 = df['middle_max']
        # data4 = df['mean_area']
        data5 = df['two_profile']

        print('middle_max: mean=%.3f stdv=%.3f' % (mean(data3), std(data3)))
        print('two_profile: mean=%.3f stdv=%.3f' % (mean(data5), std(data5)))

        fig = plt.figure(figsize=(11,11))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)


        ax1.hist(data3, bins='auto')  # arguments are passed to np.histogram
        ax1.set_xlim(0, 3000000)
        ax1.set_ylim(0, 150)
        #ax1.title("middle_max")

        ax2.hist(data5, bins='auto')  # arguments are passed to np.histogram
        ax2.set_xlim(0, 3000000)
        ax2.set_ylim(0, 150)
        #ax2.title(" two_profile")
        fig.savefig(os.path.join(ssfolder, '%s_hist.png' % (subName)))
        # print('aaa')
        #
        # fig = plt.figure(figsize=(12, 12))
        # #ax1 = fig.add_subplot(221)
        # ax2 = fig.add_subplot(221)
        # ax3 = fig.add_subplot(222)
        # #ax4 = fig.add_subplot(224)
        # ax5 = fig.add_subplot(223)
        # ax6 = fig.add_subplot(224)
        #
        # x = np.linspace(0, 800000)
        # #y = x
        #
        # corr1, _ = spearmanr(data3, data3)
        # #ax3.set_title('Corr. = %02f' % corr)
        # ax3.scatter(data2, data1,edgecolor="black")
        # #ax3.plot(x, y,  color='r')
        # #ax3.set_xlabel('Manual')
        # #ax3.set_ylabel('Our method')
        # ax3.set_ylim([0, 800000])
        # ax3.set_xlim([0, 800000])
        #
        # m, b = np.polyfit(data2, data1, 1)
        # ax3.plot(x, m*x + b, color='r')
        #
        # #ax3.set_yticks([])
        # #ax3.set_xticks([])
        #
        # corr2, _ = spearmanr(data1, data3)
        # #ax2.set_title('Corr. = %02f' % corr)
        # #ax2.set_ylabel('MPA')
        # ax2.scatter(data3, data1,edgecolor="black")
        # #ax2.set_xlabel('Manual')
        # #ax2.plot(x, y,  color='r')
        # ax2.set_ylim([0, 800000])
        # ax2.set_xlim([0, 800000])
        # #ax2.set_yticks([])
        # #ax2.set_xticks([])
        # m, b = np.polyfit(data3, data1, 1)
        # ax2.plot(x, m * x + b, color='r')
        #
        # # corr, _ = spearmanr(data1, data4)
        # # ax3.set_title('Corr. = %02f' % corr)
        # # ax3.set_ylabel('mean_area')
        # # ax3.scatter(data1, data4)
        # # ax3.set_xlabel('Manual')
        # # ax3.plot(x, y, color='r')
        # # ax3.set_ylim([0, 160000])
        #
        # #corr4, _ = spearmanr(data1, data5)
        # #ax1.set_title('Corr. = %02f' % corr)
        # #ax1.set_ylabel('2-profile')
        # #ax1.scatter(data5, data1,edgecolor="black")
        # #ax1.set_xlabel('Manual')
        # #ax1.plot(x, y, color='r')
        # #ax1.set_ylim([0, 800000])
        # #ax1.set_xlim([0, 800000])
        # #ax1.set_yticks([])
        # #ax1.set_xticks([])
        # #m, b = np.polyfit(data5, data1, 1)
        # #ax1.plot(x, m * x + b, color='r')
        #
        # # pyplot.show()
        # # #ax.show()
        # #
        # # fig2 = plt.figure(figsize=(6, 6))
        # # ax4 = fig2.add_subplot(131)
        # # ax5 = fig2.add_subplot(132)
        # # ax6 = fig2.add_subplot(133)
        #
        # # fig2 = plt.figure(figsize=(6, 6))
        # # ax = fig2.add_subplot(221)
        # # ax2 = fig2.add_subplot(222)
        # # ax3 = fig2.add_subplot(223)
        # # ax4 = fig2.add_subplot(224)
        #
        # bland_altman_plot(ax6, data2,data1)
        # #ax.set_title('slice')
        # #ax6.set_ylabel('difference')
        # #ax6.set_xlabel('Our method')
        # ax6.set_xlim([0, 800000])
        # ax6.set_ylim([-500000, 1500000])
        # #ax6.set_yticks([])
        # #ax6.set_xticks([])
        # #ax.show()
        #
        # bland_altman_plot(ax5, data3, data1)
        # #ax2.set_title('Bland-Altman Plot')
        # #ax5.set_ylabel('difference')
        # #ax5.set_xlabel('MPA')
        # ax5.set_xlim([0, 800000])
        # ax5.set_ylim([-500000, 1500000])
        # #ax5.set_yticks([])
        # #ax5.set_xticks([])
        # #ax.show()
        #
        # # bland_altman_plot(ax6, data1, data4)
        # # #ax3.set_title('Bland-Altman Plot')
        # # ax6.set_ylabel('difference')
        # # ax6.set_xlabel('mean_area')
        # # #ax.show()
        #
        # bland_altman_plot(ax4, data5, data1)
        # #ax4.set_ylabel('difference')
        # #ax4.set_xlabel('2-profile')
        # ax4.set_xlim([0, 800000])
        # ax4.set_ylim([-3000000, 3000000])
        # #ax4.set_yticks([])
        # #ax4.set_xticks([])
        #
        # #ax4.set_title('Bland-Altman Plot')
        # #ax.show()
        #
        #
        #
        # # pyplot.scatter(data1, data2)
        # # pyplot.ylim([0, 100000])
        # # pyplot.show()
        #
        #
        # corr, _ = pearsonr(data1, data2)
        # print('Pearsons correlation: %.3f' % corr)


