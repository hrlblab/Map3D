import cv2 as cv2
import numpy as np
from PIL import Image
import os
import SimpleITK as sitk

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from skimage.transform import resize
import glob
import openslide
import matplotlib.pyplot as plt
import xmltodict
import pandas as pd
import time

def get_contour(img, contour,down_rate, shift):
    vertices = contour['Vertices']['Vertex']
    cnt = np.zeros((4,1,2))

    cnt[0, 0, 0] = vertices[0]['@X']
    cnt[0, 0, 1] = vertices[0]['@Y']
    cnt[1, 0, 0] = vertices[1]['@X']
    cnt[1, 0, 1] = vertices[0]['@Y']
    cnt[2, 0, 0] = vertices[1]['@X']
    cnt[2, 0, 1] = vertices[1]['@Y']
    cnt[3, 0, 0] = vertices[0]['@X']
    cnt[3, 0, 1] = vertices[1]['@Y']

    cnt = cnt / down_rate

    cnt[0, 0, 0] = cnt[0, 0, 0] - shift
    cnt[1, 0, 0] = cnt[1, 0, 0] - shift
    cnt[2, 0, 0] = cnt[2, 0, 0] - shift
    cnt[3, 0, 0] = cnt[3, 0, 0] - shift

    glom = img[int(cnt[0, 0, 1]):int(cnt[3, 0, 1]), int(cnt[0, 0, 0]):int(cnt[1, 0, 0])]

    return glom, cnt

def scn_to_png_Leeds(svs_file,xml_file,output_dir,lv,t):
    simg = openslide.open_slide(svs_file)

    lv = 1
    start_x = 0
    start_y = 0
    width_x = np.int(simg.properties['openslide.level[0].width'])
    height_y = np.int(simg.properties['openslide.level[0].height'])

    # img = simg.read_region((start_x, start_y), lv, (height_y, width_x))

    down_rate = simg.level_downsamples[lv]
    end_x = start_x + width_x
    end_y = start_y + height_y

    # max_height = width_x
    # max_widths = height_y

    patch_size = 32
    read_patch = 0

    if read_patch:
        whole_width_lv2 = int(np.ceil((end_x - start_x) / down_rate))
        whole_height_lv2 = int(np.ceil((end_y - start_y) / down_rate))
        num_patch_x_lv = np.int(np.ceil(width_x / down_rate / patch_size))
        num_patch_y_lv = np.int(np.ceil(height_y / down_rate / patch_size))
        whole_width_lv = num_patch_x_lv * patch_size
        whole_height_lv = num_patch_y_lv * patch_size
        img = np.zeros((whole_height_lv2, whole_width_lv2, 3), dtype=np.uint8)

        for xi in range(0,num_patch_y_lv):
            for yi in range(0,num_patch_x_lv):
                patch_size_x = patch_size
                patch_size_y = patch_size

                if xi == num_patch_y_lv - 1:
                    patch_size_x = whole_height_lv2 - xi * patch_size

                if yi == num_patch_x_lv - 1:
                    patch_size_y = whole_width_lv2 - yi * patch_size

                low_res_offset_x = np.int(xi * patch_size)
                low_res_offset_y = np.int(yi * patch_size)

                patch_start_x = start_x + np.int(low_res_offset_y * down_rate)
                patch_start_y = start_y + np.int(low_res_offset_x * down_rate)

                img_lv = simg.read_region((patch_start_x, patch_start_y), lv, (patch_size_y, patch_size_x))
                img_lv = np.array(img_lv.convert('RGB'))
                # if (low_res_offset_x + patch_size) <= whole_height_lv and (
                #         low_res_offset_y + patch_size) <= whole_width_lv:
                img[low_res_offset_x:(low_res_offset_x + patch_size),
                low_res_offset_y:(low_res_offset_y + patch_size), :] = img_lv

    else:
        whole_height_lv2 = int(np.ceil((end_x - start_x) / down_rate))
        whole_width_lv2 = int(np.ceil((end_y - start_y) / down_rate))
        img = simg.read_region((start_x, start_y), lv, (whole_height_lv2, whole_width_lv2))
        img = np.array(img.convert('RGB'))
        cimg = img.copy()

    # black_pixels = np.where(
    #     (img[:, :, 0] == 0) | (img[:, :, 1] == 0) | (img[:, :, 2] == 0))
    # img[black_pixels] = [246, 246, 246]

    name = os.path.basename(svs_file).replace('.svs','.png')

    #X40_output_folder = os.path.join(output_dir, '40X')
    #X20_output_folder = os.path.join(output_dir, '20X')
    X10_output_folder = os.path.join(output_dir, '10X')
    #
    # if not os.path.exists(X40_output_folder):
    #     os.makedirs(X40_output_folder)

    # if not os.path.exists(X20_output_folder):
    #     os.makedirs(X20_output_folder)

    if not os.path.exists(X10_output_folder):
        os.makedirs(X10_output_folder)

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # img_20X = resize(cimg, (int(cimg.shape[0] / 2), int(cimg.shape[1] / 2), cimg.shape[2]))
    # img_10X = resize(cimg, (int(cimg.shape[0] / 4), int(cimg.shape[1] / 4), cimg.shape[2]))

    # plt.imsave(os.path.join(X40_output_folder, name), cimg)
    # plt.imsave(os.path.join(X20_output_folder, name), img_20X)
    plt.imsave(os.path.join(X10_output_folder, name), cimg)


    # read region
    # with open(xml_file) as fd:
    #     doc = xmltodict.parse(fd.read())
    # layers = doc['Annotations']['Annotation']
    # try :
    #     contours = layers['Regions']['Region']
    # except:
    #     if len(layers) == 2:
    #         BBlayer = layers[0]
    #         regions = BBlayer['Regions']['Region']
    #         Masklayer = layers[1]
    #     else:
    #         Masklayer = layers[0]
    #     contours = Masklayer['Regions']['Region']
    #
    # start_x = 0
    # start_y = 0
    #
    # df = pd.DataFrame(columns = ['x', 'y', 't', 'l'])
    #
    # for ci in range(len(contours)):
    #     contour = contours[ci]
    #     glom, cnt = get_contour(img, contour, down_rate, 0)
    #
    #     for si in range(len(cnt)):
    #         row = len(df)
    #         df.loc[row] = [cnt[si, 0, 0], cnt[si, 0, 1], t, ci]   # be careful to the x y coordinates
    #
    #     png_name = '%s-x-contour%03d-x-%d-x-%d.png'%(name,ci,cnt[0,0,0],cnt[0,0,1])
    #     patch_check = 0
    #     if patch_check:
    #         patch_output = os.path.join(output_folder, 'patch')
    #         if not os.path.exists(patch_output):
    #             os.makedirs(patch_output)
    #
    #         plt.imsave(os.path.join(patch_output,png_name),glom)
    #
    # df.to_csv(os.path.join(output_dir, name.replace('.png', '.csv')), index = False)


def scn_to_png_R24(svs_file,xml_file,output_dir,lv,t):
    # simg = openslide.open_slide(svs_file)
    #
    # start_x = 0
    # start_y = 0
    # width_x = np.int(simg.properties['openslide.level[0].width'])
    # height_y = np.int(simg.properties['openslide.level[0].height'])
    #
    # # img = simg.read_region((start_x, start_y), lv, (height_y, width_x))
    #
    # down_rate = simg.level_downsamples[lv]
    # end_x = start_x + width_x
    # end_y = start_y + height_y
    #
    # # max_height = width_x
    # # max_widths = height_y
    #
    # patch_size = 64
    # read_patch = 1
    #
    # if read_patch:
    #     whole_width_lv2 = int(np.ceil((end_x - start_x) / down_rate))
    #     whole_height_lv2 = int(np.ceil((end_y - start_y) / down_rate))
    #     num_patch_x_lv = np.int(np.ceil(width_x / down_rate / patch_size))
    #     num_patch_y_lv = np.int(np.ceil(height_y / down_rate / patch_size))
    #     whole_width_lv = num_patch_x_lv * patch_size
    #     whole_height_lv = num_patch_y_lv * patch_size
    #     img = np.zeros((whole_height_lv2, whole_width_lv2, 3), dtype=np.uint8)
    #
    #     for xi in range(0,num_patch_y_lv):
    #         for yi in range(0,num_patch_x_lv):
    #             patch_size_x = patch_size
    #             patch_size_y = patch_size
    #
    #             if xi == num_patch_y_lv - 1:
    #                 patch_size_x = whole_height_lv2 - xi * patch_size
    #
    #             if yi == num_patch_x_lv - 1:
    #                 patch_size_y = whole_width_lv2 - yi * patch_size
    #
    #             low_res_offset_x = np.int(xi * patch_size)
    #             low_res_offset_y = np.int(yi * patch_size)
    #
    #             patch_start_x = start_x + np.int(low_res_offset_y * down_rate)
    #             patch_start_y = start_y + np.int(low_res_offset_x * down_rate)
    #
    #             img_lv = simg.read_region((patch_start_x, patch_start_y), lv, (patch_size_y, patch_size_x))
    #             img_lv = np.array(img_lv.convert('RGB'))
    #             # if (low_res_offset_x + patch_size) <= whole_height_lv and (
    #             #         low_res_offset_y + patch_size) <= whole_width_lv:
    #             img[low_res_offset_x:(low_res_offset_x + patch_size),
    #             low_res_offset_y:(low_res_offset_y + patch_size), :] = img_lv
    #
    # else:
    #     whole_height_lv2 = int(np.ceil((end_x - start_x) / down_rate))
    #     whole_width_lv2 = int(np.ceil((end_y - start_y) / down_rate))
    #     img = simg.read_region((start_x, start_y), lv, (whole_height_lv2, whole_width_lv2))
    #     img = np.array(img.convert('RGB'))
    #     cimg = img.copy()
    #
    # black_pixels = np.where(
    #     (img[:, :, 0] == 0) | (img[:, :, 1] == 0) | (img[:, :, 2] == 0))
    # img[black_pixels] = [246, 246, 246]

    for pi in range(0,3):
        # print(pi)
        # half_w = int(img.shape[1] / 3)
        # shift = pi * half_w
        #
        #
        # img_patch = img[:,int(pi * half_w):int((pi + 1) * half_w),:]
        #
        # #img_patch =
        # name = os.path.basename(svs_file).replace('.svs','_%s.png' % (pi))
        # plt.imsave(os.path.join(output_dir, name), img_patch)


        #### just load patch rather than scn
        name = os.path.basename(svs_file).replace('.svs','_%s.png' % (pi))
        img_patch = plt.imread(os.path.join(output_dir, name))
        half_w = int(img_patch.shape[1])
        shift = pi * half_w
        down_rate = 16

        # read region
        with open(xml_file) as fd:
            doc = xmltodict.parse(fd.read())
        layers = doc['Annotations']['Annotation']
        try :
            contours = layers['Regions']['Region']
        except:
            if len(layers) == 2:
                BBlayer = layers[0]
                regions = BBlayer['Regions']['Region']
                Masklayer = layers[1]
            else:
                Masklayer = layers[0]
            contours = Masklayer['Regions']['Region']

        start_x = 0
        start_y = 0

        df = pd.DataFrame(columns = ['x', 'y', 't', 'l'])

        for ci in range(len(contours)):
            contour = contours[ci]
            glom, cnt = get_contour(img_patch, contour, down_rate, shift)

            if max(cnt[:,0,0]) >= half_w or min(cnt[:,0,0]) <=0:
                continue

            for si in range(len(cnt)):
                row = len(df)
                df.loc[row] = [cnt[si, 0, 0], cnt[si, 0, 1] , t, ci]

            png_name = '%s-x-contour%03d-x-%d-x-%d.png'%(name,ci,cnt[0,0,0],cnt[0,0,1])
            patch_check = 1
            if patch_check:
                try:
                    patch_output = os.path.join(output_folder, 'patch_%s' % (pi))
                    if not os.path.exists(patch_output):
                        os.makedirs(patch_output)

                    plt.imsave(os.path.join(patch_output,png_name),glom)
                except:
                    print('error patch: %s' % (png_name))

        df.to_csv(os.path.join(output_dir, name.replace('.png', '.csv')), index = False)



if __name__ == "__main__":

    # moving = '/Data/fromHaichun/major_review/test/moving.jpg'
    # fix = '/Data/fromHaichun/major_review/test/fix.jpg'
    # overlay_dir = '/Data/fromHaichun/major_review/test'
    # sg_affine(moving, fix, overlay_dir)
    Leeds = 1
    lv = 0
    start_time = time.time()

    if Leeds:
        data_dir = '/Data3/Peize/Leeds-repo/Transplant Renal Biopsies - Set1'
        output_dir = '/Data3/Peize/Leeds_png_pre-selection'

        sections = glob.glob(os.path.join(data_dir, '*'))
        sections.sort()


        for si in range(len(sections)):
            print("working on %s , it has been %s seconds ---" % (si + 1, time.time() - start_time))
            section_num = glob.glob(os.path.join(sections[si], '*'))
            for sci in range(len(section_num)):

                svs_files = glob.glob(os.path.join(section_num[sci], '*.svs'))
                svs_files.sort()

                output_folder = section_num[sci].replace(data_dir, output_dir)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                for xi in range(len(svs_files)):
                    svs_dir = svs_files[xi]
                    xml_dir = svs_files[xi].replace('.svs', '.xml')
                    name = os.path.basename(svs_files[xi])
                    print(svs_dir)

                    scn_to_png_Leeds(svs_dir,xml_dir,output_folder,lv,sci)


    # else:
    #     data_dir = '/media/dengr/Data2/HumanKidney/2profile/circlenet/src/data'
    #     svs_folder = '/media/dengr/Data2/HumanKidney/2profile/circlenet/scn'
    #     output_dir = '/media/dengr/Data2/HumanKidney/2profile/R24_png'
    #
    #     sections = glob.glob(os.path.join(data_dir, '*'))
    #
    #     for si in range(len(sections)):
    #         name = os.path.basename(sections[si])
    #         print(name)
    #         xml_files = glob.glob(os.path.join(sections[si], '%s.xml' % (name)))[0]
    #
    #         output_folder = sections[si].replace(data_dir, output_dir)
    #         if not os.path.exists(output_folder):
    #             os.makedirs(output_folder)
    #
    #         xml_dir = xml_files
    #         svs_dir = os.path.join(svs_folder, name + '.svs')
    #
    #         scn_to_png_R24(svs_dir, xml_dir, output_folder, lv, si)