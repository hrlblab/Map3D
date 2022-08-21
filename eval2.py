import numpy as np
import csv
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
import matplotlib.cm as cm
import glob
from math import dist

if __name__ == "__main__":

    folder = '/Data3/Peize/Leeds_png_5.5/section12/no3/'
    annotation_csv=os.path.join(folder, 'annotation_10X.csv')
    output_txt = os.path.join(folder, 'distance_10X.txt')

    num_images = 5
    num_tissues = 2
    middle_idx = 2

    affine_folder = os.path.join(folder, 'sg_affine')
    ants_output_mat_folder = os.path.join(folder, 'ANTs_final_affine')
    output_mat_folder = os.path.join(folder, 'final_affine')
    # sections = glob.glob(os.path.join(folder, '*'))
    # sections.sort()
    # middle_idx_list = [3, 3, 3]

    txt = open(output_txt, 'w')
    with open(annotation_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = list(reader)

    get_center = lambda x:np.mean(x,axis=[0,0,0])
    lala = np.array([[[1,2]],[[2,3]],[[3,4]],[[4,5]]])
    print(lala.reshape(4,-1).mean(axis=0))
    #print(get_center(np.array([[[1,2]],[[2,3]],[[3,4]],[[4,5]]])))
    points=[]
    count=0
    for point in data:
        pt1_x=float(point[5].split(",")[1].split(":")[1])
        pt1_y=float(point[5].split(",")[2].split(":")[1])
        width=float(point[5].split(",")[3].split(":")[1])
        height=float(point[5].split(",")[4].split("}")[0].split(":")[1])

        if count % num_tissues == 0:
            if count//num_tissues==middle_idx:
                middle_points=np.array([[pt1_x+width/2,pt1_y+height/2]])
            cur_points=np.array([[[[pt1_x,pt1_y]],[[pt1_x+width,pt1_y]],[[pt1_x,pt1_y+height]],[[pt1_x+width,pt1_y+height]]]])
        else:
            if count//num_tissues==middle_idx:
                middle_points=np.append(middle_points,[[pt1_x+width/2,pt1_y+height/2]],axis=0)
            cur_points=np.append(cur_points,[[[[pt1_x,pt1_y]],[[pt1_x+width,pt1_y]],[[pt1_x,pt1_y+height]],[[pt1_x+width,pt1_y+height]]]],axis=0)
            if count%num_tissues==num_tissues-1:
                points.append(cur_points)
        # print("x: ",pt1_x,", y: ",pt1_y,", width: ",width,", height: ",height)
        if count == 4:
            print("middle: ", cur_points)
        count+=1
    #print(points[1])
    #print(middle_points)

    sg_distances = 0
    ants_distances = 0
    final_distances = 0

    for image in range(num_images):
        affine_matrix = np.zeros((3, 3))
        affine_matrix[2, 2] = 1.

        ants_M_new = np.zeros((3, 3))
        ants_M_new[2, 2] = 1.

        M_new = np.zeros((3, 3))
        M_new[2, 2] = 1.

        if image != middle_idx:
            if image < middle_idx:
                for sgi in range(image, middle_idx):
                    ants_affine_root = os.path.join(ants_output_mat_folder, '%d.mat' % (sgi+1))
                    ants_m = sitk.ReadTransform(ants_affine_root)
                    ants_FixP = ants_m.GetFixedParameters()
                    ants_M_vec = ants_m.GetParameters()
                    ants_M_inv = np.zeros((2, 3))

                    ants_M_inv[0, 0] = ants_M_vec[0]
                    ants_M_inv[0, 1] = ants_M_vec[1]
                    ants_M_inv[1, 0] = ants_M_vec[2]
                    ants_M_inv[1, 1] = ants_M_vec[3]
                    ants_M_inv[0, 2] = ants_M_vec[4] - (ants_FixP[0] * ants_M_inv[0, 0] + ants_FixP[1] * ants_M_inv[0, 1] - ants_FixP[0])
                    ants_M_inv[1, 2] = ants_M_vec[5] - (ants_FixP[0] * ants_M_inv[1, 0] + ants_FixP[1] * ants_M_inv[1, 1] - ants_FixP[1])

                    ants_M1 = np.zeros((3, 3))
                    ants_M1[2, 2] = 1.
                    ants_M1[:2, :] = cv2.invertAffineTransform(ants_M_inv)
                    #print(ants_M1)

                    affine_root = os.path.join(output_mat_folder, '%d.mat' % (sgi+1))
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

                    if sgi == image:
                        matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 2, sgi + 1), 'sg_affine_init.npy')
                        affine_matrix[:2, :] = cv2.invertAffineTransform(np.load(matrix_root))
                        ants_M_new = ants_M1.copy()
                        M_new = M1.copy()
                    else:
                        matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 2, sgi + 1), 'sg_affine_init.npy')
                        new_affine = np.zeros((3, 3))
                        new_affine[2, 2] = 1.
                        new_affine[:2, :] = cv2.invertAffineTransform(np.load(matrix_root))
                        affine_matrix = affine_matrix.dot(new_affine)

                        ants_M_new = ants_M_new.dot(ants_M1)
                        M_new = M_new.dot(M1)

                ants_M_done = np.zeros((3, 3))
                ants_M_done[2, 2] = 1.
                ants_M_done[:2, :] = cv2.invertAffineTransform(ants_M_new[:2, :])

                M_done = np.zeros((3, 3))
                M_done[2, 2] = 1.
                M_done[:2, :] = cv2.invertAffineTransform(M_new[:2, :])
                #print(ants_M_done)
            else:
                for sgi in range(middle_idx, image):
                    ants_affine_root = os.path.join(ants_output_mat_folder, '%d.mat' % (sgi + 1))
                    ants_m = sitk.ReadTransform(ants_affine_root)
                    ants_FixP = ants_m.GetFixedParameters()
                    ants_M_vec = ants_m.GetParameters()
                    ants_M_inv = np.zeros((2, 3))

                    ants_M_inv[0, 0] = ants_M_vec[0]
                    ants_M_inv[0, 1] = ants_M_vec[1]
                    ants_M_inv[1, 0] = ants_M_vec[2]
                    ants_M_inv[1, 1] = ants_M_vec[3]
                    ants_M_inv[0, 2] = ants_M_vec[4] - (ants_FixP[0] * ants_M_inv[0, 0] + ants_FixP[1] * ants_M_inv[0, 1] - ants_FixP[0])
                    ants_M_inv[1, 2] = ants_M_vec[5] - (ants_FixP[0] * ants_M_inv[1, 0] + ants_FixP[1] * ants_M_inv[1, 1] - ants_FixP[1])

                    ants_M1 = np.zeros((3, 3))
                    ants_M1[2, 2] = 1.
                    ants_M1[:2, :] = cv2.invertAffineTransform(ants_M_inv)

                    affine_root = os.path.join(output_mat_folder, '%d.mat' % (sgi+1))
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
                    if sgi == middle_idx:
                        matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 2, sgi+1), 'sg_affine_init.npy')
                        affine_matrix[:2, :] = np.load(matrix_root)

                        ants_M_new = ants_M1.copy()
                        M_new = M1.copy()
                    else:
                        new_affine = np.zeros((3, 3))
                        new_affine[2, 2] = 1.
                        matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 2, sgi+1), 'sg_affine_init.npy')
                        new_affine[:2, :] = np.load(matrix_root)
                        affine_matrix = affine_matrix.dot(new_affine)

                        ants_M_new = ants_M_new.dot(ants_M1)
                        M_new = M_new.dot(M1)

                ants_M_done=ants_M_new
                M_done = M_new
            points_on_image=points[image]

            sg_distance=0
            ants_distance=0
            final_distance=0
            for tissue in range(num_tissues):
                point = points_on_image[tissue]
                point_center=point.reshape(4,-1).mean(axis=0)

                sg_point = cv2.perspectiveTransform(point,affine_matrix)
                sg_center=sg_point.reshape(4,-1).mean(axis=0)

                ants_point = cv2.perspectiveTransform(point,ants_M_done)
                ants_center = ants_point.reshape(4, -1).mean(axis=0)

                final_point=cv2.perspectiveTransform(point,M_done)
                final_center = final_point.reshape(4, -1).mean(axis=0)

                #print(sg_point)
                #print(ants_point)
                print("image: ",image,"tissue: ",tissue)
                sg_distance+=dist(sg_center,middle_points[tissue])
                ants_distance+=dist(ants_center, middle_points[tissue])
                final_distance += dist(final_center, middle_points[tissue])
                #print(dist(sg_center,middle_points[tissue]))
                #print(dist(ants_center, middle_points[tissue]))
                #print(dist(final_center, middle_points[tissue]))

                if image==3 and tissue==0:
                    print(dist(point_center,middle_points[tissue]))
                    print(dist(sg_center,middle_points[tissue]))
                    print(dist(ants_center, middle_points[tissue]))
                    print(dist(final_center, middle_points[tissue]))

                    print("10X: ",point)
                    print("sg: ",sg_point)
                    print("ants: ",ants_point)
                    print("final: ",final_point)

            sg_distances+=sg_distance/num_tissues
            ants_distances+=ants_distance/num_tissues
            final_distances+=final_distance/num_tissues
    print("sg distance: ",sg_distances/(num_images-1))
    print("ants distance: ", ants_distances / (num_images - 1))
    print("final distance: ", final_distances / (num_images - 1))
    print("sg distance: ", sg_distances / (num_images - 1), file=txt)
    print("ants distance: ", ants_distances / (num_images - 1), file=txt)
    print("final distance: ", final_distances / (num_images - 1), file=txt)
    txt.close()