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

    folder = '/Data3/Peize/Leeds_png_5.5/section13/no3/'
    annotation_csv=os.path.join(folder, 'try.csv')
    output_txt = os.path.join(folder, 'try.txt')

    num_images = 5
    num_tissues = 2
    middle_idx =2
    #sections = glob.glob(os.path.join(folder, '*'))
    #sections.sort()
    #middle_idx_list = [3, 3, 3]

    txt = open(output_txt, 'w')
    with open(annotation_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = list(reader)

    x=[]
    y=[]
    r=[]
    count=0
    for point in data:
        if count%num_tissues==0:
            cur=[]
        x.append(float(point[5].split(",")[1].split(":")[1]))
        y.append(float(point[5].split(",")[2].split(":")[1]))
        r.append(float(point[5].split(",")[3].split("}")[0].split(":")[1]))
    count = 0
    distances =[]
    radiuses=[]
    ratioes=[]
    for baseline in range(3):
        distance=0
        radius=0
        for image in range(num_images):
            for point in range(num_tissues):
                cur = count%(num_images*num_tissues)
                if cur//num_tissues!=middle_idx:
                    middle= count+(middle_idx-count%(num_images*num_tissues)//num_tissues)*num_tissues
                    middle_pt=(x[middle],y[middle])
                    now_pt=(x[count],y[count])
                    #print(middle_pt)
                    #print(now_pt)
                    distance+=dist(now_pt,middle_pt)
                    #print()
                radius+=r[count]
                count+=1
        distance=distance/num_tissues/(num_images-1)
        radius=radius/num_tissues/num_images
        ratio=distance/radius

        distances.append(distance)
        radiuses.append(radius)
        ratioes.append(ratio)
        a=np.array([[[4614,6277]],[[4879,6277]],[[4614,6446]],[[4879,6446]]],dtype='float32')
        print(a.shape)
        affine_matrix = np.zeros((3, 3))
        affine_matrix[2, 2] = 1.
        matrix_root = os.path.join(folder, 'sg_affine/4-to-3/sg_affine_init.npy')
        affine_matrix[:2,:] = np.load(matrix_root)
        print(np.load(matrix_root))
        print(affine_matrix)
        b=cv2.perspectiveTransform(a,affine_matrix)
        print(b)

        print("distance: ",distance,", radius: ",radius,", ratio: ",ratio, file=txt)
        print("distance: ",distance,", radius: ",radius,", ratio: ",ratio)
    txt.close()



