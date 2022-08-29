import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import SimpleITK as sitk
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import glob

from SuperGlue.models.matching import Matching
from SuperGlue.models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)
import re

def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def saveKpt(overlay_dir, src_kpt, dst_kpt):
    df = pd.DataFrame(columns=['src_x', 'src_y','dst_x','dst_y'])
    for ki in range(len(src_kpt)):
        df.loc[ki] = [src_kpt[ki,0,0], src_kpt[ki,0,1], dst_kpt[ki,0,0], dst_kpt[ki,0,1]]

    csv_root = os.path.join(overlay_dir, 'keypoints.csv')
    df.to_csv(csv_root, index = False)


def sg_affine(img1_file,img2_file,overlay_dir, small_image_res = 1000):

    img1 = cv2.imread(img1_file, 0)
    img2 = cv2.imread(img2_file, 0)
    

    img1_highres = img1
    img2_highres = img2

    ratio_img1 = max(img1_highres.shape[0], img1_highres.shape[1]) / float(small_image_res)
    width1 = int(img1.shape[1] / ratio_img1)
    height1 = int(img1.shape[0] / ratio_img1)
    dim1 = (width1, height1)
    # resize image
    img1 = cv2.resize(img1, dim1, interpolation = cv2.INTER_AREA)


    ratio_img2 = max(img2_highres.shape[0], img2_highres.shape[1]) / float(small_image_res)
    width2 = int(img2.shape[1] / ratio_img2)
    height2 = int(img2.shape[0] / ratio_img2)
    dim2 = (width2, height2)
    # resize image

    img2 = cv2.resize(img2, dim2, interpolation = cv2.INTER_AREA)

    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    # with open(opt.input_pairs, 'r') as f:
    #     pairs = [l.split() for l in f.readlines()]

    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    # rot0, rot1 = 0,0
    # img1, inp1, scales0 = read_image(
    #     img1_file, device, opt.resize, rot0, opt.resize_float)
    # img2, inp2, scales1 = read_image(
    #     img2_file, device, opt.resize, rot1, opt.resize_float)

    inp1 = frame2tensor(img1, device)
    inp2 = frame2tensor(img2, device)

    matching = Matching(config).eval().to(device)
    timer = AverageTimer(newline=True)

    # sift = cv2.xfeatures2d.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)

    #surf = cv2.xfeatures2d.SURF_create()
    #kp1, des1 = surf.detectAndCompute(img1, None)
    #kp2, des2 = surf.detectAndCompute(img2, None)

    # orb = cv2.ORB_create(nfeatures=2000)
    # kp1, des1 = orb.detectAndCompute(img1, None)
    # kp2, des2 = orb.detectAndCompute(img2, None)

    # akaze = cv2.AKAZE_create()
    # kp1, des1 = akaze.detectAndCompute(img1, None)
    # kp2, des2 = akaze.detectAndCompute(img2, None)

    # Perform the matching.
    pred = matching({'image0': inp1, 'image1': inp2})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    kp1, kp2 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    timer.update('matcher')

    # Write the matches to disk.
    #out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
    #               'matches': matches, 'match_confidence': conf}
    #np.savez(str(matches_path), **out_matches)


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    #src_pts = kp1
    #dst_pts = kp2

    kp1 = [cv2.KeyPoint(x[0], x[1], 1) for x in kp1]
    kp2 = [cv2.KeyPoint(x[0], x[1], 1) for x in kp2]
    #kp1 = cv2.KeyPoint_convert(kp1)
    #kp2 = cv2.KeyPoint_convert(kp2)

    des1 = pred['des0'].transpose()
    des2 = pred['des1'].transpose()

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m,n in matches:
        print(m.distance,n.distance)
        #if m.distance < 0.5*n.distance:
        if abs(m.distance - n.distance) <= 7:
            good_matches.append(m)

    good_matches_show = [good_matches]
    # Draw matches
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_show, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if not os.path.exists(overlay_dir):
        os.makedirs(overlay_dir)
    # Image.fromarray(img3).show()
    cv2.imwrite(os.path.join(overlay_dir,'match_raw.jpg'), img3)


    MIN_MATCH_COUNT = 3
    if len(good_matches)>=MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M0, mask0 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask0.ravel().tolist()

        # h,w = img1.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts,M0)
        # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        #print ("Not enough matches are found - %d/%d") % (len(good_matches),MIN_MATCH_COUNT)
        matchesMask = None
        return

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img1 = cv2.imread(img1_file)
    img2 = cv2.imread(img2_file)

    img1_highres = img1
    img2_highres = img2

    ratio_img1 = max(img1_highres.shape[0], img1_highres.shape[1]) / float(small_image_res)
    width1 = int(img1.shape[1] / ratio_img1)
    height1 = int(img1.shape[0] / ratio_img1)
    dim1 = (width1, height1)
    # resize image
    img1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)

    ratio_img2 = max(img2_highres.shape[0], img2_highres.shape[1]) / float(small_image_res)
    width2 = int(img2.shape[1] / ratio_img2)
    height2 = int(img2.shape[0] / ratio_img2)
    dim2 = (width2, height2)
    # resize image
    img2 = cv2.resize(img2, dim2, interpolation=cv2.INTER_AREA)
    img4 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,**draw_params)
    cv2.imwrite(os.path.join(overlay_dir, 'match_inliers.jpg'), img4)
    # Image.fromarray(img4).show()

    #get inlier
    inlier_matches = []
    for mm in range(len(matchesMask)):
        if matchesMask[mm] == 1:
            inlier_matches.append(good_matches[mm])
    good_matches = inlier_matches

    if len(good_matches) == 0:
        return

    # # Select good matched keypoints
    source_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    target_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #

    # cv2.drawContours(img1, [source_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
    # Image.fromarray(img1).show()

    source_matched_kpts_new = np.copy(source_matched_kpts)
    target_matched_kpts_new = np.copy(target_matched_kpts)

    x_scale = (img1_highres.shape[0]/float(img1.shape[0]))
    y_scale = (img1_highres.shape[1]/float(img1.shape[1]))
    for si in range(len(source_matched_kpts_new)):
        for sj in range(len(source_matched_kpts_new[si])):
            source_matched_kpts_new[si][sj][0] = source_matched_kpts_new[si][sj][0] * x_scale
            source_matched_kpts_new[si][sj][1] = source_matched_kpts_new[si][sj][1] * y_scale

    x_scale = (img2_highres.shape[0]/float(img2.shape[0]))
    y_scale = (img2_highres.shape[1]/float(img2.shape[1]))
    for si in range(len(target_matched_kpts_new)):
        for sj in range(len(target_matched_kpts_new[si])):
            target_matched_kpts_new[si][sj][0] = target_matched_kpts_new[si][sj][0] * x_scale
            target_matched_kpts_new[si][sj][1] = target_matched_kpts_new[si][sj][1] * y_scale

    # cv2.drawContours(img2_highres, [target_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
    # Image.fromarray(img2_highres).show()


    # Compute affine
    print(source_matched_kpts_new.shape)
    k = source_matched_kpts_new.shape[0]
    if k == 3:
        M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2], :],
                                             target_matched_kpts_new[[0, 1, 2], :])
    elif k == 4 or k == 5:
        M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3], :],
                                             target_matched_kpts_new[[0, 1, 2, 3], :])
    elif k == 6:
        M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4], :],
                                             target_matched_kpts_new[[0, 1, 2, 3, 4], :])
    elif k == 7:
        M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4, 5], :],
                                             target_matched_kpts_new[[0, 1, 2, 3, 4, 5], :])
    else:
        M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4, 5, 6], :],
                                             target_matched_kpts_new[[0, 1, 2, 3, 4, 5, 6], :])

    #M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0,1,2,3,5,6],:], target_matched_kpts_new[[0,1,2,3,5,6],:])   #####deleted 7
    #M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 5, 6], :],
                                         #target_matched_kpts_new[[0, 1, 2, 3, 5, 6], :])    #0~7 are larger than 7

    try:
        warped_image = cv2.warpAffine(img1_highres, M, (img1_highres.shape[1], img1_highres.shape[0]))
        # Image.fromarray(warped_image).show()
        cv2.imwrite(os.path.join(overlay_dir, 'swift_affine.jpg'), warped_image)
        cv2.imwrite(os.path.join(overlay_dir, 'original_affine.jpg'), img2_highres)

        M_inv, rigid_mask_inv = cv2.estimateAffine2D(target_matched_kpts_new, source_matched_kpts_new)
        M_0, _ = cv2.estimateAffine2D(source_matched_kpts_new, target_matched_kpts_new)
        np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M_0)

        saveKpt(overlay_dir, source_matched_kpts_new, target_matched_kpts_new)

        # M_vector = np.zeros([6])
        # # print('%.14f %.14f %.14f %.14f %.14f %.14f' % (M[0,0],M[1,0],M[0,1],M[1,1],M[0,2],M[1,2]))
        # #
        # #world coordinate to itk
        # FixParameters = [img2_highres.shape[1]/2.0, img2_highres.shape[0]/2.0]
        # M_vector[0] = M_inv[0, 0]
        # M_vector[1] = M_inv[0, 1]
        # M_vector[2] = M_inv[1, 0]
        # M_vector[3] = M_inv[1, 1]
        # M_vector[4] = (FixParameters[0] * M_inv[0, 0] + FixParameters[1] * M_inv[0, 1] - FixParameters[0]) + M_inv[0, 2]
        # M_vector[5] = (FixParameters[0] * M_inv[1, 0] + FixParameters[1] * M_inv[1, 1] - FixParameters[1]) + M_inv[1, 2]
        #
        # # 494.8512691713917 * 0.9635579976977282 + 289.0072193722117 * 0.267199 - 494.8512691713917 - 19.872210408065968
        # # 494.8512691713917 * -0.266946 + 289.0072193722117 * 0.9637 - 289.0072193722117 - 10.765536847388375
        #
        # m = sitk.ReadTransform('test.mat')
        #
        # m.SetParameters(M_vector)
        # m.SetFixedParameters(FixParameters)
        #
        # sitk.WriteTransform(m, os.path.join(overlay_dir, 'sg_affine_init.mat'))

    except:
        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # store all the good matches as per Lowe's ratio test.
            good_matches = []
            for m, n in matches:
                print(m.distance, n.distance)
                # if m.distance < 0.5*n.distance:
                if abs(m.distance - n.distance) >= 7:
                    good_matches.append(m)

            good_matches_show = [good_matches]
            # Draw matches
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_show, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            if not os.path.exists(overlay_dir):
                os.makedirs(overlay_dir)
            # Image.fromarray(img3).show()
            cv2.imwrite(os.path.join(overlay_dir, 'match_raw.jpg'), img3)

            MIN_MATCH_COUNT = 3
            if len(good_matches) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M0, mask0 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask0.ravel().tolist()

                # h,w = img1.shape
                # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                # dst = cv2.perspectiveTransform(pts,M0)
                # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            else:
                # print ("Not enough matches are found - %d/%d") % (len(good_matches),MIN_MATCH_COUNT)
                matchesMask = None
                return

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            #output 3band
            img1 = cv2.imread(img1_file)
            img2 = cv2.imread(img2_file)

            img1_highres = img1
            img2_highres = img2

            ratio_img1 = max(img1_highres.shape[0], img1_highres.shape[1]) / float(small_image_res)
            width1 = int(img1.shape[1] / ratio_img1)
            height1 = int(img1.shape[0] / ratio_img1)
            dim1 = (width1, height1)
            # resize image
            img1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)

            ratio_img2 = max(img2_highres.shape[0], img2_highres.shape[1]) / float(small_image_res)
            width2 = int(img2.shape[1] / ratio_img2)
            height2 = int(img2.shape[0] / ratio_img2)
            dim2 = (width2, height2)
            # resize image
            img2 = cv2.resize(img2, dim2, interpolation=cv2.INTER_AREA)


            img4 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
            cv2.imwrite(os.path.join(overlay_dir, 'match_inliers.jpg'), img4)
            # Image.fromarray(img4).show()

            # get inlier
            inlier_matches = []
            for mm in range(len(matchesMask)):
                if matchesMask[mm] == 1:
                    inlier_matches.append(good_matches[mm])
            good_matches = inlier_matches

            if len(good_matches) == 0:
                return

            # # Select good matched keypoints
            source_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            target_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            #

            # cv2.drawContours(img1, [source_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
            # Image.fromarray(img1).show()

            source_matched_kpts_new = np.copy(source_matched_kpts)
            target_matched_kpts_new = np.copy(target_matched_kpts)

            x_scale = (img1_highres.shape[0] / float(img1.shape[0]))
            y_scale = (img1_highres.shape[1] / float(img1.shape[1]))
            for si in range(len(source_matched_kpts_new)):
                for sj in range(len(source_matched_kpts_new[si])):
                    source_matched_kpts_new[si][sj][0] = source_matched_kpts_new[si][sj][0] * x_scale
                    source_matched_kpts_new[si][sj][1] = source_matched_kpts_new[si][sj][1] * y_scale

            x_scale = (img2_highres.shape[0] / float(img2.shape[0]))
            y_scale = (img2_highres.shape[1] / float(img2.shape[1]))
            for si in range(len(target_matched_kpts_new)):
                for sj in range(len(target_matched_kpts_new[si])):
                    target_matched_kpts_new[si][sj][0] = target_matched_kpts_new[si][sj][0] * x_scale
                    target_matched_kpts_new[si][sj][1] = target_matched_kpts_new[si][sj][1] * y_scale

            # cv2.drawContours(img2_highres, [target_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
            # Image.fromarray(img2_highres).show()

            # Compute affine
            print(source_matched_kpts_new.shape)
            k = source_matched_kpts_new.shape[0]
            if k == 3:
                M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2], :],
                                                     target_matched_kpts_new[[0, 1, 2], :])
            elif k == 4 or k == 5:
                M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3], :],
                                                     target_matched_kpts_new[[0, 1, 2, 3], :])
            elif k == 6:
                M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4], :],
                                                     target_matched_kpts_new[[0, 1, 2, 3, 4], :])
            elif k == 7:
                M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4, 5], :],
                                                     target_matched_kpts_new[[0, 1, 2, 3, 4, 5], :])
            else:
                M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4, 5, 6], :],
                                                     target_matched_kpts_new[[0, 1, 2, 3, 4, 5, 6], :])

            # M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0,1,2,3,5,6],:], target_matched_kpts_new[[0,1,2,3,5,6],:])   #####deleted 7
            # M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 5, 6], :],
            # target_matched_kpts_new[[0, 1, 2, 3, 5, 6], :])    #0~7 are larger than 7
            warped_image = cv2.warpAffine(img1_highres, M, (img1_highres.shape[1], img1_highres.shape[0]))
            # Image.fromarray(warped_image).show()
            cv2.imwrite(os.path.join(overlay_dir, 'swift_affine.jpg'), warped_image)
            cv2.imwrite(os.path.join(overlay_dir, 'original_affine.jpg'), img2_highres)

            M_inv, rigid_mask_inv = cv2.estimateAffine2D(target_matched_kpts_new, source_matched_kpts_new)
            M_0, _ = cv2.estimateAffine2D(source_matched_kpts_new, target_matched_kpts_new)
            np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M_0)

            saveKpt(overlay_dir, source_matched_kpts_new, target_matched_kpts_new)
            # M_vector = np.zeros([6])
            # # print('%.14f %.14f %.14f %.14f %.14f %.14f' % (M[0,0],M[1,0],M[0,1],M[1,1],M[0,2],M[1,2]))
            # #
            # # world coordinate to itk
            # FixParameters = [img2_highres.shape[1] / 2.0, img2_highres.shape[0] / 2.0]
            # M_vector[0] = M_inv[0, 0]
            # M_vector[1] = M_inv[0, 1]
            # M_vector[2] = M_inv[1, 0]
            # M_vector[3] = M_inv[1, 1]
            # M_vector[4] = (FixParameters[0] * M_inv[0, 0] + FixParameters[1] * M_inv[0, 1] - FixParameters[0]) + M_inv[0, 2]
            # M_vector[5] = (FixParameters[0] * M_inv[1, 0] + FixParameters[1] * M_inv[1, 1] - FixParameters[1]) + M_inv[1, 2]
            #
            # # 494.8512691713917 * 0.9635579976977282 + 289.0072193722117 * 0.267199 - 494.8512691713917 - 19.872210408065968
            # # 494.8512691713917 * -0.266946 + 289.0072193722117 * 0.9637 - 289.0072193722117 - 10.765536847388375
            #
            # m = sitk.ReadTransform('test.mat')
            #
            # m.SetParameters(M_vector)
            # m.SetFixedParameters(FixParameters)
            #
            # sitk.WriteTransform(m, os.path.join(overlay_dir, 'sg_affine_init.mat'))

        except:
            try:
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

                # store all the good matches as per Lowe's ratio test.
                good_matches = []
                for m, n in matches:
                    print(m.distance, n.distance)
                    # if m.distance < 0.5*n.distance:
                    if abs(m.distance - n.distance) <= 3:
                        good_matches.append(m)

                good_matches_show = [good_matches]
                # Draw matches
                img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_show, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                if not os.path.exists(overlay_dir):
                    os.makedirs(overlay_dir)
                # Image.fromarray(img3).show()
                cv2.imwrite(os.path.join(overlay_dir, 'match_raw.jpg'), img3)

                MIN_MATCH_COUNT = 3
                if len(good_matches) >= MIN_MATCH_COUNT:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M0, mask0 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask0.ravel().tolist()

                    # h,w = img1.shape
                    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    # dst = cv2.perspectiveTransform(pts,M0)
                    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

                else:
                    # print ("Not enough matches are found - %d/%d") % (len(good_matches),MIN_MATCH_COUNT)
                    matchesMask = None
                    return

                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)

                # output 3band
                img1 = cv2.imread(img1_file)
                img2 = cv2.imread(img2_file)

                img1_highres = img1
                img2_highres = img2

                ratio_img1 = max(img1_highres.shape[0], img1_highres.shape[1]) / float(small_image_res)
                width1 = int(img1.shape[1] / ratio_img1)
                height1 = int(img1.shape[0] / ratio_img1)
                dim1 = (width1, height1)
                # resize image
                img1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)

                ratio_img2 = max(img2_highres.shape[0], img2_highres.shape[1]) / float(small_image_res)
                width2 = int(img2.shape[1] / ratio_img2)
                height2 = int(img2.shape[0] / ratio_img2)
                dim2 = (width2, height2)
                # resize image
                img2 = cv2.resize(img2, dim2, interpolation=cv2.INTER_AREA)

                img4 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
                cv2.imwrite(os.path.join(overlay_dir, 'match_inliers.jpg'), img4)
                # Image.fromarray(img4).show()

                # get inlier
                inlier_matches = []
                for mm in range(len(matchesMask)):
                    if matchesMask[mm] == 1:
                        inlier_matches.append(good_matches[mm])
                good_matches = inlier_matches

                if len(good_matches) == 0:
                    return

                # # Select good matched keypoints
                source_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                target_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                #

                # cv2.drawContours(img1, [source_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
                # Image.fromarray(img1).show()

                source_matched_kpts_new = np.copy(source_matched_kpts)
                target_matched_kpts_new = np.copy(target_matched_kpts)

                x_scale = (img1_highres.shape[0] / float(img1.shape[0]))
                y_scale = (img1_highres.shape[1] / float(img1.shape[1]))
                for si in range(len(source_matched_kpts_new)):
                    for sj in range(len(source_matched_kpts_new[si])):
                        source_matched_kpts_new[si][sj][0] = source_matched_kpts_new[si][sj][0] * x_scale
                        source_matched_kpts_new[si][sj][1] = source_matched_kpts_new[si][sj][1] * y_scale

                x_scale = (img2_highres.shape[0] / float(img2.shape[0]))
                y_scale = (img2_highres.shape[1] / float(img2.shape[1]))
                for si in range(len(target_matched_kpts_new)):
                    for sj in range(len(target_matched_kpts_new[si])):
                        target_matched_kpts_new[si][sj][0] = target_matched_kpts_new[si][sj][0] * x_scale
                        target_matched_kpts_new[si][sj][1] = target_matched_kpts_new[si][sj][1] * y_scale

                # cv2.drawContours(img2_highres, [target_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
                # Image.fromarray(img2_highres).show()

                # Compute affine
                print(source_matched_kpts_new.shape)
                k = source_matched_kpts_new.shape[0]
                if k == 3:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2], :],
                                                         target_matched_kpts_new[[0, 1, 2], :])
                elif k == 4 or k == 5:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3], :],
                                                         target_matched_kpts_new[[0, 1, 2, 3], :])
                elif k == 6:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4], :],
                                                         target_matched_kpts_new[[0, 1, 2, 3, 4], :])
                elif k == 7:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4, 5], :],
                                                         target_matched_kpts_new[[0, 1, 2, 3, 4, 5], :])
                else:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4, 5, 6], :],
                                                         target_matched_kpts_new[[0, 1, 2, 3, 4, 5, 6], :])

                # M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0,1,2,3,5,6],:], target_matched_kpts_new[[0,1,2,3,5,6],:])   #####deleted 7
                # M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 5, 6], :],
                # target_matched_kpts_new[[0, 1, 2, 3, 5, 6], :])    #0~7 are larger than 7
                warped_image = cv2.warpAffine(img1_highres, M, (img1_highres.shape[1], img1_highres.shape[0]))
                # Image.fromarray(warped_image).show()
                cv2.imwrite(os.path.join(overlay_dir, 'swift_affine.jpg'), warped_image)
                cv2.imwrite(os.path.join(overlay_dir, 'original_affine.jpg'), img2_highres)

                M_inv, rigid_mask_inv = cv2.estimateAffine2D(target_matched_kpts_new, source_matched_kpts_new)
                M_0, _ = cv2.estimateAffine2D(source_matched_kpts_new, target_matched_kpts_new)
                np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M_0)

                saveKpt(overlay_dir, source_matched_kpts_new, target_matched_kpts_new)
                # M_vector = np.zeros([6])
                # # print('%.14f %.14f %.14f %.14f %.14f %.14f' % (M[0,0],M[1,0],M[0,1],M[1,1],M[0,2],M[1,2]))
                # #
                # # world coordinate to itk
                # FixParameters = [img2_highres.shape[1] / 2.0, img2_highres.shape[0] / 2.0]
                # M_vector[0] = M_inv[0, 0]
                # M_vector[1] = M_inv[0, 1]
                # M_vector[2] = M_inv[1, 0]
                # M_vector[3] = M_inv[1, 1]
                # M_vector[4] = (FixParameters[0] * M_inv[0, 0] + FixParameters[1] * M_inv[0, 1] - FixParameters[0]) + M_inv[0, 2]
                # M_vector[5] = (FixParameters[0] * M_inv[1, 0] + FixParameters[1] * M_inv[1, 1] - FixParameters[1]) + M_inv[1, 2]
                #
                # # 494.8512691713917 * 0.9635579976977282 + 289.0072193722117 * 0.267199 - 494.8512691713917 - 19.872210408065968
                # # 494.8512691713917 * -0.266946 + 289.0072193722117 * 0.9637 - 289.0072193722117 - 10.765536847388375
                #
                # m = sitk.ReadTransform('test.mat')
                #
                # m.SetParameters(M_vector)
                # m.SetFixedParameters(FixParameters)
                #
                # sitk.WriteTransform(m, os.path.join(overlay_dir, 'sg_affine_init.mat'))

            except:
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

                # store all the good matches as per Lowe's ratio test.
                good_matches = []
                for m, n in matches:
                    print(m.distance, n.distance)
                    # if m.distance < 0.5*n.distance:
                    if abs(m.distance - n.distance) >= 3:
                        good_matches.append(m)

                good_matches_show = [good_matches]
                # Draw matches
                img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches_show, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                if not os.path.exists(overlay_dir):
                    os.makedirs(overlay_dir)
                # Image.fromarray(img3).show()
                cv2.imwrite(os.path.join(overlay_dir, 'match_raw.jpg'), img3)

                MIN_MATCH_COUNT = 3
                if len(good_matches) >= MIN_MATCH_COUNT:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M0, mask0 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matchesMask = mask0.ravel().tolist()

                    # h,w = img1.shape
                    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    # dst = cv2.perspectiveTransform(pts,M0)
                    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

                else:
                    # print ("Not enough matches are found - %d/%d") % (len(good_matches),MIN_MATCH_COUNT)
                    matchesMask = None
                    return

                draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)

                # output 3band
                img1 = cv2.imread(img1_file)
                img2 = cv2.imread(img2_file)

                img1_highres = img1
                img2_highres = img2

                ratio_img1 = max(img1_highres.shape[0], img1_highres.shape[1]) / float(small_image_res)
                width1 = int(img1.shape[1] / ratio_img1)
                height1 = int(img1.shape[0] / ratio_img1)
                dim1 = (width1, height1)
                # resize image
                img1 = cv2.resize(img1, dim1, interpolation=cv2.INTER_AREA)

                ratio_img2 = max(img2_highres.shape[0], img2_highres.shape[1]) / float(small_image_res)
                width2 = int(img2.shape[1] / ratio_img2)
                height2 = int(img2.shape[0] / ratio_img2)
                dim2 = (width2, height2)
                # resize image
                img2 = cv2.resize(img2, dim2, interpolation=cv2.INTER_AREA)


                img4 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
                cv2.imwrite(os.path.join(overlay_dir, 'match_inliers.jpg'), img4)
                # Image.fromarray(img4).show()

                # get inlier
                inlier_matches = []
                for mm in range(len(matchesMask)):
                    if matchesMask[mm] == 1:
                        inlier_matches.append(good_matches[mm])
                good_matches = inlier_matches

                if len(good_matches) == 0:
                    return

                # # Select good matched keypoints
                source_matched_kpts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                target_matched_kpts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                #

                # cv2.drawContours(img1, [source_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
                # Image.fromarray(img1).show()

                source_matched_kpts_new = np.copy(source_matched_kpts)
                target_matched_kpts_new = np.copy(target_matched_kpts)

                x_scale = (img1_highres.shape[0] / float(img1.shape[0]))
                y_scale = (img1_highres.shape[1] / float(img1.shape[1]))
                for si in range(len(source_matched_kpts_new)):
                    for sj in range(len(source_matched_kpts_new[si])):
                        source_matched_kpts_new[si][sj][0] = source_matched_kpts_new[si][sj][0] * x_scale
                        source_matched_kpts_new[si][sj][1] = source_matched_kpts_new[si][sj][1] * y_scale

                x_scale = (img2_highres.shape[0] / float(img2.shape[0]))
                y_scale = (img2_highres.shape[1] / float(img2.shape[1]))
                for si in range(len(target_matched_kpts_new)):
                    for sj in range(len(target_matched_kpts_new[si])):
                        target_matched_kpts_new[si][sj][0] = target_matched_kpts_new[si][sj][0] * x_scale
                        target_matched_kpts_new[si][sj][1] = target_matched_kpts_new[si][sj][1] * y_scale

                # cv2.drawContours(img2_highres, [target_matched_kpts.astype(int)], -1, (0, 255, 0), 3)
                # Image.fromarray(img2_highres).show()

                # Compute affine
                print(source_matched_kpts_new.shape)
                k = source_matched_kpts_new.shape[0]
                if k == 3:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2], :],
                                                         target_matched_kpts_new[[0, 1, 2], :])
                elif k == 4 or k == 5:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3], :],
                                                         target_matched_kpts_new[[0, 1, 2, 3], :])
                elif k == 6:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4], :],
                                                         target_matched_kpts_new[[0, 1, 2, 3, 4], :])
                elif k == 7:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4, 5], :],
                                                         target_matched_kpts_new[[0, 1, 2, 3, 4, 5], :])
                else:
                    M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 4, 5, 6], :],
                                                         target_matched_kpts_new[[0, 1, 2, 3, 4, 5, 6], :])

                # M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0,1,2,3,5,6],:], target_matched_kpts_new[[0,1,2,3,5,6],:])   #####deleted 7
                # M, rigid_mask = cv2.estimateAffine2D(source_matched_kpts_new[[0, 1, 2, 3, 5, 6], :],
                # target_matched_kpts_new[[0, 1, 2, 3, 5, 6], :])    #0~7 are larger than 7
                warped_image = cv2.warpAffine(img1_highres, M, (img1_highres.shape[1], img1_highres.shape[0]))
                # Image.fromarray(warped_image).show()
                cv2.imwrite(os.path.join(overlay_dir, 'swift_affine.jpg'), warped_image)
                cv2.imwrite(os.path.join(overlay_dir, 'original_affine.jpg'), img2_highres)

                M_inv, rigid_mask_inv = cv2.estimateAffine2D(target_matched_kpts_new, source_matched_kpts_new)
                M_0, _ = cv2.estimateAffine2D(source_matched_kpts_new, target_matched_kpts_new)
                np.save(os.path.join(overlay_dir, 'sg_affine_init.npy'), M_0)

                # M_vector = np.zeros([6])
                #
                # # world coordinate to itk
                # FixParameters = [img2_highres.shape[1] / 2.0, img2_highres.shape[0] / 2.0]
                # M_vector[0] = M_inv[0, 0]
                # M_vector[1] = M_inv[0, 1]
                # M_vector[2] = M_inv[1, 0]
                # M_vector[3] = M_inv[1, 1]
                # M_vector[4] = (FixParameters[0] * M_inv[0, 0] + FixParameters[1] * M_inv[0, 1] - FixParameters[0]) + \
                #               M_inv[0, 2]
                # M_vector[5] = (FixParameters[0] * M_inv[1, 0] + FixParameters[1] * M_inv[1, 1] - FixParameters[1]) + \
                #               M_inv[1, 2]
                #
                # m = sitk.ReadTransform('test.mat')
                #
                # m.SetParameters(M_vector)
                # m.SetFixedParameters(FixParameters)
                #
                # sitk.WriteTransform(m, os.path.join(overlay_dir, 'sg_affine_init.mat'))

                saveKpt(overlay_dir, source_matched_kpts_new, target_matched_kpts_new)

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

    folder = 'input_png'
    output = 'data'
    cases = glob.glob(os.path.join(folder, '*'))
    cases.sort()
    print("Step 2 is running.")

    for ki in range(len(cases)):
        now_case = os.path.basename(cases[ki])
        # slices = glob.glob(os.path.join(cases[ki], '10X', '*'))
        slices = glob.glob(os.path.join(cases[ki], '*'))
        slices.sort(key=natural_keys)

        for si in range(1, len(slices)):
            moving = slices[si]
            fix = slices[si - 1]
            print('%s to %s is %s to %s' % (si + 1, si, moving, fix))

            overlay_dir = os.path.join(output, '%s/sg_affine/%d-to-%d/' % (now_case, si + 1, si))
            sg_affine(moving, fix, overlay_dir)
