import cv2
import numpy as np
from pycocotools.coco import COCO
import os
import pandas as pd
import pycocotools.mask as mask_util
from densepose.structures import DensePoseDataRelative
from visualize_rect_segm import (
    _get_dict_of_midpoints, _get_dict_of_rotated_angles, _segm_xy,
    _remove_outlier, _get_min_bounding_rect, _euclidian, _rotate,
    COARSE_TO_COLOR, COARSE_ID, JOINT_ID
)


# dataset setting
coco_folder = os.path.join('datasets', 'coco')

# dense_pose annotation
dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_train2014.json'))

# caption annotation
caption_coco = COCO(os.path.join(coco_folder, 'annotations', 'captions_train2014.json'))


def filter_by_caption(yes_word_list, no_word_list):

    # dense_pose image ids
    dp_img_ids = dp_coco.getImgIds()
    # caption image ids
    caption_img_ids = caption_coco.getImgIds()

    print('Number of dp_images:', len(dp_img_ids))
    print('Number of caption_images:', len(caption_img_ids))

    # common image ids between dense_pose and caption images
    common_img_ids = list(set(dp_img_ids) & set(caption_img_ids))

    print('Number of common images:', len(common_img_ids))

    # convert word lists to lower-case
    yes_word_list = [word.lower() for word in yes_word_list]
    no_word_list = [word.lower() for word in no_word_list]

    yes_word_size = len(yes_word_list)

    filtered_img_ids = []

    for img_id in common_img_ids:

        annotation_ids = caption_coco.getAnnIds(imgIds=img_id)
        annotations = caption_coco.loadAnns(annotation_ids)

        # one image has more than ONE annotations!
        match_count = 0
        for annotation in annotations:
            # caption = a list of lower-case words
            caption = annotation['caption'].lower().split()

            # check for words, which must be ALL included
            filtered_yes_word_list = list(set(yes_word_list) & set(caption))

            # check for words, which must NOT be included
            filtered_no_word_list = list(set(no_word_list) & set(caption))

            # strict match
            if len(filtered_yes_word_list) == yes_word_size and len(filtered_no_word_list) == 0:
                match_count += 1

        # condition: if ALL annotations are matched!
        # match_count > 0
        # match_count == len(annotations)
        if match_count == len(annotations):
            filtered_img_ids.append(img_id)

    return filtered_img_ids


def get_img_ids_by_caption(dp_img_category, dp_img_range):

    if dp_img_category == 'man':

        # images of only men
        man_list_img_ids = filter_by_caption(yes_word_list=['man'], no_word_list=['woman'])

        print('Number of images with only men:', len(man_list_img_ids))

        dp_img_ids = man_list_img_ids[dp_img_range]

    elif dp_img_category == 'woman':

        # images of only women
        woman_list_img_ids = filter_by_caption(yes_word_list=['woman'], no_word_list=['man'])

        print('Number of images with only women:', len(woman_list_img_ids))

        dp_img_ids = woman_list_img_ids[dp_img_range]

    return dp_img_ids


def _get_dp_mask(polys):

    mask_gen = np.zeros([256,256])

    for i in range(1,15):

        if(polys[i-1]):
            current_mask = mask_util.decode(polys[i-1])
            mask_gen[current_mask>0] = i

    return mask_gen


def _get_dict_of_segm_and_keypoints(segm, keypoints, box_xywh):

    segm_xy_list = []

    bg_xy = [] # 0
    segm_xy_list.append(bg_xy)

    torso_xy = _segm_xy(segm=segm, segm_id=1, box_xywh=box_xywh)
    segm_xy_list.append(torso_xy)

    r_hand_xy = [] # 2
    l_hand_xy = [] # 3
    l_foot_xy = [] # 4
    r_foot_xy = [] # 5
    segm_xy_list.append(r_hand_xy)
    segm_xy_list.append(l_hand_xy)
    segm_xy_list.append(l_foot_xy)
    segm_xy_list.append(r_foot_xy)

    r_thigh_xy = _segm_xy(segm=segm, segm_id=6, box_xywh=box_xywh)
    l_thigh_xy = _segm_xy(segm=segm, segm_id=7, box_xywh=box_xywh)
    r_calf_xy = _segm_xy(segm=segm, segm_id=8, box_xywh=box_xywh)
    l_calf_xy = _segm_xy(segm=segm, segm_id=9, box_xywh=box_xywh)
    segm_xy_list.append(r_thigh_xy)
    segm_xy_list.append(l_thigh_xy)
    segm_xy_list.append(r_calf_xy)
    segm_xy_list.append(l_calf_xy)

    l_upper_arm_xy = _segm_xy(segm=segm, segm_id=10, box_xywh=box_xywh)
    r_upper_arm_xy = _segm_xy(segm=segm, segm_id=11, box_xywh=box_xywh)
    l_lower_arm_xy = _segm_xy(segm=segm, segm_id=12, box_xywh=box_xywh)
    r_lower_arm_xy = _segm_xy(segm=segm, segm_id=13, box_xywh=box_xywh)
    segm_xy_list.append(l_upper_arm_xy)
    segm_xy_list.append(r_upper_arm_xy)
    segm_xy_list.append(l_lower_arm_xy)
    segm_xy_list.append(r_lower_arm_xy)

    head_xy = _segm_xy(segm=segm, segm_id=14, box_xywh=box_xywh)
    segm_xy_list.append(head_xy)

    # segments dictionary
    segm_xy_dict = dict(zip(COARSE_ID, segm_xy_list))

    # keypoints dictionary
    keypoints_dict = dict(zip(JOINT_ID, zip(keypoints[0::3].copy(), keypoints[1::3].copy(), keypoints[2::3].copy())))
    keypoints_dict = {key:np.array(value) for key, value in keypoints_dict.items()}

    # infer the keypoints of neck and midhip, which are missing!
    keypoints_dict['Neck'] = ((keypoints_dict['LShoulder'] + keypoints_dict['RShoulder']) / 2).astype(int)
    keypoints_dict['MidHip'] = ((keypoints_dict['LHip'] + keypoints_dict['RHip']) / 2).astype(int)

    return segm_xy_dict, keypoints_dict


def _is_valid(keypoints_dict):

    # check the scores for each main keypoint, which MUST exist!
    # main_keypoints = BODY BOX
    main_keypoints = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'MidHip']

    # filter the main keypoints by score > 0
    filtered_keypoints = [key for key, value in keypoints_dict.items() if key in main_keypoints and value[2] > 0]
    print('Number of valid keypoints (must be equal to 7):', len(filtered_keypoints))

    if len(filtered_keypoints) != 7:
        return False
    else:
        return True


def _draw_one_segm_bbox(image, segm_id, segm_xy):

    global bbox_segm_dict

    # remove outliers
    print('Before removing outliers:', len(segm_xy))
    segm_xy = np.array(_remove_outlier(segm_xy=segm_xy)).astype(int)
    print('After removing outliers:', len(segm_xy))

    # get the minimum bounding rectangle of segm_xy
    try:
        rect_xy = _get_min_bounding_rect(segm_xy)
    except:
        return
    cv2.fillPoly(image, [rect_xy], COARSE_TO_COLOR[segm_id])

    dist1 = _euclidian(rect_xy[0], rect_xy[1])
    dist2 = _euclidian(rect_xy[1], rect_xy[2])
    if 'Arm' in segm_id:
        w = max(dist1, dist2)
        h = min(dist1, dist2)
    elif segm_id != 'Head':
        w = min(dist1, dist2)
        h = max(dist1, dist2)
    else:
        w = 0
        h = 0

    bbox_segm_dict[segm_id]  = {}
    bbox_segm_dict[segm_id]['half_w'] = int(w / 2)
    bbox_segm_dict[segm_id]['half_h'] = int(h / 2)


def _draw_segm_bbox(image, segm_xy_dict, keypoints_dict):

    # head
    _draw_one_segm_bbox(image, segm_id='Head', segm_xy=segm_xy_dict['Head'])

    # torso
    _draw_one_segm_bbox(image, segm_id='Torso', segm_xy=segm_xy_dict['Torso'])

    # upper limbs
    _draw_one_segm_bbox(image, segm_id='RUpperArm', segm_xy=segm_xy_dict['RUpperArm'])
    _draw_one_segm_bbox(image, segm_id='RLowerArm', segm_xy=segm_xy_dict['RLowerArm'])
    _draw_one_segm_bbox(image, segm_id='LUpperArm', segm_xy=segm_xy_dict['LUpperArm'])
    _draw_one_segm_bbox(image, segm_id='LLowerArm', segm_xy=segm_xy_dict['LLowerArm'])

    # lower limbs
    _draw_one_segm_bbox(image, segm_id='RThigh', segm_xy=segm_xy_dict['RThigh'])
    _draw_one_segm_bbox(image, segm_id='RCalf', segm_xy=segm_xy_dict['RCalf'])
    _draw_one_segm_bbox(image, segm_id='LThigh', segm_xy=segm_xy_dict['LThigh'])
    _draw_one_segm_bbox(image, segm_id='LCalf', segm_xy=segm_xy_dict['LCalf'])

    if show:
        cv2.imshow('segm bbox', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _draw_segm_and_keypoints(image, segm_xy_dict, keypoints_dict):

    # head
    for x, y in segm_xy_dict['Head']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['Head'], -1)
    cv2.circle(image, (keypoints_dict['Nose'][0], keypoints_dict['Nose'][1]), 5, (255, 0, 255), -1)

    # torso
    for x, y in segm_xy_dict['Torso']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['Torso'], -1)
    cv2.circle(image, (keypoints_dict['Neck'][0], keypoints_dict['Neck'][1]), 5, (255, 0, 255), -1)

    # upper limbs
    for x, y in segm_xy_dict['RUpperArm']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['RUpperArm'], -1)
    for x, y in segm_xy_dict['RLowerArm']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['RLowerArm'], -1)
    for x, y in segm_xy_dict['LUpperArm']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['LUpperArm'], -1)
    for x, y in segm_xy_dict['LLowerArm']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['LLowerArm'], -1)
    cv2.circle(image, (keypoints_dict['RShoulder'][0], keypoints_dict['RShoulder'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RElbow'][0], keypoints_dict['RElbow'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RWrist'][0], keypoints_dict['RWrist'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LShoulder'][0], keypoints_dict['LShoulder'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LElbow'][0], keypoints_dict['LElbow'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LWrist'][0], keypoints_dict['LWrist'][1]), 5, (255, 0, 255), -1)

    # lower limbs
    for x, y in segm_xy_dict['RThigh']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['RThigh'], -1)
    for x, y in segm_xy_dict['RCalf']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['RCalf'], -1)
    for x, y in segm_xy_dict['LThigh']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['LThigh'], -1)
    for x, y in segm_xy_dict['LCalf']:
        cv2.circle(image, (x, y), 1, COARSE_TO_COLOR['LCalf'], -1)
    cv2.circle(image, (keypoints_dict['MidHip'][0], keypoints_dict['MidHip'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RHip'][0], keypoints_dict['RHip'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RKnee'][0], keypoints_dict['RKnee'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['RAnkle'][0], keypoints_dict['RAnkle'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LHip'][0], keypoints_dict['LHip'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LKnee'][0], keypoints_dict['LKnee'][1]), 5, (255, 0, 255), -1)
    cv2.circle(image, (keypoints_dict['LAnkle'][0], keypoints_dict['LAnkle'][1]), 5, (255, 0, 255), -1)

    if show:
        cv2.imshow('original', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _draw_one_rotated_and_scaled_segm(segm_id, segm_xy, midpoint, scaler):

    global temp_segm_dict

    # for the segments not inferred by DensePose
    if len(segm_xy) < 1:
        temp_segm_dict[segm_id] = {}
        temp_segm_dict[segm_id]['half_w'] = 0
        temp_segm_dict[segm_id]['half_h'] = 0
        return

    min_x, min_y = np.min(segm_xy, axis=0).astype(int)
    max_x, max_y = np.max(segm_xy, axis=0).astype(int)

    # translate and scale w + h
    w = int((max_x - min_x) * scaler)
    h = int((max_y - min_y) * scaler)

    img_bg = np.empty((h, w, 4), np.uint8)
    img_bg.fill(255)
    img_bg[:, :] = COARSE_TO_COLOR[segm_id]

    # translate and scale midpoint
    translated_midpoint = ((midpoint[0:2] - np.array([min_x, min_y])) * scaler).astype(int)
    cv2.circle(img_bg, (translated_midpoint[0], translated_midpoint[1]), 5, (255, 0, 255), -1)

    if show:
        cv2.imshow(segm_id, img_bg)
        cv2.waitKey()
        cv2.destroyAllWindows()

    temp_segm_dict[segm_id] = {}
    if segm_id != 'Head':
        temp_segm_dict[segm_id]['half_w'] = min(max((w - translated_midpoint[0]), translated_midpoint[0]),
                                                int(bbox_segm_dict[segm_id]['half_w'] * scaler))
        temp_segm_dict[segm_id]['half_h'] = min(max((h - translated_midpoint[1]), translated_midpoint[1]),
                                                int(bbox_segm_dict[segm_id]['half_h'] * scaler))
    else:
        temp_segm_dict[segm_id]['half_w'] = max((w - translated_midpoint[0]), translated_midpoint[0])
        temp_segm_dict[segm_id]['half_h'] = max((h - translated_midpoint[1]), translated_midpoint[1])


def _draw_one_norm_segm(image, segm_id, norm_midpoint):

    global temp_segm_dict
    global norm_segm_dict

    if segm_id == 'RUpperArm' or segm_id == 'LUpperArm':
        half_h = max(temp_segm_dict['RUpperArm']['half_h'], temp_segm_dict['LUpperArm']['half_h'])
        half_w = max(temp_segm_dict['RUpperArm']['half_w'], temp_segm_dict['LUpperArm']['half_w'])

    elif segm_id == 'RLowerArm' or segm_id == 'LLowerArm':
        half_h = max(temp_segm_dict['RLowerArm']['half_h'], temp_segm_dict['LLowerArm']['half_h'])
        half_w = max(temp_segm_dict['RLowerArm']['half_w'], temp_segm_dict['LLowerArm']['half_w'])

    elif segm_id == 'RThigh' or segm_id == 'LThigh':
        half_h = max(temp_segm_dict['RThigh']['half_h'], temp_segm_dict['LThigh']['half_h'])
        half_w = max(temp_segm_dict['RThigh']['half_w'], temp_segm_dict['LThigh']['half_w'])

    elif segm_id == 'RCalf' or segm_id == 'LCalf':
        half_h = max(temp_segm_dict['RCalf']['half_h'], temp_segm_dict['LCalf']['half_h'])
        half_w = max(temp_segm_dict['RCalf']['half_w'], temp_segm_dict['LCalf']['half_w'])

    else:
        half_h = temp_segm_dict[segm_id]['half_h']
        half_w = temp_segm_dict[segm_id]['half_w']

    # if the segment does not exist on both left and right side, return!!!
    if half_h < 1 or half_w < 1:
        return

    norm_segm_dict[segm_id + '_w'] = int(half_w * 2)
    norm_segm_dict[segm_id + '_h'] = int(half_h * 2)

    img_bg = np.empty((int(half_h*2), int(half_w*2), 4), np.uint8)
    img_bg.fill(255)
    img_bg[:, :] = COARSE_TO_COLOR[segm_id]

    norm_midpoint_x, norm_midpoint_y = norm_midpoint
    min_x = int(norm_midpoint_x - half_w)
    max_x = int(norm_midpoint_x + half_w)
    min_y = int(norm_midpoint_y - half_h)
    max_y = int(norm_midpoint_y + half_h)

    # debug
    # print(image[min_y:max_y, min_x:max_x, :].shape)
    # print(img_bg.shape)

    try:
        # draw the normalized segment
        added_image = cv2.addWeighted(image[min_y:max_y, min_x:max_x, :], 0.1, img_bg, 0.9, 0)
        image[min_y:max_y, min_x:max_x, :] = added_image
        # draw the normalized midpoint
        cv2.circle(image, tuple(norm_midpoint), radius=2, color=(255, 0, 255), thickness=-1)
    except:
        pass


def _draw_norm_segm(segm_xy_dict, keypoints_dict, midpoints_dict, rotated_angles_dict, gender):

    global temp_segm_dict
    global norm_segm_dict

    # white image
    image = np.empty((624, 624, 4), np.uint8)
    image.fill(255)

    # common settings
    # coordinates [x, y] coming from distribution_segm.extract_contour_on_vitruve()
    # nose_y 146
    # torso_y 281
    # rupper_arm_x 218
    # rlower_arm_x 149
    # lupper_arm_x 405
    # llower_arm_x 474
    # thigh_y 427
    # calf_y 544

    # [x, y]
    mid_x = 312
    arm_line_y = 217
    right_leg_x = 288
    left_leg_x = 336

    norm_nose_xy = [mid_x, 146]
    norm_mid_torso_xy = [mid_x, 281]

    norm_mid_rupper_arm_xy = [218, arm_line_y]
    norm_mid_rlower_arm_xy = [149, arm_line_y]
    norm_mid_lupper_arm_xy = [405, arm_line_y]
    norm_mid_llower_arm_xy = [474, arm_line_y]

    norm_mid_rthigh_xy = [right_leg_x, 427]
    norm_mid_rcalf_xy = [right_leg_x, 544]
    norm_mid_lthigh_xy = [left_leg_x, 427]
    norm_mid_lcalf_xy = [left_leg_x, 544]

    # rotated segments
    rotated_segm_xy_dict = {}
    # rotated midpoints
    rotated_midpoints_dict = {}

    # head
    rotated_segm_xy_dict['Head'] = np.array([_rotate((x, y), keypoints_dict['Neck'], rotated_angles_dict['Head']) for (x, y) in segm_xy_dict['Head']])
    rotated_midpoints_dict['Head'] = np.array(_rotate(midpoints_dict['Head'], keypoints_dict['Neck'], rotated_angles_dict['Head']))

    # calculate scaler based on the segment 'head'
    min_x, min_y = np.min(rotated_segm_xy_dict['Head'], axis=0).astype(int)
    max_x, max_y = np.max(rotated_segm_xy_dict['Head'], axis=0).astype(int)
    h = max((rotated_midpoints_dict['Head'][1] - min_y) * 2, (max_y - rotated_midpoints_dict['Head'][1]) * 2)
    if h > 0:
        if gender == 'man':
            scaler = 62 / h
        elif gender == 'woman':
            scaler = 58 / h
    norm_segm_dict['scaler'] = scaler

    _draw_one_rotated_and_scaled_segm('Head', rotated_segm_xy_dict['Head'], rotated_midpoints_dict['Head'], scaler)

    # torso
    rotated_segm_xy_dict['Torso'] = np.array([_rotate((x, y), keypoints_dict['MidHip'], rotated_angles_dict['Torso']) for (x, y) in segm_xy_dict['Torso']])
    rotated_midpoints_dict['Torso'] = np.array(_rotate(midpoints_dict['Torso'], keypoints_dict['MidHip'], rotated_angles_dict['Torso']))
    _draw_one_rotated_and_scaled_segm('Torso', rotated_segm_xy_dict['Torso'], rotated_midpoints_dict['Torso'], scaler)

    # upper limbs
    rotated_segm_xy_dict['RUpperArm'] = np.array([_rotate((x, y), keypoints_dict['RShoulder'], rotated_angles_dict['RUpperArm']) for (x, y) in segm_xy_dict['RUpperArm']])
    rotated_midpoints_dict['RUpperArm'] = np.array(_rotate(midpoints_dict['RUpperArm'], keypoints_dict['RShoulder'], rotated_angles_dict['RUpperArm']))
    _draw_one_rotated_and_scaled_segm('RUpperArm', rotated_segm_xy_dict['RUpperArm'], rotated_midpoints_dict['RUpperArm'], scaler)

    rotated_segm_xy_dict['RLowerArm'] = np.array([_rotate((x, y), keypoints_dict['RElbow'], rotated_angles_dict['RLowerArm']) for (x, y) in segm_xy_dict['RLowerArm']])
    rotated_midpoints_dict['RLowerArm'] = np.array(_rotate(midpoints_dict['RLowerArm'], keypoints_dict['RElbow'], rotated_angles_dict['RLowerArm']))
    _draw_one_rotated_and_scaled_segm('RLowerArm', rotated_segm_xy_dict['RLowerArm'], rotated_midpoints_dict['RLowerArm'], scaler)

    rotated_segm_xy_dict['LUpperArm'] = np.array([_rotate((x, y), keypoints_dict['LShoulder'], rotated_angles_dict['LUpperArm']) for (x, y) in segm_xy_dict['LUpperArm']])
    rotated_midpoints_dict['LUpperArm'] = np.array(_rotate(midpoints_dict['LUpperArm'], keypoints_dict['LShoulder'], rotated_angles_dict['LUpperArm']))
    _draw_one_rotated_and_scaled_segm('LUpperArm', rotated_segm_xy_dict['LUpperArm'], rotated_midpoints_dict['LUpperArm'], scaler)

    rotated_segm_xy_dict['LLowerArm'] = np.array([_rotate((x, y), keypoints_dict['LElbow'], rotated_angles_dict['LLowerArm']) for (x, y) in segm_xy_dict['LLowerArm']])
    rotated_midpoints_dict['LLowerArm'] = np.array(_rotate(midpoints_dict['LLowerArm'], keypoints_dict['LElbow'], rotated_angles_dict['LLowerArm']))
    _draw_one_rotated_and_scaled_segm('LLowerArm', rotated_segm_xy_dict['LLowerArm'], rotated_midpoints_dict['LLowerArm'], scaler)

    # lower limbs
    rotated_segm_xy_dict['RThigh'] = np.array([_rotate((x, y), keypoints_dict['RHip'], rotated_angles_dict['RThigh']) for (x, y) in segm_xy_dict['RThigh']])
    rotated_midpoints_dict['RThigh'] = np.array(_rotate(midpoints_dict['RThigh'], keypoints_dict['RHip'], rotated_angles_dict['RThigh']))
    _draw_one_rotated_and_scaled_segm('RThigh', rotated_segm_xy_dict['RThigh'], rotated_midpoints_dict['RThigh'], scaler)

    rotated_segm_xy_dict['RCalf'] = np.array([_rotate((x, y), keypoints_dict['RKnee'], rotated_angles_dict['RCalf']) for (x, y) in segm_xy_dict['RCalf']])
    rotated_midpoints_dict['RCalf'] = np.array(_rotate(midpoints_dict['RCalf'], keypoints_dict['RKnee'], rotated_angles_dict['RCalf']))
    _draw_one_rotated_and_scaled_segm('RCalf', rotated_segm_xy_dict['RCalf'], rotated_midpoints_dict['RCalf'], scaler)

    rotated_segm_xy_dict['LThigh'] = np.array([_rotate((x, y), keypoints_dict['LHip'], rotated_angles_dict['LThigh']) for (x, y) in segm_xy_dict['LThigh']])
    rotated_midpoints_dict['LThigh'] = np.array(_rotate(midpoints_dict['LThigh'], keypoints_dict['LHip'], rotated_angles_dict['LThigh']))
    _draw_one_rotated_and_scaled_segm('LThigh', rotated_segm_xy_dict['LThigh'], rotated_midpoints_dict['LThigh'], scaler)

    rotated_segm_xy_dict['LCalf'] = np.array([_rotate((x, y), keypoints_dict['LKnee'], rotated_angles_dict['LCalf']) for (x, y) in segm_xy_dict['LCalf']])
    rotated_midpoints_dict['LCalf'] = np.array(_rotate(midpoints_dict['LCalf'], keypoints_dict['LKnee'], rotated_angles_dict['LCalf']))
    _draw_one_rotated_and_scaled_segm('LCalf', rotated_segm_xy_dict['LCalf'], rotated_midpoints_dict['LCalf'], scaler)

    # head
    _draw_one_norm_segm(image, 'Head', norm_nose_xy)

    # torso
    _draw_one_norm_segm(image, 'Torso', norm_mid_torso_xy)

    # upper limbs
    _draw_one_norm_segm(image, 'RUpperArm', norm_mid_rupper_arm_xy)
    _draw_one_norm_segm(image, 'RLowerArm', norm_mid_rlower_arm_xy)
    _draw_one_norm_segm(image, 'LUpperArm', norm_mid_lupper_arm_xy)
    _draw_one_norm_segm(image, 'LLowerArm', norm_mid_llower_arm_xy)

    # lower limbs
    _draw_one_norm_segm(image, 'RThigh', norm_mid_rthigh_xy)
    _draw_one_norm_segm(image, 'RCalf', norm_mid_rcalf_xy)
    _draw_one_norm_segm(image, 'LThigh', norm_mid_lthigh_xy)
    _draw_one_norm_segm(image, 'LCalf', norm_mid_lcalf_xy)

    if show:
        cv2.imshow('norm image', image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return image


def generate(dp_img_ids, gender):

    global people_count
    global norm_segm_dict

    # iterate through all the images
    for image_id in dp_img_ids:

        entry = dp_coco.loadImgs(image_id)[0]

        dataset_name = entry['file_name'][entry['file_name'].find('_') + 1:entry['file_name'].rfind('_')]
        image_fpath = os.path.join(coco_folder, dataset_name, entry['file_name'])

        print('image_fpath:', image_fpath)

        try:
            im_gray = cv2.imread(image_fpath, cv2.IMREAD_GRAYSCALE)
            im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])
        except:
            continue

        dp_annotation_ids = dp_coco.getAnnIds(imgIds=entry['id'])
        dp_annotations = dp_coco.loadAnns(dp_annotation_ids)

        # iterate through all the people in one image
        person_index = 0
        for dp_annotation in dp_annotations:

            # check the validity of annotation
            is_valid, _ = DensePoseDataRelative.validate_annotation(dp_annotation)

            if not is_valid:
                continue

            # 1. keypoints
            keypoints = dp_annotation['keypoints']

            # 2. bbox
            bbox_xywh = np.array(dp_annotation["bbox"]).astype(int)

            # 3. segments of dense_pose
            if ('dp_masks' in dp_annotation.keys()):
                mask = _get_dp_mask(dp_annotation['dp_masks'])

                x1, y1, x2, y2 = bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]

                x2 = min([x2, im_gray.shape[1]])
                y2 = min([y2, im_gray.shape[0]])

                segm = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)

            # get segm_xy + keypoints dict
            segm_xy_dict, keypoints_dict = _get_dict_of_segm_and_keypoints(segm, keypoints, bbox_xywh)

            # if the head does not exist, continue to the next person!!!
            if len(segm_xy_dict['Head']) < 1:
                continue

            # if the body box is not full, continue to the next person!!!
            if not _is_valid(keypoints_dict):
                continue

            people_count += 1
            person_index += 1

            # get midpoints dict
            midpoints_dict = _get_dict_of_midpoints(segm_xy_dict, keypoints_dict)

            # get rotated angles dict
            rotated_angles_dict = _get_dict_of_rotated_angles(keypoints_dict, midpoints_dict)

            # draw the segments on the image
            _draw_segm_and_keypoints(im_gray.copy(), segm_xy_dict, keypoints_dict)

            # draw the bbox of segments on the image
            _draw_segm_bbox(im_gray.copy(), segm_xy_dict, keypoints_dict)

            # draw the normalized segments
            image = _draw_norm_segm(segm_xy_dict, keypoints_dict, midpoints_dict, rotated_angles_dict, gender)

            # save the normalized data
            index_name = _generate_index_name(image_id, person_index)
            df = pd.DataFrame(data=norm_segm_dict, index=[index_name])
            with open(os.path.join('output', 'norm_segm_coco_{}.csv'.format(gender)), 'a') as csv_file:
                df.to_csv(csv_file, index=True, header=False)
            # empty the data
            norm_segm_dict = {}


def _generate_index_name(image_id, person_index):

    index_name = '{}_{}'.format(image_id, person_index)

    return index_name


if __name__ == '__main__':

    # common setting
    dp_img_category = 'woman' # man or woman
    show = False

    # images within a range
    dp_img_range = slice(0, None)
    dp_img_ids = get_img_ids_by_caption(dp_img_category=dp_img_category, dp_img_range=dp_img_range)

    # count of total people
    people_count = 0

    bbox_segm_dict = {}
    temp_segm_dict = {}
    norm_segm_dict = {}

    generate(dp_img_ids, dp_img_category)

    # log
    print('total number of valid people:', people_count)