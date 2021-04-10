from detectron2.utils.logger import setup_logger
setup_logger()

import cv2, os
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from densepose.config import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer, DensePoseResultsVisualizer
from densepose.vis.densepose_data_points import DensePoseDataCoarseSegmentationVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.extractor import CompoundExtractor, DensePoseResultExtractor, create_extractor
from densepose.vis.extractor import extract_boxes_xywh_from_instances
from densepose.converters import ToChartResultConverterWithConfidences
from densepose.vis.base import MatrixVisualizer
import torch

import argparse
from pathlib import Path


# setting
gray_val_scale = 10.625
cmap = cv2.COLORMAP_PARULA

# files of config
keypoints_dir = os.path.join('output', 'data')

# format
# x, y => int, score => float
# keypoints = [x, y, score]
# segments_xy = [(x1, y1), (x2, y2), ...]

# coarse segmentation:
# 0 = Background
# 1 = Torso,
# 2 = Right Hand, 3 = Left Hand, 4 = Left Foot, 5 = Right Foot,
# 6 = Upper Leg Right, 7 = Upper Leg Left, 8 = Lower Leg Right, 9 = Lower Leg Left,
# 10 = Upper Arm Left, 11 = Upper Arm Right, 12 = Lower Arm Left, 13 = Lower Arm Right,
# 14 = Head

COARSE_ID = [
    'Background',
    'Torso',
    'RHand', 'LHand', 'LFoot', 'RFoot',
    'RThigh', 'LThigh', 'RCalf', 'LCalf',
    'LUpperArm', 'RUpperArm', 'LLowerArm', 'RLowerArm',
    'Head'
]

# fine segmentation:
# 0 = Background
# 1, 2 = Torso,
# 3 = Right Hand, 4 = Left Hand, 5 = Left Foot, 6 = Right Foot,
# 7, 9 = Upper Leg Right, 8, 10 = Upper Leg Left, 11, 13 = Lower Leg Right, 12, 14 = Lower Leg Left,
# 15, 17 = Upper Arm Left, 16, 18 = Upper Arm Right, 19, 21 = Lower Arm Left, 20, 22 = Lower Arm Right,
# 23, 24 = Head

FINE_TO_COARSE_SEGMENTATION = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 8,
    14: 9,
    15: 10,
    16: 11,
    17: 10,
    18: 11,
    19: 12,
    20: 13,
    21: 12,
    22: 13,
    23: 14,
    24: 14
}


# Body 25 Keypoints
JOINT_ID = [
    'Nose', 'Neck',
    'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist',
    'MidHip',
    'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
    'REye', 'LEye', 'REar', 'LEar',
    'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel',
    'Background'
]


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


def _extract_u_from_iuvarr(iuv_arr):
    return iuv_arr[1, :, :]


def _extract_v_from_iuvarr(iuv_arr):
    return iuv_arr[2, :, :]


def extract_segm(result_densepose, is_coarse=True):

    iuv_array = torch.cat(
        (result_densepose.labels[None].type(torch.float32), result_densepose.uv * 255.0)
    ).type(torch.uint8)

    iuv_array = iuv_array.cpu().numpy()

    segm = _extract_i_from_iuvarr(iuv_array)

    if is_coarse:
        for fine_idx, coarse_idx in FINE_TO_COARSE_SEGMENTATION.items():
            segm[segm == fine_idx] = coarse_idx

    mask = np.zeros(segm.shape, dtype=np.uint8)
    mask[segm > 0] = 1

    # matrix = _extract_v_from_iuvarr(iuv_array)

    return mask, segm


def _resize(mask, segm, w, h):

    interp_method_mask = cv2.INTER_NEAREST
    interp_method_segm = cv2.INTER_LINEAR,

    if (w != mask.shape[1]) or (h != mask.shape[0]):
        mask = cv2.resize(mask, (w, h), interp_method_mask)

    if (w != segm.shape[1]) or (h != segm.shape[0]):
        segm = cv2.resize(segm, (w, h), interp_method_segm)

    return mask, segm


def _calc_angle(point1, center, point2):

    try:
        a = np.array(point1)[0:2] - np.array(center)[0:2]
        b = np.array(point2)[0:2] - np.array(center)[0:2]

        cos_theta = np.dot(a, b)
        sin_theta = np.cross(a, b)

        rad = np.arctan2(sin_theta, cos_theta)
        deg = np.rad2deg(rad)

        if np.isnan(rad):
            return 0, 0

        return rad, deg

    except:
        return 0, 0


def _rotate(point, center, rad):

    x = ((point[0] - center[0]) * np.cos(rad)) - ((point[1] - center[1]) * np.sin(rad)) + center[0];
    y = ((point[0] - center[0]) * np.sin(rad)) + ((point[1] - center[1]) * np.cos(rad)) + center[1];

    if len(point) == 3:
        return [int(x), int(y), point[2]] # for keypoints with score
    elif len(point) == 2:
        return [int(x), int(y)] # for segments x, y without score


def _segm_xy(segm, segm_id_list, is_equal=True):

    if len(segm_id_list) == 1:

        segm_id = segm_id_list[0]

        if is_equal:
            y, x = np.where(segm == segm_id)
        else:
            y, x = np.where(segm != segm_id)

    elif len(segm_id_list) > 1:

        if is_equal:
            cond = []
            for segm_id in segm_id_list:
                cond.append(segm == segm_id)
            y, x = np.where(np.logical_or.reduce(tuple(cond)))
        else:
            cond = []
            for segm_id in segm_id_list:
                cond.append(segm != segm_id)
            y, x = np.where(np.logical_or.reduce(tuple(cond)))

    return list(zip(x, y))


def _get_segments_xy(segm):

    head_xy = _segm_xy(segm=segm, segm_id_list=[14])
    torso_xy = _segm_xy(segm=segm, segm_id_list=[1])
    body_xy = _segm_xy(segm=segm, segm_id_list=[0], is_equal=False)

    r_thigh_xy = _segm_xy(segm=segm, segm_id_list=[6])
    l_thigh_xy = _segm_xy(segm=segm, segm_id_list=[7])
    r_calf_xy = _segm_xy(segm=segm, segm_id_list=[8])
    l_calf_xy = _segm_xy(segm=segm, segm_id_list=[9])

    l_upper_arm_xy = _segm_xy(segm=segm, segm_id_list=[10])
    r_upper_arm_xy = _segm_xy(segm=segm, segm_id_list=[11])
    l_lower_arm_xy = _segm_xy(segm=segm, segm_id_list=[12])
    r_lower_arm_xy = _segm_xy(segm=segm, segm_id_list=[13])

    return (head_xy, torso_xy, body_xy,
            r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
            l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy)


def _rotate_to_vertical_segments_xy_with_keypoints(segments_xy, keypoints):

    (head_xy, torso_xy, body_xy,
     r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
     l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy) = segments_xy

    # calculate the angle for rotation to vertical pose
    reference_point = np.array(keypoints['MidHip']) + np.array([0, -100, 0])
    rad, deg = _calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'], point2=reference_point)

    # rotate segments to vertical pose
    head_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in head_xy]
    torso_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in torso_xy]

    r_thigh_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in r_thigh_xy]
    l_thigh_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in l_thigh_xy]
    r_calf_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in r_calf_xy]
    l_calf_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in l_calf_xy]

    l_upper_arm_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in l_upper_arm_xy]
    r_upper_arm_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in r_upper_arm_xy]
    l_lower_arm_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in l_lower_arm_xy]
    r_lower_arm_xy = [_rotate([x, y], keypoints['MidHip'], rad) for (x, y) in r_lower_arm_xy]

    # rotate keypoints to vertical pose
    rotated_keypoints = {key: _rotate(value, keypoints['MidHip'], rad) for key, value in keypoints.items()}

    return (head_xy, torso_xy, body_xy,
            r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
            l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy), rotated_keypoints


def _rotate_to_tpose_segments_xy_with_keypoints(segments_xy, keypoints):

    tpose_lower_limb_rad_factor = 10

    (head_xy, torso_xy, body_xy,
     r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
     l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy) = segments_xy

    tpose_keypoints = keypoints

    # nose -> head (BUT nose is not at the middle point of face, e.g., face right, face left!!!)
    # midhip -> torso (DONE in vertical rotation)
    # elbow -> upper arm
    # wrist -> lower arm
    # knee -> thigh
    # ankle -> calf

    # head
    # rad, deg = _calc_angle(keypoints.get('Nose'), keypoints.get('Neck'), keypoints.get('MidHip'))
    # rad = rad + np.pi
    # head_xy = [_rotate([x, y], keypoints.get('Neck'), rad) for (x, y) in head_xy]

    # RIGHT
    # Upper Limb
    # rotate lower arm to align with upper arm
    rad, deg = _calc_angle(keypoints.get('RWrist'), keypoints.get('RElbow'), keypoints.get('RShoulder'))
    rad = rad + np.pi
    r_lower_arm_xy = [_rotate([x, y], keypoints.get('RElbow'), rad) for (x, y) in r_lower_arm_xy]

    tpose_keypoints['RWrist'] = _rotate(keypoints.get('RWrist'), keypoints.get('RElbow'), rad)

    # rotate upper limb to align with shoulder
    rad, deg = _calc_angle(keypoints.get('RElbow'), keypoints.get('RShoulder'), keypoints.get('Neck'))
    rad = rad + np.pi
    r_upper_arm_xy = [_rotate([x, y], keypoints.get('RShoulder'), rad) for (x, y) in r_upper_arm_xy]
    r_lower_arm_xy = [_rotate([x, y], keypoints.get('RShoulder'), rad) for (x, y) in r_lower_arm_xy]

    tpose_keypoints['RWrist'] = _rotate(keypoints.get('RWrist'), keypoints.get('RShoulder'), rad)
    tpose_keypoints['RElbow'] = _rotate(keypoints.get('RElbow'), keypoints.get('RShoulder'), rad)

    # rotate shoulder to horizontal pose
    rad, deg = _calc_angle(keypoints.get('RShoulder'), keypoints.get('Neck'), keypoints.get('MidHip'))
    rad = rad + np.pi/2
    r_upper_arm_xy = [_rotate([x, y], keypoints.get('Neck'), rad) for (x, y) in r_upper_arm_xy]
    r_lower_arm_xy = [_rotate([x, y], keypoints.get('Neck'), rad) for (x, y) in r_lower_arm_xy]

    tpose_keypoints['RShoulder'] = _rotate(keypoints.get('RShoulder'), keypoints.get('Neck'), rad)
    tpose_keypoints['RWrist'] = _rotate(keypoints.get('RWrist'), keypoints.get('Neck'), rad)
    tpose_keypoints['RElbow'] = _rotate(keypoints.get('RElbow'), keypoints.get('Neck'), rad)

    # Lower Limb
    # rotate calf to align with thigh
    rad, deg = _calc_angle(keypoints.get('RAnkle'), keypoints.get('RKnee'), keypoints.get('RHip'))
    rad = rad + np.pi
    r_calf_xy = [_rotate([x, y], keypoints.get('RKnee'), rad) for (x, y) in r_calf_xy]

    tpose_keypoints['RAnkle'] = _rotate(keypoints.get('RAnkle'), keypoints.get('RKnee'), rad)

    # rotate hip to horizontal
    rad, deg = _calc_angle(keypoints.get('RHip'), keypoints.get('MidHip'), keypoints.get('Neck'))
    rad = rad - np.pi/2
    r_thigh_xy = [_rotate([x, y], keypoints.get('MidHip'), rad) for (x, y) in r_thigh_xy]
    r_calf_xy = [_rotate([x, y], keypoints.get('MidHip'), rad) for (x, y) in r_calf_xy]

    tpose_keypoints['RHip'] = _rotate(keypoints.get('RHip'), keypoints.get('MidHip'), rad)
    tpose_keypoints['RKnee'] = _rotate(keypoints.get('RKnee'), keypoints.get('MidHip'), rad)
    tpose_keypoints['RAnkle'] = _rotate(keypoints.get('RAnkle'), keypoints.get('MidHip'), rad)

    # rotate lower limb to degree np.pi/10 from vertical line
    rhip_ref = np.array(keypoints.get('RHip')[0:2]) + np.array([0, 100])
    rad, deg = _calc_angle(keypoints.get('RKnee'), keypoints.get('RHip'), rhip_ref)
    rad = rad + np.pi/tpose_lower_limb_rad_factor
    r_thigh_xy = [_rotate([x, y], keypoints.get('RHip'), rad) for (x, y) in r_thigh_xy]
    r_calf_xy = [_rotate([x, y], keypoints.get('RHip'), rad) for (x, y) in r_calf_xy]

    tpose_keypoints['RKnee'] = _rotate(keypoints.get('RKnee'), keypoints.get('RHip'), rad)
    tpose_keypoints['RAnkle'] = _rotate(keypoints.get('RAnkle'), keypoints.get('RHip'), rad)

    # LEFT
    # Upper Limb
    # rotate lower arm to align with upper arm
    rad, deg = _calc_angle(keypoints.get('LWrist'), keypoints.get('LElbow'), keypoints.get('LShoulder'))
    rad = rad + np.pi
    l_lower_arm_xy = [_rotate([x, y], keypoints.get('LElbow'), rad) for (x, y) in l_lower_arm_xy]

    tpose_keypoints['LWrist'] = _rotate(keypoints.get('LWrist'), keypoints.get('LElbow'), rad)

    # rotate upper limb to align with shoulder
    rad, deg = _calc_angle(keypoints.get('LElbow'), keypoints.get('LShoulder'), keypoints.get('Neck'))
    rad = rad + np.pi
    l_upper_arm_xy = [_rotate([x, y], keypoints.get('LShoulder'), rad) for (x, y) in l_upper_arm_xy]
    l_lower_arm_xy = [_rotate([x, y], keypoints.get('LShoulder'), rad) for (x, y) in l_lower_arm_xy]

    tpose_keypoints['LWrist'] = _rotate(keypoints.get('LWrist'), keypoints.get('LShoulder'), rad)
    tpose_keypoints['LElbow'] = _rotate(keypoints.get('LElbow'), keypoints.get('LShoulder'), rad)

    # rotate shoulder to horizontal
    rad, deg = _calc_angle(keypoints.get('LShoulder'), keypoints.get('Neck'), keypoints.get('MidHip'))
    rad = rad - np.pi/2
    l_upper_arm_xy = [_rotate([x, y], keypoints.get('Neck'), rad) for (x, y) in l_upper_arm_xy]
    l_lower_arm_xy = [_rotate([x, y], keypoints.get('Neck'), rad) for (x, y) in l_lower_arm_xy]

    tpose_keypoints['LShoulder'] = _rotate(keypoints.get('LShoulder'), keypoints.get('Neck'), rad)
    tpose_keypoints['LWrist'] = _rotate(keypoints.get('LWrist'), keypoints.get('Neck'), rad)
    tpose_keypoints['LElbow'] = _rotate(keypoints.get('LElbow'), keypoints.get('Neck'), rad)

    # Lower Limb
    # rotate calf to align with thigh
    rad, deg = _calc_angle(keypoints.get('LAnkle'), keypoints.get('LKnee'), keypoints.get('LHip'))
    rad = rad + np.pi
    l_calf_xy = [_rotate([x, y], keypoints.get('LKnee'), rad) for (x, y) in l_calf_xy]

    tpose_keypoints['LAnkle'] = _rotate(keypoints.get('LAnkle'), keypoints.get('LKnee'), rad)

    # rotate hip to horizontal
    rad, deg = _calc_angle(keypoints.get('LHip'), keypoints.get('MidHip'), keypoints.get('Neck'))
    rad = rad + np.pi/2
    l_thigh_xy = [_rotate([x, y], keypoints.get('MidHip'), rad) for (x, y) in l_thigh_xy]
    l_calf_xy = [_rotate([x, y], keypoints.get('MidHip'), rad) for (x, y) in l_calf_xy]

    tpose_keypoints['LHip'] = _rotate(keypoints.get('LHip'), keypoints.get('MidHip'), rad)
    tpose_keypoints['LKnee'] = _rotate(keypoints.get('LKnee'), keypoints.get('MidHip'), rad)
    tpose_keypoints['LAnkle'] = _rotate(keypoints.get('LAnkle'), keypoints.get('MidHip'), rad)

    # rotate lower limb to degree np.pi / 8 from vertical line
    lhip_ref = np.array(keypoints.get('LHip')[0:2]) + np.array([0, 100])
    rad, deg = _calc_angle(keypoints.get('LKnee'), keypoints.get('LHip'), lhip_ref)
    rad = rad - np.pi/tpose_lower_limb_rad_factor
    l_thigh_xy = [_rotate([x, y], keypoints.get('LHip'), rad) for (x, y) in l_thigh_xy]
    l_calf_xy = [_rotate([x, y], keypoints.get('LHip'), rad) for (x, y) in l_calf_xy]

    tpose_keypoints['LKnee'] = _rotate(keypoints.get('LKnee'), keypoints.get('LHip'), rad)
    tpose_keypoints['LAnkle'] = _rotate(keypoints.get('LAnkle'), keypoints.get('LHip'), rad)

    return (head_xy, torso_xy, body_xy,
            r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
            l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy), tpose_keypoints


def rotate_to_tpose(segm, keypoints):

    # Issue ONE: cannot rotate body to [Face-front + Torso-front] view!!!
    # Issue TWO: cannot have the same person -> so it can be a fat person or a thin person!!!
    # Issue THREE: NO mapped HAND and FOOT keypoints to rotate them!!!
    # Issue FOUR: NOSE is not at the middle point of the head, e.g., face right, face left, so cannot normalize HEAD!!!

    # STEP 1: rotated any pose to a vertical pose, i.e., stand up, sit up, etc...
    # extract original segment's x, y
    segments_xy = _get_segments_xy(segm=segm)

    # rotated segment to vertical pose, i.e., stand up, sit up, etc...
    rotated_segments_xy, rotated_keypoints = _rotate_to_vertical_segments_xy_with_keypoints(segments_xy=segments_xy, keypoints=keypoints)

    # STEP 2: rotate specific segment further to t-pose
    tpose_segments_xy, tpose_keypoints = _rotate_to_tpose_segments_xy_with_keypoints(segments_xy=rotated_segments_xy, keypoints=rotated_keypoints)

    return tpose_segments_xy, tpose_keypoints


def _translate_segments_xy_with_keypoints(segments_xy, keypoints):

    mid_xy = np.array(keypoints['MidHip'])[0:2]
    diff_xy = np.array([1000, 1000]) - mid_xy

    translated_segments_xy = (np.array(segment) + diff_xy for segment in segments_xy)
    translated_keypoints = {key: (np.array(value) + np.append(diff_xy, 0)) for key, value in keypoints.items()}

    return translated_segments_xy, translated_keypoints


def _draw_segments_xy_with_keypoint(image, segments_xy, keypoints):

    head_xy, torso_xy, body_xy, \
    r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy, \
    l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy = segments_xy

    # first draw body
    for x, y in body_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=10, color=(192, 192, 192), thickness=-1)

    # then draw head + torso
    for x, y in head_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    for x, y in torso_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    # last draw limbs
    for x, y in r_thigh_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(255, 255, 0), thickness=-1)
    for x, y in l_thigh_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 255), thickness=-1)
    for x, y in r_calf_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(255, 255, 0), thickness=-1)
    for x, y in l_calf_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 255), thickness=-1)

    for x, y in l_upper_arm_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(255, 0, 255), thickness=-1)
    for x, y in r_upper_arm_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for x, y in l_lower_arm_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(255, 0, 255), thickness=-1)
    for x, y in r_lower_arm_xy.astype(int):
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    if keypoints:
        for keypoint in keypoints.values():
            x, y, score = keypoint
            if score > 0:
                image = cv2.circle(image, (int(x), int(y)), radius=10, color=(0, 0, 0), thickness=-1)

    return image


def visualize_norm_segm(image_bgr, mask, segm, bbox_xywh, keypoints):

    x, y, w, h = [int(v) for v in bbox_xywh]

    mask, segm = _resize(mask, segm, w, h)

    # mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])

    # scaled keypoints
    keypoints = np.array(keypoints) - np.array([x, y, 0.0])
    # dict keypoints
    keypoints = dict(zip(JOINT_ID, keypoints))

    # visualize original pose by bbox
    segm_scaled = segm.astype(np.float32) * gray_val_scale
    segm_scaled_8u = segm_scaled.clip(0, 255).astype(np.uint8)

    # apply cmap
    segm_vis = cv2.applyColorMap(segm_scaled_8u, cmap)

    cv2.imshow('bbox:', segm_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # visualize normalized pose
    # rotate to t-pose
    tpose_segments_xy, tpose_keypoints = rotate_to_tpose(segm=segm, keypoints=keypoints)

    # white background image
    image = np.empty((2000, 2000, 3), np.uint8)
    image.fill(255)

    translated_segments_xy, translated_keypoints = _translate_segments_xy_with_keypoints(segments_xy = tpose_segments_xy, keypoints=tpose_keypoints)

    _draw_segments_xy_with_keypoint(image, segments_xy=translated_segments_xy, keypoints=translated_keypoints)

    resized = cv2.resize(image, (1000, 1000), interpolation=cv2.INTER_AREA)

    cv2.imshow('norm', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_norm_segm(infile, score_cutoff, show):

    print('input:', infile)

    image = cv2.imread(infile)
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.MODEL.DEVICE = 'cpu'

    cfg.merge_from_file('./configs/densepose_rcnn_R_50_FPN_s1x.yaml')
    cfg.MODEL.WEIGHTS = './models/densepose_rcnn_R_50_FPN_s1x.pkl'

    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    # filter the probabilities of scores for each bbox > score_cutoff
    instances = outputs['instances']
    confident_detections = instances[instances.scores > score_cutoff]

    # extractor
    extractor = DensePoseResultExtractor()
    results_densepose, boxes_xywh = extractor(confident_detections)

    # boxes_xywh: tensor -> numpy array
    boxes_xywh = boxes_xywh.numpy()

    # load keypoints
    file_keypoints = os.path.join(keypoints_dir, '{}_keypoints.npy'.format(infile[infile.find('/') + 1:infile.rfind('.')]))
    data_keypoints = np.load(file_keypoints, allow_pickle='TRUE').item()['keypoints']

    for result_densepose, box_xywh, keypoints in zip(results_densepose, boxes_xywh, data_keypoints):

        # extract segm + mask
        mask, segm = extract_segm(result_densepose=result_densepose)

        # visualizer
        visualize_norm_segm(image_bgr=im_gray, mask=mask, segm=segm, bbox_xywh=box_xywh, keypoints=keypoints)


def generate_segm(infile, score_cutoff, show):

    print('input:', infile)

    image = cv2.imread(infile)
    # cv2.imshow('input', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.MODEL.DEVICE = 'cpu'

    cfg.merge_from_file('./configs/densepose_rcnn_R_50_FPN_s1x.yaml')
    cfg.MODEL.WEIGHTS = './models/densepose_rcnn_R_50_FPN_s1x.pkl'

    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    # print(outputs["instances"].pred_densepose)

    # filter the probabilities of scores for each bbox > 90%
    instances = outputs['instances']
    confident_detections = instances[instances.scores > score_cutoff]

    visualizers = []

    visualizer_segm = DensePoseResultsFineSegmentationVisualizer(cfg=cfg)
    visualizer_bbox = ScoredBoundingBoxVisualizer(cfg=cfg)

    visualizers.append(visualizer_segm)
    visualizers.append(visualizer_bbox)

    extractors = []

    extractor_segm = create_extractor(visualizer_segm)
    extractor_bbox = create_extractor(visualizer_bbox)

    extractors.append(extractor_segm)
    extractors.append(extractor_bbox)

    visualizer = CompoundVisualizer(visualizers)
    extractor = CompoundExtractor(extractors)

    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])

    # instances
    # data = extractor(instances)

    # confident detections
    data = extractor(confident_detections)

    image_vis = visualizer.visualize(im_gray, data)

    if show:
        cv2.imshow('segm', image_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        outfile = generate_outfile(infile)
        cv2.imwrite(outfile, image_vis)
        print('output:', outfile)


def generate_outfile(infile):

    outdir = os.path.join('output', infile[infile.find('/') + 1:infile.rfind('/')])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = infile[infile.find('/') + 1:infile.rfind('.')]
    outfile = os.path.join('output', '{}_segm.jpg'.format(fname))

    return outfile


if __name__ == '__main__':

    # python infer_segm.py --input datasets/classical
    # python infer_segm.py --input datasets/modern

    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/80019.jpg

    parser = argparse.ArgumentParser(description='DensePose - Infer the segments')
    parser.add_argument('--input', help='Path to image file or directory')
    args = parser.parse_args()

    if os.path.isfile(args.input):
        # generate_segm(infile=args.input, score_cutoff=0.95, show=True)
        generate_norm_segm(infile=args.input, score_cutoff=0.95, show=True)

    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.jpg'):
            try:
                generate_segm(infile=str(path), score_cutoff=0.9, show=False)
            except:
                continue
    else:
        pass