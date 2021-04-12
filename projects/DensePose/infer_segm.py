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
import matplotlib.pyplot as plt


# setting
gray_val_scale = 10.625
cmap = cv2.COLORMAP_PARULA

norm_image_shape = (5000, 5000, 3)
norm_image_center = (2500, 2500)

resized_image_dim = (1000, 1000)

keypoints_radius = 5
keypoints_color = (0, 0, 0)

# files of config
densepose_keypoints_dir = os.path.join('output', 'keypoints')
openpose_keypoints_dir = os.path.join('output', 'data')
norm_segm_dir = os.path.join('output', 'pix')

# data type
# keypoints = {key: (x, y, score)}
# pixel = (x, y)
# segments_xy = [(x1, y1), (x2, y2), ...]
# segm = [[x1, y1]=(b,g,r), [x2, y2]=(b,g,r), ...] -> 2D np.ndarray

# coarse segmentation:
# 0 = Background
# 1 = Torso,
# 2 = Right Hand, 3 = Left Hand, 4 = Left Foot, 5 = Right Foot,
# 6 = Upper Leg Right, 7 = Upper Leg Left, 8 = Lower Leg Right, 9 = Lower Leg Left,
# 10 = Upper Arm Left, 11 = Upper Arm Right, 12 = Lower Arm Left, 13 = Lower Arm Right,
# 14 = Head

COARSE_ID = {
    'Background': 0,
    'Torso': 1,
    'RHand': 2,
    'LHand': 3,
    'LFoot': 4,
    'RFoot':5,
    'RThigh': 6,
    'LThigh': 7,
    'RCalf': 8,
    'LCalf': 9,
    'LUpperArm': 10,
    'RUpperArm': 11,
    'LLowerArm': 12,
    'RLowerArm': 13,
    'Head': 14
}

# implicit cmap = cv2.COLORMAP_PARULA <= hard-coded!!! ugh!!!
COARSE_TO_COLOR = {
    'Background': [255, 255, 255],
    'Torso': [191, 78, 22],
    'RThigh': [167, 181, 44],
    'LThigh': [141, 187, 91],
    'RCalf': [114, 191, 147],
    'LCalf': [96, 188, 192],
    'LUpperArm': [87, 207, 112],
    'RUpperArm': [55, 218, 162],
    'LLowerArm': [25, 226, 216],
    'RLowerArm': [37, 231, 253],
    'Head': [14, 251, 249]
}

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
        return (int(x), int(y), point[2]) # for keypoints with score
    elif len(point) == 2:
        return (int(x), int(y)) # for segments (x, y) without score


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


def _segments_xy_centroid(segments_xy):

    x = [segment_xy[0] for segment_xy in segments_xy if not np.isnan(segment_xy[0])]
    y = [segment_xy[1] for segment_xy in segments_xy if not np.isnan(segment_xy[1])]
    centroid = (sum(x) / len(segments_xy), sum(y) / len(segments_xy))

    return centroid


def get_segments_xy(segm):

    head_xy = _segm_xy(segm=segm, segm_id_list=[14])
    torso_xy = _segm_xy(segm=segm, segm_id_list=[1])

    r_thigh_xy = _segm_xy(segm=segm, segm_id_list=[6])
    l_thigh_xy = _segm_xy(segm=segm, segm_id_list=[7])
    r_calf_xy = _segm_xy(segm=segm, segm_id_list=[8])
    l_calf_xy = _segm_xy(segm=segm, segm_id_list=[9])

    l_upper_arm_xy = _segm_xy(segm=segm, segm_id_list=[10])
    r_upper_arm_xy = _segm_xy(segm=segm, segm_id_list=[11])
    l_lower_arm_xy = _segm_xy(segm=segm, segm_id_list=[12])
    r_lower_arm_xy = _segm_xy(segm=segm, segm_id_list=[13])


    if np.array(head_xy).size > 0:
        head_xy = np.array(head_xy)
    else:
        head_xy = np.array([])

    if np.array(torso_xy).size > 0:
        torso_xy = np.array(torso_xy)
    else:
        torso_xy = np.array([])


    if np.array(r_thigh_xy).size > 0:
        r_thigh_xy = np.array(r_thigh_xy)
    else:
        r_thigh_xy = np.array([])

    if np.array(l_thigh_xy).size > 0:
        l_thigh_xy = np.array(l_thigh_xy)
    else:
        l_thigh_xy = np.array([])

    if np.array(r_calf_xy).size > 0:
        r_calf_xy = np.array(r_calf_xy)
    else:
        r_calf_xy = np.array([])

    if np.array(l_calf_xy).size > 0:
        l_calf_xy = np.array(l_calf_xy)
    else:
        l_calf_xy = np.array([])


    if np.array(l_upper_arm_xy).size > 0:
        l_upper_arm_xy = np.array(l_upper_arm_xy)
    else:
        l_upper_arm_xy = np.array([])

    if np.array(r_upper_arm_xy).size > 0:
        r_upper_arm_xy = np.array(r_upper_arm_xy)
    else:
        r_upper_arm_xy = np.array([])

    if np.array(l_lower_arm_xy).size > 0:
        l_lower_arm_xy = np.array(l_lower_arm_xy)
    else:
        l_lower_arm_xy = np.array([])

    if np.array(r_lower_arm_xy).size > 0:
        r_lower_arm_xy = np.array(r_lower_arm_xy)
    else:
        r_lower_arm_xy = np.array([])


    return (head_xy, torso_xy,
            r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
            l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy)


def rotate_to_vertical_segments_xy_with_keypoints(segments_xy, keypoints):

    (head_xy, torso_xy,
     r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
     l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy) = segments_xy

    # calculate the angle for rotation to vertical pose
    reference_point = np.array(keypoints['MidHip']) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'], point2=reference_point)

    # rotate segments to vertical pose
    if head_xy.size > 0:
        head_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in head_xy])

    if torso_xy.size > 0:
        torso_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in torso_xy])


    if r_thigh_xy.size > 0:
        r_thigh_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in r_thigh_xy])

    if l_thigh_xy.size > 0:
        l_thigh_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in l_thigh_xy])

    if r_calf_xy.size > 0:
        r_calf_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in r_calf_xy])

    if l_calf_xy.size > 0:
        l_calf_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in l_calf_xy])


    if l_upper_arm_xy.size > 0:
        l_upper_arm_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in l_upper_arm_xy])

    if r_upper_arm_xy.size > 0:
        r_upper_arm_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in r_upper_arm_xy])

    if l_lower_arm_xy.size > 0:
        l_lower_arm_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in l_lower_arm_xy])

    if r_lower_arm_xy.size > 0:
        r_lower_arm_xy = np.array([_rotate((x, y), keypoints['MidHip'], rad) for (x, y) in r_lower_arm_xy])


    # rotate keypoints to vertical pose
    vertical_keypoints = {key: _rotate(value, keypoints['MidHip'], rad) for key, value in keypoints.items()}

    return (head_xy, torso_xy,
            r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
            l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy), vertical_keypoints


def rotate_to_tpose_segments_xy_with_keypoints(segments_xy, keypoints):

    tpose_lower_limb_rad_factor = 10

    (head_xy, torso_xy,
     r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
     l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy) = segments_xy

    tpose_keypoints = keypoints

    # nose -> head (BUT nose is not at the middle point of face, e.g., face right, face left!!!)
    # midhip -> torso (DONE in vertical rotation)
    # elbow -> upper arm
    # wrist -> lower arm
    # knee -> thigh
    # ankle -> calf

    # head -> NOT use Nose, use Centroid of head_xy!!!
    # ONE solution to Issue FOUR: NOSE is not at the middle point of the head!!!
    if head_xy.size > 0:
        head_centroid = _segments_xy_centroid(head_xy)
        rad, deg = _calc_angle(head_centroid, keypoints.get('Neck'), keypoints.get('MidHip'))
        rad = rad + np.pi
        head_xy = np.array([_rotate([x, y], keypoints.get('Neck'), rad) for (x, y) in head_xy])

    # RIGHT
    # Upper Limb
    # rotate lower arm to align with upper arm
    if r_lower_arm_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('RWrist'), keypoints.get('RElbow'), keypoints.get('RShoulder'))
        rad = rad + np.pi
        r_lower_arm_xy = np.array([_rotate((x, y), keypoints.get('RElbow'), rad) for (x, y) in r_lower_arm_xy])

        tpose_keypoints['RWrist'] = _rotate(keypoints.get('RWrist'), keypoints.get('RElbow'), rad)

    # rotate upper limb to align with shoulder
    if r_lower_arm_xy.size > 0 and r_upper_arm_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('RElbow'), keypoints.get('RShoulder'), keypoints.get('Neck'))
        rad = rad + np.pi
        r_upper_arm_xy = np.array([_rotate((x, y), keypoints.get('RShoulder'), rad) for (x, y) in r_upper_arm_xy])
        r_lower_arm_xy = np.array([_rotate((x, y), keypoints.get('RShoulder'), rad) for (x, y) in r_lower_arm_xy])

        tpose_keypoints['RWrist'] = _rotate(keypoints.get('RWrist'), keypoints.get('RShoulder'), rad)
        tpose_keypoints['RElbow'] = _rotate(keypoints.get('RElbow'), keypoints.get('RShoulder'), rad)

    # rotate shoulder to horizontal pose
    if r_lower_arm_xy.size > 0 and r_upper_arm_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('RShoulder'), keypoints.get('Neck'), keypoints.get('MidHip'))
        rad = rad + np.pi / 2
        r_upper_arm_xy = np.array([_rotate((x, y), keypoints.get('Neck'), rad) for (x, y) in r_upper_arm_xy])
        r_lower_arm_xy = np.array([_rotate((x, y), keypoints.get('Neck'), rad) for (x, y) in r_lower_arm_xy])

        tpose_keypoints['RShoulder'] = _rotate(keypoints.get('RShoulder'), keypoints.get('Neck'), rad)
        tpose_keypoints['RWrist'] = _rotate(keypoints.get('RWrist'), keypoints.get('Neck'), rad)
        tpose_keypoints['RElbow'] = _rotate(keypoints.get('RElbow'), keypoints.get('Neck'), rad)

    # Lower Limb
    # rotate calf to align with thigh
    if r_calf_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('RAnkle'), keypoints.get('RKnee'), keypoints.get('RHip'))
        rad = rad + np.pi
        r_calf_xy = np.array([_rotate((x, y), keypoints.get('RKnee'), rad) for (x, y) in r_calf_xy])

        tpose_keypoints['RAnkle'] = _rotate(keypoints.get('RAnkle'), keypoints.get('RKnee'), rad)

    # rotate hip to horizontal
    if r_calf_xy.size > 0 and r_thigh_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('RHip'), keypoints.get('MidHip'), keypoints.get('Neck'))
        rad = rad - np.pi / 2
        r_thigh_xy = np.array([_rotate([x, y], keypoints.get('MidHip'), rad) for (x, y) in r_thigh_xy])
        r_calf_xy = np.array([_rotate([x, y], keypoints.get('MidHip'), rad) for (x, y) in r_calf_xy])

        tpose_keypoints['RHip'] = _rotate(keypoints.get('RHip'), keypoints.get('MidHip'), rad)
        tpose_keypoints['RKnee'] = _rotate(keypoints.get('RKnee'), keypoints.get('MidHip'), rad)
        tpose_keypoints['RAnkle'] = _rotate(keypoints.get('RAnkle'), keypoints.get('MidHip'), rad)

    # rotate lower limb to degree np.pi/10 from vertical line
    if r_calf_xy.size > 0 and r_thigh_xy.size > 0:
        rhip_ref = np.array(keypoints.get('RHip')[0:2]) + np.array([0, 100])
        rad, deg = _calc_angle(keypoints.get('RKnee'), keypoints.get('RHip'), rhip_ref)
        rad = rad + np.pi / tpose_lower_limb_rad_factor
        r_thigh_xy = np.array([_rotate((x, y), keypoints.get('RHip'), rad) for (x, y) in r_thigh_xy])
        r_calf_xy = np.array([_rotate((x, y), keypoints.get('RHip'), rad) for (x, y) in r_calf_xy])

        tpose_keypoints['RKnee'] = _rotate(keypoints.get('RKnee'), keypoints.get('RHip'), rad)
        tpose_keypoints['RAnkle'] = _rotate(keypoints.get('RAnkle'), keypoints.get('RHip'), rad)

    # LEFT
    # Upper Limb
    # rotate lower arm to align with upper arm
    if l_lower_arm_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('LWrist'), keypoints.get('LElbow'), keypoints.get('LShoulder'))
        rad = rad + np.pi
        l_lower_arm_xy = np.array([_rotate((x, y), keypoints.get('LElbow'), rad) for (x, y) in l_lower_arm_xy])

        tpose_keypoints['LWrist'] = _rotate(keypoints.get('LWrist'), keypoints.get('LElbow'), rad)

    # rotate upper limb to align with shoulder
    if l_lower_arm_xy.size > 0 and l_upper_arm_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('LElbow'), keypoints.get('LShoulder'), keypoints.get('Neck'))
        rad = rad + np.pi
        l_upper_arm_xy = np.array([_rotate((x, y), keypoints.get('LShoulder'), rad) for (x, y) in l_upper_arm_xy])
        l_lower_arm_xy = np.array([_rotate((x, y), keypoints.get('LShoulder'), rad) for (x, y) in l_lower_arm_xy])

        tpose_keypoints['LWrist'] = _rotate(keypoints.get('LWrist'), keypoints.get('LShoulder'), rad)
        tpose_keypoints['LElbow'] = _rotate(keypoints.get('LElbow'), keypoints.get('LShoulder'), rad)

    # rotate shoulder to horizontal
    if l_lower_arm_xy.size > 0 and l_upper_arm_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('LShoulder'), keypoints.get('Neck'), keypoints.get('MidHip'))
        rad = rad - np.pi / 2
        l_upper_arm_xy = np.array([_rotate((x, y), keypoints.get('Neck'), rad) for (x, y) in l_upper_arm_xy])
        l_lower_arm_xy = np.array([_rotate((x, y), keypoints.get('Neck'), rad) for (x, y) in l_lower_arm_xy])

        tpose_keypoints['LShoulder'] = _rotate(keypoints.get('LShoulder'), keypoints.get('Neck'), rad)
        tpose_keypoints['LWrist'] = _rotate(keypoints.get('LWrist'), keypoints.get('Neck'), rad)
        tpose_keypoints['LElbow'] = _rotate(keypoints.get('LElbow'), keypoints.get('Neck'), rad)

    # Lower Limb
    # rotate calf to align with thigh
    if l_calf_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('LAnkle'), keypoints.get('LKnee'), keypoints.get('LHip'))
        rad = rad + np.pi
        l_calf_xy = np.array([_rotate((x, y), keypoints.get('LKnee'), rad) for (x, y) in l_calf_xy])

        tpose_keypoints['LAnkle'] = _rotate(keypoints.get('LAnkle'), keypoints.get('LKnee'), rad)

    # rotate hip to horizontal
    if l_calf_xy.size > 0 and l_thigh_xy.size > 0:
        rad, deg = _calc_angle(keypoints.get('LHip'), keypoints.get('MidHip'), keypoints.get('Neck'))
        rad = rad + np.pi / 2
        l_thigh_xy = np.array([_rotate((x, y), keypoints.get('MidHip'), rad) for (x, y) in l_thigh_xy])
        l_calf_xy = np.array([_rotate((x, y), keypoints.get('MidHip'), rad) for (x, y) in l_calf_xy])

        tpose_keypoints['LHip'] = _rotate(keypoints.get('LHip'), keypoints.get('MidHip'), rad)
        tpose_keypoints['LKnee'] = _rotate(keypoints.get('LKnee'), keypoints.get('MidHip'), rad)
        tpose_keypoints['LAnkle'] = _rotate(keypoints.get('LAnkle'), keypoints.get('MidHip'), rad)

    # rotate lower limb to degree np.pi / 8 from vertical line
    if l_calf_xy.size > 0 and l_thigh_xy.size > 0:
        lhip_ref = np.array(keypoints.get('LHip')[0:2]) + np.array([0, 100])
        rad, deg = _calc_angle(keypoints.get('LKnee'), keypoints.get('LHip'), lhip_ref)
        rad = rad - np.pi / tpose_lower_limb_rad_factor
        l_thigh_xy = np.array([_rotate((x, y), keypoints.get('LHip'), rad) for (x, y) in l_thigh_xy])
        l_calf_xy = np.array([_rotate((x, y), keypoints.get('LHip'), rad) for (x, y) in l_calf_xy])

        tpose_keypoints['LKnee'] = _rotate(keypoints.get('LKnee'), keypoints.get('LHip'), rad)
        tpose_keypoints['LAnkle'] = _rotate(keypoints.get('LAnkle'), keypoints.get('LHip'), rad)

    return (head_xy, torso_xy,
            r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
            l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy), tpose_keypoints


def rotate_to_tpose(segm, keypoints):

    # Issue ONE: cannot rotate body to [Face-front + Torso-front] view!!!
    # Issue TWO: cannot have the same person -> so it can be a fat person or a thin person!!!
    # Issue THREE: NO mapped HAND and FOOT keypoints to rotate them!!!
    # *Issue FOUR*: NOSE is not at the middle point of the head, e.g., face right, face left, so cannot normalize HEAD!!!

    # STEP 1: rotated any pose to a vertical pose, i.e., stand up, sit up, etc...
    # extract original segment's x, y
    segments_xy = get_segments_xy(segm=segm)

    # rotated segment to vertical pose, i.e., stand up, sit up, etc...
    vertical_segments_xy, vertical_keypoints = rotate_to_vertical_segments_xy_with_keypoints(segments_xy=segments_xy, keypoints=keypoints)

    # STEP 2: rotate specific segment further to t-pose
    tpose_segments_xy, tpose_keypoints = rotate_to_tpose_segments_xy_with_keypoints(segments_xy=vertical_segments_xy, keypoints=vertical_keypoints)

    return tpose_segments_xy, tpose_keypoints


def translate_segments_xy_with_keypoints(segments_xy, keypoints):

    mid_xy = np.array(keypoints['MidHip'])[0:2]
    diff_xy = np.array(norm_image_center) - mid_xy

    translated_segments_xy = np.array([np.array(segment_xy) + diff_xy if segment_xy.size > 0 else np.array([]) for segment_xy in segments_xy])
    translated_keypoints = {key: (np.array(value) + np.append(diff_xy, 0)) for key, value in keypoints.items()}

    return translated_segments_xy, translated_keypoints


def draw_segments_xy(segments_xy):

    (head_xy, torso_xy,
     r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
     l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy) = segments_xy

    # initial background image
    image = np.empty(norm_image_shape, np.uint8)
    image.fill(255)  # => white (255, 255, 255) = background

    # first: draw head + torso
    for x, y in head_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['Head']
    for x, y in torso_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['Torso']

    # second: draw limbs
    for x, y in r_thigh_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['RThigh']
    for x, y in l_thigh_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['LThigh']
    for x, y in r_calf_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['RCalf']
    for x, y in l_calf_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['LCalf']

    for x, y in l_upper_arm_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['LUpperArm']
    for x, y in r_upper_arm_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['RUpperArm']
    for x, y in l_lower_arm_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['LLowerArm']
    for x, y in r_lower_arm_xy.astype(int):
        if x < norm_image_shape[0] and y < norm_image_shape[1]:
            image[y][x] = COARSE_TO_COLOR['RLowerArm']

    return image


def crop_and_resize_image_with_keypoints(image, keypoints):

    bbox = np.array(list(keypoints.values()))

    min_x, min_y, _ = bbox.min(axis=0).astype(int)
    max_x, max_y, _ = bbox.max(axis=0).astype(int)
    edge_x = max_x - min_x
    edge_y = max_y - min_y

    defaut_margin = 50
    if edge_x > edge_y:
        margin_x = defaut_margin
        margin_y = int((edge_x + defaut_margin * 2 - edge_y) / 2)
    elif edge_y > edge_x:
        margin_y = defaut_margin
        margin_x = int((edge_y + defaut_margin * 2 - edge_x) / 2)
    else:
        margin_x = defaut_margin
        margin_y = defaut_margin

    # crop image
    min_x, min_y = min_x - margin_x, min_y - margin_y
    max_x, max_y = max_x + margin_x, max_y + margin_y
    cropped_image = image[min_y:max_y, min_x:max_x]

    # resize image
    resized_image = cv2.resize(cropped_image, resized_image_dim, interpolation=cv2.INTER_AREA)

    # draw keypoints on resized image
    cropped_height, cropped_width, _ = cropped_image.shape
    resized_height, resized_width, _ = resized_image.shape

    ignore_keypoints_ids = ['Nose', 'REye', 'LEye', 'REar', 'LEar', 'LBigToe', 'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel']

    if keypoints:
        for id, keypoint in keypoints.items():
                x, y, score = keypoint
                if score > 0 and id not in ignore_keypoints_ids:
                    # translate + scale
                    x = (x - min_x) / cropped_width * resized_width
                    y = (y - min_y) / cropped_height * resized_height

                    resized_image = cv2.circle(resized_image, (int(x), int(y)), radius=keypoints_radius, color=keypoints_color, thickness=-1)

    return resized_image


def visualize_norm_segm(image_bgr, mask, segm, bbox_xywh, keypoints, infile, show=False):

    x, y, w, h = [int(v) for v in bbox_xywh]

    mask, segm = _resize(mask, segm, w, h)

    # mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])

    # translate keypoints to bbox
    keypoints = np.array(keypoints) - np.array((x, y, 0.0))
    # dict keypoints
    keypoints = dict(zip(JOINT_ID, keypoints))

    # visualize original pose by bbox
    if show:
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

    # translate to center
    translated_segments_xy, translated_keypoints = translate_segments_xy_with_keypoints(segments_xy = tpose_segments_xy, keypoints=tpose_keypoints)

    # draw segments in image
    image = draw_segments_xy(segments_xy=translated_segments_xy)

    # crop and resize image with keypoints
    resized_image = crop_and_resize_image_with_keypoints(image=image, keypoints=translated_keypoints)

    if show:
        cv2.imshow('norm', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        outfile = generate_norm_segm_outfile(infile)
        cv2.imwrite(outfile, resized_image)
        print('output', outfile)


def stitch_data(results_densepose, boxes_xywh, data_keypoints, image, show):

    # print('length of results_densepose:', len(results_densepose))
    # print('length of boxes_xywh:', len(boxes_xywh))
    # print('length of data_keypoints:', len(data_keypoints))

    matched_results_densepose = []
    matched_boxes_xywh = []
    matched_data_keypoints = []

    image_h, image_w, _ = image.shape

    for result_densepose, box_xywh in zip(results_densepose, boxes_xywh):

        x, y, w, h = box_xywh.astype(int)
        prop_w = w / image_w
        prop_h = h / image_h

        # condition 1: height of bbox > 0.5 * im_h
        if prop_h >= 0.6:

            for keypoints in data_keypoints:

                keypoints = [[x, y, score] for x, y, score in keypoints if score != 0]
                centroid_x, centroid_y = _segments_xy_centroid(keypoints)

                # condition 2: centroid (x, y) of keypoints within bbox
                if centroid_x > x and centroid_x < (x + w) and centroid_y > y and centroid_y < (y + h):

                    matched_results_densepose.append(result_densepose)
                    matched_boxes_xywh.append(box_xywh)
                    matched_data_keypoints.append(keypoints)

                    cv2.circle(image, (int(centroid_x), int(centroid_y)), radius=5, color=(255, 0, 255), thickness=5)

                    cv2.line(image, (x, y), (int(x + w), y), color=(0, 255, 0), thickness=5)
                    cv2.line(image, (x, y), (x, int(y + h)), color=(0, 255, 0), thickness=5)
                    cv2.line(image, (int(x + w), int(y + h)), (x, int(y + h)), color=(0, 255, 0), thickness=5)
                    cv2.line(image, (int(x + w), int(y + h)), (int(x + w), y), color=(0, 255, 0), thickness=5)

                    for keypoint in keypoints:
                        x, y, _ = keypoint
                        cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 255), thickness=5)

                    break

    if show:
        cv2.imshow('stitched data', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # print('length of matched_results_densepose:', len(matched_results_densepose))
    # print('length of matched_boxes_xywh:', len(matched_boxes_xywh))
    # print('length of matched_data_keypoints:', len(matched_data_keypoints))

    return matched_results_densepose, matched_boxes_xywh, matched_data_keypoints


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
    file_keypoints = os.path.join(openpose_keypoints_dir, '{}_keypoints.npy'.format(infile[infile.find('/') + 1:infile.rfind('.')]))
    data_keypoints = np.load(file_keypoints, allow_pickle='TRUE').item()['keypoints']

    # stitch DensePose segments with OpenPose keypoints!
    matched_results_densepose, matched_boxes_xywh, matched_data_keypoints = stitch_data(results_densepose, boxes_xywh, data_keypoints, im_gray, show=show)

    for result_densepose, box_xywh, keypoints in zip(matched_results_densepose, matched_boxes_xywh, matched_data_keypoints):

        # extract segm + mask
        mask, segm = extract_segm(result_densepose=result_densepose)

        # visualizer
        visualize_norm_segm(image_bgr=im_gray, mask=mask, segm=segm, bbox_xywh=box_xywh, keypoints=keypoints, infile=infile, show=show)


def generate_norm_segm_outfile(infile):

    outdir = os.path.join(norm_segm_dir, infile[infile.find('/') + 1:infile.rfind('/')])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = infile[infile.find('/') + 1:infile.rfind('.')]
    outfile = os.path.join(norm_segm_dir, '{}_norm.jpg'.format(fname))

    return outfile


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
        outfile = generate_segm_outfile(infile)
        cv2.imwrite(outfile, image_vis)
        print('output:', outfile)


def generate_segm_outfile(infile):

    outdir = os.path.join(densepose_keypoints_dir, infile[infile.find('/') + 1:infile.rfind('/')])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = infile[infile.find('/') + 1:infile.rfind('.')]
    outfile = os.path.join(densepose_keypoints_dir, '{}_segm.jpg'.format(fname))

    return outfile


if __name__ == '__main__':

    # python infer_segm.py --input datasets/classical
    # python infer_segm.py --input datasets/modern

    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/80019.jpg
    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/81903.jpg
    # buggy
    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/25239.jpg
    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/16338.jpg

    parser = argparse.ArgumentParser(description='DensePose - Infer the segments')
    parser.add_argument('--input', help='Path to image file or directory')
    args = parser.parse_args()

    if os.path.isfile(args.input):
        # generate_segm(infile=args.input, score_cutoff=0.95, show=True)
        generate_norm_segm(infile=args.input, score_cutoff=0.95, show=True)

    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.jpg'):
            try:
                # generate_segm(infile=str(path), score_cutoff=0.9, show=False)
                generate_norm_segm(infile=args.input, score_cutoff=0.95, show=False)
            except:
                continue
    else:
        pass