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
import collections
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


# setting
gray_val_scale = 10.625
cmap = cv2.COLORMAP_PARULA

norm_img_shape = (2000, 2000, 3)

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

COARSE_ID = [
    'Background',
    'Torso',
    'RHand', 'LHand', 'LFoot', 'RFoot',
    'RThigh', 'LThigh', 'RCalf', 'LCalf',
    'LUpperArm', 'RUpperArm', 'LLowerArm', 'RLowerArm',
    'Head'
]

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
        return [int(x), int(y), point[2]] # for keypoints with score
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


def is_valid(keypoints):

    # check the scores for each main keypoint, which MUST exist!
    # main_keypoints = BODY BOX
    main_keypoints = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'MidHip']

    keypoints = dict(zip(JOINT_ID, keypoints))

    # filter the main keypoints by score = 0
    filtered_keypoints = [key for key, value in keypoints.items() if key in main_keypoints and value[2] == 0]

    if len(filtered_keypoints) != 0:
        return False
    else:
        return True


def get_segments_xy(segm, keypoints):

    segments_xy = []

    bg_xy = [] # 0
    segments_xy.append(bg_xy)

    torso_xy = _segm_xy(segm=segm, segm_id_list=[1])
    segments_xy.append(torso_xy)

    r_hand_xy = [] # 2
    l_hand_xy = [] # 3
    l_foot_xy = [] # 4
    r_foot_xy = [] # 5
    segments_xy.append(r_hand_xy)
    segments_xy.append(l_hand_xy)
    segments_xy.append(l_foot_xy)
    segments_xy.append(r_foot_xy)

    r_thigh_xy = _segm_xy(segm=segm, segm_id_list=[6])
    l_thigh_xy = _segm_xy(segm=segm, segm_id_list=[7])
    r_calf_xy = _segm_xy(segm=segm, segm_id_list=[8])
    l_calf_xy = _segm_xy(segm=segm, segm_id_list=[9])
    segments_xy.append(r_thigh_xy)
    segments_xy.append(l_thigh_xy)
    segments_xy.append(r_calf_xy)
    segments_xy.append(l_calf_xy)

    l_upper_arm_xy = _segm_xy(segm=segm, segm_id_list=[10])
    r_upper_arm_xy = _segm_xy(segm=segm, segm_id_list=[11])
    l_lower_arm_xy = _segm_xy(segm=segm, segm_id_list=[12])
    r_lower_arm_xy = _segm_xy(segm=segm, segm_id_list=[13])
    segments_xy.append(l_upper_arm_xy)
    segments_xy.append(r_upper_arm_xy)
    segments_xy.append(l_lower_arm_xy)
    segments_xy.append(r_lower_arm_xy)

    head_xy = _segm_xy(segm=segm, segm_id_list=[14])
    segments_xy.append(head_xy)

    # valid segments with keypoints
    dict_segments_xy = dict(zip(COARSE_ID, segments_xy))
    segments_xy = {}

    # head
    if len(dict_segments_xy['Head']) > 0 and keypoints['Nose'][2] > 0:
        segments_xy['Head'] = {'segm_xy': dict_segments_xy['Head'],
                               'keypoints':
                                   {'Nose': keypoints['Nose']}
                               }

    # torso
    if len(dict_segments_xy['Torso']) > 0:
        segments_xy['Torso'] = {'segm_xy': dict_segments_xy['Torso'],
                                'keypoints':
                                    {'Neck': keypoints['Neck'],
                                     'RShoulder': keypoints['RShoulder'],
                                     'LShoulder': keypoints['LShoulder'],
                                     'MidHip': keypoints['MidHip'],
                                     'RHip': keypoints['RHip'],
                                     'LHip': keypoints['LHip']}
                                }

    # upper limbs
    if len(dict_segments_xy['RThigh']) > 0 and keypoints['RKnee'][2] > 0:
        segments_xy['RThigh'] = {'segm_xy': dict_segments_xy['RThigh'],
                                 'keypoints':
                                     {'RKnee': keypoints['RKnee']}
                                 }

    if len(dict_segments_xy['LThigh']) > 0 and keypoints['LKnee'][2] > 0:
        segments_xy['LThigh'] = {'segm_xy': dict_segments_xy['LThigh'],
                                 'keypoints':
                                     {'LKnee': keypoints['LKnee']}
                                 }

    if len(dict_segments_xy['RCalf']) > 0 and keypoints['RAnkle'][2] > 0:
        segments_xy['RCalf'] = {'segm_xy': dict_segments_xy['RCalf'],
                                'keypoints':
                                    {'RAnkle': keypoints['RAnkle']}
                                }

    if len(dict_segments_xy['LCalf']) > 0 and keypoints['LAnkle'][2] > 0:
        segments_xy['LCalf'] = {'segm_xy': dict_segments_xy['LCalf'],
                                'keypoints':
                                    {'LAnkle': keypoints['LAnkle']}
                                }

    # lower limbs
    if len(dict_segments_xy['RUpperArm']) > 0 and keypoints['RElbow'][2] > 0:
        segments_xy['RUpperArm'] = {'segm_xy': dict_segments_xy['RUpperArm'],
                                    'keypoints':
                                        {'RElbow': keypoints['RElbow']}
                                    }

    if len(dict_segments_xy['LUpperArm']) > 0 and keypoints['LElbow'][2] > 0:
        segments_xy['LUpperArm'] = {'segm_xy': dict_segments_xy['LUpperArm'],
                                    'keypoints':
                                        {'LElbow': keypoints['LElbow']}
                                    }

    if len(dict_segments_xy['RLowerArm']) > 0 and keypoints['RWrist'][2] > 0:
        segments_xy['RLowerArm'] = {'segm_xy': dict_segments_xy['RLowerArm'],
                                    'keypoints':
                                        {'RWrist': keypoints['RWrist']}
                                    }

    if len(dict_segments_xy['LLowerArm']) > 0 and keypoints['LWrist'][2] > 0:
        segments_xy['LLowerArm'] = {'segm_xy': dict_segments_xy['LLowerArm'],
                                    'keypoints':
                                        {'LWrist': keypoints['LWrist']}
                                    }

    return segments_xy


def _rotate_to_vertical_pose(segments_xy):

    midhip_keypoint = segments_xy['Torso']['keypoints']['MidHip']
    neck_keypoint = segments_xy['Torso']['keypoints']['Neck']

    # calculate the angle for rotation to vertical pose
    reference_point = np.array(midhip_keypoint) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=neck_keypoint, center=midhip_keypoint, point2=reference_point)

    for segment_id, segment in segments_xy.items():
        segments_xy[segment_id]['segm_xy'] = np.array([_rotate((x, y), midhip_keypoint, rad) for (x, y) in segment['segm_xy']])

        for keypoints_id, keypoints in segment['keypoints'].items():
            segments_xy[segment_id]['keypoints'][keypoints_id] = _rotate(keypoints, midhip_keypoint, rad)

    return segments_xy


def _rotate_to_tpose(segments_xy):

    # nose -> head (BUT nose is not at the middle point of face, e.g., face right, face left!!!)
    # midhip -> torso (DONE in vertical rotation)
    # elbow -> upper arm
    # wrist -> lower arm
    # knee -> thigh
    # ankle -> calf

    # valid keypoints confirmed by is_valid()
    nose_keypoint = segments_xy['Head']['keypoints']['Nose']

    neck_keypoint = segments_xy['Torso']['keypoints']['Neck']
    rsho_keypoint = segments_xy['Torso']['keypoints']['RShoulder']
    lsho_keypoint = segments_xy['Torso']['keypoints']['LShoulder']

    midhip_keypoint = segments_xy['Torso']['keypoints']['MidHip']
    rhip_keypoint = segments_xy['Torso']['keypoints']['RHip']
    lhip_keypoint = segments_xy['Torso']['keypoints']['LHip']

    # update midhip keypoint = mid torso point
    if 'Torso' in segments_xy and len(segments_xy['Torso']['segm_xy']) > 0:
        neck_keypoint = np.array(segments_xy['Torso']['keypoints']['Neck'])[0:2]
        midhip_keypoint = np.array(segments_xy['Torso']['keypoints']['MidHip'])[0:2]
        segments_xy['Torso']['keypoints']['MidHip'] = ((neck_keypoint + midhip_keypoint)/2).astype(int)

    # head -> NOT use Nose, use Centroid of head_xy!!!
    # ONE solution to Issue FOUR: NOSE is not at the middle point of the head!!!
    # so nose keypoint = head centroid
    if 'Head' in segments_xy and len(segments_xy['Head']['segm_xy']) > 0:

        head_xy = segments_xy['Head']['segm_xy']
        head_centroid = _segments_xy_centroid(head_xy)

        rad, deg = _calc_angle(head_centroid, neck_keypoint, midhip_keypoint)
        rad = rad + np.pi

        segments_xy['Head']['segm_xy'] = np.array([_rotate([x, y], neck_keypoint, rad) for (x, y) in head_xy])
        segments_xy['Head']['keypoints']['Nose'] = _rotate(head_centroid, neck_keypoint, rad)

    # Upper Limb
    # Right
    # wrist keypoint = lower arm midpoint
    if 'RLowerArm' in segments_xy and 'RUpperArm' in segments_xy and len(segments_xy['RLowerArm']['segm_xy']) > 0 and segments_xy['RLowerArm']['keypoints']['RWrist'][2] > 0 and segments_xy['RUpperArm']['keypoints']['RElbow'][2] > 0:

        rlower_arm_xy = segments_xy['RLowerArm']['segm_xy']
        rwrist_keypoint = segments_xy['RLowerArm']['keypoints']['RWrist']
        relb_keypoint = segments_xy['RUpperArm']['keypoints']['RElbow']
        mid_rlower_arm_keypoint = ((np.array(relb_keypoint) + np.array(rwrist_keypoint)) / 2).astype(int)

        # rotate to horizontal
        mid_rlower_arm_ref = mid_rlower_arm_keypoint + np.array([50, 0, 0])
        rad, deg = _calc_angle(relb_keypoint, mid_rlower_arm_keypoint, mid_rlower_arm_ref)

        segments_xy['RLowerArm']['segm_xy'] = np.array([_rotate([x, y], mid_rlower_arm_keypoint, rad) for (x, y) in rlower_arm_xy])
        segments_xy['RLowerArm']['keypoints']['RWrist'] = mid_rlower_arm_keypoint

    # elbow keypoint = upper arm midpoint
    if 'RUpperArm' in segments_xy and len(segments_xy['RUpperArm']['segm_xy']) > 0 and segments_xy['RUpperArm']['keypoints']['RElbow'][2] > 0:

        rupper_arm_xy = segments_xy['RUpperArm']['segm_xy']
        relb_keypoint = segments_xy['RUpperArm']['keypoints']['RElbow']
        mid_rupper_arm_keypoint = ((np.array(rsho_keypoint) + np.array(relb_keypoint))/2).astype(int)

        # rotate to horizontal
        mid_rupper_arm_ref = mid_rupper_arm_keypoint + np.array([50, 0, 0])
        rad, deg = _calc_angle(rsho_keypoint, mid_rupper_arm_keypoint, mid_rupper_arm_ref)

        segments_xy['RUpperArm']['segm_xy'] = np.array([_rotate([x, y], mid_rupper_arm_keypoint, rad) for (x, y) in rupper_arm_xy])
        segments_xy['RUpperArm']['keypoints']['RElbow'] = mid_rupper_arm_keypoint

    # Left
    # wrist keypoint = lower arm midpoint
    if 'LLowerArm' in segments_xy and 'LUpperArm' in segments_xy and len(segments_xy['LLowerArm']['segm_xy']) > 0 and segments_xy['LLowerArm']['keypoints']['LWrist'][2] > 0 and segments_xy['LUpperArm']['keypoints']['LElbow'][2] > 0:
        llower_arm_xy = segments_xy['LLowerArm']['segm_xy']
        lwrist_keypoint = segments_xy['LLowerArm']['keypoints']['LWrist']
        lelb_keypoint = segments_xy['LUpperArm']['keypoints']['LElbow']
        mid_llower_arm_keypoint = ((np.array(lelb_keypoint) + np.array(lwrist_keypoint)) / 2).astype(int)

        # rotate to horizontal
        mid_llower_arm_ref = mid_llower_arm_keypoint + np.array([50, 0, 0])
        rad, deg = _calc_angle(lwrist_keypoint, mid_llower_arm_keypoint, mid_llower_arm_ref)

        segments_xy['LLowerArm']['segm_xy'] = np.array([_rotate([x, y], mid_llower_arm_keypoint, rad) for (x, y) in llower_arm_xy])
        segments_xy['LLowerArm']['keypoints']['LWrist'] = mid_llower_arm_keypoint

    # elbow keypoint = upper arm midpoint
    if 'LUpperArm' in segments_xy and len(segments_xy['LUpperArm']['segm_xy']) > 0 and segments_xy['LUpperArm']['keypoints']['LElbow'][2] > 0:
        lupper_arm_xy = segments_xy['LUpperArm']['segm_xy']
        lelb_keypoint = segments_xy['LUpperArm']['keypoints']['LElbow']
        mid_lupper_arm_keypoint = ((np.array(lsho_keypoint) + np.array(lelb_keypoint)) / 2).astype(int)

        # rotate to horizontal
        mid_lupper_arm_ref = mid_lupper_arm_keypoint + np.array([50, 0, 0])
        rad, deg = _calc_angle(lelb_keypoint, mid_lupper_arm_keypoint, mid_lupper_arm_ref)

        segments_xy['LUpperArm']['segm_xy'] = np.array([_rotate([x, y], mid_lupper_arm_keypoint, rad) for (x, y) in lupper_arm_xy])
        segments_xy['LUpperArm']['keypoints']['LElbow'] = mid_lupper_arm_keypoint

    # Lower Limb
    # Right
    # ankle keypoint = calf midpoint
    if 'RCalf' in segments_xy and 'RThigh' in segments_xy and len(segments_xy['RCalf']['segm_xy']) > 0 and segments_xy['RCalf']['keypoints']['RAnkle'][2] > 0 and segments_xy['RThigh']['keypoints']['RKnee'][2] > 0:

        rcalf_xy = segments_xy['RCalf']['segm_xy']
        rankle_keypoint = segments_xy['RCalf']['keypoints']['RAnkle']
        rknee_keypoint = segments_xy['RThigh']['keypoints']['RKnee']
        mid_rcalf_keypoint = ((np.array(rknee_keypoint) + np.array(rankle_keypoint)) / 2).astype(int)

        # rotate to horizontal
        mid_rcalf_ref = mid_rcalf_keypoint + np.array([50, 0, 0])
        rad, deg = _calc_angle(rankle_keypoint, mid_rcalf_keypoint, mid_rcalf_ref)
        rad = rad + np.pi/2

        segments_xy['RCalf']['segm_xy'] = np.array([_rotate([x, y], mid_rcalf_keypoint, rad) for (x, y) in rcalf_xy])
        segments_xy['RCalf']['keypoints']['RAnkle'] = mid_rcalf_keypoint

    # knee keypoint = thigh midpoint
    if 'RThigh' in segments_xy and len(segments_xy['RThigh']['segm_xy']) > 0 and segments_xy['RThigh']['keypoints']['RKnee'][2] > 0:

        rthigh_xy = segments_xy['RThigh']['segm_xy']
        rknee_keypoint = segments_xy['RThigh']['keypoints']['RKnee']
        mid_rthigh_keypoint = ((np.array(rhip_keypoint) + np.array(rknee_keypoint))/2).astype(int)

        # rotate to horizontal
        mid_rthigh_ref = mid_rthigh_keypoint + np.array([50, 0, 0])
        rad, deg = _calc_angle(rknee_keypoint, mid_rthigh_keypoint, mid_rthigh_ref)
        rad = rad + np.pi/2

        segments_xy['RThigh']['segm_xy'] = np.array([_rotate([x, y], mid_rthigh_keypoint, rad) for (x, y) in rthigh_xy])
        segments_xy['RThigh']['keypoints']['RKnee'] = mid_rthigh_keypoint

    # Left
    # ankle keypoint = calf midpoint
    if 'LCalf' in segments_xy and 'LThigh' in segments_xy and len(segments_xy['LCalf']['segm_xy']) > 0 and segments_xy['LCalf']['keypoints']['LAnkle'][2] > 0 and segments_xy['LThigh']['keypoints']['LKnee'][2] > 0:

        lcalf_xy = segments_xy['LCalf']['segm_xy']
        lankle_keypoint = segments_xy['LCalf']['keypoints']['LAnkle']
        lknee_keypoint = segments_xy['LThigh']['keypoints']['LKnee']
        mid_lcalf_keypoint = ((np.array(lknee_keypoint) + np.array(lankle_keypoint)) / 2).astype(int)

        # rotate to horizontal
        mid_lcalf_ref = mid_lcalf_keypoint + np.array([50, 0, 0])
        rad, deg = _calc_angle(lankle_keypoint, mid_lcalf_keypoint, mid_lcalf_ref)
        rad = rad + np.pi/2

        segments_xy['LCalf']['segm_xy'] = np.array([_rotate([x, y], mid_lcalf_keypoint, rad) for (x, y) in lcalf_xy])
        segments_xy['LCalf']['keypoints']['LAnkle'] = mid_lcalf_keypoint

    # knee keypoint = thigh midpoint
    if 'LThigh' in segments_xy and len(segments_xy['LThigh']['segm_xy']) > 0 and segments_xy['LThigh']['keypoints']['LKnee'][2] > 0:

        lthigh_xy = segments_xy['LThigh']['segm_xy']
        lknee_keypoint = segments_xy['LThigh']['keypoints']['LKnee']
        mid_lthigh_keypoint = ((np.array(lhip_keypoint) + np.array(lknee_keypoint))/2).astype(int)

        # rotate to horizontal
        mid_lthigh_ref = mid_lthigh_keypoint + np.array([50, 0, 0])
        rad, deg = _calc_angle(lknee_keypoint, mid_lthigh_keypoint, mid_lthigh_ref)
        rad = rad + np.pi/2

        segments_xy['LThigh']['segm_xy'] = np.array([_rotate([x, y], mid_lthigh_keypoint, rad) for (x, y) in lthigh_xy])
        segments_xy['LThigh']['keypoints']['LKnee'] = mid_lthigh_keypoint

    return segments_xy


def rotate_segments_xy(segm, keypoints):

    # Issue ONE: cannot rotate body to [Face-front + Torso-front] view!!!
    # Issue TWO: cannot have the same person -> so it can be a fat person or a thin person!!!
    # Issue THREE: NO mapped HAND and FOOT keypoints to rotate them!!!
    # *Issue FOUR*: NOSE is not at the middle point of the head, e.g., face right, face left, so cannot normalize HEAD!!!

    # STEP 1: rotated any pose to a vertical pose, i.e., stand up, sit up, etc...
    # extract original segment's x, y
    segments_xy = get_segments_xy(segm=segm, keypoints=keypoints)

    # rotated segment to vertical pose, i.e., stand up, sit up, etc...
    vertical_segments_xy = _rotate_to_vertical_pose(segments_xy=segments_xy)

    # STEP 2: rotate specific segment further to t-pose
    tpose_segments_xy = _rotate_to_tpose(segments_xy=vertical_segments_xy)

    return tpose_segments_xy


def _translate_and_scale_segm(image, segm_id, segm_xy, keypoint, ref_point, scaler):

    # translate
    diff_xy = np.array(ref_point) - np.array(keypoint)[0:2]
    segm_xy = np.array([np.array(xy) + np.array(diff_xy) for xy in segm_xy])

    # draw on background image
    min_x, min_y = np.min(segm_xy, axis=0).astype(int)
    max_x, max_y = np.max(segm_xy, axis=0).astype(int)

    img_bg = np.empty(norm_img_shape, np.uint8)
    img_bg.fill(255)

    for x, y in segm_xy.astype(int):
        cv2.circle(img_bg, (x, y), radius=5, color=COARSE_TO_COLOR[segm_id], thickness=-1)

    # crop
    img_bg = img_bg[min_y:max_y, min_x:max_x]
    h, w, _ = img_bg.shape

    if segm_id == 'Head':
        scaler = 250 / w

    # resize
    img_bg = cv2.resize(img_bg, (int(w * scaler), int(h * scaler)), cv2.INTER_LINEAR)
    h, w, _ = img_bg.shape

    x, y = ref_point
    min_x = int(x - w / 2)
    max_x = int(x + w / 2)
    min_y = int(y - h / 2)
    max_y = int(y + h / 2)

    image[min_y:max_y, min_x:max_x] = img_bg

    if segm_id == 'Head':
        return scaler


def draw_segments_xy(segments_xy):

    # normalized image
    image = np.empty(norm_img_shape, np.uint8)
    image.fill(255)  # => white (255, 255, 255) = background

    norm_nose_xy = [1000, 250]

    norm_mid_torso_xy = [1000, 750]

    norm_mid_rupper_arm_xy = [600, 500]
    norm_mid_rlower_arm_xy = [300, 500]
    norm_mid_lupper_arm_xy = [1400, 500]
    norm_mid_llower_arm_xy = [1700, 500]

    norm_mid_rthigh_xy = [900, 1200]
    norm_mid_rcalf_xy = [900, 1600]
    norm_mid_lthigh_xy = [1100, 1200]
    norm_mid_lcalf_xy = [1100, 1600]

    scaler = None # Assumption!!! Size of head for all people is the same!!!

    # translate first, scale second!
    # head
    if 'Head' in segments_xy:
        scaler = _translate_and_scale_segm(image=image,
                                           segm_id='Head', segm_xy=segments_xy['Head']['segm_xy'],
                                           keypoint=segments_xy['Head']['keypoints']['Nose'],
                                           ref_point=norm_nose_xy,
                                           scaler=None)

    # torso
    if 'Torso' in segments_xy:
        _translate_and_scale_segm(image=image,
                                  segm_id='Torso',
                                  segm_xy=segments_xy['Torso']['segm_xy'],
                                  keypoint=segments_xy['Torso']['keypoints']['MidHip'],
                                  ref_point=norm_mid_torso_xy,
                                  scaler=scaler)

    # upper limbs
    if 'RUpperArm' in segments_xy:
        _translate_and_scale_segm(image=image,
                                  segm_id='RUpperArm',
                                  segm_xy=segments_xy['RUpperArm']['segm_xy'],
                                  keypoint=segments_xy['RUpperArm']['keypoints']['RElbow'],
                                  ref_point=norm_mid_rupper_arm_xy,
                                  scaler=scaler)

    if 'RLowerArm' in segments_xy:
        _translate_and_scale_segm(image=image,
                                  segm_id='RLowerArm',
                                  segm_xy=segments_xy['RLowerArm']['segm_xy'],
                                  keypoint=segments_xy['RLowerArm']['keypoints']['RWrist'],
                                  ref_point=norm_mid_rlower_arm_xy,
                                  scaler=scaler)

    if 'LUpperArm' in segments_xy:
        _translate_and_scale_segm(image=image,
                                  segm_id='LUpperArm',
                                  segm_xy=segments_xy['LUpperArm']['segm_xy'],
                                  keypoint=segments_xy['LUpperArm']['keypoints']['LElbow'],
                                  ref_point=norm_mid_lupper_arm_xy,
                                  scaler=scaler)

    if 'LLowerArm' in segments_xy:
        _translate_and_scale_segm(image=image,
                                  segm_id='LLowerArm',
                                  segm_xy=segments_xy['LLowerArm']['segm_xy'],
                                  keypoint=segments_xy['LLowerArm']['keypoints']['LWrist'],
                                  ref_point=norm_mid_llower_arm_xy,
                                  scaler=scaler)

    # lower limbs
    if 'RThigh' in segments_xy:
        _translate_and_scale_segm(image=image,
                                  segm_id='RThigh',
                                  segm_xy=segments_xy['RThigh']['segm_xy'],
                                  keypoint=segments_xy['RThigh']['keypoints']['RKnee'],
                                  ref_point=norm_mid_rthigh_xy,
                                  scaler=scaler)

    if 'RCalf' in segments_xy:
        _translate_and_scale_segm(image=image,
                                  segm_id='RCalf',
                                  segm_xy=segments_xy['RCalf']['segm_xy'],
                                  keypoint=segments_xy['RCalf']['keypoints']['RAnkle'],
                                  ref_point=norm_mid_rcalf_xy,
                                  scaler=scaler)

    if 'LThigh' in segments_xy:
        _translate_and_scale_segm(image=image,
                                  segm_id='LThigh',
                                  segm_xy=segments_xy['LThigh']['segm_xy'],
                                  keypoint=segments_xy['LThigh']['keypoints']['LKnee'],
                                  ref_point=norm_mid_lthigh_xy,
                                  scaler=scaler)

    if 'LCalf' in segments_xy:
        _translate_and_scale_segm(image=image,
                                  segm_id='LCalf',
                                  segm_xy=segments_xy['LCalf']['segm_xy'],
                                  keypoint=segments_xy['LCalf']['keypoints']['LAnkle'],
                                  ref_point=norm_mid_lcalf_xy,
                                  scaler=scaler)

    # draw centers
    # head center
    cv2.circle(image, tuple(norm_nose_xy), radius=10, color=(255, 0, 255), thickness=-1)

    # torso center
    cv2.circle(image, tuple(norm_mid_torso_xy), radius=10, color=(255, 0, 255), thickness=-1)

    # upper limbs
    cv2.circle(image, tuple(norm_mid_lupper_arm_xy), radius=10, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_rlower_arm_xy), radius=10, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_lupper_arm_xy), radius=10, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_llower_arm_xy), radius=10, color=(255, 0, 255), thickness=-1)

    # lower limbs
    cv2.circle(image, tuple(norm_mid_rthigh_xy), radius=10, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_rcalf_xy), radius=10, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_lthigh_xy), radius=10, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_lcalf_xy), radius=10, color=(255, 0, 255), thickness=-1)

    cv2.imshow('test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_norm_segm(image_bg, mask, segm, bbox_xywh, keypoints, infile, show=False):

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
    segments_xy = rotate_segments_xy(segm=segm, keypoints=keypoints)

    # draw segments in normalized image
    draw_segments_xy(segments_xy=segments_xy)

    # if show:
    #     cv2.imshow('norm', resized_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     outfile = generate_norm_segm_outfile(infile)
    #     cv2.imwrite(outfile, resized_image)
    #     print('output', outfile)


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

        # condition: valid body box!
        if is_valid(keypoints):
            # extract segm + mask
            mask, segm = extract_segm(result_densepose=result_densepose)

            # visualizer
            visualize_norm_segm(image_bg=im_gray, mask=mask, segm=segm, bbox_xywh=box_xywh, keypoints=keypoints, infile=infile, show=show)
        else:
            continue


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