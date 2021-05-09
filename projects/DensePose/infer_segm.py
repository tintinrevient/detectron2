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
from scipy import ndimage
from scipy.ndimage.interpolation import rotate
from scipy.spatial import ConvexHull

# window setting
window_segm = 'segm'
window_bbox = 'bbox'
window_norm = 'norm'
window_dilation = 'dilation'
window_stitched_data = 'stitched data'

# setting
gray_val_scale = 10.625
cmap = cv2.COLORMAP_PARULA

norm_img_shape = (2000, 2000, 4)

keypoints_radius = 5
keypoints_color = (0, 0, 0)

# files of config
densepose_keypoints_dir = os.path.join('output', 'keypoints')
openpose_keypoints_dir = os.path.join('output', 'data')
norm_segm_dir = os.path.join('output', 'pix')

fname_vitruve_norm = os.path.join('pix', 'vitruve_norm.png')

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
# BGRA -> alpha channel: 0 = transparent, 255 = non-transparent
COARSE_TO_COLOR = {
    'Background': [255, 255, 255, 255],
    'Torso': [191, 78, 22, 255],
    'RThigh': [167, 181, 44, 255],
    'LThigh': [141, 187, 91, 255],
    'RCalf': [114, 191, 147, 255],
    'LCalf': [96, 188, 192, 255],
    'LUpperArm': [87, 207, 112, 255],
    'RUpperArm': [55, 218, 162, 255],
    'LLowerArm': [25, 226, 216, 255],
    'RLowerArm': [37, 231, 253, 255],
    'Head': [14, 251, 249, 255]
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

    # print(point)

    x = ((point[0] - center[0]) * np.cos(rad)) - ((point[1] - center[1]) * np.sin(rad)) + center[0]
    y = ((point[0] - center[0]) * np.sin(rad)) + ((point[1] - center[1]) * np.cos(rad)) + center[1]

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


def _keypoints_midpoint(keypoint1, keypoint2):

    return ((np.array(keypoint1) + np.array(keypoint2)) / 2).astype(int)


def is_valid(keypoints):

    # check the scores for each main keypoint, which MUST exist!
    # main_keypoints = BODY BOX
    main_keypoints = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'MidHip']

    keypoints = dict(zip(JOINT_ID, keypoints))

    # filter the main keypoints by score > 0
    filtered_keypoints = [key for key, value in keypoints.items() if key in main_keypoints and value[2] > 0]
    print('Number of valid keypoints (must be equal to 7):', len(filtered_keypoints))

    if len(filtered_keypoints) != 7:
        return False
    else:
        return True


def _get_segments_xy(segm, keypoints):

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


def _rotate_head_around_centroid(segm_xy, keypoint1_ref, keypoint2_ref):

    # midpoint of vertical line and horizontal line
    centroid = _segments_xy_centroid(segm_xy)

    rad, deg = _calc_angle(centroid, keypoint1_ref, keypoint2_ref)
    rad += np.pi

    segm_xy = np.array([_rotate([x, y], keypoint1_ref, rad) for (x, y) in segm_xy])
    keypoint = _rotate(centroid, keypoint1_ref, rad)

    return segm_xy, keypoint


def _rotate_limbs_around_midpoint(segm_xy, keypoint, ref_keypoint, is_right, is_leg):

    # mid-keypoint
    midpoint = _keypoints_midpoint(keypoint1=keypoint, keypoint2=ref_keypoint)

    # rotate to horizontal
    ref_midpoint = midpoint + np.array([50, 0, 0])

    if is_right:
        rad, deg = _calc_angle(ref_keypoint, midpoint, ref_midpoint)

        if is_leg:
            rad -= np.pi/2

    else:
        rad, deg = _calc_angle(keypoint, midpoint, ref_midpoint)

        if is_leg:
            rad += np.pi / 2

    segm_xy = np.array([_rotate([x, y], midpoint, rad) for (x, y) in segm_xy])
    keypoint = midpoint

    return segm_xy, keypoint


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


    # update midhip keypoint = (midhip + neck) / 2
    if 'Torso' in segments_xy and len(segments_xy['Torso']['segm_xy']) > 0:
        segments_xy['Torso']['keypoints']['MidHip'] =  _keypoints_midpoint(neck_keypoint, midhip_keypoint)


    # head -> NOT use Nose, use Centroid of head_xy!!!
    # ONE solution to Issue FOUR: NOSE is not at the middle point of the head!!!
    # so nose keypoint = head centroid
    if 'Head' in segments_xy and len(segments_xy['Head']['segm_xy']) > 0:

        segm_xy, keypoint = _rotate_head_around_centroid(segm_xy=segments_xy['Head']['segm_xy'],
                                                         keypoint1_ref=neck_keypoint,
                                                         keypoint2_ref=midhip_keypoint)

        segments_xy['Head']['segm_xy'] = segm_xy
        segments_xy['Head']['keypoints']['Nose'] = keypoint


    # Upper Limb
    # Right
    # wrist keypoint = lower arm midpoint
    if 'RLowerArm' in segments_xy and 'RUpperArm' in segments_xy and len(segments_xy['RLowerArm']['segm_xy']) > 0 and segments_xy['RLowerArm']['keypoints']['RWrist'][2] > 0 and segments_xy['RUpperArm']['keypoints']['RElbow'][2] > 0:

        segm_xy, keypoint = _rotate_limbs_around_midpoint(segm_xy=segments_xy['RLowerArm']['segm_xy'],
                                                          keypoint=segments_xy['RLowerArm']['keypoints']['RWrist'],
                                                          ref_keypoint=segments_xy['RUpperArm']['keypoints']['RElbow'],
                                                          is_right=True,
                                                          is_leg=False)

        segments_xy['RLowerArm']['segm_xy'] = segm_xy
        segments_xy['RLowerArm']['keypoints']['RWrist'] = keypoint


    # elbow keypoint = upper arm midpoint
    if 'RUpperArm' in segments_xy and len(segments_xy['RUpperArm']['segm_xy']) > 0 and segments_xy['RUpperArm']['keypoints']['RElbow'][2] > 0:

        segm_xy, keypoint = _rotate_limbs_around_midpoint(segm_xy=segments_xy['RUpperArm']['segm_xy'],
                                                          keypoint=segments_xy['RUpperArm']['keypoints']['RElbow'],
                                                          ref_keypoint=rsho_keypoint,
                                                          is_right=True,
                                                          is_leg=False)

        segments_xy['RUpperArm']['segm_xy'] = segm_xy
        segments_xy['RUpperArm']['keypoints']['RElbow'] = keypoint


    # Left
    # wrist keypoint = lower arm midpoint
    if 'LLowerArm' in segments_xy and 'LUpperArm' in segments_xy and len(segments_xy['LLowerArm']['segm_xy']) > 0 and segments_xy['LLowerArm']['keypoints']['LWrist'][2] > 0 and segments_xy['LUpperArm']['keypoints']['LElbow'][2] > 0:

        segm_xy, keypoint = _rotate_limbs_around_midpoint(segm_xy=segments_xy['LLowerArm']['segm_xy'],
                                                          keypoint=segments_xy['LLowerArm']['keypoints']['LWrist'],
                                                          ref_keypoint=segments_xy['LUpperArm']['keypoints']['LElbow'],
                                                          is_right=False,
                                                          is_leg=False)

        segments_xy['LLowerArm']['segm_xy'] = segm_xy
        segments_xy['LLowerArm']['keypoints']['LWrist'] = keypoint


    # elbow keypoint = upper arm midpoint
    if 'LUpperArm' in segments_xy and len(segments_xy['LUpperArm']['segm_xy']) > 0 and segments_xy['LUpperArm']['keypoints']['LElbow'][2] > 0:

        segm_xy, keypoint = _rotate_limbs_around_midpoint(segm_xy=segments_xy['LUpperArm']['segm_xy'],
                                                          keypoint=segments_xy['LUpperArm']['keypoints']['LElbow'],
                                                          ref_keypoint=lsho_keypoint,
                                                          is_right=False,
                                                          is_leg=False)

        segments_xy['LUpperArm']['segm_xy'] = segm_xy
        segments_xy['LUpperArm']['keypoints']['LElbow'] = keypoint


    # Lower Limb
    # Right
    # ankle keypoint = calf midpoint
    if 'RCalf' in segments_xy and 'RThigh' in segments_xy and len(segments_xy['RCalf']['segm_xy']) > 0 and segments_xy['RCalf']['keypoints']['RAnkle'][2] > 0 and segments_xy['RThigh']['keypoints']['RKnee'][2] > 0:

        segm_xy, keypoint = _rotate_limbs_around_midpoint(segm_xy=segments_xy['RCalf']['segm_xy'],
                                                          keypoint=segments_xy['RCalf']['keypoints']['RAnkle'],
                                                          ref_keypoint=segments_xy['RThigh']['keypoints']['RKnee'],
                                                          is_right=True,
                                                          is_leg=True)

        segments_xy['RCalf']['segm_xy'] = segm_xy
        segments_xy['RCalf']['keypoints']['RAnkle'] = keypoint


    # knee keypoint = thigh midpoint
    if 'RThigh' in segments_xy and len(segments_xy['RThigh']['segm_xy']) > 0 and segments_xy['RThigh']['keypoints']['RKnee'][2] > 0:

        segm_xy, keypoint = _rotate_limbs_around_midpoint(segm_xy=segments_xy['RThigh']['segm_xy'],
                                                          keypoint=segments_xy['RThigh']['keypoints']['RKnee'],
                                                          ref_keypoint=rhip_keypoint,
                                                          is_right=True,
                                                          is_leg=True)

        segments_xy['RThigh']['segm_xy'] = segm_xy
        segments_xy['RThigh']['keypoints']['RKnee'] = keypoint


    # Left
    # ankle keypoint = calf midpoint
    if 'LCalf' in segments_xy and 'LThigh' in segments_xy and len(segments_xy['LCalf']['segm_xy']) > 0 and segments_xy['LCalf']['keypoints']['LAnkle'][2] > 0 and segments_xy['LThigh']['keypoints']['LKnee'][2] > 0:

        segm_xy, keypoint = _rotate_limbs_around_midpoint(segm_xy=segments_xy['LCalf']['segm_xy'],
                                                          keypoint=segments_xy['LCalf']['keypoints']['LAnkle'],
                                                          ref_keypoint=segments_xy['LThigh']['keypoints']['LKnee'],
                                                          is_right=False,
                                                          is_leg=True)

        segments_xy['LCalf']['segm_xy'] = segm_xy
        segments_xy['LCalf']['keypoints']['LAnkle'] = keypoint


    # knee keypoint = thigh midpoint
    if 'LThigh' in segments_xy and len(segments_xy['LThigh']['segm_xy']) > 0 and segments_xy['LThigh']['keypoints']['LKnee'][2] > 0:

        segm_xy, keypoint = _rotate_limbs_around_midpoint(segm_xy=segments_xy['LThigh']['segm_xy'],
                                                          keypoint=segments_xy['LThigh']['keypoints']['LKnee'],
                                                          ref_keypoint=lhip_keypoint,
                                                          is_right=False,
                                                          is_leg=True)

        segments_xy['LThigh']['segm_xy'] = segm_xy
        segments_xy['LThigh']['keypoints']['LKnee'] = keypoint


    return segments_xy


def rotate_segments_xy(segm, keypoints):

    # Issue ONE: cannot rotate body to [Face-front + Torso-front] view!!!
    # Issue TWO: cannot have the same person -> so it can be a fat person or a thin person!!!
    # *Issue THREE*: NO mapped HAND and FOOT keypoints to rotate them - hands are feet are ignored in analysis!!!
    # *Issue FOUR*: NOSE is not at the middle point of the head, e.g., face right, face left, so cannot normalize HEAD!!!

    # STEP 1: rotated any pose to a vertical pose, i.e., stand up, sit up, etc...
    # extract original segment's x, y
    segments_xy = _get_segments_xy(segm=segm, keypoints=keypoints)

    # rotated segment to vertical pose, i.e., stand up, sit up, etc...
    vertical_segments_xy = _rotate_to_vertical_pose(segments_xy=segments_xy)

    # STEP 2: rotate specific segment further to t-pose
    tpose_segments_xy = _rotate_to_tpose(segments_xy=vertical_segments_xy)

    return tpose_segments_xy


def _euclidian(point1, point2):

    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


def _remove_outlier(segm_xy):

    # outlier factor
    factor = 3

    # mean of [x, y]
    xy_mean = np.mean(segm_xy, axis=0)

    # mean distance between [x, y] and mean of [x, y]
    distance_mean = np.mean([_euclidian(xy, xy_mean) for xy in segm_xy])

    # remove outliers from segm_xy
    segm_xy_without_outliers = [xy for xy in segm_xy if _euclidian(xy, xy_mean) <= distance_mean * factor]

    return segm_xy_without_outliers


def _translate_and_scale_segm_to_convex(image, segm_id, segm_xy, keypoint, ref_point, is_man, is_rect_symmetrical, segm_symmetry_dict, scaler):

    # test each segment
    # print('Segment ID:', segm_id)

    # remove outliers
    print('Before removing outliers:', len(segm_xy))
    segm_xy = np.array(_remove_outlier(segm_xy=segm_xy)).astype(int)
    print('After removing outliers:', len(segm_xy))

    min_x, min_y = np.min(segm_xy, axis=0).astype(int)
    max_x, max_y = np.max(segm_xy, axis=0).astype(int)

    margin = 5

    w = int(max_x - min_x + margin*2)
    h = int(max_y - min_y + margin*2)

    img_bg = np.empty((h, w, 4), np.uint8)
    img_bg.fill(255)
    img_bg[:, :, 3] = 0  # alpha channel = 0 -> transparent

    # fill the segment with the segment color
    contours = [[int(x - min_x + margin), int(y - min_y + margin)] for x, y in segm_xy]
    # option 1 - convex hull of [x, y]
    contours = np.array(contours, np.int32)
    cv2.fillConvexPoly(img_bg, cv2.convexHull(contours), color=COARSE_TO_COLOR[segm_id])
    # option 2 - dots on [x, y]
    # for x, y in contours:
    #     cv2.circle(img_bg, (x, y), color=COARSE_TO_COLOR[segm_id], radius=2, thickness=-2)

    # assumption: head_radius = 31 -> head_height = 31*2 = 62 -> men; 58 -> women
    if segm_id == 'Head' and h > 0:
        if is_man:
            scaler = 62 / h
        else:
            scaler = 58 / h

    img_bg = cv2.resize(img_bg, (int(w * scaler), int(h * scaler)), cv2.INTER_LINEAR)
    h, w, _ = img_bg.shape

    # midpoint [x, y] in the scaled coordinates of img_bg
    # distance between the center point and the left/upper boundaries
    midpoint_x, midpoint_y = ((np.array(keypoint)[0:2] - np.array([min_x, min_y]) + np.array([margin, margin])) * scaler).astype(int)

    x, y = ref_point
    min_x = int(x - midpoint_x)
    max_x = int(x + w - midpoint_x)
    min_y = int(y - midpoint_y)
    max_y = int(y + h - midpoint_y)

    cond_bg = img_bg[:, :, 3] > 0  # condition for already-drawn segment pixels

    try:
        image[min_y:max_y, min_x:max_x, :][cond_bg] = img_bg[cond_bg]
    except:
        if segm_id == 'Head':
            return scaler

    # test each segment
    # cv2.circle(img_bg, (midpoint_x, midpoint_y), radius=5,color=(255, 255, 0), thickness=-1)
    # cv2.imshow('test', img_bg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if segm_id == 'Head':
        return scaler, None
    else:
        return None


def _symmetrize_rect_segm(segm_id, w, h, midpoint_x, midpoint_y, segm_symmetry_dict):

    if segm_id == 'Head':
        segm_symmetry_dict['Head'] = (w, h)

    else:

        if midpoint_x < w/2:
            w = int((w - midpoint_x) * 2)
        else:
            w = int(midpoint_x * 2)

        if midpoint_y < h/2:
            h = int((h - midpoint_y) * 2)
        else:
            h = int(midpoint_y * 2)

        if segm_id == 'Torso':
            segm_symmetry_dict['Torso'] = (w, h)

        elif segm_id == 'RUpperArm':
            segm_symmetry_dict['RUpperArm'] = (w, h)

        elif segm_id == 'RLowerArm':
            segm_symmetry_dict['RLowerArm'] = (w, h)

        elif segm_id == 'LUpperArm':

            ref_w, ref_h = segm_symmetry_dict['RUpperArm']

            if w < ref_w:
                segm_symmetry_dict['LUpperArm'] = segm_symmetry_dict['RUpperArm']
            else:
                segm_symmetry_dict['LUpperArm'] = (w, h)
                segm_symmetry_dict['RUpperArm'] = (w, h)

        elif segm_id == 'LLowerArm':

            ref_w, ref_h = segm_symmetry_dict['RLowerArm']

            if w < ref_w:
                segm_symmetry_dict['LLowerArm'] = segm_symmetry_dict['RLowerArm']
            else:
                segm_symmetry_dict['LLowerArm'] = (w, h)
                segm_symmetry_dict['RLowerArm'] = (w, h)

        elif segm_id == 'RThigh':
            segm_symmetry_dict['RThigh'] = (w, h)

        elif segm_id == 'RCalf':
            segm_symmetry_dict['RCalf'] = (w, h)

        elif segm_id == 'LThigh':

            ref_w, ref_h = segm_symmetry_dict['RThigh']

            if h < ref_h:
                segm_symmetry_dict['LThigh'] = segm_symmetry_dict['RThigh']
            else:
                segm_symmetry_dict['LThigh'] = (w, h)
                segm_symmetry_dict['RThigh'] = (w, h)

        elif segm_id == 'LCalf':

            ref_w, ref_h = segm_symmetry_dict['RCalf']

            if h < ref_h:
                segm_symmetry_dict['LCalf'] = segm_symmetry_dict['RCalf']
            else:
                segm_symmetry_dict['LCalf'] = (w, h)
                segm_symmetry_dict['RCalf'] = (w, h)


def _draw_symmetrical_rect_segm(image, segm_id, w_and_h, ref_point):

    w, h = w_and_h

    img_bg = np.empty((h, w, 4), np.uint8)
    img_bg.fill(255)
    img_bg[:, :] = COARSE_TO_COLOR[segm_id]

    midpoint_x = w / 2
    midpoint_y = h / 2

    x, y = ref_point
    min_x = int(x - midpoint_x)
    max_x = int(x + midpoint_x)
    min_y = int(y - midpoint_y)
    max_y = int(y + midpoint_y)

    try:
        added_image = cv2.addWeighted(image[min_y:max_y, min_x:max_x, :], 0.1, img_bg, 0.9, 0)
        image[min_y:max_y, min_x:max_x, :] = added_image
    except:
        pass


def _translate_and_scale_segm_to_rect(image, segm_id, segm_xy, keypoint, ref_point, is_man, is_rect_symmetrical, segm_symmetry_dict, scaler):

    # test each segment
    # print('Segment ID:', segm_id)

    # remove outliers
    print('Before removing outliers:', len(segm_xy))
    segm_xy = np.array(_remove_outlier(segm_xy=segm_xy)).astype(int)
    print('After removing outliers:', len(segm_xy))

    min_x, min_y = np.min(segm_xy, axis=0).astype(int)
    max_x, max_y = np.max(segm_xy, axis=0).astype(int)

    w = int(max_x - min_x)
    h = int(max_y - min_y)

    img_bg = np.empty((h, w, 4), np.uint8)
    img_bg.fill(255)
    img_bg[:, :] = COARSE_TO_COLOR[segm_id]

    if segm_id == 'Head' and h > 0:
        if is_man:
            scaler = 62 / h
        else:
            scaler = 58 / h

    img_bg = cv2.resize(img_bg, (int(w * scaler), int(h * scaler)), cv2.INTER_LINEAR)
    h, w, _ = img_bg.shape

    # midpoint [x, y] in the scaled coordinates of img_bg
    # distance between the center point and the left/upper boundaries
    midpoint_x, midpoint_y = ((np.array(keypoint)[0:2] - np.array([min_x, min_y])) * scaler).astype(int)

    if is_rect_symmetrical:
        _symmetrize_rect_segm(segm_id=segm_id, w=w, h=h, midpoint_x=midpoint_x, midpoint_y=midpoint_y, segm_symmetry_dict=segm_symmetry_dict)

    else:
        x, y = ref_point
        min_x = int(x - midpoint_x)
        max_x = int(x + w - midpoint_x)
        min_y = int(y - midpoint_y)
        max_y = int(y + h - midpoint_y)

        try:
            image[min_y:max_y, min_x:max_x, :] = img_bg
        except:
            if segm_id == 'Head':
                return scaler

                # test each segment
                # cv2.circle(img_bg, (midpoint_x, midpoint_y), radius=5,color=(255, 255, 0), thickness=-1)
                # cv2.imshow('test', img_bg)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    if segm_id == 'Head':
        return scaler, segm_symmetry_dict
    else:
        return segm_symmetry_dict


def draw_segments_xy(segments_xy, is_vitruve, is_rect, is_man, is_rect_symmetrical):

    segm_symmetry_dict = {}

    if is_vitruve:
        # normalized image = (624, 624, 4)
        image = cv2.imread(fname_vitruve_norm, 0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

        # assumption -> default height of head = 63 pixels!
        # scaler = 63 / actual head height

    else:
        # normalized image = (624, 624, 4)
        image = np.empty((624, 624, 4), np.uint8)
        image.fill(255)  # => white (255, 255, 255, 255) = background with non-transparency

        # assumption -> default height of head = 63 pixels!
        # scaler = 63 / actual head height

    # common settings
    # coordinates [x, y] coming from dist_segm.draw_contour_on_vitruve()
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
    norm_mid_lthigh_xy = [left_leg_x, 427]
    norm_mid_rcalf_xy = [right_leg_x, 544]
    norm_mid_lcalf_xy = [left_leg_x, 544]

    # mid-point radius for keypoints
    radius = 2

    # assumption -> size of head for all people is the same!!!
    scaler = None

    dispatcher = {
        'segm_function_rect': _translate_and_scale_segm_to_rect,
        'segm_function_convex': _translate_and_scale_segm_to_convex
    }

    if is_rect:
        dispatcher['segm_function'] = dispatcher['segm_function_rect']
    else:
        dispatcher['segm_function'] = dispatcher['segm_function_convex']

    # translate first, scale second!
    # head
    if 'Head' in segments_xy:
        scaler, segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                             segm_id='Head', segm_xy=segments_xy['Head']['segm_xy'],
                                             keypoint=segments_xy['Head']['keypoints']['Nose'],
                                             ref_point=norm_nose_xy,
                                             is_man=is_man,
                                             is_rect_symmetrical=is_rect_symmetrical,
                                             segm_symmetry_dict=segm_symmetry_dict,
                                             scaler=None)

    # torso
    if 'Torso' in segments_xy:
        segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                    segm_id='Torso',
                                    segm_xy=segments_xy['Torso']['segm_xy'],
                                    keypoint=segments_xy['Torso']['keypoints']['MidHip'],
                                    ref_point=norm_mid_torso_xy,
                                    is_man=is_man,
                                    is_rect_symmetrical=is_rect_symmetrical,
                                    segm_symmetry_dict=segm_symmetry_dict,
                                    scaler=scaler)

    # upper limbs
    if 'RUpperArm' in segments_xy:
        segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                    segm_id='RUpperArm',
                                    segm_xy=segments_xy['RUpperArm']['segm_xy'],
                                    keypoint=segments_xy['RUpperArm']['keypoints']['RElbow'],
                                    ref_point=norm_mid_rupper_arm_xy,
                                    is_man=is_man,
                                    is_rect_symmetrical=is_rect_symmetrical,
                                    segm_symmetry_dict=segm_symmetry_dict,
                                    scaler=scaler)

    if 'RLowerArm' in segments_xy:
        segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                    segm_id='RLowerArm',
                                    segm_xy=segments_xy['RLowerArm']['segm_xy'],
                                    keypoint=segments_xy['RLowerArm']['keypoints']['RWrist'],
                                    ref_point=norm_mid_rlower_arm_xy,
                                    is_man=is_man,
                                    is_rect_symmetrical=is_rect_symmetrical,
                                    segm_symmetry_dict=segm_symmetry_dict,
                                    scaler=scaler)

    if 'LUpperArm' in segments_xy:
        segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                    segm_id='LUpperArm',
                                    segm_xy=segments_xy['LUpperArm']['segm_xy'],
                                    keypoint=segments_xy['LUpperArm']['keypoints']['LElbow'],
                                    ref_point=norm_mid_lupper_arm_xy,
                                    is_man=is_man,
                                    is_rect_symmetrical=is_rect_symmetrical,
                                    segm_symmetry_dict=segm_symmetry_dict,
                                    scaler=scaler)

    if 'LLowerArm' in segments_xy:
        segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                    segm_id='LLowerArm',
                                    segm_xy=segments_xy['LLowerArm']['segm_xy'],
                                    keypoint=segments_xy['LLowerArm']['keypoints']['LWrist'],
                                    ref_point=norm_mid_llower_arm_xy,
                                    is_man=is_man,
                                    is_rect_symmetrical=is_rect_symmetrical,
                                    segm_symmetry_dict=segm_symmetry_dict,
                                    scaler=scaler)

    # lower limbs
    if 'RThigh' in segments_xy:
        segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                    segm_id='RThigh',
                                    segm_xy=segments_xy['RThigh']['segm_xy'],
                                    keypoint=segments_xy['RThigh']['keypoints']['RKnee'],
                                    ref_point=norm_mid_rthigh_xy,
                                    is_man=is_man,
                                    is_rect_symmetrical=is_rect_symmetrical,
                                    segm_symmetry_dict=segm_symmetry_dict,
                                    scaler=scaler)

    if 'RCalf' in segments_xy:
        segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                    segm_id='RCalf',
                                    segm_xy=segments_xy['RCalf']['segm_xy'],
                                    keypoint=segments_xy['RCalf']['keypoints']['RAnkle'],
                                    ref_point=norm_mid_rcalf_xy,
                                    is_man=is_man,
                                    is_rect_symmetrical=is_rect_symmetrical,
                                    segm_symmetry_dict=segm_symmetry_dict,
                                    scaler=scaler)

    if 'LThigh' in segments_xy:
        segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                    segm_id='LThigh',
                                    segm_xy=segments_xy['LThigh']['segm_xy'],
                                    keypoint=segments_xy['LThigh']['keypoints']['LKnee'],
                                    ref_point=norm_mid_lthigh_xy,
                                    is_man=is_man,
                                    is_rect_symmetrical=is_rect_symmetrical,
                                    segm_symmetry_dict=segm_symmetry_dict,
                                    scaler=scaler)

    if 'LCalf' in segments_xy:
        segm_symmetry_dict = dispatcher['segm_function'](image=image,
                                    segm_id='LCalf',
                                    segm_xy=segments_xy['LCalf']['segm_xy'],
                                    keypoint=segments_xy['LCalf']['keypoints']['LAnkle'],
                                    ref_point=norm_mid_lcalf_xy,
                                    is_man=is_man,
                                    is_rect_symmetrical=is_rect_symmetrical,
                                    segm_symmetry_dict=segm_symmetry_dict,
                                    scaler=scaler)

    # draw the segments at last, after the symmetry of all segments has been checked
    if is_rect_symmetrical:
        # head
        _draw_symmetrical_rect_segm(image=image, segm_id='Head', w_and_h=segm_symmetry_dict['Head'],
                                    ref_point=norm_nose_xy)

        # torso
        _draw_symmetrical_rect_segm(image=image, segm_id='Torso', w_and_h=segm_symmetry_dict['Torso'],
                                    ref_point=norm_mid_torso_xy)

        # arms
        _draw_symmetrical_rect_segm(image=image, segm_id='RUpperArm', w_and_h=segm_symmetry_dict['RUpperArm'],
                                    ref_point=norm_mid_rupper_arm_xy)
        _draw_symmetrical_rect_segm(image=image, segm_id='RLowerArm', w_and_h=segm_symmetry_dict['RLowerArm'],
                                    ref_point=norm_mid_rlower_arm_xy)
        _draw_symmetrical_rect_segm(image=image, segm_id='LUpperArm', w_and_h=segm_symmetry_dict['LUpperArm'],
                                    ref_point=norm_mid_lupper_arm_xy)
        _draw_symmetrical_rect_segm(image=image, segm_id='LLowerArm', w_and_h=segm_symmetry_dict['LLowerArm'],
                                    ref_point=norm_mid_llower_arm_xy)

        # legs
        _draw_symmetrical_rect_segm(image=image, segm_id='RThigh', w_and_h=segm_symmetry_dict['RThigh'],
                                    ref_point=norm_mid_rthigh_xy)
        _draw_symmetrical_rect_segm(image=image, segm_id='RCalf', w_and_h=segm_symmetry_dict['RCalf'],
                                    ref_point=norm_mid_rcalf_xy)
        _draw_symmetrical_rect_segm(image=image, segm_id='LThigh', w_and_h=segm_symmetry_dict['LThigh'],
                                    ref_point=norm_mid_lthigh_xy)
        _draw_symmetrical_rect_segm(image=image, segm_id='LCalf', w_and_h=segm_symmetry_dict['LCalf'],
                                    ref_point=norm_mid_lcalf_xy)

    # draw centers
    # head center
    cv2.circle(image, tuple(norm_nose_xy), radius=radius, color=(255, 0, 255), thickness=-1)

    # torso center
    cv2.circle(image, tuple(norm_mid_torso_xy), radius=radius, color=(255, 0, 255), thickness=-1)

    # upper limbs
    cv2.circle(image, tuple(norm_mid_rupper_arm_xy), radius=radius, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_rlower_arm_xy), radius=radius, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_lupper_arm_xy), radius=radius, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_llower_arm_xy), radius=radius, color=(255, 0, 255), thickness=-1)

    # lower limbs
    cv2.circle(image, tuple(norm_mid_rthigh_xy), radius=radius, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_rcalf_xy), radius=radius, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_lthigh_xy), radius=radius, color=(255, 0, 255), thickness=-1)
    cv2.circle(image, tuple(norm_mid_lcalf_xy), radius=radius, color=(255, 0, 255), thickness=-1)

    return image


def visualize_norm_segm(image_bg, mask, segm, bbox_xywh, keypoints, infile, is_vitruve, is_rect, is_man, is_rect_symmetrical, show=False):

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

        cv2.imshow(window_bbox, segm_vis)
        cv2.setWindowProperty(window_bbox, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # visualize normalized pose
    # rotate to t-pose
    segments_xy = rotate_segments_xy(segm=segm, keypoints=keypoints)

    # draw segments in normalized image
    image = draw_segments_xy(segments_xy=segments_xy, is_vitruve=is_vitruve,
                             is_rect=is_rect, is_man=is_man, is_rect_symmetrical=is_rect_symmetrical)

    if show:
        outfile = generate_norm_segm_outfile(infile, is_rect)
        cv2.imwrite(outfile, image)
        print('output', outfile)

        cv2.imshow(window_norm, image)
        cv2.setWindowProperty(window_norm, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        outfile = generate_norm_segm_outfile(infile, is_rect)
        cv2.imwrite(outfile, image)
        print('output', outfile)


def _dilate_segm_to_convex(image, segm_id, segm_xy, bbox_xywh):

    # test each segment
    # print('Segment ID:', segm_id)

    # remove outliers
    print('Before removing outliers:', len(segm_xy))
    segm_xy = np.array(_remove_outlier(segm_xy=segm_xy)).astype(int)
    print('After removing outliers:', len(segm_xy))

    min_x, min_y = np.min(segm_xy, axis=0).astype(int)
    max_x, max_y = np.max(segm_xy, axis=0).astype(int)

    margin = 5

    w = int(max_x - min_x + margin * 2)
    h = int(max_y - min_y + margin * 2)

    img_bg = np.empty((h, w, 4), np.uint8)
    img_bg.fill(255)
    img_bg[:, :, 3] = 0  # alpha channel = 0 -> transparent

    # fill the segment with the segment color
    contours = [[int(x - min_x + margin), int(y - min_y + margin)] for x, y in segm_xy]
    contours = np.array(contours, np.int32)
    cv2.fillConvexPoly(img_bg, cv2.convexHull(contours), color=COARSE_TO_COLOR[segm_id])

    # translate from the bbox's coordinate to the image's coordinate
    bbox_x, bbox_y, bbox_w, bbox_h = bbox_xywh

    # stack two images
    cond_bg = img_bg[:, :, 3] > 0  # condition for already-drawn segment pixels
    image[int(min_y - margin + bbox_y):int(max_y + margin + bbox_y), int(min_x - margin + bbox_x):int(max_x + margin + bbox_x), :][cond_bg] = img_bg[cond_bg]


def _get_min_bounding_rect(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    """

    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def _dilate_segm_to_rect(image, segm_id, segm_xy, bbox_xywh):

    # test each segment
    # print('Segment ID:', segm_id)

    # remove outliers
    print('Before removing outliers:', len(segm_xy))
    segm_xy = np.array(_remove_outlier(segm_xy=segm_xy)).astype(int)
    print('After removing outliers:', len(segm_xy))

    # get the minimum bounding rectangle of segm_xy
    rect_xy = _get_min_bounding_rect(segm_xy)

    min_x, min_y = np.min(rect_xy, axis=0).astype(int)
    max_x, max_y = np.max(rect_xy, axis=0).astype(int)

    w = int(max_x - min_x)
    h = int(max_y - min_y)

    img_bg = np.empty((h, w, 4), np.uint8)
    img_bg.fill(255)
    img_bg[:, :, 3] = 0  # alpha channel = 0 -> transparent

    # fit in the coordinate of img_bg
    contours = [[int(x - min_x), int(y - min_y)] for x, y in rect_xy]
    contours = np.array(contours, np.int32)
    # convex hull = rectangle
    cv2.fillConvexPoly(img_bg, cv2.convexHull(contours), color=COARSE_TO_COLOR[segm_id])

    # translate from the bbox's coordinate to the image's coordinate
    bbox_x, bbox_y, bbox_w, bbox_h = bbox_xywh

    # stack two images
    cond_bg = img_bg[:, :, 3] > 0  # condition for already-drawn segment pixels
    image[int(min_y + bbox_y):int(max_y + bbox_y), int(min_x + bbox_x):int(max_x + bbox_x), :][cond_bg] = img_bg[cond_bg]


def dilate_segm(image, mask, segm, bbox_xywh, keypoints, infile, is_rect, show):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_overlay = image.copy()

    keypoints = dict(zip(JOINT_ID, keypoints))
    segments_xy = _get_segments_xy(segm=segm, keypoints=keypoints)

    dispatcher = {
        'dilate_to_rect': _dilate_segm_to_rect,
        'dilate_to_convex': _dilate_segm_to_convex
    }

    if is_rect:
        dispatcher['dilate_function'] = dispatcher['dilate_to_rect']
    else:
        dispatcher['dilate_function'] = dispatcher['dilate_to_convex']


    # draw segments in the original image
    if 'Head' in segments_xy and 'Torso' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='Head',
                                      segm_xy=segments_xy['Head']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    # torso
    if 'Torso' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='Torso',
                                      segm_xy=segments_xy['Torso']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    # upper limbs
    if 'RUpperArm' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='RUpperArm',
                                      segm_xy=segments_xy['RUpperArm']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    if 'RLowerArm' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='RLowerArm',
                                      segm_xy=segments_xy['RLowerArm']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    if 'LUpperArm' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='LUpperArm',
                                      segm_xy=segments_xy['LUpperArm']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    if 'LLowerArm' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='LLowerArm',
                                      segm_xy=segments_xy['LLowerArm']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    # lower limbs
    if 'RThigh' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='RThigh',
                                      segm_xy=segments_xy['RThigh']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    if 'RCalf' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='RCalf',
                                      segm_xy=segments_xy['RCalf']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    if 'LThigh' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='LThigh',
                                      segm_xy=segments_xy['LThigh']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    if 'LCalf' in segments_xy:

        dispatcher['dilate_function'](image=image,
                                      segm_id='LCalf',
                                      segm_xy=segments_xy['LCalf']['segm_xy'],
                                      bbox_xywh=bbox_xywh)

    added_image = cv2.addWeighted(image_overlay, 0.5, image, 0.5, 0)

    if show:
        outfile = generate_dilated_segm_outfile(infile, is_rect)
        cv2.imwrite(outfile, added_image)
        print('output', outfile)

        cv2.imshow(window_dilation, added_image)
        cv2.setWindowProperty(window_dilation, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        outfile = generate_dilated_segm_outfile(infile, is_rect)
        cv2.imwrite(outfile, added_image)
        print('output', outfile)


def stitch_data(results_densepose, boxes_xywh, data_keypoints, image, show):

    image = image.copy()

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

                    # draw the centroid
                    cv2.circle(image, (int(centroid_x), int(centroid_y)), radius=5, color=(255, 0, 255), thickness=5)

                    # draw the bbox
                    cv2.line(image, (x, y), (int(x + w), y), color=(0, 255, 0), thickness=5)
                    cv2.line(image, (x, y), (x, int(y + h)), color=(0, 255, 0), thickness=5)
                    cv2.line(image, (int(x + w), int(y + h)), (x, int(y + h)), color=(0, 255, 0), thickness=5)
                    cv2.line(image, (int(x + w), int(y + h)), (int(x + w), y), color=(0, 255, 0), thickness=5)

                    # draw the keypoints
                    for keypoint in keypoints:
                        x, y, _ = keypoint
                        cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 255, 255), thickness=5)

                    break

    if show:
        cv2.imshow(window_stitched_data, image)
        cv2.setWindowProperty(window_stitched_data, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # print('length of matched_results_densepose:', len(matched_results_densepose))
    # print('length of matched_boxes_xywh:', len(matched_boxes_xywh))
    # print('length of matched_data_keypoints:', len(matched_data_keypoints))

    return matched_results_densepose, matched_boxes_xywh, matched_data_keypoints


def generate_norm_segm(infile, score_cutoff, is_vitruve, is_rect, is_man, is_rect_symmetrical, show):

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

            # dilate segments
            dilate_segm(image=im_gray, mask=mask, segm=segm, bbox_xywh=box_xywh, keypoints=keypoints, infile=infile,
                        is_rect=is_rect, show=show)

            # visualizer
            visualize_norm_segm(image_bg=im_gray, mask=mask, segm=segm, bbox_xywh=box_xywh, keypoints=keypoints, infile=infile,
                                is_vitruve=is_vitruve, is_rect=is_rect, is_man=is_man, is_rect_symmetrical=is_rect_symmetrical,
                                show=show)
        else:
            continue


def generate_norm_segm_outfile(infile, is_rect):

    outdir = os.path.join(norm_segm_dir, infile[infile.find('/') + 1:infile.rfind('/')])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = infile[infile.find('/') + 1:infile.rfind('.')]

    if is_rect:
        outfile = os.path.join(norm_segm_dir, '{}_norm_rect.jpg'.format(fname))
    else:
        outfile = os.path.join(norm_segm_dir, '{}_norm_convex.jpg'.format(fname))

    return outfile


def generate_dilated_segm_outfile(infile, is_rect):

    outdir = os.path.join(norm_segm_dir, infile[infile.find('/') + 1:infile.rfind('/')])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = infile[infile.find('/') + 1:infile.rfind('.')]

    if is_rect:
        outfile = os.path.join(norm_segm_dir, '{}_dilated_rect.jpg'.format(fname))
    else:
        outfile = os.path.join(norm_segm_dir, '{}_dilated_convex.jpg'.format(fname))

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
        cv2.imshow(window_segm, image_vis)
        cv2.setWindowProperty(window_segm, cv2.WND_PROP_TOPMOST, 1)
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

    # example cases
    # modern
    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/90551.jpg --gender woman --output norm
    # python infer_segm.py --input datasets/modern/Paul\ Gauguin/30963.jpg --gender woman --output norm

    # classical
    # python infer_segm.py --input datasets/classical/Michelangelo/12758.jpg --gender man --output norm
    # python infer_segm.py --input datasets/classical/Artemisia\ Gentileschi/45093.jpg --gender man --output norm

    # python infer_segm.py --input datasets/classical/Pierre-Auguste\ Renoir/96672.jpg --output norm
    # python infer_segm.py --input datasets/classical/Pierre-Auguste\ Renoir/90411.jpg --output norm
    # python infer_segm.py --input datasets/classical/Pierre-Auguste\ Renoir/79467.jpg --output norm
    # python infer_segm.py --input datasets/classical/El\ Greco/4651.jpg --output norm
    # python infer_segm.py --input datasets/classical/Pierre-Paul\ Prud\'hon/48529.jpg --output norm

    # test cases
    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/80019.jpg --output norm
    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/81903.jpg --output norm

    # buggy cases
    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/25239.jpg --output norm
    # python infer_segm.py --input datasets/modern/Paul\ Delvaux/16338.jpg --output norm
    # python infer_segm.py --input datasets/modern/Tamara\ de\ Lempicka/61475.jpg --output norm

    # failed cases
    # python infer_segm.py --input datasets/modern/Felix\ Vallotton/55787.jpg --output norm
    # python infer_segm.py --input datasets/classical/Michelangelo/6834.jpg --output norm
    # python infer_segm.py --input datasets/classical/Michelangelo/26362.jpg --output norm
    # python infer_segm.py --input datasets/classical/Michelangelo/44006.jpg --output norm
    # python infer_segm.py --input datasets/classical/Michelangelo/62566.jpg --output norm

    parser = argparse.ArgumentParser(description='DensePose - Infer the segments')
    parser.add_argument('--input', help='Path to image file or directory')
    parser.add_argument('--gender', help='Gender of the figure')
    parser.add_argument('--output', help='segm is segment only, norm is normalized segment')
    args = parser.parse_args()

    if args.gender == 'man':
        is_man = True
    elif args.gender == 'woman':
        is_man = False

    # visualize the normalized pose
    if os.path.isfile(args.input):
        if args.output == 'segm':
            generate_segm(infile=args.input, score_cutoff=0.95, show=True)
        elif args.output == 'norm':
            generate_norm_segm(infile=args.input, score_cutoff=0.95,
                               is_vitruve=False, is_rect=True, is_man=is_man, is_rect_symmetrical=True,
                               show=True)

    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.jpg'):
            try:
                if args.output == 'segm':
                    generate_segm(infile=str(path), score_cutoff=0.9, show=False)
                elif args.output == 'norm':
                    generate_norm_segm(infile=args.input, score_cutoff=0.95,
                                       is_vitruve=False, is_rect=True, is_man=is_man, is_rect_symmetrical=False,
                                       show=False)
            except:
                continue
    else:
        pass