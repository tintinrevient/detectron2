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


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


def _extract_u_from_iuvarr(iuv_arr):
    return iuv_arr[1, :, :]


def _extract_v_from_iuvarr(iuv_arr):
    return iuv_arr[2, :, :]


def resize(mask, matrix, w, h):

    interp_method_matrix = cv2.INTER_LINEAR,
    interp_method_mask = cv2.INTER_NEAREST

    if (w != mask.shape[1]) or (h != mask.shape[0]):
        mask = cv2.resize(mask, (w, h), interp_method_mask)
    if (w != matrix.shape[1]) or (h != matrix.shape[0]):
        matrix = cv2.resize(matrix, (w, h), interp_method_matrix)
    return mask, matrix


def extract_segm(result, is_coarse=True):

    iuv_array = torch.cat(
        (result.labels[None].type(torch.float32), result.uv * 255.0)
    ).type(torch.uint8)

    iuv_array = iuv_array.cpu().numpy()

    segm = _extract_i_from_iuvarr(iuv_array)

    if is_coarse:
        for fine_idx, coarse_idx in FINE_TO_COARSE_SEGMENTATION.items():
            segm[segm == fine_idx] = coarse_idx

    mask = np.zeros(segm.shape, dtype=np.uint8)
    mask[segm > 0] = 1

    # matrix = _extract_v_from_iuvarr(iuv_array)

    return segm, mask


def calc_angle(point1, center, point2):

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


def rotate(point, center, rad):

    x = ((point[0] - center[0]) * np.cos(rad)) - ((point[1] - center[1]) * np.sin(rad)) + center[0];
    y = ((point[0] - center[0]) * np.sin(rad)) + ((point[1] - center[1]) * np.cos(rad)) + center[1];

    return [int(x), int(y)]


def translate_segments(segments_xy, keypoints):

    mid_xy = np.array(keypoints['MidHip'])[0:2]
    diff_xy = np.array([1000, 1000]) - mid_xy

    segments_xy = (np.array(segment) + diff_xy for segment in segments_xy)

    return segments_xy


def rotate_to_t_pose(segm, keypoints):

    # Issue ONE: cannot rotate body to FACE-FRONT TORSO-FRONT
    # Issue TWO: cannot have the same person!!! so can be fat, or can be thin!!!

    # rotated angle: transform any pose to neck-midhip-vertical pose, i.e., stand up, sit up, etc...
    reference_point = np.array(keypoints['MidHip']) + np.array([0, -100, 0])
    rad, deg = calc_angle(point1=keypoints['Neck'], center=keypoints['MidHip'], point2=reference_point)

    # original x, y
    # single segment
    y, x = np.where(segm == 1)
    torso_xy = list(zip(x, y))

    y, x = np.where(segm == 2)
    r_hand_xy = list(zip(x, y))

    y, x = np.where(segm == 3)
    l_hand_xy = list(zip(x, y))

    y, x = np.where(segm == 4)
    l_foot_xy = list(zip(x, y))

    y, x = np.where(segm == 5)
    r_foot_xy = list(zip(x, y))

    y, x = np.where(segm == 6)
    r_thigh_xy = list(zip(x, y))

    y, x = np.where(segm == 7)
    l_thigh_xy = list(zip(x, y))

    y, x = np.where(segm == 8)
    r_calf_xy = list(zip(x, y))

    y, x = np.where(segm == 9)
    l_calf_xy = list(zip(x, y))

    y, x = np.where(segm == 10)
    l_upper_arm_xy = list(zip(x, y))

    y, x = np.where(segm == 11)
    r_upper_arm_xy = list(zip(x, y))

    y, x = np.where(segm == 12)
    l_lower_arm_xy = list(zip(x, y))

    y, x = np.where(segm == 13)
    r_lower_arm_xy = list(zip(x, y))

    y, x = np.where(segm == 14)
    head_xy = list(zip(x, y))

    # combined segments
    y, x = np.where(segm != 0)
    body_xy = list(zip(x, y))

    # y, x = np.where(np.logical_or.reduce((segm == 2, segm == 11, segm == 13)))
    # r_upper_limb_xy = list(zip(x, y))
    #
    # y, x = np.where(np.logical_or.reduce((segm == 5, segm == 6, segm == 8)))
    # r_lower_limb_xy = list(zip(x, y))
    #
    # y, x = np.where(np.logical_or.reduce((segm == 3, segm == 10, segm == 12)))
    # l_upper_limb_xy = list(zip(x, y))
    #
    # y, x = np.where(np.logical_or.reduce((segm == 4, segm == 7, segm == 9)))
    # l_lower_limb_xy = list(zip(x, y))

    # rotated each single segment to vertical pose, i.e., stand up, sit up, etc...
    torso_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in torso_xy]

    r_thigh_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in r_thigh_xy]

    l_thigh_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in l_thigh_xy]

    r_calf_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in r_calf_xy]

    l_calf_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in l_calf_xy]

    l_upper_arm_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in l_upper_arm_xy]

    r_upper_arm_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in r_upper_arm_xy]

    l_lower_arm_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in l_lower_arm_xy]

    r_lower_arm_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in r_lower_arm_xy]

    head_xy = [rotate([x, y], keypoints['MidHip'], rad) for (x, y) in head_xy]

    # rotate keypoints to vertical pose
    rotated_keypoints = {key: rotate(value, keypoints['MidHip'], rad) for key, value in keypoints.items()}

    # NO mapped hand and foot keypoints to rotate them!!!
    # nose -> head (DONE in vertical rotation)
    # midhip -> torso (DONE in vertical rotation)
    # elbow -> upper arm
    # wrist -> lower arm
    # knee -> thigh
    # ankle -> calf

    # rotated specific single segment further to t-pose

    # RIGHT!!!
    # upper limb
    rad, deg = calc_angle(rotated_keypoints.get('RWrist'), rotated_keypoints.get('RElbow'), rotated_keypoints.get('RShoulder'))
    rad = np.pi + rad
    r_lower_arm_xy = [rotate([x, y], rotated_keypoints.get('RElbow'), rad) for (x, y) in r_lower_arm_xy]

    rsho_ref = np.array(rotated_keypoints.get('RShoulder')[0:2]) + np.array([-50, 0])  # reference line to calculate angles
    rad, deg = calc_angle(rotated_keypoints.get('RElbow'), rotated_keypoints.get('RShoulder'), rsho_ref)
    r_upper_arm_xy = [rotate([x, y], rotated_keypoints.get('RShoulder'), rad) for (x, y) in r_upper_arm_xy]
    r_lower_arm_xy = [rotate([x, y], rotated_keypoints.get('RShoulder'), rad) for (x, y) in r_lower_arm_xy]

    # lower limb
    # rotate calf to align with thigh
    rad, deg = calc_angle(rotated_keypoints.get('RAnkle'), rotated_keypoints.get('RKnee'),
                          rotated_keypoints.get('RHip'))
    rad = np.pi + rad
    r_calf_xy = [rotate([x, y], rotated_keypoints.get('RKnee'), rad) for (x, y) in r_calf_xy]

    # rotate hip to horizontal
    midhip_ref = np.array(rotated_keypoints.get('MidHip')[0:2]) + np.array([-50, 0])  # reference line to calculate angles
    rad, deg = calc_angle(rotated_keypoints.get('RHip'), rotated_keypoints.get('MidHip'), midhip_ref)
    r_thigh_xy = [rotate([x, y], rotated_keypoints.get('MidHip'), rad) for (x, y) in r_thigh_xy]
    r_calf_xy = [rotate([x, y], rotated_keypoints.get('MidHip'), rad) for (x, y) in r_calf_xy]

    # rotate lower limb to degree np.pi / 8 from vertical line
    rhip_ref = np.array(rotated_keypoints.get('RHip')[0:2]) + np.array([0, 100])
    rad, deg = calc_angle(rotated_keypoints.get('RKnee'), rotated_keypoints.get('RHip'), rhip_ref)
    rad = rad + np.pi / 8
    r_thigh_xy = [rotate([x, y], rotated_keypoints.get('RHip'), rad) for (x, y) in r_thigh_xy]
    r_calf_xy = [rotate([x, y], rotated_keypoints.get('RHip'), rad) for (x, y) in r_calf_xy]

    # LEFT!!!
    # upper limb
    rad, deg = calc_angle(rotated_keypoints.get('LWrist'), rotated_keypoints.get('LElbow'),
                          rotated_keypoints.get('LShoulder'))
    rad = np.pi + rad
    l_lower_arm_xy = [rotate([x, y], rotated_keypoints.get('LElbow'), rad) for (x, y) in l_lower_arm_xy]

    lsho_ref = np.array(rotated_keypoints.get('LShoulder')[0:2]) + np.array([50, 0])  # reference line to calculate angles
    rad, deg = calc_angle(rotated_keypoints.get('LElbow'), rotated_keypoints.get('LShoulder'), lsho_ref)
    l_upper_arm_xy = [rotate([x, y], rotated_keypoints.get('LShoulder'), rad) for (x, y) in l_upper_arm_xy]
    l_lower_arm_xy = [rotate([x, y], rotated_keypoints.get('LShoulder'), rad) for (x, y) in l_lower_arm_xy]

    # lower limb
    # rotate calf to align with thigh
    rad, deg = calc_angle(rotated_keypoints.get('LAnkle'), rotated_keypoints.get('LKnee'),
                          rotated_keypoints.get('LHip'))
    rad = np.pi + rad
    l_calf_xy = [rotate([x, y], rotated_keypoints.get('LKnee'), rad) for (x, y) in l_calf_xy]

    # rotate hip to horizontal
    midhip_ref = np.array(rotated_keypoints.get('MidHip')[0:2]) + np.array(
        [50, 0])  # reference line to calculate angles
    rad, deg = calc_angle(rotated_keypoints.get('LHip'), rotated_keypoints.get('MidHip'), midhip_ref)
    l_thigh_xy = [rotate([x, y], rotated_keypoints.get('MidHip'), rad) for (x, y) in l_thigh_xy]
    l_calf_xy = [rotate([x, y], rotated_keypoints.get('MidHip'), rad) for (x, y) in l_calf_xy]

    # rotate lower limb to degree np.pi / 8 from vertical line
    lhip_ref = np.array(rotated_keypoints.get('LHip')[0:2]) + np.array([0, 100])
    rad, deg = calc_angle(rotated_keypoints.get('LKnee'), rotated_keypoints.get('LHip'), lhip_ref)
    rad = rad - np.pi / 8
    l_thigh_xy = [rotate([x, y], rotated_keypoints.get('LHip'), rad) for (x, y) in l_thigh_xy]
    l_calf_xy = [rotate([x, y], rotated_keypoints.get('LHip'), rad) for (x, y) in l_calf_xy]


    return (head_xy, torso_xy, body_xy, r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy,
            l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy), (rotated_keypoints)


def draw_segments(image, segments_xy):

    head_xy, torso_xy, body_xy, r_thigh_xy, l_thigh_xy, r_calf_xy, l_calf_xy, l_upper_arm_xy, r_upper_arm_xy, l_lower_arm_xy, r_lower_arm_xy = segments_xy

    for x, y in body_xy:
        image = cv2.circle(image, (x, y), radius=10, color=(192, 192, 192), thickness=-1)
    for x, y in head_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    for x, y in torso_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    for x, y in r_thigh_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for x, y in l_thigh_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for x, y in r_calf_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for x, y in l_calf_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for x, y in l_upper_arm_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for x, y in r_upper_arm_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for x, y in l_lower_arm_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for x, y in r_lower_arm_xy:
        image = cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    return image


def visualize_segm(image_bgr, mask, segm, bbox_xywh):

    val_scale = 10.625
    cmap = cv2.COLORMAP_PARULA

    x, y, w, h = [int(v) for v in bbox_xywh]
    if w <= 0 or h <= 0:
        return image_bgr
    mask, segm = resize(mask, segm, w, h)

    cv2.imshow('image_bgr', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    infile = os.path.join('output', 'data', 'modern', 'Paul Delvaux', '80019_keypoints.npy')
    data = np.load(infile, allow_pickle='TRUE').item()
    keypoints = data['keypoints'][0]

    # scaled keypoints
    keypoints = np.array(keypoints) - np.array([x, y, 0.0])
    # dict keypoints
    keypoints = dict(zip(JOINT_ID, keypoints))

    mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])
    segm_scaled = segm.astype(np.float32) * val_scale
    segm_scaled_8u = segm_scaled.clip(0, 255).astype(np.uint8)

    cv2.imshow('segm_scaled_8u test:', segm_scaled_8u)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    segm_vis = cv2.applyColorMap(segm_scaled_8u, cmap)

    # ROTATION test!!!
    # function test!!!
    segments_xy, rotated_keypoints = rotate_to_t_pose(segm=segm, keypoints=keypoints)

    draw_segments(segm_vis, segments_xy=segments_xy)

    # for keypoint in keypoints.values():
    #     x, y, score = keypoint
    #     if score > 0:
    #         segm_vis = cv2.circle(segm_vis, (int(x), int(y)), radius=10, color=(0, 255, 0), thickness=-1)

    cv2.imshow('matrix_vis', segm_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # white background image
    image = np.empty((2000, 2000, 3), np.uint8)
    image.fill(255)

    segments_xy = translate_segments(segments_xy = segments_xy, keypoints=rotated_keypoints)

    draw_segments(image, segments_xy=segments_xy)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_rotated_segm(infile, score_cutoff, show):

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

    # filter the probabilities of scores for each bbox > 90%
    instances = outputs['instances']
    confident_detections = instances[instances.scores > score_cutoff]

    # extractor
    extractor = DensePoseResultExtractor()
    densepose_result, boxes_xywh = extractor(confident_detections)

    # bbox
    bbox = boxes_xywh.numpy()
    bbox = bbox[0] # -> first bbox
    print('bbox:', bbox)

    # result
    result = densepose_result[0] # -> first densepose_result

    # extract mask + matrix
    segm, mask = extract_segm(result=result)

    # visualizer
    visualize_segm(image_bgr=im_gray, mask=mask, segm=segm, bbox_xywh=bbox)


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
        generate_rotated_segm(infile=args.input, score_cutoff=0.95, show=True)

    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.jpg'):
            try:
                generate_segm(infile=str(path), score_cutoff=0.9, show=False)
            except:
                continue
    else:
        pass