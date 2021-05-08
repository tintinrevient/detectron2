import cv2
import numpy as np
from pycocotools.coco import COCO
import os
import glob
import pycocotools.mask as mask_util
from densepose.structures import DensePoseDataRelative
from infer_segm import _calc_angle, _rotate
from distribution_segm import (
    coco_folder, dp_coco, caption_coco, JOINT_ID,
    _is_valid, _translate_keypoints_to_bbox, filter_by_caption, get_img_ids_by_caption, get_img_ids_by_dir
)


# image shape
image_w_and_h = 624


def _rotate_to_vertical_pose(keypoints):

    midhip_keypoint = keypoints['MidHip']
    neck_keypoint = keypoints['Neck']

    # calculate the angle for rotation to vertical pose
    reference_point = np.array(midhip_keypoint) + np.array((0, -100, 0))
    rad, deg = _calc_angle(point1=neck_keypoint, center=midhip_keypoint, point2=reference_point)

    for keypoints_id, keypoints_xy in keypoints.items():
        keypoints[keypoints_id] = _rotate(keypoints_xy, midhip_keypoint, rad)

    return keypoints


def _rotate_to_tpose(keypoints):

    nose_keypoint = keypoints['Nose']
    neck_keypoint = keypoints['Neck']

    rsho_keypoint = keypoints['RShoulder']
    relb_keypoint = keypoints['RElbow']
    rwrist_keypoint = keypoints['RWrist']

    lsho_keypoint = keypoints['LShoulder']
    lelb_keypoint = keypoints['LElbow']
    lwrist_keypoint = keypoints['LWrist']

    midhip_keypoint = keypoints['MidHip']
    rhip_keypoint = keypoints['RHip']
    lhip_keypoint = keypoints['LHip']

    rknee_keypoint = keypoints['RKnee']
    rankle_keypoint = keypoints['RAnkle']

    lknee_keypoint = keypoints['LKnee']
    lankle_keypoint = keypoints['LAnkle']

    # Nose
    reference_point = np.array(neck_keypoint) + np.array((0, -50, 0))
    rad, deg = _calc_angle(point1=nose_keypoint, center=neck_keypoint, point2=reference_point)
    keypoints['Nose'] = _rotate(nose_keypoint, neck_keypoint, rad)

    # Right upper limb
    reference_point = np.array(neck_keypoint) + np.array((-50, 0, 0))
    rad, deg = _calc_angle(point1=rsho_keypoint, center=neck_keypoint, point2=reference_point)
    keypoints['RShoulder'] = _rotate(rsho_keypoint, neck_keypoint, rad)

    relb_keypoint = _rotate(relb_keypoint, neck_keypoint, rad)
    rwrist_keypoint = _rotate(rwrist_keypoint, neck_keypoint, rad)
    reference_point = np.array(keypoints['RShoulder']) + np.array((-50, 0, 0))
    rad, deg = _calc_angle(point1=relb_keypoint, center=keypoints['RShoulder'], point2=reference_point)
    keypoints['RElbow'] = _rotate(relb_keypoint, keypoints['RShoulder'], rad)

    rwrist_keypoint = _rotate(rwrist_keypoint, keypoints['RShoulder'], rad)
    reference_point = np.array(keypoints['RElbow']) + np.array((-50, 0, 0))
    rad, deg = _calc_angle(point1=rwrist_keypoint, center=keypoints['RElbow'], point2=reference_point)
    keypoints['RWrist'] = _rotate(rwrist_keypoint, keypoints['RElbow'], rad)

    # Left upper limb
    reference_point = np.array(neck_keypoint) + np.array((50, 0, 0))
    rad, deg = _calc_angle(point1=lsho_keypoint, center=neck_keypoint, point2=reference_point)
    keypoints['LShoulder'] = _rotate(lsho_keypoint, neck_keypoint, rad)

    lelb_keypoint = _rotate(lelb_keypoint, neck_keypoint, rad)
    lwrist_keypoint = _rotate(lwrist_keypoint, neck_keypoint, rad)
    reference_point = np.array(keypoints['LShoulder']) + np.array((50, 0, 0))
    rad, deg = _calc_angle(point1=lelb_keypoint, center=keypoints['LShoulder'], point2=reference_point)
    keypoints['LElbow'] = _rotate(lelb_keypoint, keypoints['LShoulder'], rad)

    lwrist_keypoint = _rotate(lwrist_keypoint, keypoints['LShoulder'], rad)
    reference_point = np.array(keypoints['LElbow']) + np.array((50, 0, 0))
    rad, deg = _calc_angle(point1=lwrist_keypoint, center=keypoints['LElbow'], point2=reference_point)
    keypoints['LWrist'] = _rotate(lwrist_keypoint, keypoints['LElbow'], rad)

    # Right lower limb
    reference_point = np.array(midhip_keypoint) + np.array((-50, 0, 0))
    rad, deg = _calc_angle(point1=rhip_keypoint, center=midhip_keypoint, point2=reference_point)
    keypoints['RHip'] = _rotate(rhip_keypoint, midhip_keypoint, rad)

    rknee_keypoint = _rotate(rknee_keypoint, midhip_keypoint, rad)
    rankle_keypoint = _rotate(rankle_keypoint, midhip_keypoint, rad)
    reference_point = np.array(keypoints['RHip']) + np.array((0, 50, 0))
    rad, deg = _calc_angle(point1=rknee_keypoint, center=keypoints['RHip'], point2=reference_point)
    keypoints['RKnee'] = _rotate(rknee_keypoint, keypoints['RHip'], rad)

    rankle_keypoint = _rotate(rankle_keypoint, keypoints['RHip'], rad)
    reference_point = np.array(keypoints['RKnee']) + np.array((0, 50, 0))
    rad, deg = _calc_angle(point1=rankle_keypoint, center=keypoints['RKnee'], point2=reference_point)
    keypoints['RAnkle'] = _rotate(rankle_keypoint, keypoints['RKnee'], rad)

    # Left lower limb
    reference_point = np.array(midhip_keypoint) + np.array((50, 0, 0))
    rad, deg = _calc_angle(point1=lhip_keypoint, center=midhip_keypoint, point2=reference_point)
    keypoints['LHip'] = _rotate(lhip_keypoint, midhip_keypoint, rad)

    lknee_keypoint = _rotate(lknee_keypoint, midhip_keypoint, rad)
    lankle_keypoint = _rotate(lankle_keypoint, midhip_keypoint, rad)
    reference_point = np.array(keypoints['LHip']) + np.array((0, 50, 0))
    rad, deg = _calc_angle(point1=lknee_keypoint, center=keypoints['LHip'], point2=reference_point)
    keypoints['LKnee'] = _rotate(lknee_keypoint, keypoints['LHip'], rad)

    lankle_keypoint = _rotate(lankle_keypoint, keypoints['LHip'], rad)
    reference_point = np.array(keypoints['LKnee']) + np.array((0, 50, 0))
    rad, deg = _calc_angle(point1=lankle_keypoint, center=keypoints['LKnee'], point2=reference_point)
    keypoints['LAnkle'] = _rotate(lankle_keypoint, keypoints['LKnee'], rad)

    return keypoints


def rotate_keypoints(keypoints):

    keypoints = _rotate_to_vertical_pose(keypoints)
    keypoints = _rotate_to_tpose(keypoints)

    return keypoints


def translate_keypoints(keypoints):

    for keypoints_id, keypoints_xy in keypoints.items():
        keypoints[keypoints_id] = np.array(keypoints_xy) + np.array([50, 50, 0])

    return keypoints


def visualize_keypoints(image_id):

    entry = dp_coco.loadImgs(image_id)[0]

    dataset_name = entry['file_name'][entry['file_name'].find('_') + 1:entry['file_name'].rfind('_')]
    image_fpath = os.path.join(coco_folder, dataset_name, entry['file_name'])

    print('image_fpath:', image_fpath)

    dp_annotation_ids = dp_coco.getAnnIds(imgIds=entry['id'])
    dp_annotations = dp_coco.loadAnns(dp_annotation_ids)

    # iterate through all the people in one image
    for dp_annotation in dp_annotations:

        # check the validity of annotation
        is_valid, _ = DensePoseDataRelative.validate_annotation(dp_annotation)

        if not is_valid:
            continue

        # bbox
        bbox_xywh = np.array(dp_annotation["bbox"]).astype(int)
        x1, y1, x2, y2 = bbox_xywh[0], bbox_xywh[1], int(bbox_xywh[0] + bbox_xywh[2]), int(bbox_xywh[1] + bbox_xywh[3])

        # keypoints
        keypoints = np.array(dp_annotation['keypoints']).astype(int)
        keypoints = _translate_keypoints_to_bbox(keypoints=keypoints, bbox_xywh=bbox_xywh)

        # check the validity of keypoints
        if not _is_valid(keypoints=keypoints):
            continue

        # print annotations
        caption_annotation_ids = caption_coco.getAnnIds(imgIds=dp_annotation['image_id'])
        caption_annotations = caption_coco.loadAnns(caption_annotation_ids)
        print([caption_annotation['caption'] for caption_annotation in caption_annotations])

        # load the original image
        im_gray = cv2.imread(image_fpath, cv2.IMREAD_GRAYSCALE)
        im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])

        # crop the image within bbox
        im_output = im_gray[y1:y2, x1:x2, :].copy()

        not_drawn_joints = ['LEye', 'REye', 'LEar', 'REar']

        # draw keypoints
        for keypoints_id, keypoints_xy in keypoints.items():
            x, y, score = keypoints_xy
            if score > 0 and keypoints_id not in not_drawn_joints:
                cv2.circle(im_output, (int(x), int(y)), radius=3, color=(255, 0, 255), thickness=-1)

        window_input = 'input'
        cv2.imshow(window_input, im_output)
        cv2.setWindowProperty(window_input, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # draw t-pose keypoints
        keypoints = rotate_keypoints(keypoints)
        keypoints = translate_keypoints(keypoints)

        im_norm = np.empty((624, 624, 4), np.uint8)
        im_norm.fill(255)
        for keypoints_id, keypoints_xy in keypoints.items():
            x, y, score = keypoints_xy
            if score > 0 and keypoints_id not in not_drawn_joints:
                cv2.circle(im_norm, (int(x), int(y)), radius=3, color=(255, 0, 255), thickness=-1)

        window_norm = 'norm'
        cv2.imshow(window_norm, im_norm)
        cv2.setWindowProperty(window_norm, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    # common setting
    dp_img_category = 'man'  # man or woman

    # option 1 - images within a range
    dp_img_range = slice(0, 10)
    dp_img_ids = get_img_ids_by_caption(dp_img_category=dp_img_category, dp_img_range=dp_img_range)

    # option 2 - image from a directory
    # img_dir = os.path.join('datasets', dp_img_category)
    # dp_img_ids = get_img_ids_by_dir(indir=img_dir)

    for image_id in dp_img_ids:
        visualize_keypoints(image_id=image_id)