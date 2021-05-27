import cv2
import numpy as np
import os
from densepose.structures import DensePoseDataRelative
from infer_segm import _calc_angle, _rotate, _euclidian, COARSE_TO_COLOR
from distribution_segm import (
    coco_folder, dp_coco, image_w_and_h,
    _is_valid, _translate_keypoints_to_bbox, get_img_ids_by_caption, get_img_ids_by_dir
)
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


JOINT_COLOR = {
    'Nose': 'gold',
    'RShoulder': 'yellowgreen',
    'RElbow': 'palegreen',
    'RWrist': 'khaki',
    'LShoulder': 'yellowgreen',
    'LElbow': 'palegreen',
    'LWrist': 'khaki',
    'MidHip': 'thistle',
    'RHip': 'plum',
    'LHip': 'violet',
    'RKnee': 'mediumturquoise',
    'RAnkle': 'paleturquoise',
    'LKnee': 'deepskyblue',
    'LAnkle': 'lightskyblue'
}


def show_keypoints_by_bbox(image_fpath, bbox_xywh, keypoints):

    # load the original image
    im_gray = cv2.imread(image_fpath, cv2.IMREAD_GRAYSCALE)
    im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])

    # bbox
    x1, y1, x2, y2 = bbox_xywh[0], bbox_xywh[1], int(bbox_xywh[0] + bbox_xywh[2]), int(bbox_xywh[1] + bbox_xywh[3])

    # crop the image within bbox
    im_output = im_gray[y1:y2, x1:x2, :].copy()

    # draw the keypoints in bbox
    for keypoints_id, keypoints_xy in keypoints.items():
        x, y, score = keypoints_xy
        if score > 0:
            cv2.circle(im_output, (int(x), int(y)), radius=3, color=(255, 0, 255), thickness=-1)

    window_input = 'bbox'
    cv2.imshow(window_input, im_output)
    cv2.setWindowProperty(window_input, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_keypoints_in_norm(norm_neck_xy, norm_nose_xy, norm_rsho_xy, norm_relb_xy, norm_rwrist_xy,
                           norm_lsho_xy, norm_lelb_xy, norm_lwrist_xy, norm_midhip_xy, norm_rhip_xy, norm_lhip_xy,
                           norm_rknee_xy, norm_rankle_xy, norm_lknee_xy, norm_lankle_xy):

    # white image
    im_norm = np.empty((image_w_and_h, image_w_and_h, 4), np.uint8)
    im_norm.fill(255)

    # drawing setting
    radius_keypoint = 3
    thickness_keypoint = -1

    cv2.circle(im_norm, tuple(norm_neck_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['Head']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_nose_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['Head']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_rsho_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['Head']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_relb_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['RUpperArm']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_rwrist_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['RLowerArm']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_lsho_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['RLowerArm']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_lelb_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['LUpperArm']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_lwrist_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['LLowerArm']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_midhip_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['Torso']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_rhip_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['Torso']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_lhip_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['Torso']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_rknee_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['RThigh']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_rankle_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['RCalf']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_lknee_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['LThigh']),
               thickness=thickness_keypoint)

    cv2.circle(im_norm, tuple(norm_lankle_xy), radius=radius_keypoint, color=tuple(COARSE_TO_COLOR['LCalf']),
               thickness=thickness_keypoint)

    # debug to show the normalized image
    window_norm = 'norm'
    cv2.imshow(window_norm, im_norm)
    cv2.setWindowProperty(window_norm, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', alpha=0.5, **kwargs):

    # scatter all the points
    # ax.scatter(x, y, s=0.5)

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, alpha=0.5, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)

    ax.add_patch(ellipse)

    ax.scatter([mean_x], [mean_y], s=1, c='black')

    return mean_x, mean_y


def _draw_std_ellipse(keypoint_id, ax, n_std):

    print('Keypoint:', keypoint_id)

    xy = np.array(dict_norm_keypoints_xy[keypoint_id])
    x = xy[:, 0]
    y = xy[:, 1]

    # plot
    mean_x, mean_y = _confidence_ellipse(x, y, ax, n_std=n_std, facecolor=JOINT_COLOR[keypoint_id])

    return mean_x, mean_y


def show_std_image(dict_norm_keypoints_xy, n_std, gender):

    # empty figure
    fig, ax_nstd = plt.subplots(figsize=(6, 6))

    # range
    plt.xlim(0, 600)

    if n_std == 1:
        plt.ylim(100, 800)
    else:
        plt.ylim(100, 750)

    # descend y-axis
    ax_nstd.set_ylim(ax_nstd.get_ylim()[::-1])

    # Neck
    ax_nstd.scatter([312], [198], s=1, c='black')

    # Nose
    nose_mean_x, nose_mean_y = _draw_std_ellipse('Nose', ax_nstd, n_std)

    # Line between neck and nose
    plt.plot([312, nose_mean_x], [198, nose_mean_y], linewidth=0.5, c='black')

    # RShoulder
    rsho_mean_x, rsho_mean_y = _draw_std_ellipse('RShoulder', ax_nstd, n_std)

    # Line between neck and shoulder
    plt.plot([312, rsho_mean_x], [198, rsho_mean_y], linewidth=0.5, c='black')

    # RElbow
    relb_mean_x, relb_mean_y = _draw_std_ellipse('RElbow', ax_nstd, n_std)

    # Line between shoulder and elbow
    plt.plot([relb_mean_x, rsho_mean_x], [relb_mean_y, rsho_mean_y], linewidth=0.5, c='black')

    # RWrist
    rwrist_mean_x, rwrist_mean_y = _draw_std_ellipse('RWrist', ax_nstd, n_std)

    # Line between wrist and elbow
    plt.plot([relb_mean_x, rwrist_mean_x], [relb_mean_y, rwrist_mean_y], linewidth=0.5, c='black')

    # LShoulder
    lsho_mean_x, lsho_mean_y = _draw_std_ellipse('LShoulder', ax_nstd, n_std)

    # Line between neck and shoulder
    plt.plot([312, lsho_mean_x], [198, lsho_mean_y], linewidth=0.5, c='black')

    # LElbow
    lelb_mean_x, lelb_mean_y = _draw_std_ellipse('LElbow', ax_nstd, n_std)

    # Line between elbow and shoulder
    plt.plot([lelb_mean_x, lsho_mean_x], [lelb_mean_y, lsho_mean_y], linewidth=0.5, c='black')

    # LWrist
    lwrist_mean_x, lwrist_mean_y = _draw_std_ellipse('LWrist', ax_nstd, n_std)

    # Line between elbow and wrist
    plt.plot([lelb_mean_x, lwrist_mean_x], [lelb_mean_y, lwrist_mean_y], linewidth=0.5, c='black')

    # MidHip
    xy = np.array(dict_norm_keypoints_xy['MidHip'])[:, 0:2]
    midhip_mean_x, midhip_mean_y = np.mean(xy, axis=0)
    ax_nstd.scatter([midhip_mean_x], [midhip_mean_y], s=1, c='black')

    # Line between neck and midhip
    plt.plot([312, midhip_mean_x], [198, midhip_mean_y], linewidth=0.5, c='black')

    # RHip
    rhip_mean_x, rhip_mean_y = _draw_std_ellipse('RHip', ax_nstd, n_std)

    # Line between rhip and midhip
    plt.plot([rhip_mean_x, midhip_mean_x], [rhip_mean_y, midhip_mean_y], linewidth=0.5, c='black')

    # LHip
    lhip_mean_x, lhip_mean_y = _draw_std_ellipse('LHip', ax_nstd, n_std)

    # Line between lhip and midhip
    plt.plot([lhip_mean_x, midhip_mean_x], [lhip_mean_y, midhip_mean_y], linewidth=0.5, c='black')

    # RKnee
    rknee_mean_x, rknee_mean_y = _draw_std_ellipse('RKnee', ax_nstd, n_std)

    # Line between rhip and knee
    plt.plot([rhip_mean_x, rknee_mean_x], [rhip_mean_y, rknee_mean_y], linewidth=0.5, c='black')

    # RAnkle
    rankle_mean_x, rankle_mean_y = _draw_std_ellipse('RAnkle', ax_nstd, n_std)

    # Line between ankle and knee
    plt.plot([rankle_mean_x, rknee_mean_x], [rankle_mean_y, rknee_mean_y], linewidth=0.5, c='black')

    # LKnee
    lknee_mean_x, lknee_mean_y = _draw_std_ellipse('LKnee', ax_nstd, n_std)

    # Line between lhip and knee
    plt.plot([lhip_mean_x, lknee_mean_x], [lhip_mean_y, lknee_mean_y], linewidth=0.5, c='black')

    # LAnkle
    lankle_mean_x, lankle_mean_y = _draw_std_ellipse('LAnkle', ax_nstd, n_std)

    # Line between ankle and knee
    plt.plot([lankle_mean_x, lknee_mean_x], [lankle_mean_y, lknee_mean_y], linewidth=0.5, c='black')

    fname = 'pose_std{}_{}.png'.format(n_std, gender)
    plt.savefig(os.path.join('pix', fname))


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
    # keypoints = _rotate_to_tpose(keypoints)

    return keypoints


def translate_keypoints(keypoints, dict_norm_keypoints_xy, show):

    # Scaler
    dist_from_nose_to_neck = _euclidian(keypoints['Nose'], keypoints['Neck'])
    scaler = std_head_height / dist_from_nose_to_neck

    # Universal reference point
    reference_point = np.array(keypoints['Neck']) + np.array([0, -50, 0])

    # Neck
    # neck_y: 198
    norm_neck_xy = [312, 198]

    # Nose
    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['Nose'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_nose_to_neck*scaler])
    norm_nose_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['Nose'].append(norm_nose_xy)

    # RShoulder
    dist_from_rsho_to_neck = _euclidian(keypoints['RShoulder'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['RShoulder'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_rsho_to_neck*scaler])
    norm_rsho_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['RShoulder'].append(norm_rsho_xy)

    # RElbow
    dist_from_relb_to_neck = _euclidian(keypoints['RElbow'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['RElbow'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_relb_to_neck * scaler])
    norm_relb_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['RElbow'].append(norm_relb_xy)

    # RWrist
    dist_from_rwrist_to_neck = _euclidian(keypoints['RWrist'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['RWrist'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_rwrist_to_neck * scaler])
    norm_rwrist_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['RWrist'].append(norm_rwrist_xy)

    # LShoulder
    dist_from_lsho_to_neck = _euclidian(keypoints['LShoulder'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['LShoulder'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_lsho_to_neck * scaler])
    norm_lsho_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['LShoulder'].append(norm_lsho_xy)

    # LElbow
    dist_from_lelb_to_neck = _euclidian(keypoints['LElbow'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['LElbow'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_lelb_to_neck * scaler])
    norm_lelb_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['LElbow'].append(norm_lelb_xy)

    # LWrist
    dist_from_lwrist_to_neck = _euclidian(keypoints['LWrist'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['LWrist'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_lwrist_to_neck * scaler])
    norm_lwrist_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['LWrist'].append(norm_lwrist_xy)

    # MidHip
    dist_from_midhip_to_neck = _euclidian(keypoints['MidHip'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['MidHip'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_midhip_to_neck * scaler])
    norm_midhip_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['MidHip'].append(norm_midhip_xy)

    # RHip
    dist_from_rhip_to_neck = _euclidian(keypoints['RHip'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['RHip'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_rhip_to_neck * scaler])
    norm_rhip_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['RHip'].append(norm_rhip_xy)

    # LHip
    dist_from_lhip_to_neck = _euclidian(keypoints['LHip'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['LHip'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_lhip_to_neck * scaler])
    norm_lhip_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['LHip'].append(norm_lhip_xy)

    # RKnee
    dist_from_rknee_to_neck = _euclidian(keypoints['RKnee'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['RKnee'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_rknee_to_neck * scaler])
    norm_rknee_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['RKnee'].append(norm_rknee_xy)

    # RAnkle
    dist_from_rankle_to_neck = _euclidian(keypoints['RAnkle'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['RAnkle'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_rankle_to_neck * scaler])
    norm_rankle_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['RAnkle'].append(norm_rankle_xy)

    # LKnee
    dist_from_lknee_to_neck = _euclidian(keypoints['LKnee'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['LKnee'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_lknee_to_neck * scaler])
    norm_lknee_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['LKnee'].append(norm_lknee_xy)

    # LAnkle
    dist_from_lankle_to_neck = _euclidian(keypoints['LAnkle'], keypoints['Neck'])

    rad, deg = _calc_angle(point1=reference_point, center=keypoints['Neck'], point2=keypoints['LAnkle'])

    reference_norm_point = np.array(norm_neck_xy) + np.array([0, -dist_from_lankle_to_neck * scaler])
    norm_lankle_xy = _rotate(point=reference_norm_point, center=norm_neck_xy, rad=rad)

    dict_norm_keypoints_xy['LAnkle'].append(norm_lankle_xy)


    if show:
        show_keypoints_in_norm(norm_neck_xy, norm_nose_xy, norm_rsho_xy, norm_relb_xy, norm_rwrist_xy,
                               norm_lsho_xy, norm_lelb_xy, norm_lwrist_xy, norm_midhip_xy, norm_rhip_xy, norm_lhip_xy,
                               norm_rknee_xy, norm_rankle_xy, norm_lknee_xy, norm_lankle_xy)

    return dict_norm_keypoints_xy


def normalize_keypoints(image_id, dict_norm_keypoints_xy, show):

    global people_count

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

        # keypoints
        keypoints = np.array(dp_annotation['keypoints']).astype(int)
        keypoints = _translate_keypoints_to_bbox(keypoints=keypoints, bbox_xywh=bbox_xywh)

        # check the validity of keypoints
        if not _is_valid(keypoints=keypoints):
            continue

        if show:
            show_keypoints_by_bbox(image_fpath, bbox_xywh, keypoints)

        # get the normalized image
        keypoints = rotate_keypoints(keypoints)

        dict_norm_keypoints_xy = translate_keypoints(keypoints,
                                                     dict_norm_keypoints_xy=dict_norm_keypoints_xy,
                                                     show=show)
        # valid person -> count_of_people++
        people_count += 1

    return dict_norm_keypoints_xy


if __name__ == '__main__':

    # common setting
    dp_img_category = 'man'  # man or woman
    n_std = 0.5

    # standard head height to calcuclate scaler!
    if dp_img_category == 'man':
        std_head_height = 62
    elif dp_img_category == 'woman':
        std_head_height = 58
    else:
        std_head_height = 50

    # option 1 - images within a range
    dp_img_range = slice(0, None)
    dp_img_ids = get_img_ids_by_caption(dp_img_category=dp_img_category, dp_img_range=dp_img_range)

    # option 2 - image from a directory
    # img_dir = os.path.join('datasets', dp_img_category)
    # dp_img_ids = get_img_ids_by_dir(indir=img_dir)

    # count of people
    people_count = 0

    # list of keypoints
    dict_norm_keypoints_xy = {
        'Nose': [],
        'RShoulder': [],
        'RElbow': [],
        'RWrist': [],
        'LShoulder': [],
        'LElbow': [],
        'LWrist': [],
        'MidHip': [],
        'RHip': [],
        'LHip': [],
        'RKnee': [],
        'RAnkle': [],
        'LKnee': [],
        'LAnkle': []
    }

    for image_id in dp_img_ids:
        dict_norm_keypoints_xy = normalize_keypoints(image_id=image_id,
                                                     dict_norm_keypoints_xy=dict_norm_keypoints_xy,
                                                     show=False)

    # show the standard deviation image
    show_std_image(dict_norm_keypoints_xy, n_std=n_std, gender=dp_img_category)

    # logs
    print('Total number of images:', len(dp_img_ids))
    print('Total number of people:', people_count)