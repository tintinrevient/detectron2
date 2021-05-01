import cv2
import numpy as np
from pycocotools.coco import COCO
import os
import glob
import pycocotools.mask as mask_util
import infer_segm
from densepose.structures import DensePoseDataRelative


# dataset setting
coco_folder = os.path.join('datasets', 'coco')

# dense_pose annotation
dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_train2014.json'))

# caption annotation
caption_coco = COCO(os.path.join(coco_folder, 'annotations', 'captions_train2014.json'))

# image shape
image_w_and_h = 624

# joint id
JOINT_ID = [
    'Nose', 'LEye', 'REye', 'LEar', 'REar',
    'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
    'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
]


def _is_valid(keypoints):

    # check the scores for each main keypoint, which MUST exist!
    # main_keypoints = BODY BOX
    main_keypoints = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'MidHip',
                      'RElbow', 'LElbow', 'RWrist', 'LWrist', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle']

    # filter the main keypoints by score > 0
    filtered_keypoints = [key for key, value in keypoints.items() if key in main_keypoints and value[2] > 0]
    print('Number of valid keypoints (must be equal to 15):', len(filtered_keypoints))

    if len(filtered_keypoints) != 15:
        return False
    else:
        return True


def _get_dp_mask(polys):

    mask_gen = np.zeros([256,256])

    for i in range(1,15):

        if(polys[i-1]):
            current_mask = mask_util.decode(polys[i-1])
            mask_gen[current_mask>0] = i

    return mask_gen


def _translate_keypoints_to_bbox(keypoints, bbox_xywh):

    # translate keypoints to bbox
    x, y, w, h = bbox_xywh
    # numpy's slicing = start:stop:step.
    keypoints = dict(zip(JOINT_ID, zip(keypoints[0::3].copy() - x, keypoints[1::3].copy() - y, keypoints[2::3].copy())))

    # infer the keypoints of neck and midhip, which are missing!
    keypoints['Neck'] = tuple(((np.array(keypoints['LShoulder']) + np.array(keypoints['RShoulder'])) / 2).astype(int))
    keypoints['MidHip'] = tuple(((np.array(keypoints['LHip']) + np.array(keypoints['RHip'])) / 2).astype(int))

    return keypoints


def _show_bbox(segm):

    # show bbox
    segm_scaled = segm.astype(np.float32) * 15
    segm_scaled_8u = segm_scaled.clip(0, 255).astype(np.uint8)

    # apply cmap
    segm_vis = cv2.applyColorMap(segm_scaled_8u, cv2.COLORMAP_PARULA)

    window_bbox = 'bbox'
    cv2.imshow(window_bbox, segm_vis)
    cv2.setWindowProperty(window_bbox, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _show_full_image(segm, annotation, im_output, caption_coco):

    # print annotations
    caption_annotation_ids = caption_coco.getAnnIds(imgIds=annotation['image_id'])
    caption_annotations = caption_coco.loadAnns(caption_annotation_ids)
    print([caption_annotation['caption'] for caption_annotation in caption_annotations])

    # bbox
    bbox_xywh = np.array(annotation["bbox"]).astype(int)
    x1, y1, x2, y2 = bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]

    x2 = min([x2, im_output.shape[1]])
    y2 = min([y2, im_output.shape[0]])

    # show original gray image
    mask_bool = np.tile((segm == 0)[:, :, np.newaxis], [1, 1, 3])

    # replace the visualized mask image with I_vis.
    mask_vis = cv2.applyColorMap((segm * 15).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
    mask_vis[mask_bool] = im_output[y1:y2, x1:x2, :][mask_bool]
    im_output[y1:y2, x1:x2, :] = im_output[y1:y2, x1:x2, :] * 0.3 + mask_vis * 0.7

    # draw keypoints
    keypoints = np.array(annotation['keypoints']).astype(int)

    for x, y, score in list(zip(keypoints[0::3], keypoints[1::3], keypoints[2::3])):
        if score > 0:
            cv2.circle(im_output, (x, y), radius=3, color=(255, 0, 255), thickness=-1)

    window_input = 'input'
    cv2.imshow(window_input, im_output)
    cv2.setWindowProperty(window_input, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_norm_segm_from_coco(dp_coco, caption_coco, image_id, image_mean, is_vitruve, is_rect, is_man, show):

    entry = dp_coco.loadImgs(image_id)[0]

    dataset_name = entry['file_name'][entry['file_name'].find('_')+1:entry['file_name'].rfind('_')]
    image_fpath = os.path.join(coco_folder, dataset_name, entry['file_name'])

    print('image_fpath:', image_fpath)

    dp_annotation_ids = dp_coco.getAnnIds(imgIds=entry['id'])
    dp_annotations = dp_coco.loadAnns(dp_annotation_ids)

    im_gray = cv2.imread(image_fpath, cv2.IMREAD_GRAYSCALE)
    im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])

    im_output = im_gray.copy()

    # imposed image for all people in one image!
    # sum of images
    image_sum = np.empty((image_w_and_h, image_w_and_h, 4), np.float32)
    image_sum.fill(0)
    # count of images
    count = 0
    is_updated = False

    # deviation of images
    image_deviation_sum = np.empty((image_w_and_h, image_w_and_h), np.float32)
    image_deviation_sum.fill(0)

    # iterate through all the people in one image
    for dp_annotation in dp_annotations:

        is_valid, _ = DensePoseDataRelative.validate_annotation(dp_annotation)

        if not is_valid:
            continue

        # bbox
        bbox_xywh = np.array(dp_annotation["bbox"]).astype(int)

        # keypoints
        keypoints = np.array(dp_annotation['keypoints']).astype(int)
        keypoints = _translate_keypoints_to_bbox(keypoints=keypoints, bbox_xywh=bbox_xywh)

        if not _is_valid(keypoints=keypoints):
            continue

        # if we have dense_pose annotation for this annotation
        if ('dp_masks' in dp_annotation.keys()):

            mask = _get_dp_mask(dp_annotation['dp_masks'])

            x1, y1, x2, y2 = bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]

            x2 = min([x2, im_gray.shape[1]])
            y2 = min([y2, im_gray.shape[0]])

            segm = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)

            if show:
                # show the original gray image
                dp_annotation['image_id'] = entry['id']
                _show_full_image(segm, dp_annotation, im_output, caption_coco)

                # show bbox
                # _show_bbox(segm)

        # visualize normalized pose - per person
        # rotate to t-pose
        segments_xy = infer_segm.rotate_segments_xy(segm=segm, keypoints=keypoints)

        # draw segments in normalized image
        image = infer_segm.draw_segments_xy(segments_xy=segments_xy, is_vitruve=is_vitruve, is_rect=is_rect, is_man=is_man)

        if show:
            window_norm = 'norm'
            cv2.imshow(window_norm, image)
            cv2.setWindowProperty(window_norm, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # update the sum and count of images
        image_sum = np.array(image) + np.array(image_sum)
        count += 1
        is_updated = True

        # update the deviation of images
        if image_mean is not None:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            image_deviation_sum = ((np.array(image_gray) - np.array(image_mean)) ** 2) + np.array(image_deviation_sum)

    if is_updated:
        return image_sum, count, image_deviation_sum
    else:
        return None, 0, None


def visualize_mean(dp_coco, caption_coco, image_ids, output_fn, is_vitruve, is_rect, is_man, show):

    # calculate the mean of the COCO poses
    image_mean = np.empty((image_w_and_h, image_w_and_h, 4), np.float32)
    image_mean.fill(0)

    # total count
    count = 0

    # per image
    for idx, image_id in enumerate(image_ids):
        print('Current number of images:', (idx+1))

        try:
            # per person
            image_sum, image_count, _ = generate_norm_segm_from_coco(dp_coco=dp_coco, caption_coco=caption_coco, image_id=image_id, image_mean=None,
                                                                     is_vitruve=is_vitruve, is_rect=is_rect, is_man=is_man, show=show)

            if image_count > 0:
                count += image_count
                print('Current number of people:', count)
                image_mean = np.array(image_sum) + np.array(image_mean)
        except:
            continue

    if count > 0:
        print('Total number of people:', count)
        image_mean = (np.array(image_mean) / count).astype(int)
        # image_mean[..., :] = np.clip(image_mean[..., :], 0, 255)
        # image_mean_norm = np.array(image_mean, dtype=np.uint8)
        image_mean_norm = cv2.normalize(image_mean, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # show the image
        window_mean = 'image_mean'
        cv2.imshow(window_mean, image_mean_norm)
        cv2.setWindowProperty(window_mean, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save the image
        cv2.imwrite(output_fn, image_mean_norm)

        return image_mean_norm


def visualize_std(dp_coco, caption_coco, image_ids, image_mean, output_fn, is_vitruve, is_rect, is_man, show):

    # convert to grayscale
    image_mean = cv2.cvtColor(image_mean, cv2.COLOR_BGRA2GRAY)

    # calculate the standard deviation of the COCO poses
    image_std = np.empty((image_w_and_h, image_w_and_h), np.float32)
    image_std.fill(0)

    # total count
    count = 0

    # per image
    for idx, image_id in enumerate(image_ids):
        print('Current number of images:', (idx + 1))

        try:
            # per person
            _, image_count, image_deviation_sum = generate_norm_segm_from_coco(dp_coco=dp_coco, caption_coco=caption_coco, image_id=image_id, image_mean=image_mean,
                                                                               is_vitruve=is_vitruve, is_rect=is_rect, is_man=is_man, show=show)

            if image_count > 0:
                count += image_count
                print('Current number of people:', count)
                image_std = np.array(image_deviation_sum) + np.array(image_std)
        except:
            continue

    if count > 1:
        print('Total number of people:', count)
        image_std = np.sqrt((np.array(image_std) / (count - 1))).astype(int)
        image_std_norm = cv2.normalize(image_std, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # show the image
        window_std = 'image_std'
        cv2.imshow(window_std, image_std_norm)
        cv2.setWindowProperty(window_std, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # save the image
        cv2.imwrite(output_fn, image_std_norm)


def get_img_ids_by_caption(dp_img_category, dp_img_range):

    if dp_img_category == 'man':

        # images of only men
        man_list_img_ids = filter_by_caption(dp_coco=dp_coco, caption_coco=caption_coco, yes_word_list=['man'],
                                             no_word_list=['woman'])

        print('Number of images with only men:', len(man_list_img_ids))

        dp_img_ids = man_list_img_ids[dp_img_range]

    elif dp_img_category == 'woman':

        # images of only women
        woman_list_img_ids = filter_by_caption(dp_coco=dp_coco, caption_coco=caption_coco, yes_word_list=['woman'],
                                               no_word_list=['man'])

        print('Number of images with only women:', len(woman_list_img_ids))

        dp_img_ids = woman_list_img_ids[dp_img_range]

    return dp_img_ids


def get_img_ids_by_dir(indir):

    dp_img_ids = []

    for fname in glob.glob(indir + '/*.jpg'):

        img_id = int(fname[fname.rfind('_')+1:fname.rfind('.')])

        dp_img_ids.append(img_id)

    return dp_img_ids


def visualize_dist(dp_img_category, dp_img_ids, is_vitruve, is_rect, show):

    if dp_img_category == 'man':
        is_man = True
    elif dp_img_category == 'woman':
        is_man = False

    if is_rect:
        dp_img_block = 'rect'
    else:
        dp_img_block = 'convex'

    # visualize the mean of images
    image_mean_output_fn = os.path.join('pix', '{}_vitruve_mean_{}.png'.format(dp_img_category, dp_img_block))
    image_mean = visualize_mean(dp_coco=dp_coco, caption_coco=caption_coco,
                                image_ids=dp_img_ids, output_fn=image_mean_output_fn,
                                is_vitruve=is_vitruve, is_rect=is_rect, is_man=is_man, show=show)

    # visualize the standard deviation of images
    image_std_output_fn = os.path.join('pix', '{}_vitruve_std_{}.png'.format(dp_img_category, dp_img_block))
    visualize_std(dp_coco=dp_coco, caption_coco=caption_coco,
                  image_ids=dp_img_ids, image_mean=image_mean, output_fn=image_std_output_fn,
                  is_vitruve=is_vitruve, is_rect=is_rect, is_man=is_man, show=False)


def filter_by_caption(dp_coco, caption_coco, yes_word_list, no_word_list):

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


def extract_contour_on_vitruve():

    # output the image
    fname_vitruve_contour = os.path.join('pix', 'vitruve_contour.png')

    # read the image
    fname_vitruve_norm = os.path.join('pix', 'vitruve_norm.png')

    image = cv2.imread(fname_vitruve_norm, 0)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

    # drawing setting
    radius_keypoint = 3
    radius_midpoint = 3
    thickness_contour = 1

    color_keypoint = (0, 255, 0)
    color_midpoint = (255, 0, 255)
    color_contour = (255, 0, 255)

    color_line = (0, 255, 255)

    # height of the man
    height = 500

    # default setting
    head_top_y = 115
    mid_x = 312

    arm_line_y = 217
    arm_half_width = 15
    arm_line_upper_y = int(arm_line_y - arm_half_width)
    arm_line_lower_y = int(arm_line_y + arm_half_width)

    right_leg_x = 282
    left_leg_x = 330
    leg_half_width = 20

    right_leg_line_left_x = int(right_leg_x - leg_half_width)
    right_leg_line_right_x = int(right_leg_x + leg_half_width)
    left_leg_line_left_x = int(left_leg_x - leg_half_width)
    left_leg_line_right_x = int(left_leg_x + leg_half_width)

    ankle_margin_y = 17
    torso_margin_y = 10

    # lines
    # line along the height
    feet_bottom_y = int(head_top_y + height)
    cv2.line(image, (mid_x, head_top_y), (mid_x, feet_bottom_y), color=color_line, thickness=1)

    # line along the arm
    right_arm_x = int(mid_x - height/2)
    left_arm_x = int(mid_x + height/2)
    cv2.line(image, (right_arm_x, arm_line_y), (left_arm_x, arm_line_y), color=color_line, thickness=1)

    # line along the top of the head
    cv2.line(image, (right_arm_x, head_top_y), (left_arm_x, head_top_y), color=color_line, thickness=1)

    # line along the bottom of the feet
    cv2.line(image, (right_arm_x, feet_bottom_y), (left_arm_x, feet_bottom_y), color=color_line, thickness=1)

    # line along the chin
    chin_y = int(head_top_y + height/8)
    cv2.line(image, (right_arm_x, chin_y), (left_arm_x, chin_y), color=color_line, thickness=1)

    # keypoints
    # keypoint of neck
    neck_y = int(head_top_y + height/6)
    cv2.circle(image, (mid_x, neck_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    # keypoint of midhip
    midhip_y = int(feet_bottom_y - height/2)
    cv2.circle(image, (mid_x, midhip_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    # keypoint of knees
    knee_y = int(midhip_y + height/4)
    cv2.circle(image, (mid_x, knee_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    # keypoint of ankles
    ankle_y = feet_bottom_y - ankle_margin_y
    cv2.circle(image, (mid_x, ankle_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    # keypoint of wrists
    rwrist_x = int(right_arm_x + height/10)
    cv2.circle(image, (rwrist_x, arm_line_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    lwrist_x = int(left_arm_x - height/10)
    cv2.circle(image, (lwrist_x, arm_line_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    # keypoint of elbows
    relb_x = int(right_arm_x + height/4)
    cv2.circle(image, (relb_x, arm_line_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    lelb_x = int(left_arm_x - height/4)
    cv2.circle(image, (lelb_x, arm_line_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    # keypoint of shoulders = armpits
    rsho_x = int(right_arm_x + height*3/8)
    cv2.circle(image, (rsho_x, arm_line_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    lsho_x = int(left_arm_x - height * 3 / 8)
    cv2.circle(image, (lsho_x, arm_line_y), radius=radius_keypoint, color=color_keypoint, thickness=-1)

    # draw midpoints and contour on a white image
    # white image
    image = np.empty((624, 624, 4), np.uint8)
    image.fill(255)

    # midpoints
    # centroid of head
    nose_y = int((head_top_y + chin_y) / 2)
    cv2.circle(image, (mid_x, nose_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)

    # midpoint of torso
    mid_torso_y = int((midhip_y + neck_y) / 2)
    cv2.circle(image, (mid_x, mid_torso_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)

    # midpoint of thighs
    mid_thigh_y = int((midhip_y + knee_y) / 2)
    cv2.circle(image, (right_leg_x, mid_thigh_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)
    cv2.circle(image, (left_leg_x, mid_thigh_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)

    # midpoint of calves
    mid_calf_y = int((knee_y + ankle_y) / 2)
    cv2.circle(image, (right_leg_x, mid_calf_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)
    cv2.circle(image, (left_leg_x, mid_calf_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)

    # midpoint of arms
    mid_rlower_arm_x = int((rwrist_x + relb_x)/2)
    mid_rupper_arm_x = int((relb_x + rsho_x)/2)
    mid_llower_arm_x = int((lwrist_x + lelb_x)/2)
    mid_lupper_arm_x = int((lelb_x + lsho_x)/2)

    cv2.circle(image, (mid_rlower_arm_x, arm_line_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)
    cv2.circle(image, (mid_rupper_arm_x, arm_line_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)
    cv2.circle(image, (mid_llower_arm_x, arm_line_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)
    cv2.circle(image, (mid_lupper_arm_x, arm_line_y), radius=radius_midpoint, color=color_midpoint, thickness=-1)

    print('nose_y', nose_y)
    print('torso_y', mid_torso_y)
    print('rupper_arm_x', mid_rupper_arm_x)
    print('rlower_arm_x', mid_rlower_arm_x)
    print('lupper_arm_x', mid_lupper_arm_x)
    print('llower_arm_x', mid_llower_arm_x)
    print('thigh_y', mid_thigh_y)
    print('calf_y', mid_calf_y)

    # contour of head
    head_radius = int(chin_y - nose_y)
    print('head_radius:', head_radius)
    cv2.circle(image, (mid_x, nose_y), radius=head_radius, color=color_contour, thickness=thickness_contour)

    # contour of torso
    cv2.rectangle(image, (int(rsho_x + torso_margin_y), neck_y), (int(lsho_x - torso_margin_y), midhip_y), color=color_contour, thickness=thickness_contour)

    # contour of arms
    cv2.rectangle(image, (rwrist_x, arm_line_upper_y), (relb_x, arm_line_lower_y), color=color_contour, thickness=thickness_contour)
    cv2.rectangle(image, (relb_x, arm_line_upper_y), (rsho_x, arm_line_lower_y), color=color_contour, thickness=thickness_contour)
    cv2.rectangle(image, (lwrist_x, arm_line_upper_y), (lelb_x, arm_line_lower_y), color=color_contour, thickness=thickness_contour)
    cv2.rectangle(image, (lelb_x, arm_line_upper_y), (lsho_x, arm_line_lower_y), color=color_contour, thickness=thickness_contour)

    # contour of legs
    cv2.rectangle(image, (right_leg_line_left_x, midhip_y), (right_leg_line_right_x, knee_y), color=color_contour, thickness=thickness_contour)
    cv2.rectangle(image, (right_leg_line_left_x, knee_y), (right_leg_line_right_x, ankle_y), color=color_contour, thickness=thickness_contour)
    cv2.rectangle(image, (left_leg_line_left_x, midhip_y), (left_leg_line_right_x, knee_y), color=color_contour, thickness=thickness_contour)
    cv2.rectangle(image, (left_leg_line_left_x, knee_y), (left_leg_line_right_x, ankle_y), color=color_contour, thickness=thickness_contour)

    winname = 'vitruve'
    cv2.imshow(winname, image)
    cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite(fname_vitruve_contour, image)


def impose_dist_on_vitruve(fname_dist):

    # output
    fname_output = '{}_bounded.png'.format(fname_dist[:fname_dist.rfind('.')])

    # read the contour of vitruve
    fname_vitruve_contour = os.path.join('pix', 'vitruve_contour.png')
    image_vitruve_contour = cv2.imread(fname_vitruve_contour)

    # read the distribution of the poses
    image_dist = cv2.imread(fname_dist)

    # overlay
    added_image = cv2.addWeighted(image_vitruve_contour, 0.1, image_dist, 0.9, 0)

    winname = 'imposed dist'
    cv2.imshow(winname, added_image)
    cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite(fname_output, added_image)


if __name__ == '__main__':

    # bugs
    # dp_img_ids = [558114, 262710]

    # common setting
    dp_img_category = 'woman' # man or woman
    is_vitruve = False
    is_rect = False

    # option 1 - images within a range
    # dp_img_range = slice(0, 10)
    # dp_img_ids = get_img_ids_by_caption(dp_img_category=dp_img_category, dp_img_range=dp_img_range)

    # option 2 - image from a directory
    img_dir = os.path.join('datasets', dp_img_category)
    dp_img_ids = get_img_ids_by_dir(indir=img_dir)


    # visualize the mean and std of all the poses
    visualize_dist(dp_img_category=dp_img_category, dp_img_ids=dp_img_ids,
                   is_vitruve=is_vitruve, is_rect=is_rect, show=True)


    # superimpose the distribution on the contour of vitruve
    # fname_dist = os.path.join('pix', '{}_vitruve_mean_{}.png'.format(dp_img_category, 'rect'))
    # impose_dist_on_vitruve(fname_dist)


    # used only for ONCE to generate the contour and midpoints for the vitruve canon!
    # extract the contour from vitruve
    # extract_contour_on_vitruve()