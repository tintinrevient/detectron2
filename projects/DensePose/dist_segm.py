import cv2
import numpy as np
from pycocotools.coco import COCO
import os
import pycocotools.mask as mask_util
import infer_segm
from densepose.structures import DensePoseDataRelative

# image_w_and_h = 2000
image_w_and_h = 624

JOINT_ID = [
    'Nose', 'LEye', 'REye', 'LEar', 'REar',
    'LShoulder', 'RShoulder', 'LElbow', 'RElbow', 'LWrist', 'RWrist',
    'LHip', 'RHip', 'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
]


def _is_valid(keypoints):

    # check the scores for each main keypoint, which MUST exist!
    # main_keypoints = BODY BOX
    main_keypoints = ['Nose', 'Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip', 'MidHip']

    # filter the main keypoints by score > 0
    filtered_keypoints = [key for key, value in keypoints.items() if key in main_keypoints and value[2] > 0]
    print('Number of valid keypoints (must be equal to 7):', len(filtered_keypoints))

    if len(filtered_keypoints) != 7:
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


def generate_norm_segm_from_coco(image_id, show, image_mean):

    print('image_id:', image_id)

    entry = dp_coco.loadImgs(image_id)[0]
    image_fpath = os.path.join(coco_folder, 'val2014', entry['file_name'])
    print('image_fpath:', image_fpath)

    annotation_ids = dp_coco.getAnnIds(imgIds=entry['id'])
    annotations = dp_coco.loadAnns(annotation_ids)

    im_gray = cv2.imread(image_fpath, cv2.IMREAD_GRAYSCALE)
    im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])

    im_output = im_gray.copy()

    # imposed image for all people in one image!
    image_sum = np.empty((image_w_and_h, image_w_and_h, 4), np.float32)
    image_sum.fill(0)
    image_deviation_sum = np.empty((image_w_and_h, image_w_and_h, 4), np.float32)
    image_deviation_sum.fill(0)
    count = 0
    is_updated = False

    # iterate through all the people in one image
    for annotation in annotations:

        is_valid, _ = DensePoseDataRelative.validate_annotation(annotation)

        if not is_valid:
            continue

        # bbox
        bbox_xywh = np.array(annotation["bbox"]).astype(int)

        # keypoints
        keypoints = np.array(annotation['keypoints']).astype(int)
        keypoints = _translate_keypoints_to_bbox(keypoints=keypoints, bbox_xywh=bbox_xywh)

        if not _is_valid(keypoints=keypoints):
            continue

        if ('dp_masks' in annotation.keys()):  # If we have densepose annotation for this ann,

            mask = _get_dp_mask(annotation['dp_masks'])

            x1, y1, x2, y2 = bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]

            x2 = min([x2, im_gray.shape[1]])
            y2 = min([y2, im_gray.shape[0]])

            segm = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)

            if show:
                # show original gray image
                mask_bool = np.tile((segm == 0)[:, :, np.newaxis], [1, 1, 3])

                #  replace the visualized mask image with I_vis.
                mask_vis = cv2.applyColorMap((segm * 15).astype(np.uint8), cv2.COLORMAP_PARULA)[:, :, :]
                mask_vis[mask_bool] = im_output[y1:y2, x1:x2, :][mask_bool]
                im_output[y1:y2, x1:x2, :] = im_output[y1:y2, x1:x2, :] * 0.3 + mask_vis * 0.7

                window_input = 'input'
                cv2.imshow(window_input, im_output)
                cv2.setWindowProperty(window_input, cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

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


        # visualize normalized pose
        # rotate to t-pose
        segments_xy = infer_segm.rotate_segments_xy(segm=segm, keypoints=keypoints)

        # draw segments in normalized image
        image = infer_segm.draw_segments_xy(segments_xy=segments_xy, is_vitruve=True)

        if show:
            window_norm = 'norm'
            cv2.imshow(window_norm, image)
            cv2.setWindowProperty(window_norm, cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        count += 1
        image_sum = np.array(image) + np.array(image_sum)
        is_updated = True

        if image_mean is not None:
            image_deviation_sum = ((np.array(image) - np.array(image_mean)) ** 2) + np.array(image_deviation_sum)

    if is_updated:
        return image_sum, count, image_deviation_sum
    else:
        return None, 0, None


if __name__ == '__main__':

    coco_folder = os.path.join('datasets', 'coco')

    # caption
    caption_coco = COCO(os.path.join(coco_folder, 'annotations', 'captions_train2014.json'))
    caption_img_ids = caption_coco.getImgIds()

    # dense_pose
    # dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_minival2014.json'))
    dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_train2014.json'))
    dp_img_ids = dp_coco.getImgIds()

    # common images
    common_img_ids = list(set(caption_img_ids) & set(dp_img_ids))

    print('Number of caption images:', len(caption_img_ids))
    print('Number of dense_pose images:', len(dp_img_ids))
    print('Number of common images:', len(common_img_ids))

    man_img_ids = []
    woman_img_ids = []

    for img_id in common_img_ids:

        annotation_ids = caption_coco.getAnnIds(imgIds=img_id)
        annotations = caption_coco.loadAnns(annotation_ids)

        for annotation in annotations:
            # image id
            image_id = annotation['image_id']
            # caption = a list of lower words
            caption = annotation['caption'].lower().split()

            if 'man' in caption and 'woman' not in caption:
                man_img_ids.append(image_id)
                break
            elif 'woman' in caption and 'man' not in caption:
                woman_img_ids.append(image_id)
                break

    print('Number of images of man:', len(man_img_ids))
    print('Number of images of woman:', len(woman_img_ids))
    common_people_img_ids = list(set(man_img_ids) & set(woman_img_ids))
    print('Number of images of man and woman:', len(common_people_img_ids))

    # bugs
    # dp_img_ids = [558114]

    # test
    # dp_img_ids = [437239, 438304, 438774, 438862, 303713, 295138]
    dp_img_ids = [303713, 295138]

    # calculate the mean of the COCO poses
    image_mean = np.empty((image_w_and_h, image_w_and_h, 4), np.float32)
    image_mean.fill(0)
    image_std = np.empty((image_w_and_h, image_w_and_h, 4), np.float32)
    image_std.fill(0)
    count = 0

    for img_id in man_img_ids:

        try:
            image_sum, image_count, _ = generate_norm_segm_from_coco(image_id=img_id, show=False, image_mean=None)
            if image_count > 0:
                count += image_count
                image_mean = np.array(image_sum) + np.array(image_mean)
        except:
            continue

    image_mean = (np.array(image_mean) / count).astype(int)
    image_mean_norm = cv2.normalize(image_mean, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    window_mean = 'image_mean'
    cv2.imshow(window_mean, image_mean_norm)
    cv2.setWindowProperty(window_mean, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # calculate the standard deviation of the COCO poses
    # count = 0
    # for img_id in img_ids:
    #     try:
    #         _, image_count, image_deviation_sum = generate_norm_segm_from_coco(image_id=img_id, show=False, image_mean=image_mean_norm)
    #
    #         if count is not None:
    #             count += image_count
    #             image_std = np.array(image_deviation_sum) + np.array(image_std)
    #     except:
    #         continue
    #
    # image_std = np.sqrt((np.array(image_std) / (count-1))).astype(int)
    # image_std_norm = cv2.normalize(image_std, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # image_std_norm = cv2.cvtColor(image_std_norm, cv2.COLOR_RGBA2GRAY)
    #
    # window_std = 'image_std'
    # cv2.imshow(window_std, image_std_norm)
    # cv2.setWindowProperty(window_std, cv2.WND_PROP_TOPMOST, 1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()