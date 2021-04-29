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


def generate_norm_segm_from_coco(dp_coco, caption_coco, image_id, image_mean, is_vitruve, is_rect, show):

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
        image = infer_segm.draw_segments_xy(segments_xy=segments_xy, is_vitruve=is_vitruve, is_rect=is_rect)

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


def visualize_mean(dp_coco, caption_coco, image_ids, output_fn, is_vitruve, is_rect, show):

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
                                                                     is_vitruve=is_vitruve, is_rect=is_rect, show=show)

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


def visualize_std(dp_coco, caption_coco, image_ids, image_mean, output_fn, is_vitruve, is_rect, show):

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
                                                                               is_vitruve=is_vitruve, is_rect=is_rect, show=show)

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


def visualize_dist(dp_ds_name, caption_ds_name, dp_img_category, dp_img_range, is_vitruve, is_rect):

    # caption
    caption_coco = COCO(os.path.join(coco_folder, 'annotations', caption_ds_name))

    # dense_pose
    # dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_minival2014.json'))
    dp_coco = COCO(os.path.join(coco_folder, 'annotations', dp_ds_name))

    # images of only men
    man_list_img_ids = filter_by_caption(dp_coco=dp_coco, caption_coco=caption_coco, yes_word_list=['man'],
                                         no_word_list=['woman'])
    print('Number of images with only men:', len(man_list_img_ids))

    # images of only women
    woman_list_img_ids = filter_by_caption(dp_coco=dp_coco, caption_coco=caption_coco, yes_word_list=['woman'],
                                           no_word_list=['man'])
    print('Number of images with only women:', len(woman_list_img_ids))

    common_people_img_ids = list(set(man_list_img_ids) & set(woman_list_img_ids))
    print('Number of images with men and women:', len(common_people_img_ids))

    # bugs
    # dp_img_ids = [558114]

    if dp_img_category == 'man':
        dp_img_ids = man_list_img_ids[dp_img_range]
    else:
        dp_img_ids = woman_list_img_ids[dp_img_range]

    if is_rect:
        dp_img_block = 'rect'
    else:
        dp_img_block = 'convex'

    # visualize the mean of images
    image_mean_output_fn = os.path.join('pix', '{}_vitruve_mean_{}.png'.format(dp_img_category, dp_img_block))
    image_mean = visualize_mean(dp_coco=dp_coco, caption_coco=caption_coco,
                                image_ids=dp_img_ids, output_fn=image_mean_output_fn,
                                is_vitruve=is_vitruve, is_rect=is_rect, show=False)

    # dilate within contour - option 1
    # image_mean_contour_output_fn = os.path.join('pix', '{}_vitruve_mean_contour.png'.format(dp_img_name))
    # image_mean_gray = cv2.cvtColor(image_mean, cv2.COLOR_BGRA2GRAY)
    # ret, thresh = cv2.threshold(image_mean_gray, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # image_mean_contour = image_mean.copy()
    # cv2.drawContours(image_mean_contour, contours, -1, (0, 255, 0), 3)
    # cv2.imwrite(image_mean_contour_output_fn, image_mean_contour)

    # visualize the standard deviation of images
    image_std_output_fn = os.path.join('pix', '{}_vitruve_std_{}.png'.format(dp_img_category, dp_img_block))
    visualize_std(dp_coco=dp_coco, caption_coco=caption_coco,
                  image_ids=dp_img_ids, image_mean=image_mean, output_fn=image_std_output_fn,
                  is_vitruve=is_vitruve, is_rect=is_rect, show=False)


if __name__ == '__main__':

    # dataset setting
    coco_folder = os.path.join('datasets', 'coco')
    dp_ds_name = 'densepose_train2014.json'
    caption_ds_name = 'captions_train2014.json'

    # visualize the mean and std of all the poses
    dp_img_category = 'woman'
    dp_img_range = slice(0, 10)
    is_vitruve = False
    is_rect = False

    visualize_dist(dp_ds_name=dp_ds_name, caption_ds_name=caption_ds_name,
                   dp_img_category=dp_img_category, dp_img_range=dp_img_range,
                   is_vitruve=is_vitruve, is_rect=is_rect)
