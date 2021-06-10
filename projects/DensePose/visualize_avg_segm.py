import os, cv2, re
import numpy as np
import argparse
import pandas as pd
from infer_segm import (
    _draw_symmetrical_rect_segm, COARSE_TO_COLOR
)


# the path to the data of norm_segm.csv
fname_norm_segm = os.path.join('output', 'norm_segm.csv')


def _calc_avg_contour(df_norm_segm, artist):

    dict_avg_contour = {
        'Head_w': 0,
        'Head_h': 0,
        'Torso_w': 0,
        'Torso_h': 0,
        'RUpperArm_w': 0,
        'RUpperArm_h': 0,
        'RLowerArm_w': 0,
        'RLowerArm_h': 0,
        'LUpperArm_w': 0,
        'LUpperArm_h': 0,
        'LLowerArm_w': 0,
        'LLowerArm_h': 0,
        'RThigh_w': 0,
        'RThigh_h': 0,
        'RCalf_w': 0,
        'RCalf_h': 0,
        'LThigh_w': 0,
        'LThigh_h': 0,
        'LCalf_w': 0,
        'LCalf_h': 0
    }

    # step 1: calculate the sum of width and height
    dict_norm_segm_by_artist = df_norm_segm[df_norm_segm.index.str.contains(artist)]
    for index, row in dict_norm_segm_by_artist.iterrows():

        dict_avg_contour['Head_w'] += row['Head_w']
        dict_avg_contour['Head_h'] += row['Head_h']
        dict_avg_contour['Torso_w'] += row['Torso_w']
        dict_avg_contour['Torso_h'] += row['Torso_h']

        dict_avg_contour['RUpperArm_w'] += row['RUpperArm_w']
        dict_avg_contour['RUpperArm_h'] += row['RUpperArm_h']
        dict_avg_contour['RLowerArm_w'] += row['RLowerArm_w']
        dict_avg_contour['RLowerArm_h'] += row['RLowerArm_h']

        dict_avg_contour['LUpperArm_w'] += row['LUpperArm_w']
        dict_avg_contour['LUpperArm_h'] += row['LUpperArm_h']
        dict_avg_contour['LLowerArm_w'] += row['LLowerArm_w']
        dict_avg_contour['LLowerArm_h'] += row['LLowerArm_h']

        dict_avg_contour['RThigh_w'] += row['RThigh_w']
        dict_avg_contour['RThigh_h'] += row['RThigh_h']
        dict_avg_contour['RCalf_w'] += row['RCalf_w']
        dict_avg_contour['RCalf_h'] += row['RCalf_h']

        dict_avg_contour['LThigh_w'] += row['LThigh_w']
        dict_avg_contour['LThigh_h'] += row['LThigh_h']
        dict_avg_contour['LCalf_w'] += row['LCalf_w']
        dict_avg_contour['LCalf_h'] += row['LCalf_h']

    # step 2: calculate the average width and height
    count = len(dict_norm_segm_by_artist.index)
    for key, value in dict_avg_contour.items():
        dict_avg_contour[key] = int(value / count)

    return dict_avg_contour


def _draw_norm_segm_on_avg_contour(dict_norm_segm, dict_avg_contour, infile):

    # normalized image = (624, 624, 4)
    image = np.empty((624, 624, 4), np.uint8)
    image.fill(255)  # => white (255, 255, 255, 255) = background with non-transparency

    # common settings
    # coordinates [x, y] coming from distribution_segm.extract_contour_on_vitruve()
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

    # head segment
    _draw_symmetrical_rect_segm(image,
                                segm_id='Head',
                                w_and_h=(int(dict_norm_segm['Head_w']), int(dict_norm_segm['Head_h'])),
                                ref_point=norm_nose_xy,
                                update_dict=False)

    # head contour
    rect = (norm_nose_xy,
            (dict_avg_contour['Head_w'] , dict_avg_contour['Head_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    # torso segment
    _draw_symmetrical_rect_segm(image,
                                segm_id='Torso',
                                w_and_h=(int(dict_norm_segm['Torso_w']), int(dict_norm_segm['Torso_h'])),
                                ref_point=norm_mid_torso_xy,
                                update_dict=False)

    # torso contour
    rect = (norm_mid_torso_xy,
            (dict_avg_contour['Torso_w'], dict_avg_contour['Torso_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    # upper limbs - segments
    _draw_symmetrical_rect_segm(image,
                                segm_id='RUpperArm',
                                w_and_h=(int(dict_norm_segm['RUpperArm_w']), int(dict_norm_segm['RUpperArm_h'])),
                                ref_point=norm_mid_rupper_arm_xy,
                                update_dict=False)

    _draw_symmetrical_rect_segm(image,
                                segm_id='RLowerArm',
                                w_and_h=(int(dict_norm_segm['RLowerArm_w']), int(dict_norm_segm['RLowerArm_h'])),
                                ref_point=norm_mid_rlower_arm_xy,
                                update_dict=False)

    _draw_symmetrical_rect_segm(image,
                                segm_id='LUpperArm',
                                w_and_h=(int(dict_norm_segm['LUpperArm_w']), int(dict_norm_segm['LUpperArm_h'])),
                                ref_point=norm_mid_lupper_arm_xy,
                                update_dict=False)

    _draw_symmetrical_rect_segm(image,
                                segm_id='LLowerArm',
                                w_and_h=(int(dict_norm_segm['LLowerArm_w']), int(dict_norm_segm['LLowerArm_h'])),
                                ref_point=norm_mid_llower_arm_xy,
                                update_dict=False)

    # upper limbs - contours
    rect = (norm_mid_rupper_arm_xy,
            (dict_avg_contour['RUpperArm_w'], dict_avg_contour['RUpperArm_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    rect = (norm_mid_rlower_arm_xy,
            (dict_avg_contour['RLowerArm_w'], dict_avg_contour['RLowerArm_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    rect = (norm_mid_lupper_arm_xy,
            (dict_avg_contour['LUpperArm_w'], dict_avg_contour['LUpperArm_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    rect = (norm_mid_llower_arm_xy,
            (dict_avg_contour['LLowerArm_w'], dict_avg_contour['LLowerArm_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    # lower limbs
    _draw_symmetrical_rect_segm(image,
                                segm_id='RThigh',
                                w_and_h=(int(dict_norm_segm['RThigh_w']), int(dict_norm_segm['RThigh_h'])),
                                ref_point=norm_mid_rthigh_xy,
                                update_dict=False)

    _draw_symmetrical_rect_segm(image,
                                segm_id='RCalf',
                                w_and_h=(int(dict_norm_segm['RCalf_w']), int(dict_norm_segm['RCalf_h'])),
                                ref_point=norm_mid_rcalf_xy,
                                update_dict=False)

    _draw_symmetrical_rect_segm(image,
                                segm_id='LThigh',
                                w_and_h=(int(dict_norm_segm['LThigh_w']), int(dict_norm_segm['LThigh_h'])),
                                ref_point=norm_mid_lthigh_xy,
                                update_dict=False)

    _draw_symmetrical_rect_segm(image,
                                segm_id='LCalf',
                                w_and_h=(int(dict_norm_segm['LCalf_w']), int(dict_norm_segm['LCalf_h'])),
                                ref_point=norm_mid_lcalf_xy,
                                update_dict=False)

    # lower limbs - contours
    rect = (norm_mid_rthigh_xy,
            (dict_avg_contour['RThigh_w'], dict_avg_contour['RThigh_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    rect = (norm_mid_rcalf_xy,
            (dict_avg_contour['RCalf_w'], dict_avg_contour['RCalf_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    rect = (norm_mid_lthigh_xy,
            (dict_avg_contour['LThigh_w'], dict_avg_contour['LThigh_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    rect = (norm_mid_lcalf_xy,
            (dict_avg_contour['LCalf_w'], dict_avg_contour['LCalf_h']),
            0)
    box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, color=color, thickness=thickness)

    # save and show the final image
    outfile = generate_outfile(infile)
    cv2.imwrite(outfile, image)

    image_window = 'image'
    cv2.imshow(image_window, image)
    cv2.setWindowProperty(image_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_outfile(infile):

    iter_list = [iter.start() for iter in re.finditer(r"/", infile)]
    category = infile[iter_list[0] + 1:iter_list[1]]
    artist = infile[iter_list[1] + 1:iter_list[2]]
    painting_number = infile[iter_list[2] + 1:infile.rfind('.')]

    outfile = os.path.join('output', 'pix', '', category, artist, '{}_on_avg_contour.jpg'.format(painting_number))

    return outfile


def visualize(infile, openpose_idx):

    # step 1: load the data of norm_segm
    df_norm_segm = pd.read_csv(fname_norm_segm, index_col=0)

    iter_list = [iter.start() for iter in re.finditer(r"/", infile)]
    artist = args.input[iter_list[1] + 1:iter_list[2]]
    painting_number = args.input[iter_list[2] + 1:args.input.rfind('.')]
    index_name = '{}_{}_{}'.format(artist, painting_number, openpose_idx)

    dict_norm_segm = df_norm_segm.loc[index_name]
    print(dict_norm_segm)

    # step 2: calculate the average contour for this artist
    dict_avg_contour = _calc_avg_contour(df_norm_segm, artist)
    print(dict_avg_contour)

    # step 3: draw the norm_segm over the average contour
    _draw_norm_segm_on_avg_contour(dict_norm_segm, dict_avg_contour, infile)


if __name__ == '__main__':

    # settings
    thickness = 2
    color = (0, 255, 0)

    # modern
    # python visualize_avg_segm.py --input datasets/modern/Paul\ Delvaux/90551.jpg
    # python visualize_avg_segm.py --input datasets/modern/Paul\ Gauguin/30963.jpg

    # classical
    # python visualize_avg_segm.py --input datasets/classical/Michelangelo/12758.jpg
    # python visualize_avg_segm.py --input datasets/classical/Artemisia\ Gentileschi/45093.jpg

    parser = argparse.ArgumentParser(description='DensePose - Visualize the dilated and symmetrical segment')
    parser.add_argument('--input', help='Path to image file')
    args = parser.parse_args()

    visualize(infile=args.input, openpose_idx=1)