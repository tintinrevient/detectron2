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


fname_vitruve = os.path.join('pix', 'vitruve.png')
fname_vitruve_norm = os.path.join('pix', 'vitruve_norm.png')


def vitruve_circle():

    img = cv2.imread(fname_vitruve, 0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    blur_img = cv2.medianBlur(img, 5)
    edge_img = cv2.Canny(blur_img, 50, 100)

    circles = cv2.HoughCircles(edge_img, cv2.HOUGH_GRADIENT, 1, 240,
                               param1=50, param2=20, minRadius=300, maxRadius=310)

    if circles is not None:
        for circles in circles[0, :]:
            x = circles[0]
            y = circles[1]
            radius = int(circles[2])

            print('x, y:', x, y)
            print('radius:', radius)

            cv2.circle(cimg, (x, y), radius=2, color=(0, 255, 255), thickness=-1)
            cv2.circle(cimg, (x, y), radius=radius, color=(255, 0, 0), thickness=2)

    window_name = 'detected circles'
    cv2.imshow(window_name, cimg)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def vitruve_norm_circle():

    img = cv2.imread(fname_vitruve_norm, 0)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    print('image shape:', cimg.shape)

    upper_limbs_margin_y = 20
    ankle_margin_y = 19

    circle_x, circle_y = (312, 312)
    radius = 302

    cv2.circle(cimg, (circle_x, circle_y), radius=1, color=(255, 0, 255), thickness=-1)
    cv2.circle(cimg, (circle_x, circle_y), radius=radius, color=(255, 0, 255), thickness=1)

    # height = length of square
    # the length of the outspread arms is equal to the height of a man
    height = 500

    # penis = midhip
    # the root of the penis is at half the height of a man
    penis_x = circle_x
    penis_y = int(circle_y + radius - height/2)
    cv2.circle(cimg, (penis_x, penis_y), radius=1, color=(255, 0, 255), thickness=-1)

    # knee
    # from below the knee to the root of the penis is a quarter of the height of a man
    knee_x = circle_x
    knee_y = int(penis_y + height/4)
    cv2.circle(cimg, (knee_x, knee_y), radius=1, color=(255, 0, 255), thickness=-1)

    cv2.circle(cimg, (int(knee_x-30), knee_y), radius=1, color=(255, 0, 255), thickness=-1)
    cv2.circle(cimg, (int(knee_x+10), knee_y), radius=1, color=(255, 0, 255), thickness=-1)

    print('mid-upper-leg x, y:', penis_x, int((penis_y + knee_y)/2))

    # ankle -> foot?
    # from below the foot to below the knee is a quarter of the height of a man
    # the foot is one-seventh of the height of a man!
    ankle_x = circle_x
    ankle_y = int(knee_y + height/4 - ankle_margin_y)
    cv2.circle(cimg, (ankle_x, ankle_y), radius=1, color=(255, 0, 255), thickness=-1)

    cv2.circle(cimg, (int(ankle_x - 30), ankle_y), radius=1, color=(255, 0, 255), thickness=-1)
    cv2.circle(cimg, (int(ankle_x + 10), ankle_y), radius=1, color=(255, 0, 255), thickness=-1)

    print('mid-lower-leg x, y:', penis_x, int((knee_y + ankle_y) / 2))

    # neck = above the chest
    # from above the chest to the top of the head is one-sixth of the height of a man
    neck_x = circle_x
    neck_y = int(penis_y - height/2 + height/6 + upper_limbs_margin_y)
    cv2.circle(cimg, (neck_x, neck_y), radius=1, color=(255, 0, 255), thickness=-1)

    # shoulder
    # the maximum width of the shoulders is a quarter of the height of a man
    rsho_x = int(neck_x - height/(4*2))
    rsho_y = neck_y

    lsho_x = int(neck_x + height/(4*2))
    lsho_y = neck_y
    cv2.circle(cimg, (rsho_x, rsho_y), radius=1, color=(255, 0, 255), thickness=-1)
    cv2.circle(cimg, (lsho_x, lsho_y), radius=1, color=(255, 0, 255), thickness=-1)

    # elbow
    # the distance from the elbow to the armpit is one-eighth of the height of a man
    relb_x = int(rsho_x - height/8)
    relb_y = neck_y

    lelb_x = int(lsho_x + height/8)
    lelb_y = neck_y
    cv2.circle(cimg, (relb_x, relb_y), radius=1, color=(255, 0, 255), thickness=-1)
    cv2.circle(cimg, (lelb_x, lelb_y), radius=1, color=(255, 0, 255), thickness=-1)

    print('mid-right-upper-arm x, y:', int((rsho_x + relb_x)/2), rsho_y)
    print('mid-left-upper-arm x, y:', int((lsho_x + lelb_x)/2), lsho_y)

    # wrist
    # the distance from the elbow to the tip of the hand is a quarter of the height of a man
    # the length of the hand is one-tenth of the height of a man
    rwrist_x = int(relb_x - height/4 + height/10)
    rwrist_y = neck_y

    lwrist_x = int(lelb_x + height/4 - height/10)
    lwrist_y = neck_y
    cv2.circle(cimg, (rwrist_x, rwrist_y), radius=1, color=(255, 0, 255), thickness=-1)
    cv2.circle(cimg, (lwrist_x, lwrist_y), radius=1, color=(255, 0, 255), thickness=-1)

    print('mid-right-lower-arm x, y:', int((relb_x + rwrist_x)/2), rsho_y)
    print('mid-left-lower_arm x, y:', int((lelb_x + lwrist_x)/2), lsho_y)

    # mid-torso
    torso_x = circle_x
    torso_y = int((neck_y + penis_y)/2)
    cv2.circle(cimg, (torso_x, torso_y), radius=1, color=(255, 0, 255), thickness=-1)
    print('mid-torso x, y:', torso_x, torso_y)

    # mid-head
    # from the hairline to the bottom of the chin is one-tenth of the height of a man
    # from below the chin to the top of the head is one-eighth of the height of a man
    nose_x = circle_x
    nose_y = int(penis_y - height/2 + height/(8*2))
    cv2.circle(cimg, (nose_x, nose_y), radius=1, color=(255, 0, 255), thickness=-1)
    print('mid-head x, y:', nose_x, nose_y)

    window_name = 'circles'
    cv2.imshow(window_name, cimg)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # vitruve_circle()
    vitruve_norm_circle()