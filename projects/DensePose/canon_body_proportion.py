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


fname_vitruvian_man = os.path.join('pix', 'vitruve.png')


def vitruvian_man():

    img = cv2.imread(fname_vitruvian_man, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))

    circles_mat = circles[0, :]
    min_x, min_y = np.min(np.array(circles_mat[0:2]), axis=0).astype(int)
    max_x, max_y = np.max(np.array(circles_mat[0:2]), axis=0).astype(int)

    print('min_x:', min_x)
    print('max_x:', max_x)
    print('min_y:', min_y)
    print('max_y:', max_y)

    cv2.line(cimg, (min_x, min_y), (max_x, min_y), color=(0, 255, 0), thickness=2)
    cv2.line(cimg, (min_x, min_y), (min_x, max_y), color=(0, 255, 0), thickness=2)

    cv2.line(cimg, (max_x, max_y), (max_x, min_y), color=(0, 255, 0), thickness=2)
    cv2.line(cimg, (max_x, max_y), (min_x, max_y), color=(0, 255, 0), thickness=2)

    cv2.imshow('bbox', cimg)

    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    #
    # cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    vitruvian_man()