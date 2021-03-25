from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import argparse

parser = argparse.ArgumentParser(description='DensePose - Infer keypoints')
parser.add_argument('--input', help='Path to image')
args = parser.parse_args()

infile = args.input
fname = infile[infile.find('/')+1:infile.rfind('.')]
outfile = os.path.join('pix', fname + '_keypoints.jpg')

im = cv2.imread(infile)
# cv2.imshow('input', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs)
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
# print(outputs["instances"].pred_keypoints)
# print(outputs["instances"].scores)

# filter the probabilities of scores for each bbox > 90%
instances = outputs["instances"]
confident_detections = instances[instances.scores > 0.8]

# We can use `Visualizer` to draw the predictions on the image.
visualizer = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

# draw all the predictions
# out = visualizer.draw_instance_predictions(confident_detections.to("cpu"))

# draw only the keypoints
out = visualizer.draw_and_connect_keypoints(confident_detections.pred_keypoints[0,:,:])

cv2.imshow('keypoints', out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(outfile, out.get_image()[:, :, ::-1])