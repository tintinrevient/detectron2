from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import time
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import argparse
from pathlib import Path


def generate_keypoints(infile, score_cutoff, show):

    print('input:', infile)

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
    confident_detections = instances[instances.scores > score_cutoff]

    # We can use `Visualizer` to draw the predictions on the image.
    visualizer = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

    # draw all the predictions
    out = visualizer.draw_instance_predictions(confident_detections.to("cpu"))

    # draw only the keypoints for the first person only!
    # out = visualizer.draw_and_connect_keypoints(confident_detections.pred_keypoints[0, :, :])

    if show:
        cv2.imshow('keypoints', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        outfile = generate_outfile(infile)
        cv2.imwrite(outfile, out.get_image()[:, :, ::-1])
        print('output:', outfile)


def generate_outfile(infile):

    outdir = os.path.join('output', infile[infile.find('/') + 1:infile.rfind('/')])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = infile[infile.find('/') + 1:infile.rfind('.')]
    outfile = os.path.join('output', '{}_keypoints.jpg'.format(fname))

    return outfile


if __name__ == '__main__':

    # python infer_keypoints.py --input datasets/classical
    # python infer_keypoints.py --input datasets/modern

    # time the execution time
    start = time.time()

    parser = argparse.ArgumentParser(description='DensePose - Infer the keypoints')
    parser.add_argument('--input', help='Path to image file or directory')
    args = parser.parse_args()

    if os.path.isfile(args.input):
        generate_keypoints(infile=args.input, score_cutoff=0.9, show=True)

    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.jpg'):
            try:
                generate_keypoints(infile=str(path), score_cutoff=0.9, show=False)
            except:
                continue
    else:
        pass

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")