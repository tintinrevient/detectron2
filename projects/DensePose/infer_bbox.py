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
import glob
from pathlib import Path


def generate_bbox(infile, score_cutoff, clip_bbox=False):

    print('input:', infile)

    image = cv2.imread(infile)
    # cv2.imshow('input', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)

    # filter the probabilities of scores for each bbox > 90%
    instances = outputs['instances']
    confident_detections = instances[instances.scores > score_cutoff]

    if clip_bbox:
        # if bbox is detected above the threshold, which is defined in confident_detections
        if confident_detections.pred_boxes and confident_detections.pred_boxes[0] and confident_detections.pred_boxes.tensor[0][0].item() > -1:

            image_bboxes = clip_by_bboxes(image, confident_detections)

            for index, image_bbox in enumerate(image_bboxes):
                outfile = generate_outfile(infile=infile, index=index)
                cv2.imwrite(outfile, image_bbox)
                print('output:', outfile)
        else:
            return

    else:
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(confident_detections.to('cpu'))

        cv2.imshow('bbox', out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def clip_by_bboxes(image, confident_detections):

    bboxes = confident_detections.pred_boxes
    obj_classes = confident_detections.pred_classes
    image_bboxes = []

    for bbox, obj_class in zip(bboxes.tensor, obj_classes):

        print('Object class:', obj_class)

        # pred_classes = 0 -> person
        if obj_class != 0:
            continue

        x1, y1, x2, y2 = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()

        x = int(x1)
        y = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)

        image_bbox = image[y:y + h, x:x + w]
        image_bboxes.append(image_bbox)

    return image_bboxes


def generate_outfile(infile, index):

    outdir = os.path.join('output', infile[infile.find('/') + 1:infile.rfind('/')])

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fname = infile[infile.find('/') + 1:infile.rfind('.')]
    outfile = os.path.join('output', '{}_bbox_{}.jpg'.format(fname, index))

    return outfile


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DensePose - Infer bbox')
    parser.add_argument('--input', help='Path to image file or directory')
    args = parser.parse_args()

    if os.path.isfile(args.input):
        generate_bbox(infile=args.input, score_cutoff=0.9, clip_bbox=False)

    elif os.path.isdir(args.input):
        for path in Path(args.input).rglob('*.jpg'):
            generate_bbox(infile=str(path), score_cutoff=0.9, clip_bbox=True)
    else:
        pass