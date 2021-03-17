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
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.extractor import CompoundExtractor, DensePoseResultExtractor, create_extractor

fname = 'test'
infile = os.path.join('pix', fname + '.jpg')
outfile = os.path.join('pix', fname + '_segm.jpg')

im = cv2.imread(infile)
# cv2.imshow('input', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cfg = get_cfg()
add_densepose_config(cfg)
cfg.MODEL.DEVICE = 'cpu'

cfg.merge_from_file("./configs/densepose_rcnn_R_50_FPN_s1x.yaml")
cfg.MODEL.WEIGHTS = "./models/densepose_rcnn_R_50_FPN_s1x.pkl"

predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
print(outputs["instances"].pred_densepose)

visualizers = []

visualizer_segm = DensePoseResultsFineSegmentationVisualizer(cfg=cfg)
visualizer_bbox = ScoredBoundingBoxVisualizer(cfg=cfg)

visualizers.append(visualizer_segm)
visualizers.append(visualizer_bbox)

extractors = []

extractor_segm = create_extractor(visualizer_segm)
extractor_bbox = create_extractor(visualizer_bbox)

extractors.append(extractor_segm)
extractors.append(extractor_bbox)

visualizer = CompoundVisualizer(visualizers)
extractor = CompoundExtractor(extractors)

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = np.tile(im_gray[:, :, np.newaxis], [1, 1, 3])
data = extractor(outputs["instances"])
im_vis = visualizer.visualize(im_gray, data)

cv2.imshow('segm', im_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(outfile, im_vis)