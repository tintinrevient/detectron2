import os
from itertools import chain
import cv2
import tqdm
from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

img_id = 1000

cfg = get_cfg()
cfg.DATASETS.TRAIN = ('keypoints_coco_2014_val',)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
cfg.DATALOADER.NUM_WORKERS = 0
cfg.freeze()

# dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
# dicts = filter_images_with_few_keypoints(dicts, 1)


def output(vis, fname, show=True):
    if show:
        print(fname)
        cv2.imshow("window", vis.get_image()[:, :, ::-1])
        cv2.waitKey()
    else:
        filepath = os.path.join('test', fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)

# coco_folder = os.path.join('datasets', 'coco')
# dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_minival2014.json'))
#
# img_metadata = dp_coco.loadImgs(img_id)[0]
# img_name = img_metadata["file_name"]
# img_path = os.path.join(coco_folder, 'val2014', img_name)
#
# img_annotation_ids = dp_coco.getAnnIds(imgIds=img_id)
# img_annotations = dp_coco.loadAnns(img_annotation_ids)
# img = utils.read_image(img_path, "RGB")

count = 0
train_data_loader = build_detection_train_loader(cfg)
for batch in train_data_loader:
    for per_image in batch:
        count += 1
        print(count, per_image["image_id"])
        # Pytorch tensor is in (C, H, W) format
        if per_image["image_id"] == 1000:
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

            visualizer = Visualizer(img, metadata=metadata, scale=1.0)
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            output(vis, str(per_image["image_id"]) + ".jpg", show=False)
            break

# visualizer = Visualizer(img, metadata=metadata, scale=1.0)
# vis = visualizer.overlay_instances(
#     boxes=target_fields.get("gt_boxes", None),
#     masks=target_fields.get("gt_masks", None),
#     keypoints=target_fields.get("gt_keypoints", None),
# )
# output(vis, os.path.basename(img_metadata["file_name"]), show=False)

# for dic in tqdm.tqdm(dicts):
#
#     print(dic)
#
#     img = utils.read_image(dic["file_name"], "RGB")
#
#     visualizer = Visualizer(img, metadata=metadata, scale=1.0)
#     vis = visualizer.draw_dataset_dict(dic)
#     output(vis, os.path.basename(dic["file_name"]), show=False)