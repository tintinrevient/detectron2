import os, cv2
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from densepose.utils.dbhelper import EntrySelector
from detectron2.utils.file_io import PathManager
from detectron2.utils.visualizer import Visualizer

img_id = 1000
dataset_name = 'keypoints_coco_2014_val'

dataset = DatasetCatalog.get(dataset_name)
entry_selector = EntrySelector.from_string('image_id:int=' + str(img_id))

for entry in dataset:
    if entry_selector(entry):

        image_fpath = PathManager.get_local_path(entry['file_name'])
        image_fname = entry['file_name'][entry['file_name'].rfind('/')+1:]
        output_fpath = os.path.join('pix', image_fname)

        image = cv2.imread(image_fpath, cv2.IMREAD_GRAYSCALE)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])

        visualizer = Visualizer(image, MetadataCatalog.get(dataset_name), scale=1.0)
        image_vis = visualizer.draw_dataset_dict(dic=entry) # entry['annotations']

        cv2.imwrite(output_fpath, image_vis.get_image()[:, :, ::-1])
