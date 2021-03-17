from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from random import randint

# val2014: 21634 person images
# train2014: 45174 person images

coco_folder = os.path.join('datasets', 'coco')
dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'instances_train2014.json'))

# Get the category ids
cats = dp_coco.loadCats(dp_coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cat_names)))

super_cat_names = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(super_cat_names)))

# Search by category names
cat_ids = dp_coco.getCatIds(catNms=['person'])
im_ids = dp_coco.getImgIds(catIds=cat_ids)
print('COCO total images of person:', len(im_ids))