from pycocotools.coco import COCO
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from random import randint


coco_folder = os.path.join('datasets', 'coco')
dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_minival2014.json'))

# Get the category ids
cats = dp_coco.loadCats(dp_coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cat_names)))

super_cat_names = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(super_cat_names)))

# Search by category names
# cat_ids = dp_coco.getCatIds(catNms=['person','dog','skateboard'])
# im_ids = dp_coco.getImgIds(catIds=cat_ids)

# Get img id's for the minival dataset
im_ids = dp_coco.getImgIds()
print('Total images:', len(im_ids))

# Select a random image id
selected_im = im_ids[randint(0, len(im_ids))]
print('Selected image:', selected_im)

# for test only - two people in the image
# selected_im = im_ids[1200]
selected_im = 466986

# Load the image
im = dp_coco.loadImgs(selected_im)[0]

# Load annotations for the selected image
ann_ids = dp_coco.getAnnIds(imgIds=im['id'])
anns = dp_coco.loadAnns(ann_ids)
print('Image ID:', im['id'])
print('Annotation ID:', ann_ids)
print('Total annotations:', len(anns))

# Now read the image and show
im_name = os.path.join(coco_folder, 'val2014', im['file_name'])
print('Image name:', im_name)
input_im = cv2.imread(im_name)
plt.imshow(input_im[:,:,::-1]); plt.axis('off'); plt.show()


# Visualization of collected masks

# Get dense masks from the decoded masks
def getDensePoseMask(polys):

    mask_gen = np.zeros([256,256])

    for i in range(1,15):

        if(polys[i-1]):
            current_mask = mask_util.decode(polys[i-1])
            mask_gen[current_mask>0] = i

    return mask_gen

# Go over all annotations and visualize them one by one
output_mask_im = input_im.copy() # Dim the image.

for ann in anns:

    bbr =  np.array(ann['bbox']).astype(int) # the box.

    if('dp_masks' in ann.keys()): # If we have densepose annotation for this ann,

        mask = getDensePoseMask(ann['dp_masks'])

        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0]+bbr[2], bbr[1]+bbr[3]

        x2 = min([x2, input_im.shape[1]]); y2 = min([y2, input_im.shape[0]])

        mask_im = cv2.resize(mask, (int(x2-x1), int(y2-y1)), interpolation=cv2.INTER_NEAREST)
        mask_bool = np.tile((mask_im==0)[:,:,np.newaxis], [1,1,3])

        #  Replace the visualized mask image with I_vis.
        mask_vis = cv2.applyColorMap((mask_im*15).astype(np.uint8), cv2.COLORMAP_PARULA)[:,:,:]
        mask_vis[mask_bool] = output_mask_im[y1:y2,x1:x2,:][mask_bool]
        output_mask_im[y1:y2,x1:x2,:] = output_mask_im[y1:y2,x1:x2,:]*0.3 + mask_vis*0.7

plt.imshow(output_mask_im[:,:,::-1]); plt.axis('off'); plt.show()


# Visualization of collected points
# For each collected point we have the surface patch index, and UV coordinates

# Show images for each subplot
fig = plt.figure(figsize=[15, 5])

plt.subplot(1, 3, 1)
plt.imshow(input_im[:, :, ::-1]); plt.axis('off'); plt.title('Patch Indices')

plt.subplot(1, 3, 2)
plt.imshow(input_im[:, :, ::-1]); plt.axis('off'); plt.title('U coordinates')

plt.subplot(1, 3, 3)
plt.imshow(input_im[:, :, ::-1]); plt.axis('off'); plt.title('V coordinates')

## For each annotation, scatter plot the collected points
for ann in anns:

    bbr = np.round(ann['bbox'])

    if ('dp_masks' in ann.keys()):

        point_x = np.array(ann['dp_x']) / 255. * bbr[2]  # Strech the points to current box.
        point_y = np.array(ann['dp_y']) / 255. * bbr[3]  # Strech the points to current box.

        point_I = np.array(ann['dp_I'])
        point_U = np.array(ann['dp_U'])
        point_V = np.array(ann['dp_V'])

        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]

        point_x = point_x + x1; point_y = point_y + y1

        plt.subplot(1, 3, 1)
        plt.scatter(point_x, point_y, 22, point_I)

        plt.subplot(1, 3, 2)
        plt.scatter(point_x, point_y, 22, point_U)

        plt.subplot(1, 3, 3)
        plt.scatter(point_x, point_y, 22, point_V)

plt.show()