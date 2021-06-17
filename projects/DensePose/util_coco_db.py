from pycocotools.coco import COCO
import os
import shutil

# filtered words
matched_words_list = ['man', 'surf']
matched_words_size = 2
image_fpath_list = []

# dataset setting
coco_folder = os.path.join('datasets', 'coco')

# dense_pose annotation
dp_coco = COCO(os.path.join(coco_folder, 'annotations', 'densepose_train2014.json'))

# caption annotation
caption_coco = COCO(os.path.join(coco_folder, 'annotations', 'captions_train2014.json'))

# dense_pose image ids
dp_img_ids = dp_coco.getImgIds()
# caption image ids
caption_img_ids = caption_coco.getImgIds()

print('Number of dp_images:', len(dp_img_ids))
print('Number of caption_images:', len(caption_img_ids))

# common image ids between dense_pose and caption images
common_img_ids = list(set(dp_img_ids) & set(caption_img_ids))

print('Number of common images:', len(common_img_ids))

for img_id in common_img_ids:

    annotation_ids = caption_coco.getAnnIds(imgIds=img_id)
    annotations = caption_coco.loadAnns(annotation_ids)

    # one image -> multiple annotations
    for annotation in annotations:

        # caption_list = a list of lower-case words
        caption_list = annotation['caption'].lower().split()

        # check for words, which must be ALL included
        common_words_list = list(set(matched_words_list) & set(caption_list))

        if len(common_words_list) == matched_words_size:

            entry = dp_coco.loadImgs(img_id)[0]
            dataset_name = entry['file_name'][entry['file_name'].find('_') + 1:entry['file_name'].rfind('_')]
            image_fpath = os.path.join(coco_folder, dataset_name, entry['file_name'])

            image_fpath_list.append(image_fpath)

            break

print('Total images:', len(image_fpath_list))
print(image_fpath_list[:10])

# copy files from the COCO database to the specified directory
output_dir = os.path.join('coco_man_surf')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for image_fpath in image_fpath_list:
    fname = image_fpath[image_fpath.rfind('/') + 1:]
    shutil.copy(image_fpath, os.path.join(output_dir, fname))