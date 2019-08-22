import os
import sys

import numpy as np
import cv2

import skimage.io
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from imgaug import augmenters as aug

# Root directory of the project
ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/Mask_RCNN")


RGB_TRAIN_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/21_06/trainData/RGB")
SEG_TRAIN_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/21_06/trainData/SEG")

RGB_VAL_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/21_06/valData/RGB")
SEG_VAL_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/21_06/valData/SEG")


NUMBER_OF_TRAIN_DATA = 8943
NUMBER_OF_VAL_DATA = 993

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.visualize import display_images
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=tf_config))



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# imagenet_MODEL_PATH = os.path.join(ROOT_DIR, "MASK/mask_rcnn_image.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(imagenet_MODEL_PATH):
#     utils.download_trained_weights(imagenet_MODEL_PATH)


class CarlaConfig(Config):
    """
        Configuration for training on the carla data set
    """

    NAME = 'carla'
    BACKBONE = 'resnet101'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 13 + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320  #

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2048
    POST_NMS_ROIS_INFERENCE = 2048

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 1024

    # You can increase this during training to generate more proposals.
    RPN_NMS_THRESHOLD = 0.7
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 256

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 400

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.75

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3  # 0.3

    # Threshold number for mask binarization, only used in inference mode
    DETECTION_MASK_THRESHOLD = 0.35

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50




class carlaDataset(utils.Dataset):
    random_index = 0

    def load_images(self, dir, type):
        images = os.listdir(dir)

        self.add_class("carla", 1, "Unlabeled")
        self.add_class("carla", 2, "Building")
        self.add_class("carla", 3, "Fence")
        self.add_class("carla", 4, "Other")
        self.add_class("carla", 5, "Pedestrian")
        self.add_class("carla", 6, "Pole")
        self.add_class("carla", 7, "Road line")
        self.add_class("carla", 8, "Road")
        self.add_class("carla", 9, "Sidewalk")
        self.add_class("carla", 10, "Vegetation")
        self.add_class("carla", 11, "Car")
        self.add_class("carla", 12, "Wall")
        self.add_class("carla", 13, "Traffic Sign")
        if type == 'train':
            images = images[:NUMBER_OF_TRAIN_DATA]
        elif type == 'valid':
            images = images[NUMBER_OF_VAL_DATA:]

        for image in images:
            #             print("[image]",image)
            self.add_image('carla', image_id=image, path=os.path.join(dir, image))

    def load_image(self, image_id):
        #         print(self.image_info[image_id]['path'])
        image = skimage.io.imread(self.image_info[image_id]['path'])
        image = cv2.resize(image, (512, 512))
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        self.random_index += 1
        info = self.image_info[image_id]

        if info["source"] != "carla":
            print("not carla image", info["source"])
            return super(self.__class__, self).load_mask(image_id)
        info = self.image_info[image_id]
        if os.path.exists(os.path.join(SEG_TRAIN_DIR, info["id"])):
            directory = SEG_TRAIN_DIR
        elif os.path.exists(os.path.join(SEG_VAL_DIR, info["id"])):
            directory = SEG_VAL_DIR
        mask_label = skimage.io.imread(os.path.join(directory, info["id"]))
        # If grayscale. Convert to RGB for consistency.
        if mask_label.ndim != 3:
            mask_label = skimage.color.gray2rgb(mask_label)

        # If has an alpha channel, remove it for consistency
        if mask_label.shape[-1] == 4:
            mask_label = mask_label[..., :3]
        mask = self.get_labels(mask_label)
        #         print(mask.shape)
        mask = cv2.resize(mask, (512, 512))
        #         print('yo')
        #         print(mask)
        #         print(np.array([1,13], dtype=np.int32))

        return mask, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)

    def get_labels(self, labels):

        dims = labels.shape

        unlabeled_labels = np.zeros((dims[0], dims[1], 1))
        building_labels = np.zeros((dims[0], dims[1], 1))
        fence_labels = np.zeros((dims[0], dims[1], 1))
        other_labels = np.zeros((dims[0], dims[1], 1))
        pedestrian_labels = np.zeros((dims[0], dims[1], 1))
        pole_labels = np.zeros((dims[0], dims[1], 1))
        road_line_labels = np.zeros((dims[0], dims[1], 1))
        road_labels = np.zeros((dims[0], dims[1], 1))
        sidewalk_labels = np.zeros((dims[0], dims[1], 1))
        vegetation_labels = np.zeros((dims[0], dims[1], 1))
        car_labels = np.zeros((dims[0], dims[1], 1))
        wall_labels = np.zeros((dims[0], dims[1], 1))
        traffic_sign_labels = np.zeros((dims[0], dims[1], 1))

        unlabeled_index = np.all(labels == (0, 0, 0), axis=-1)
        building_index = np.all(labels == (70, 70, 70), axis=-1)
        fence_index = np.all(labels == (190, 153, 153), axis=-1)
        other_index = np.all(labels == (250, 170, 160), axis=-1)
        pedestrian_index = np.all(labels == (220, 20, 60), axis=-1)
        pole_index = np.all(labels == (153, 153, 153), axis=-1)
        road_line_index = np.all(labels == (157, 234, 50), axis=-1)
        road_index = np.all(labels == (128, 64, 128), axis=-1)
        sidewalk_index = np.all(labels == (244, 35, 232), axis=-1)
        vegetation_index = np.all(labels == (107, 142, 35), axis=-1)
        car_index = np.all(labels == (0, 0, 142), axis=-1)
        wall_index = np.all(labels == (102, 102, 156), axis=-1)
        traffic_sign_index = np.all(labels == (220, 220, 70), axis=-1)

        unlabeled_labels[unlabeled_index] = 1
        building_labels[building_index] = 10
        fence_labels[fence_index] = 10
        other_labels[other_index] = 10
        pedestrian_labels[pedestrian_index] = 10
        pole_labels[pole_index] = 10
        road_line_labels[road_line_index] = 10
        road_labels[road_index] = 10
        sidewalk_labels[sidewalk_index] = 10
        vegetation_labels[vegetation_index] = 1
        car_labels[car_index] = 10
        wall_labels[wall_index] = 10
        traffic_sign_labels[traffic_sign_index] = 10

        return np.dstack([unlabeled_labels, building_labels, fence_labels,
        return np.dstack([unlabeled_labels, building_labels, fence_labels,
                          other_labels, pedestrian_labels, pole_labels,
                          road_line_labels, road_labels, sidewalk_labels, vegetation_labels,
                          car_labels, wall_labels, traffic_sign_labels])

    def image_reference(self, image_id):
        """Return the carla data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "carla":
            return info["id"]
        else:
            super(self.__class__).image_reference(self, image_id)

config = CarlaConfig()
config.STEPS_PER_EPOCH = NUMBER_OF_TRAIN_DATA//config.BATCH_SIZE
config.VALIDATION_STEPS = NUMBER_OF_VAL_DATA//config.BATCH_SIZE
config.display()


dataset = carlaDataset()
dataset.load_images(dir=RGB_TRAIN_DIR, type='train')


# mask, a = train.load_mask(50)
# print(a)
dataset.prepare()
print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)




# Load random image and mask.
image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id)
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


# Load random image and mask.
image_id = np.random.choice(dataset.image_ids, 1)[0]
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
original_shape = image.shape
# Resize
image, window, scale, padding, _ = utils.resize_image(
    image,
    min_dim=config.IMAGE_MIN_DIM,
    max_dim=config.IMAGE_MAX_DIM,
    mode=config.IMAGE_RESIZE_MODE)
mask = utils.resize_mask(mask, scale, padding)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id: ", image_id)
print("Original shape: ", original_shape)
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)
# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)



image_id = np.random.choice(dataset.image_ids, 1)[0]
image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    dataset, config, image_id, use_mini_mask=False)

log("image", image)
log("image_meta", image_meta)
log("class_ids", class_ids)
log("bbox", bbox)
log("mask", mask)

display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])

visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

# Generate Anchors
backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                          config.RPN_ANCHOR_RATIOS,
                                          backbone_shapes,
                                          config.BACKBONE_STRIDES,
                                          config.RPN_ANCHOR_STRIDE)

# Print summary of anchors
num_levels = len(backbone_shapes)
anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
print("Count: ", anchors.shape[0])
print("Scales: ", config.RPN_ANCHOR_SCALES)
print("ratios: ", config.RPN_ANCHOR_RATIOS)
print("Anchors per Cell: ", anchors_per_cell)
print("Levels: ", num_levels)
anchors_per_level = []
for l in range(num_levels):
    num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
    anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)
    print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))
## Visualize anchors of one cell at the center of the feature map of a specific level

# Load and draw random image
image_id = np.random.choice(dataset.image_ids, 1)[0]
image, image_meta, _, _, _ = modellib.load_image_gt(dataset, config, image_id)
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)
levels = len(backbone_shapes)

for level in range(levels):
    colors = visualize.random_colors(levels)
    # Compute the index of the anchors at the center of the image
    level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
    level_anchors = anchors[level_start:level_start+anchors_per_level[level]]
    print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0],
                                                                  backbone_shapes[level]))
    center_cell = backbone_shapes[level] // 2
    center_cell_index = (center_cell[0] * backbone_shapes[level][1] + center_cell[1])
    level_center = center_cell_index * anchors_per_cell
    center_anchor = anchors_per_cell * (
        (center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE**2) \
        + center_cell[1] / config.RPN_ANCHOR_STRIDE)
    level_center = int(center_anchor)

    # Draw anchors. Brightness show the order in the array, dark to bright.
    for i, rect in enumerate(level_anchors[level_center:level_center+anchors_per_cell]):
        y1, x1, y2, x2 = rect
        p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
                              edgecolor=(i+1)*np.array(colors[level]) / anchors_per_cell)
        ax.add_patch(p)

# Create data generator
random_rois = 4000
g = modellib.data_generator(
    dataset, config, shuffle=True, random_rois=random_rois,
    batch_size=4,
    detection_targets=True)
# Get Next Image
if random_rois:
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
    [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)

    log("rois", rois)
    log("mrcnn_class_ids", mrcnn_class_ids)
    log("mrcnn_bbox", mrcnn_bbox)
    log("mrcnn_mask", mrcnn_mask)
else:
    [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)

log("gt_class_ids", gt_class_ids)
log("gt_boxes", gt_boxes)
log("gt_masks", gt_masks)
log("rpn_match", rpn_match, )
log("rpn_bbox", rpn_bbox)
image_id = modellib.parse_image_meta(image_meta)["image_id"][0]
print("image_id: ", image_id, dataset.image_reference(image_id))

# Remove the last dim in mrcnn_class_ids. It's only added
# to satisfy Keras restriction on target shape.
mrcnn_class_ids = mrcnn_class_ids[:, :, 0]


b = 0

# Restore original image (reverse normalization)
sample_image = modellib.unmold_image(normalized_images[b], config)

# Compute anchor shifts.
indices = np.where(rpn_match[b] == 1)[0]
refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
log("anchors", anchors)
log("refined_anchors", refined_anchors)

# Get list of positive anchors
positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
print("Positive anchors: {}".format(len(positive_anchor_ids)))
negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
print("Negative anchors: {}".format(len(negative_anchor_ids)))
neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

# ROI breakdown by class
for c, n in zip(dataset.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
    if n:
        print("{:23}: {}".format(c[:20], n))

# Show positive anchors
visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids],
                     refined_boxes=refined_anchors)



# Show negative anchors
visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids])


# Show neutral anchors. They don't contribute to training.
visualize.draw_boxes(sample_image, boxes=anchors[np.random.choice(neutral_anchor_ids, 100)])

if random_rois:
    # Class aware bboxes
    bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]

    # Refined ROIs
    refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:, :4] * config.BBOX_STD_DEV)

    # Class aware masks
    mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]

    visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset.class_names)

    # Any repeated ROIs?
    rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
    _, idx = np.unique(rows, return_index=True)
    print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))
if random_rois:
    # Dispalay ROIs and corresponding masks and bounding boxes
    ids = random.sample(range(rois.shape[1]), 8)

    images = []
    titles = []
    for i in ids:
        image = visualize.draw_box(sample_image.copy(), rois[b,i,:4].astype(np.int32), [255, 0, 0])
        image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
        images.append(image)
        titles.append("ROI {}".format(i))
        images.append(mask_specific[i] * 255)
        titles.append(dataset.class_names[mrcnn_class_ids[b,i]][:20])

    display_images(images, titles, cols=4, cmap="Blues", interpolation="none")
# Check ratio of positive ROIs in a set of images.
if random_rois:
    limit = 10
    temp_g = modellib.data_generator(
        dataset, config, shuffle=True, random_rois=10000,
        batch_size=1, detection_targets=True)
    total = 0
    for i in range(limit):
        _, [ids, _, _] = next(temp_g)
        positive_rois = np.sum(ids[0] > 0)
        total += positive_rois
        print("{:5} {:5.2f}".format(positive_rois, positive_rois/ids.shape[1]))
    print("Average percent: {:.2f}".format(total/(limit*ids.shape[1])))
exit()