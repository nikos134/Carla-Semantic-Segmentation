import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
from imgaug import augmenters as aug
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=tf_config))

# Root directory of the project
ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/Mask_RCNN")


RGB_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/21_06/_out_0")
SEG_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/21_06/_out_2")

NUMBER_OF_TRAIN_DATA = 9000


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "MASK/ver2.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class CarlaConfig(Config):
    """
        Configuration for training on the carla data set
    """

    NAME = 'carla'

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 13 + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1280

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = CarlaConfig()
config.display()




class_names = ["Unlabeled","Building" , "Fence", "Other", "Pedestrian", "Pole", "Road Line","Road", "Sidewalk", "Vegetation", "Car", "Wall", "Traffic Sign" ]

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

for i in range(5):

    file_names = next(os.walk(RGB_DIR))[2]
    image = skimage.io.imread(os.path.join(RGB_DIR, random.choice(file_names)))

    # Run detection
    # Remove alpha channel, if it has one
    if image.shape[-1] == 4:
        image = image[..., :3]

    results = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])