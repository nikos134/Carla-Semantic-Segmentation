import os
import sys

import numpy as np
import cv2

import skimage.io
from imgaug import augmenters as aug

# Root directory of the project
ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/unet")


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
        building_labels[building_index] = 1
        fence_labels[fence_index] = 1
        other_labels[other_index] = 1
        pedestrian_labels[pedestrian_index] = 1
        pole_labels[pole_index] = 1
        road_line_labels[road_line_index] = 1
        road_labels[road_index] = 1
        sidewalk_labels[sidewalk_index] = 1
        vegetation_labels[vegetation_index] = 1
        car_labels[car_index] = 1
        wall_labels[wall_index] = 1
        traffic_sign_labels[traffic_sign_index] = 1

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




train = carlaDataset()
train.load_images(dir=RGB_TRAIN_DIR, type='train')


# mask, a = train.load_mask(50)
# print(a)
train.prepare()
val = carlaDataset()
val.load_images(RGB_VAL_DIR, type='valid')
val.prepare()
augmentation = aug.SomeOf((0, None), [
        aug.Fliplr(0.5),
        aug.Flipud(0.5),
        aug.OneOf([aug.Affine(rotate=90),
                   aug.Affine(rotate=180),
                   aug.Affine(rotate=270)]),
        aug.Multiply((0.8, 1.5)),
        aug.GaussianBlur(sigma=(0.0, 5.0)),
        aug.Affine(scale=(0.5, 1.5)),
        aug.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
    ])

config = CarlaConfig()
config.STEPS_PER_EPOCH = NUMBER_OF_TRAIN_DATA//config.BATCH_SIZE
config.VALIDATION_STEPS = NUMBER_OF_VAL_DATA//config.BATCH_SIZE
config.display()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

weights_path = model.get_imagenet_weights()
model.load_weights(weights_path, by_name=True)
# model.load_weights(COCO_MODEL_PATH, by_name=True,
#                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
#                                 "mrcnn_bbox", "mrcnn_mask"])


model_path = os.path.join(ROOT_DIR, "mask_rcnn_carla.h5")

# # Load trained weights (fill in path to trained weights here)
# assert model_path != "", "Provide path to trained weights"
# print("Loading weights from ", model_path)
# model.load_weights(model_path, by_name=True)


print("Training ...")

model.train(train, val,
            learning_rate=1e-4,
            epochs=25,
            verbose=2,
            augmentation=augmentation,
            layers='all')

model.train(train, val,
            learning_rate=1e-5,
            epochs=50,
            verbose=2,
            augmentation=augmentation,
            layers='all')

model.train(train, val,
            learning_rate=1e-6,
            epochs=75,
            verbose=2,
            augmentation=augmentation,
            layers='all')

model_path = os.path.join(ROOT_DIR, "ver3.h5")
model.keras_model.save_weights(model_path)



exit()