import argparse
import cv2
import numpy as np
import random
import os
import skimage
from model.unet import unet
import keras
from matplotlib import image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=tf_config))


RGB_TRAIN_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/trainData/RGB"
SEG_TRAIN_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/trainData/SEG"

LIDAR_DIR =  "/media/nikos134/DATADRIVE1/CarlaData/25_07/_out_4"

RGB_VAL_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/valData/RGB"
SEG_VAL_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/valData/SEG"
ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/unet")


def getMask(b_size, images):
    mask = np.zeros((b_size, 256, 512, 13))
    for i in range(b_size):

        unlabeled_index = np.all(images == (0, 0, 0), axis=-1)
        building_index = np.all(images == (70, 70, 70), axis=-1)
        fence_index = np.all(images == (190, 153, 153), axis=-1)
        other_index = np.all(images == (250, 170, 160), axis=-1)
        pedestrian_index = np.all(images == (220, 20, 60), axis=-1)
        pole_index = np.all(images == (153, 153, 153), axis=-1)
        road_line_index = np.all(images == (157, 234, 50), axis=-1)
        road_index = np.all(images == (128, 64, 128), axis=-1)
        sidewalk_index = np.all(images == (244, 35, 232), axis=-1)
        vegetation_index = np.all(images == (107, 142, 35), axis=-1)
        car_index = np.all(images == (0, 0, 142), axis=-1)
        wall_index = np.all(images == (102, 102, 156), axis=-1)
        traffic_sign_index = np.all(images == (220, 220, 0), axis=-1)

        mask[unlabeled_index, 0] = 1
        mask[building_index, 1] = 1
        mask[fence_index, 2] = 1
        mask[other_index, 3] = 1
        mask[pedestrian_index, 4] = 1
        mask[pole_index, 5] = 1
        mask[road_line_index, 6] = 1
        mask[road_index, 7] = 1
        mask[sidewalk_index, 8] = 1
        mask[vegetation_index, 9] = 1
        mask[car_index, 10] = 1
        mask[wall_index, 11] = 1
        mask[traffic_sign_index, 12] = 1
    return mask


def data_generator(type, batch_size):
    # image_batch = np.zeros((batch_size, 256, 512, 6))
    image_batch = np.zeros((batch_size, 256, 512, 3))
    image_batch_seg = np.zeros((batch_size, 256, 512, 3))
    if type == 'train':
        images = os.listdir(RGB_TRAIN_DIR)
        directoryRGB = RGB_TRAIN_DIR
        directorySEG = SEG_TRAIN_DIR
    elif type == 'valid':
        images = os.listdir(RGB_VAL_DIR)
        directoryRGB = RGB_VAL_DIR
        directorySEG = SEG_VAL_DIR

    while True:
        random.shuffle(images)
        for i in range(batch_size):
            idx = images[i]

            image = skimage.io.imread(os.path.join(directoryRGB, idx))
            image = cv2.resize(image, (512, 256), interpolation = cv2.INTER_NEAREST)

            # imageLidar = skimage.io.imread(os.path.join(LIDAR_DIR, idx))
            # imageLidar = cv2.resize(imageLidar, (512, 256), interpolation = cv2.INTER_NEAREST)


            image_seg = skimage.io.imread(os.path.join(directorySEG, idx))
            image_seg = cv2.resize(image_seg, (512, 256), interpolation = cv2.INTER_NEAREST)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
            if image_seg.shape[-1] == 4:
                image_seg = image_seg[..., :3]
            # if imageLidar.shape[-1] == 4:
            #     imageLidar = imageLidar[..., :3]
            #
            # image_batch[i,:,:,:3] = image
            # image_batch[i, :, :, 3:] = imageLidar
            image_batch[i]=image


            # image_batch_lidar[i] = imageLidar
            image_batch_seg[i] = image_seg
        #
        # yield [image_batch, image_batch_lidar], getMask(batch_size,image_batch_seg)
        yield image_batch, getMask(batch_size,image_batch_seg)


TRAIN_BATCH = 2
VAL_BATCH = 1
NUMBER_OF_TRAIN_DATA = 27000
NUMBER_OF_VAL_DATA = 3001
#
# NUMBER_OF_TRAIN_DATA = 8943
# NUMBER_OF_VAL_DATA = 993


lr_init = 1e-4
lr_decay = 5e-4
vgg_path =  os.path.join(ROOT_DIR, "vgg16.h5")
print("START")




labels = ["Unlabeled","Building" , "Fence", "Other", "Pedestrian", "Pole", "Road Line","Road", "Sidewalk", "Vegetation", "Car", "Wall", "Traffi a Sign" ]

model = unet(input_shape=(256, 512, 3), num_classes=len(labels),
             lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)

model.summary()
callbacks_list = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,), keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,), ]

# training
history = model.fit_generator(data_generator('train',TRAIN_BATCH),
                              steps_per_epoch=NUMBER_OF_TRAIN_DATA // TRAIN_BATCH,
                              validation_data=data_generator('valid',VAL_BATCH),
                              validation_steps=NUMBER_OF_VAL_DATA // VAL_BATCH,
                              callbacks=callbacks_list,
                              epochs=30,
                              verbose=1,
                              )
model_path = os.path.join(ROOT_DIR, "ver2.h5")
model.save(model_path)

plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="val")
plt.legend(loc="best")
plt.savefig('ver2_loss.png')

plt.gcf().clear()
plt.title("dice_coef")
plt.plot(history.history["dice_coef"], color="r", label="train")
plt.plot(history.history["val_dice_coef"], color="b", label="val")
plt.savefig('ver2_dice_coef.png')


plt.legend(loc="best")