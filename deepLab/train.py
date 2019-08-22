from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from utils import Jaccard,sparse_accuracy_ignoring_last_label,sparse_crossentropy_ignoring_last_label
import cv2
import numpy as np
import random
import os
import skimage
from model import Deeplabv3
from utils import SegModel
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=tf_config))


RGB_TRAIN_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/trainData/RGB"
SEG_TRAIN_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/trainData/SEG"

LIDAR_DIR =  "/media/nikos134/DATADRIVE1/CarlaData/25_07/_out_4"

RGB_VAL_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/valData/RGB"
SEG_VAL_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/valData/SEG"
ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/deepLabTest")


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
    image_batch = np.zeros((batch_size, 256, 512, 6))
    # image_batch = np.zeros((batch_size, 256, 512, 3))
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

            imageLidar = skimage.io.imread(os.path.join(LIDAR_DIR, idx))
            imageLidar = cv2.resize(imageLidar, (512, 256), interpolation = cv2.INTER_NEAREST)


            image_seg = skimage.io.imread(os.path.join(directorySEG, idx))
            image_seg = cv2.resize(image_seg, (512, 256), interpolation = cv2.INTER_NEAREST)
            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]
            if image_seg.shape[-1] == 4:
                image_seg = image_seg[..., :3]
            if imageLidar.shape[-1] == 4:
                imageLidar = imageLidar[..., :3]

            image_batch[i,:,:,:3] = image
            image_batch[i, :, :, 3:] = imageLidar
            # image_batch[i]=image


            # image_batch_lidar[i] = imageLidar
            image_batch_seg[i] = image_seg
        #
        # yield [image_batch, image_batch_lidar], getMask(batch_size,image_batch_seg)
        yield image_batch, getMask(batch_size,image_batch_seg)


TRAIN_BATCH = 8
VAL_BATCH = 1
NUMBER_OF_TRAIN_DATA = 27000
NUMBER_OF_VAL_DATA = 3001
#
# NUMBER_OF_TRAIN_DATA = 8943
# NUMBER_OF_VAL_DATA = 993


lr_init = 1e-4
lr_decay = 5e-4

print("START")


metrics = {'pred_mask' : [Jaccard]}



labels = ["Unlabeled","Building" , "Fence", "Other", "Pedestrian", "Pole", "Road Line","Road", "Sidewalk", "Vegetation", "Car", "Wall", "Traffic Sign" ]

model = Deeplabv3(input_shape=(256, 512, 6), classes=len(labels), weights="cityscapes", backbone="xception", OS=16)
# SegClass = SegModel(ROOT_DIR, (256,512))
# SegClass.set_batch_size(TRAIN_BATCH)
# model = SegClass.create_seg_model(net='subpixel', n=len(labels), load_weights=False, multi_gpu=False, backbone="mobilenetv2")


callbacks_list = [ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,),EarlyStopping(monitor='val_loss', patience=8,), ]
# keras.callbacks.
# training
# model.compile(K.optimizers.SGD(), loss=K.losses.binary_crossentropy)
# model.compile(optimizer=K.optimizers.Adam(lr=lr_init, decay=lr_decay),
#                   loss='categorical_crossentropy', metrics=['accuracy'])


model.compile(optimizer = Adam(lr=7e-4, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
              loss = "categorical_crossentropy", metrics = metrics)




monitor = 'Jaccard'
mode = 'max'

# fine-tune model (train only last conv layers)
flag = 0
for k, l in enumerate(model.layers):
    l.trainable = False
    if l.name == 'concat_projection':
        flag = 1
    if flag:
        l.trainable = True

model.summary()
history = model.fit_generator(data_generator('train',TRAIN_BATCH),
                              steps_per_epoch=NUMBER_OF_TRAIN_DATA // TRAIN_BATCH,
                              validation_data=data_generator('valid',VAL_BATCH),
                              validation_steps=NUMBER_OF_VAL_DATA // VAL_BATCH,
                              callbacks=callbacks_list,
                              epochs=30,
                              verbose=1,
                              max_queue_size=10,
                                workers=10
                              )
model_path = os.path.join(ROOT_DIR, "ver2.h5")
model.save(model_path)

plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="val")
plt.legend(loc="best")
plt.savefig('ver2_loss.png')




plt.legend(loc="best")