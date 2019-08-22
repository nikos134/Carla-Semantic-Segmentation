import keras
from model import nvidia
from model import custom
from model import inception
import skimage.io
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=tf_config))

DIR_RGB_TRAIN = '/media/nikos134/DATADRIVE1/WarningDataVer2/trainData'

DIR_CLASS_TRAIN = ''

DIR_RGB_VAL = '/media/nikos134/DATADRIVE1/WarningDataVer2/valData'

DIR_CLASS_VAL = '/media/nikos134/DATADRIVE1/WarningDataVer2/valData'

ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/vgg16")


def getMask(b_size, names):
    # mask = np.zeros((b_size, 4))
    mask = np.zeros((b_size, 4))
    for i in range(b_size):
        if names[i].find('a') != -1:

            mask[i, 0] = 1
            # 'No Problem with the feed'

        elif names[i].find('b') != -1:

            mask[i, 1] = 1
            # "Camera Tilted or moved"

        elif names[i].find('c') != -1:

            mask[i, 2] = 1
            # "Blur Image"

        elif names[i].find('d') != -1:

            mask[i, 3] = 1
            # "Camera Occlusion"

        # elif names[i].find('e') != -1:
        #
        #     mask[i, 4] = 1

    return mask


def data_generator(type, batch_size):
    image_batch = np.zeros((batch_size, 256, 256, 3))
    if type == 'train':
        images = os.listdir(DIR_RGB_TRAIN)
        directoryRGB = DIR_RGB_TRAIN

    elif type == 'valid':
        images = os.listdir(DIR_RGB_VAL)
        directoryRGB = DIR_RGB_VAL


    while True:
        random.shuffle(images)
        names = []

        for i in range(batch_size):
            idx = images[i]
            names.append(idx)

            image = skimage.io.imread(os.path.join(directoryRGB, idx))
            image = cv2.resize(image, (256, 256))


            # If has an alpha channel, remove it for consistency
            if image.shape[-1] == 4:
                image = image[..., :3]


            image_batch[i] = image

        yield image_batch, getMask(batch_size, names)

TRAIN_BATCH = 32
VAL_BATCH = 1
NUMBER_OF_TRAIN_DATA = 26950
NUMBER_OF_VAL_DATA = 3000
lr_init = 1e-4
lr_decay = 5e-4

# model = inception((256, 256, 3), 4 ,lr_init,lr_decay)

model = custom((256, 256, 3), 4 ,lr_init,lr_decay)
# model = nvidia((256, 256, 3), 4 ,lr_init,lr_decay)
# vgg16((256, 256, 3), 5 ,lr_init,lr_decay)
model.summary()
callbacks_list = [ keras.callbas.EarlyStopping(monitor='val_loss', patience=10,), ]
# keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,),
# training
history = model.fit_generator(data_generator('train',TRAIN_BATCH),
                              steps_per_epoch=NUMBER_OF_TRAIN_DATA // TRAIN_BATCH,
                              validation_data=data_generator('valid',VAL_BATCH),
                              validation_steps=NUMBER_OF_VAL_DATA // VAL_BATCH,
                              callbacks=callbacks_list,
                              epochs=30,
                              verbose=1,
                              )
model_path = os.path.join(ROOT_DIR, "warningVer8.h5")
model.save(model_path)

plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="val")
plt.legend(loc="best")
plt.savefig('warningVer8_loss.png')

