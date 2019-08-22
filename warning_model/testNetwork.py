import keras
from keras.models import load_model
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

DIR_RGB = '/media/nikos134/DATADRIVE1/WarningDataVer3'


ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/vgg16")


batch_size = 100
number_of_false = 0
number_of_true = 0

model_path = os.path.join(ROOT_DIR,'warningVer8.h5')
model = load_model(model_path)

image_batch = np.zeros((batch_size, 256, 256, 3))

images = os.listdir(DIR_RGB)
#
for j in range(int(6000/batch_size)):
    # print("J: ", j)
    mask = np.zeros((batch_size, 4))
    for i in range(batch_size):
        images = os.listdir(DIR_RGB)
        # print("Batch: ", i)

        idx = images[j*batch_size + i]
        if idx.find('a') != -1:

            mask[i, 0] = 1
            # print(mask[i,0])
            # 'No Problem with the feed'

        elif idx.find('b') != -1:

            mask[i, 1] = 1
            # "Camera Tilted or moved"

        elif idx.find('c') != -1:

            mask[i, 2] = 1
            # "Blur Image"

        elif idx.find('d') != -1:
            mask[i, 3] = 1
        # print(idx)

        image = skimage.io.imread(os.path.join(DIR_RGB, idx))
        image = cv2.resize(image, (256, 256))

        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        image_batch[i] = image


    features = model.predict(image_batch)
    # print(features)
    for i,k in enumerate(features):
        temp = np.argmax(k)
        if temp == 0:
            temp = 'a'
            if mask[i, 0] == 1:
                number_of_true += 1
            else:
                wrong = np.where(mask[i] == 1)
                wrongPic = images[j*batch_size + i];
                print("J: %d I: %d is: %s where: %s" % (j, i, temp , wrongPic))
                number_of_false += 1
        elif temp == 1:
            temp = 'b'
            if mask[i, 1] == 1:
                number_of_true += 1
            else:
                wrong = np.where(mask[i] == 1)
                wrongPic = images[j * batch_size + i];
                print("J: %d I: %d is: %s where: %s" % (j, i, temp, wrongPic))
                number_of_false += 1
        elif temp == 2:
            temp = 'c'
            if mask[i, 2] == 1:
                number_of_true += 1
            else:
                wrong = np.where(mask[i] == 1)
                wrongPic = images[j * batch_size + i];
                print("J: %d I: %d is: %s where: %s" % (j, i, temp, wrongPic))
                number_of_false += 1
        elif temp == 3:
            temp = 'd'
            if mask[i, 3] == 1:
                number_of_true += 1
            else:
                wrong = np.where(mask[i] == 1)
                wrongPic = images[j * batch_size + i];
                print("J: %d I: %d is: %s where: %s" % (j, i, temp, wrongPic))
                number_of_false += 1


        # print("I: %d is: %s" % (i, temp))
print("True", number_of_true)
print("False", number_of_false)
print("total", (number_of_false+number_of_true))