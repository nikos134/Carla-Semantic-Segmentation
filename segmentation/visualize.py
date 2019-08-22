import os


import random


import matplotlib.pyplot as plt
import skimage.io                                     #Used for imshow function
import skimage.transform                              #Used for resize function

import numpy as np
import cv2

RGB_TRAIN_DIR = "/media/nikos134/DATADRIVE1/onedrive/21_06/trainData/RGB"
SEG_TRAIN_DIR = "/media/nikos134/DATADRIVE1/onedrive/21_06/trainData/SEG"

RGB_VAL_DIR = "/media/nikos134/DATADRIVE1/onedrive/21_06/valData/RGB"
SEG_VAL_DIR = "/media/nikos134/DATADRIVE1/onedrive/21_06/valData/SEG"
ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/deepLab")
# Illustrate the train images and masks
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


    random.shuffle(images)
    for i in range(batch_size):
        idx = images[i]

        image = skimage.io.imread(os.path.join(directoryRGB, idx))
        image = cv2.resize(image, (512, 256), interpolation = cv2.INTER_NEAREST)

        image_seg = skimage.io.imread(os.path.join(directorySEG, idx))
        image_seg = cv2.resize(image_seg, (512, 256), interpolation = cv2.INTER_NEAREST)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image_seg.shape[-1] == 4:
            image_seg = image_seg[..., :3]

        image_batch[i] = image
        image_batch_seg[i] = image_seg

    return image_batch, getMask(batch_size,image_batch_seg)


img, mask = data_generator('train',1)

labels = ["Unlabeled","Building" , "Fence", "Other", "Pedestrian", "Pole", "Road Line","Road", "Sidewalk", "Vegetation", "Car", "Wall", "Traffic Sign" ]

plt.figure(figsize=(20,10))
ax = plt.subplot(5, 4, 1)
plt.imshow(img[0]/255,)
ax.title.set_text("RGB")
for row in range(5):
    for col in range(4):
        if 4*row + col >= 13:
            continue
        ax = plt.subplot(5, 4, 4*row + col +2 )
        ax.title.set_text(labels[4*row + col])
        plt.imshow(mask[0, :, :, 4*row + col])



plt.tight_layout()

plt.show()
