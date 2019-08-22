import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random

RGB_VAL_DIR = "/media/nikos134/DATADRIVE1/CarlaData/19_08/_out_0"




RGB_VAL_DIR_TO = "/media/nikos134/DATADRIVE1/WarningDataVer3/"



b_size = 40

images_val = os.listdir(RGB_VAL_DIR)

# 539

print(len(images_val))


for i in range(int(len(images_val)/40)):
    print("BATCH: ", i)
    image_batch = np.zeros((b_size, 540, 960, 3), dtype=np.uint8)
    for j in range(b_size):
        idx = images_val[i*b_size + j]
        image = skimage.io.imread(os.path.join(RGB_VAL_DIR, idx))
        if image.shape[-1] == 4:
            image = image[..., :3]
        image_batch[j] = image

    images_aug = image_batch[0:10]
    for k in range(10):
        im = Image.fromarray(images_aug[k])
        temp = "a_" + images_val[i*b_size + k ]
        im.save(os.path.join(RGB_VAL_DIR_TO, temp))

    sometimes = lambda aug: iaa.Sometimes(0.25, aug)

    # 025 Rotation or translation

    aug = iaa.OneOf([
        iaa.Affine(rotate=(-25, 25)),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    ])


    images_aug1 = aug(images=image_batch[10:20])
    for k in range(10):
        im = Image.fromarray(images_aug1[k])
        im.save(os.path.join(RGB_VAL_DIR_TO ,("b_" + images_val[i*b_size + k + 10])))




    # 050 Rotation-Blur


    aug = iaa.Sequential([
        iaa.SomeOf((1,3),[
            iaa.AverageBlur(k=(2, 7)),
            iaa.WithChannels(0, iaa.Affine(rotate=(-20, 20))),


        ]),
        iaa.OneOf([
            iaa.Add((40, 70)),
            iaa.Add((-70, -40))
        ])


    ])


    images_aug2 = aug(images=image_batch[20:30])
    for k in range(10):
        im = Image.fromarray(images_aug2[k])
        im.save(os.path.join(RGB_VAL_DIR_TO, ("c_" + images_val[i*b_size + k + 20])))

    # 075


    aug = iaa.Sequential([
        iaa.OneOf([
            iaa.Dropout(p=(0, 0.3), per_channel=0.5),
            iaa.CoarseDropout((0.0, 0.3), size_percent=(0.01, 0.05))
        ])



    ])


    images_aug3 = aug(images=image_batch[30:40])
    for k in range(10):
        im = Image.fromarray(images_aug3[k])
        im.save(os.path.join(RGB_VAL_DIR_TO,("d_" + images_val[i*b_size + k + 30])))

