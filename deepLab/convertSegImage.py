import os
import numpy as np
import skimage
import scipy.misc
import skimage.io

dir = "/media/nikos134/DATADRIVE1/onedrive/21_06/CarlaDataset/SegmentationClass"
dirTo = "/media/nikos134/DATADRIVE1/onedrive/21_06/CarlaDataset/SegmentationClassNew"
images = os.listdir(dir)


for Path in images:
    image = skimage.io.imread(os.path.join(dir, Path))
    print('image: ', Path)
   # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
        dims = image.shape
    image_new = np.zeros((dims[0], dims[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j, 0] == 0 and image[i, j, 1] == 0 and image[i, j, 2] == 0:
                image_new[i, j] = 0
            elif image[i, j, 0] == 70 and image[i, j, 1] == 70 and image[i, j, 2] == 70:
                image_new[i, j] = 1
            elif image[i, j, 0] == 190 and image[i, j, 1] == 153 and image[i, j, 2] == 153:
                image_new[i, j] = 2
            elif image[i, j, 0] == 250 and image[i, j, 1] == 170 and image[i, j, 2] == 160:
                image_new[i, j] = 3
            elif image[i, j, 0] == 220 and image[i, j, 1] == 20 and image[i, j, 2] == 60:
                image_new[i, j] = 4
            elif image[i, j, 0] == 153 and image[i, j, 1] == 153 and image[i, j, 2] == 153:
                image_new[i, j] = 5
            elif image[i, j, 0] == 157 and image[i, j, 1] == 234 and image[i, j, 2] == 50:
                image_new[i, j] = 6
            elif image[i, j, 0] == 128 and image[i, j, 1] == 64 and image[i, j, 2] == 128:
                image_new[i, j] = 7
            elif image[i, j, 0] == 244 and image[i, j, 1] == 35 and image[i, j, 2] == 232:
                image_new[i, j] = 8
            elif image[i, j, 0] == 107 and image[i, j, 1] == 142 and image[i, j, 2] == 35:
                image_new[i, j] = 9
            elif image[i, j, 0] == 0 and image[i, j, 1] == 0 and image[i, j, 2] == 142:
                image_new[i, j] = 10
            elif image[i, j, 0] == 102 and image[i, j, 1] == 102 and image[i, j, 2] == 156:
                image_new[i, j] = 11
            elif image[i, j, 0] == 220 and image[i, j, 1] == 220 and image[i, j, 2] == 0:
                image_new[i, j] = 12
    temp = os.path.join(dirTo, Path)
    skimage.io.imsave(temp, image_new)
exit()