import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io


dirRGB = "/media/nikos134/DATADRIVE1/onedrive/21_06/_out_0"
dirLidar = "/media/nikos134/DATADRIVE1/onedrive/21_06/_out_4"



imagesRGB = os.listdir(dirRGB)
imagesLIDAR = os.listdir(dirLidar)

imageRGB = skimage.io.imread(os.path.join(dirRGB, imagesRGB[0]))
imageRGB = cv2.resize(imageRGB, (512, 512))


imageLidar = cv2.imread(os.path.join(dirLidar, imagesLIDAR[0]), cv2.IMREAD_GRAYSCALE)
imageLidar = cv2.resize(imageLidar, (512, 512))

if imageRGB.shape[-1] == 4:
    imageRGB = imageRGB[..., :3]
# if imageLidar.shape[-1] == 4:
#     imageLidar = imageLidar[..., :3]


final_image = np.zeros((512,512,4))
final_image[:, :, :3] = imageRGB
final_image[:, :, 3] = imageLidar

print(imageRGB.dtype)
print(imageLidar.dtype)

cv2.imshow('image',final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(final_image)
#
#
# plt.show()