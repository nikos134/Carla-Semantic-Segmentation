from keras.models import load_model
import cv2
import numpy as np
import os
import random
import skimage.io
import matplotlib.pyplot as plt
from model.unet import unet
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=tf_config))

random.seed(0)
class_colors = [(0,0,0),  	( 70, 70, 70), (190, 153, 153), 	(250, 170, 160), (220, 20, 60),(153, 153, 153),(157, 234, 50),(128, 64, 128),(244, 35, 232),(107, 142, 35),( 0, 0, 142),(102, 102, 156),(220, 220, 0)]

global_iu = 0
counter = 0
iu = []
RGB_DIR = "/media/nikos134/DATADRIVE1/CarlaData/19_08/_out_0"
SEG_DIR = "/media/nikos134/DATADRIVE1/CarlaData/19_08/_out_2"

LIDAR_DIR =  "/media/nikos134/DATADRIVE1/CarlaData/19_08/_out_4"
ROOT_DIR = "/media/nikos134/DATADRIVE1/onedrive/CNN/unet"


# RGB_TRAIN_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/trainData/RGB"
#
# SEG_TRAIN_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/trainData/SEG"
#
#
# LIDAR_DIR =  "/media/nikos134/DATADRIVE1/CarlaData/25_07/_out_4"
#
# RGB_VAL_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/valData/RGB"
# SEG_VAL_DIR = "/media/nikos134/DATADRIVE1/CarlaData/25_07/valData/SEG"

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def mean_iu(predicted_image, true_image):
    classes, number_of_classes = union_classes(predicted_image, true_image)
    true_classes, number_of_true_classes = get_classes(true_image)

    predicted_mask = getMask(predicted_image, classes, number_of_classes)
    true_mask = getMask(true_image,classes,number_of_classes)

    IU = list([0]) * number_of_classes


    for i, c in enumerate(classes):
        temp_predicted_mask = predicted_mask[i,:,:]
        temp_true_mask = true_mask[i, :, :]
        if (np.sum(temp_predicted_mask) == 0) or (np.sum(temp_true_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(temp_predicted_mask, temp_true_mask))
        # print(np.logical_and(temp_predicted_mask, temp_true_mask))
        t_i  = np.sum(temp_true_mask)
        n_ij = np.sum(temp_predicted_mask)
        # print(np.logical_and(temp_predicted_mask, temp_true_mask))


        IU[i] = n_ii / (t_i + n_ij - n_ii)

    iu_mean = np.sum(IU) / number_of_true_classes
    return iu_mean


def get_classes(segmetation):
    classes = np.unique(segmetation.reshape(-1, segmetation.shape[2]), axis=0)
    number_of_classes = len(classes)
    # print(classes)
    # print(number_of_classes)
    return classes, number_of_classes


def union_classes(predicted, true):
    predicted_classes, number_of_predicted_classes = get_classes(predicted)

    true_classes, number_of_true_classes = get_classes(true)

    temp = np.concatenate((predicted_classes, true_classes))

    classes = np.unique(temp, axis=0)
    number_of_classes = len(classes)

    return classes, number_of_classes


def getMask(image, classes, number_of_classes):
    h = image.shape[0]
    w = image.shape[1]

    mask = np.zeros((number_of_classes, h, w))
    for i, c in enumerate(classes):
        # print(c)
        # print(np.all(image == c, axis=-1))
        # temp = np.all(image == c, axis=-1)
        # print(temp.shape)
        # mask[i, :, :] = (image == c)
        mask[i, :, :] = np.all(image == c, axis=-1)
    return mask


def compileImage(b_size, images, images_seg):
    global  global_iu
    global counter
    global iu
    image_seg_batch = np.zeros((10, 256, 512, 3))
    image_seg_batch_final = np.zeros((10, 512, 512, 3))
    for i in range(b_size):
        predicted_image = images[i].reshape((256,512,13))

        # print(np.amax(predicted_image[:,:,1]))

        for j in range(13):

            image_seg_batch[i, :, :, 0] += ((predicted_image[:, :, j] > 0.9) * (class_colors[j][0])).astype('uint8')
            image_seg_batch[i, :, :, 1] += ((predicted_image[:, :, j] > 0.9) * (class_colors[j][1])).astype('uint8')
            image_seg_batch[i, :, :, 2] += ((predicted_image[:, :, j] > 0.9) * (class_colors[j][2])).astype('uint8')

        temp_iu = mean_iu(image_seg_batch[i], cv2.resize(images_seg[i], (512, 256), interpolation = cv2.INTER_NEAREST))
        global_iu = global_iu +  temp_iu
        iu.append(temp_iu)
        counter = counter +  1
        print("Image: %d mean IU : %f" % (i,temp_iu ))
        image_seg_batch_final[i] = cv2.resize(image_seg_batch[i], (512, 512), interpolation = cv2.INTER_NEAREST)

    # labels = ["Unlabeled", "Building", "Fence", "Other", "Pedestrian", "Pole", "Road Line", "Road", "Sidewalk",
    #           "Vegetation", "Car", "Wall", "Traffic Sign"]
    # for row in range(5):
    #     for col in range(4):
    #         if 4 * row + col >= 13:
    #             continue
    #         ax = plt.subplot(5, 4, 4 * row + col + 1)
    #         ax.title.set_text(labels[4 * row + col])
    #         plt.imshow(predicted_image[:, :, 4 * row + col])
    # plt.imshow(image_seg_batch[0])
    # plt.show()
    return image_seg_batch_final


def startVideo(string):
    temp = "ver1_" + str(string) + ".avi"
    print(temp)
    return cv2.VideoWriter(temp, 1482049860, 10, frameSize=(1536, 512))


def createVideo(out, images):
    for i in range(len(images)):
        out.write(np.uint8(images[i]))


def endVideo(out):
    out.release()

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
    if type == 'eval':
        images = os.listdir(RGB_DIR)
        directoryRGB = RGB_DIR
        directorySEG = SEG_DIR
    j = 0
    while True:
        # random.shuffle(images)
        # print("J: ", j)
        for i in range(batch_size):
            idx = images[j*batch_size + i]

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
            # if imageLidar.shape[-1] == 4:
            #     imageLidar = imageLidar[..., :3]

            # image_batch[i,:,:,:3] = image
            # image_batch[i, :, :, 3:] = imageLidar
            image_batch[i]=image


            # image_batch_lidar[i] = imageLidar
            image_batch_seg[i] = image_seg
        #
        # yield [image_batch, image_batch_lidar], getMask(batch_size,image_batch_seg)
        j += 1
        yield image_batch, getMask(batch_size,image_batch_seg)




batch = 20

model_path = os.path.join(ROOT_DIR,'ver2.h5')
model = load_model(model_path,custom_objects={'dice_coef': dice_coef})
images = os.listdir(RGB_DIR)
print(model.metrics_names)
results = model.evaluate_generator(data_generator('eval', batch), steps=len(images) // batch, verbose=1,use_multiprocessing=True,workers=18)
print(results)
# for k in range(6):
#     out = startVideo(k)
#     iu = []
#     for j in range(100):
#         print("Batch: ", j)
#         image_batch = np.zeros((batch, 256, 512, 6))
#         # image_batch = np.zeros((batch, 256, 512, 3))
#         image_batch_for_video = np.zeros((batch, 512, 512, 3))
#         image_seg_batch_for_video = np.zeros((batch, 512, 512, 3))
#
#
#         for i in range(batch):
#             images = os.listdir(RGB_DIR)
#             image = skimage.io.imread(os.path.join(RGB_DIR, images[(k * 100 + j) * batch + i]))
#             image_lidar = skimage.io.imread(os.path.join(LIDAR_DIR, images[(k * 100 + j) * batch + i]))
#             image_seg = skimage.io.imread(os.path.join(SEG_DIR, images[(k * 100 + j) * batch + i]))
#
#             if image.shape[-1] == 4:
#                 image = image[..., :3]
#             if image_seg.shape[-1] == 4:
#                 image_seg = image_seg[..., :3]
#             if image_lidar.shape[-1] == 4:
#                 image_lidar = image_lidar[..., :3]
#
#             image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_NEAREST)
#             image_seg = cv2.resize(image_seg, (512, 512), interpolation = cv2.INTER_NEAREST)
#             image_batch_for_video[i] = image
#             image_seg_batch_for_video[i] = image_seg
#             image_lidar = cv2.resize(image_lidar, (512, 256))
#             image = cv2.resize(image, (512, 256))
#             image_batch[i, :, :, :3] = image
#             image_batch[i, :, :, 3:] = image_lidar
#             # image_batch[i] = image
#         features = model.predict(image_batch)
#
#         images = compileImage(batch, features, image_seg_batch_for_video)
#
#         final_video = np.zeros((batch, 512, 1536, 3))
#
#         final_video[:, :, :512, :] = image_batch_for_video
#
#         final_video[:, :, 512:1024, :] = image_seg_batch_for_video
#         final_video[:, :, 1024:, :] = images
#         createVideo(out, final_video)
#
#     arr = np.array(iu)
#     tempString = "iu_" + str(k)
#     np.savetxt(tempString, arr)
#     print("Global IU: ", global_iu/counter)
#     endVideo(out)
