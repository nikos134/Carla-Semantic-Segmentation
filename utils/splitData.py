import os
import random

NUMBER_OF_TRAIN_DATA = 27000
NUMBER_OF_VAL_DATA = 3001


RGB_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/CarlaData/25_07/_out_0")
SEG_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/CarlaData/25_07/_out_2")

RGB_TRAIN_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/CarlaData/25_07/trainData/RGB")
SEG_TRAIN_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/CarlaData/25_07/trainData/SEG")

RGB_VAL_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/CarlaData/25_07/valData/RGB")
SEG_VAL_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/CarlaData/25_07/valData/SEG")

imagesRGB = os.listdir(RGB_DIR)
imagesSEG = os.listdir(SEG_DIR)

rand = random.sample(range(1, 30000), NUMBER_OF_VAL_DATA)
for i in rand:
    rgb = os.path.join(RGB_DIR, imagesRGB[i])
    seg = os.path.join(SEG_DIR, imagesSEG[i])
    toRgb = os.path.join(RGB_VAL_DIR, imagesRGB[i])
    toSeg = os.path.join(SEG_VAL_DIR, imagesSEG[i])
    os.rename(rgb, toRgb)
    os.rename(seg, toSeg)

imagesRGB = os.listdir(RGB_DIR)
imagesSEG = os.listdir(SEG_DIR)
for i in range(NUMBER_OF_TRAIN_DATA):
    rgb = os.path.join(RGB_DIR, imagesRGB[i])
    seg = os.path.join(SEG_DIR, imagesSEG[i])
    toRgb = os.path.join(RGB_TRAIN_DIR, imagesRGB[i])
    toSeg = os.path.join(SEG_TRAIN_DIR, imagesSEG[i])
    os.rename(rgb, toRgb)
    os.rename(seg, toSeg)

