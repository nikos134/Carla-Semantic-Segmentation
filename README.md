# Carla-Semantic-Segmentation
This repository holds my dissertation project during my time at the University of Bristol titled 'Deep Learning for Semantic Segmentation'.

## Summary

**Simulator:** Carla ver 0.9.5

**Data preprocessing:** Created front-view image representations of the Lidar Data, augment the collected datasets to create the different classes for the Warning Model

**Data augmentation:** Random rotations, translations, blur, brightness, occlusions.

**Architecture:** Modified U-net & custom CNN.

## Datasets

I gathered in total 42,000 frames, by running the Carla simulaor on an Ubuntu Machine

Ouput:<br />

<img align="left" width=200 height=150 src="/Dataset_Sample/RGB/00183310.png" alt="RGB_image" title="RGB" hspace="20"/>
<img align="left" width=200 height=150 src="./Dataset_Sample/Segmentation_Images/00183310.png" alt="Segmentation_Image" title="Segmentation" hspace="20"/>
<img align="" width=200 height=150 src="/Dataset_Sample/Lidar_Images/00183310.png" alt="Lidar_image" title="Lidar" hspace="20"/>
<br /><br />



**Data Augmentation**



<img align="left" width=200 height=150 src="/Dataset_Sample/Warning_Model_Data/b_00183670.png" alt="Rotation" title="Rotation" hspace="20"/>
<img align="left" width=200 height=150 src="/Dataset_Sample/Warning_Model_Data/c_00184044.png" alt="Blur" title="Blur" hspace="20"/>
<img align="" width=200 height=150 src="/Dataset_Sample/Warning_Model_Data/d_00183968.png" alt="Occlusion" title="Occlusion" hspace="20"/>

## Architecture

U-net modified for image containing both the RGb and Lidar channels (256 x 512 x 6)

Custom CNN model for classifing the image quality. Image input size (256 x 512 x 3)

![](/imgs/Architecture.png)


## Results

**RGB U-net**

| Dataset       |   Accuracy    | Dice Coefficient |
| ------------- | ------------- | -------------    |
| Training      |     96.91%    |       0.9806     |
| Validation    |     96.01%    |       0.9762     |
| Evaluation    |     87.69%    |       0.9485     |

| Weather ID    |   Mean IU     | 
| ------------- | ------------- | 
| CloudyNoon    |     52.51%    |       
| WetNoon       |     59.08%    |     
| WetCloudyNoon |     67.84%    |    
| MidRainyNoon  |     53.98%    |    
| HardRainNoon  |     50.53%    |      
| SoftRainNoon  |     55.05%    |      

**Sesnor-Fusion U-net**

| Dataset       |   Accuracy    | Dice Coefficient |
| ------------- | ------------- | -------------    |
| Training      |     97.51%    |       0.9858     |
| Validation    |     96.61%    |       0.9817     |
| Evaluation    |     82.52%    |       0.9498     |

| Weather ID    |   Mean IU     | 
| ------------- | ------------- | 
| CloudyNoon    |     55.01%    |       
| WetNoon       |     62.40%    |     
| WetCloudyNoon |     71.47%    |    
| MidRainyNoon  |     57.00%    |    
| HardRainNoon  |     53.33%    |      
| SoftRainNoon  |     57.70%    |  
