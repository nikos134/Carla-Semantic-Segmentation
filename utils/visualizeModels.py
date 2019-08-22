from keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=tf_config))

import os
def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

ROOT_DIR = os.path.abspath("/media/nikos134/DATADRIVE1/onedrive/CNN/unet")

model_path = os.path.join(ROOT_DIR, 'ver2.h5')

model = load_model(model_path, custom_objects={'dice_coef': dice_coef})

model.summary()
# plot_model(model, to_file='unet_model_2.png')