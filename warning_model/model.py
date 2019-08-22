from keras.applications.vgg16 import VGG16

from keras.layers import Input, Flatten, Dense,MaxPooling2D, BatchNormalization, Conv2D,Dropout, regularizers, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3

# def vgg16(image_shape, num_of_classes,lr_init,lr_decay):
#     model_vgg16 = VGG16(weights="imagenet", include_top=False)
#     input = Input(shape=image_shape, name='image_input')
#
#     model = model_vgg16(input)
#
#     x = Flatten(name='flatten')(model)
#     x = Dense(4096, activation='relu', name='fc1')(x)
#     x = Dense(4096, activation='relu', name='fc2')(x)
#     x = Dense(num_of_classes, activation='softmax', name='predictions')(x)
#
#     # Create your own model
#     model = Model(inputs=input, outputs=x)
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_init, decay=lr_decay), metrics=['accuracy'])
#
#     return model
def custom(image_shape, num_of_classes,lr_init,lr_decay):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_init, decay=lr_decay), metrics=['accuracy'])
    return model




def nvidia(image_shape, num_of_classes,lr_init,lr_decay):
    model = Sequential()
    model.add((BatchNormalization(epsilon=0.001, axis=1, input_shape=(256, 256, 3))))
    model.add(Conv2D(24, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))
    model.add(Conv2D(36, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
    model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="th"))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_init, decay=lr_decay), metrics=['accuracy'])
    return model


def inception(image_shape, num_of_classes,lr_init,lr_decay):
    base_model = InceptionV3(weights='imagenet',input_shape=image_shape, include_top=False)
    # input = Input(shape=image_shape, name='image_input')
    # base_model = base_model(input)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_of_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # for layer in base_model.layers:
    #     layer.trainable = False
    # model.compile(optimizer='rmsprop',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_init, decay=lr_decay), metrics=['accuracy'])
    return model