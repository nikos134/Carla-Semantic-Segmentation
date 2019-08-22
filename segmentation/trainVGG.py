import keras
from keras.applications.vgg16 import VGG16

from keras.layers import Input, Flatten, Dense, Dropout

from keras import models
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
validation_directory = "/media/nikos134/DATADRIVE1/CarlaData/21_06/2/"
train_directory = "/media/nikos134/DATADRIVE1/CarlaData/21_06/1/"

def main():
    """

    """
    vgg16 = VGG16(weights ='imagenet',include_top=False, input_shape=(1280, 720, 3))
    # vgg16.summary()
    for layer in vgg16.layers[:-4]:
        layer.trainable = False

    # for layer in vgg16.layers:
    #     print(layer, layer.trainable)

    # # input = Input(shape=(1280, 720, 3), name='pwd
    # # out_vgg16 = vgg16(input)
    #
    # x = Flatten(name='flatten')(out_vgg16)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dense(8, activation='softmax', name='predictions')(x)
    #
    # # Create your own model
    # my_model = Model(input=input, output=x)
    my_model = models.Sequential()
    my_model.add(vgg16)
    my_model.add(Flatten())

    my_model.add(Dense(4096, activation='relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(3, activation='softmax'))
    my_model.summary()


    train_data = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

    valid_data = ImageDataGenerator(rescale=1. / 255)

    train_batchsize = 100
    val_batchsize = 10
    print(train_directory)
    gen_train = train_data.flow_from_directory(
        train_directory,
        target_size=(1280, 720),
        batch_size=train_batchsize,
        class_mode='input',
        shuffle=False)

    valid_gen = valid_data.flow_from_directory(
        validation_directory,
        target_size=(1280, 720),
        batch_size=val_batchsize,
        class_mode='input',
        shuffle=False)

    # In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

    # Train the model
    history = my_model.fit_generator(
        gen_train,
        steps_per_epoch=gen_train.samples / gen_train.batch_size,
        epochs=30,
        validation_data=valid_gen,
        validation_steps=valid_gen.samples / valid_gen.batch_size,
        verbose=1)

    # Save the model
    my_model.save('small_last4.h5')

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    # Create a generator for prediction
    valid_gen = valid_gen.flow_from_directory(
        validation_directory,
        target_size=(1280, 720),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

    # Get the filenames from the generator
    fnames = valid_gen.filenames

    # Get the ground truth from generator
    ground_truth = valid_gen.classes

    # Get the label to class mapping from the generator
    label2index = valid_gen.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v, k) for k, v in label2index.items())

    # Get the predictions from the model using the generator
    predictions = my_model.predict_generator(valid_gen,
                                          steps=valid_gen.samples / valid_gen.batch_size,
                                          verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors), validation_generator.samples))

    # Show the errors
    for i in range(len(errors)):
        pred_class = np.argmax(predictions[errors[i]])
        pred_label = idx2label[pred_class]

        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])

        original = load_img('{}/{}'.format(validation_directory, fnames[errors[i]]))
        plt.figure(figsize=[7, 7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()


if __name__ == '__main__':
    main()
