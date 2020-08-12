import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import keras
import numpy as np

#load the training and test data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

print('max pixel value', np.max(x_train[0]))

# Scale images to the [0, 1] range
x_train = x_train / 255
x_test = x_test / 255

print('max pixel value', np.max(x_train[0]))

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#print the shape of the loaded data
print('train: ', x_train.shape, 'test:', x_test.shape)

#one-hot encode the y-values
from keras.utils import np_utils

NUM_CLASSES = 10

# print before
print(y_train[0])
y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
y_test = np_utils.to_categorical(y_test, NUM_CLASSES)
# print after
print(y_train[0])

##
# Visualize a few of the training images -- Not a necessary cell ---
##

from matplotlib import pyplot
import numpy as np


NUM_IMAGES = 250
NUM_PER_COL = 25

#grab the first NUM_PER_COL training images & show them in a grid
x = x_train[:NUM_IMAGES]
input_shape = x[0].shape

# if the number of images isn't evenly divisible by NUM_PER_COL
remainder = x.shape[0] % NUM_PER_COL
if remainder > 0:
    # make some filler zero-valued pixels to fill in the remaining empty columns in the last row
    filler = np.zeros((NUM_PER_COL-remainder, input_shape[0], input_shape[1]))
    
    # add the filler to the big image
    x = np.concatenate((x, filler), axis=0)

# NUM_IMAGES x each image width x each image height x 1 channel (it's a gray-scale image)
print(x.shape)  
x_plot = x.reshape((x.shape[0] // NUM_PER_COL, NUM_PER_COL, input_shape[0], input_shape[1]))
# Number of Rows (calculated) x NUM_PER_COL x each image width x each image height 
print(x_plot.shape)
x_plot = np.vstack([np.hstack(x) for x in x_plot])
# pixels accross x pixels down
print(x_plot.shape)
fig = pyplot.figure(figsize=(18, 16), dpi= 100,)   
pyplot.axis('off')
pyplot.imshow(x_plot, interpolation='lanczos', cmap='gray')




#
# Make the model
#

print('input shape', input_shape)
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Conv2D(16, kernel_size=(5, 5), activation="selu", padding="same", kernel_initializer='lecun_normal', strides=(2,2)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="selu", padding="same", kernel_initializer='lecun_normal'),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="selu", padding="same", kernel_initializer='lecun_normal',  strides=(2,2)),
        #keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.AlphaDropout(0.25),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="selu", padding="valid", kernel_initializer='lecun_normal'),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="selu", padding="same", kernel_initializer='lecun_normal', strides=(2,2)),
        keras.layers.AlphaDropout(0.5),
        keras.layers.Conv2D(NUM_CLASSES, kernel_size=(3, 3), activation="softmax", padding="valid", kernel_initializer='lecun_normal', use_bias=False),
        keras.layers.Flatten(),
    ]
)

model.summary()


# setup the tensorboard callback
# import datetime
# logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# callbacks = [tensorboard_callback]

##
# Train the model
##

import os

model_dir = 'models/'
model_classification_path = os.path.join(model_dir, 'model_classification')
# create directory if it doesn't exist
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

batch_size = 128
epochs = 15                                         

if os.path.exists(model_classification_path):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test,y_test))#, callbacks=callbacks)
    model.save(model_classification_path)
#keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        #keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="valid"),
        #keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu",padding='same'),
        #keras.layers.MaxPooling2D(pool_size=(2, 2)),
        #keras.layers.Conv2D(NUM_CLASSES, kernel_size=(3, 3), activation="relu", use_bias=False, padding='same'),
        #keras.layers.GlobalAveragePooling2D(),
        #keras.layers.Activation('softmax')
        #keras.layers.MaxPooling2D(pool_size=(2, 2)),