import os.path
import cv2 as opencv
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.keras as keras

#Load the model
model = keras.models.load_model('../model/ziffer_model.h5')


def image_to_array(image_name):
    img_to_array = opencv.imread(image_name)[:, :, 0]  # shape(28 x 28 x 3 )(RGB) to shape (28 x 28) (BW)
    img_to_array_invert = np.invert(np.array([img_to_array]))  # shwarz to weiß und weiß to shwarz
    return img_to_array_invert


def show_image(image_array):
    plt.imshow(image_array[0], cmap=plt.cm.binary)
    plt.show()


i = 0
while True:
    image_name = '../img/ziffer{}.png'.format(i)
    isImg = os.path.isfile(image_name)
    if isImg:
        img = image_to_array(image_name)
        show_image(img)
        i += 1
    else:
        break

