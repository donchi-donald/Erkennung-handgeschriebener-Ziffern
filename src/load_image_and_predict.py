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
    img_to_array_invert = img_to_array_invert / 255.
    return img_to_array_invert


def show_image(image_array):
    plt.imshow(np.reshape(image_array, [28,28]), cmap='gray')
    plt.show()


i = 5
while True:
    image_name = '../img/ziffer{}.png'.format(i)
    isImg = os.path.isfile(image_name)
    if isImg:
        imgToArray = image_to_array(image_name)
        prediction = model.predict(imgToArray)
        print("Die Zahl ist wahrscheinlich eine {}".format(np.argmax(prediction)))
        show_image(imgToArray)
        i += 1
    else:
        break

