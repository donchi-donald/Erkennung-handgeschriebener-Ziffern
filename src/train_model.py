import  tensorflow as tf
import tensorflow.keras as keras

#device auswählen
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#parameter
mnist = keras.datasets.mnist

#prepare data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train /= 255.
x_test /= 255.
assert x_train.shape == (60000, 28, 28) and x_test.shape == (10000, 28, 28) and y_train.shape == (60000,) and y_test.shape == (10000,)


print(y_train[102])
