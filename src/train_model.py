import  tensorflow as tf
import tensorflow.keras as keras

#device auswaehlen
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#parameter definieren
mnist = keras.datasets.mnist
layers = keras.layers

#daten vorbereiten
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.
x_test = x_test / 255.
assert x_train.shape == (60000, 28, 28) and x_test.shape == (10000, 28, 28) and y_train.shape == (60000,) and y_test.shape == (10000,)

#model erzeugen
model = keras.models.Sequential() #sequentielle model erzeugen
model.add(layers.Flatten(input_shape=(28,28))) #Eingabeschicht für die Pixel hinzufuegen (inputlayers)
model.add(layers.Dense(128, activation=tf.nn.relu))  #hidden layers mit 128 Neuronen hinzufuegen
model.add(layers.Dense(256, activation=tf.nn.relu)) #hidden layers mit 256 Neuronen hinzufuegen
model.add(layers.Dense(10, activation=tf.nn.softmax)) #ausgabeschicht mit 10 Neuronen für die 10 Ziffern  hinzufuegen

#model kompilieren und optimieren
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',#Loss/Fehler-Funktion
    metrics=['accuracy']#Accuracy Metric (Was wollen wir haben)
)



