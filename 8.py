# 8 Implement and analyse an AlexNet‑style deep convolutional neural network using TensorFlow Keras for image classification. The network should accept RGB images of size 227×227×3, consist of multiple convolutions, batch‑normalization, max‑pooling, and fully connected layers similar to the original AlexNet architecture, and output class probabilities for 10 categories using a SoftMax layer.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
inputs = keras.Input(shape = (227,227,3))
x = layers.Conv2D(filters = 96, kernel_size=(11, 11), strides=(4,4), activation = "relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(pool_size= (3,3), strides = (2,2))(x)

x = layers.Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), activation = "relu", padding = "same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(pool_size= (3,3), strides = (2,2))(x)

x = layers.Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "same")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "same")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "same")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D(pool_size= (3,3), strides = (2,2))(x)

x = layers.Flatten()(x)
x = layers.Dense(4096, activation = "relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4096, activation = "relu")(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(10, activation = "softmax")(x)
model = keras.Model(inputs = inputs, outputs = outputs)
optimizer = keras.optimizers.SGD(learning_rate = 0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"],)
model.summary()

!pip install visualkeras
import visualkeras
visualkeras.graph_view(model).show()
