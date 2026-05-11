# 6 Develop a convolutional neural network (CNN) model to classify handwritten digits using the MNIST dataset. The goal is to train a model that accurately identifies digits (0-9) from images. CO2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np

dataset = keras.datasets.mnist
class_names = ['Zero', 'one', 'two', 'three' , 'Four', 'Five', 'Six', 'seven', 'Eight', 'nine']
(x_train, y_train), (x_test, y_test) = dataset.load_data()
X_train =x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
X_test =x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
print(X_train.shape)
print(X_test.shape)

#plot five data with its class name
plt.figure(figsize = (12,5))
for i in range(9):
    plt.subplot(1,9,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i]])
    plt.axis("off")
    plt.tight_layout()
plt.show()
#convert into grayscale
X_train = X_train/255
X_test = X_test/255

model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
model.fit(x_train,y_train,epochs=5,callbacks=keras.callbacks.EarlyStopping(patience=2))
#evaluting the model
model.evaluate(x_test, y_test)
#prediction
sample_img= X_test[0]
sample_img = X_test[0]
print(sample_img.shape)
plt.imshow(sample_img)
plt.title(f"Predicted: {class_names[y_test[0]]}")
img = np.expand_dims(sample_img, axis = 0)
print(img.shape)
pred = model.predict(img)
print(f"Predicted: {class_names[y_test[0]]}")
model.summary()
