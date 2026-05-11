# 5 Develop a SoftMax classifier using Python for multi-class image classification on the CIFAR-10 dataset and evaluate its performance metrics. CO2
# Implement in python softmax Classifer for CIFAR -10 dataset
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder

# classify test and train data
# Load the cifar data set X-image, y-label for train and test

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

cifar_10_classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
print(x_train.shape)
print(x_test.shape)

import matplotlib.pyplot as plt
plt.imshow(x_train[777])
plt.title(cifar_10_classes[y_train[777][0]])
plt.axis("off")

x_train = x_train/255.0
x_test = x_test/255.0

one_hot_encoder = OneHotEncoder()
y_train = one_hot_encoder.fit_transform(y_train).toarray()
y_test = one_hot_encoder.fit_transform(y_test).toarray()

softmax_model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(32,32,3)), tf.keras.layers.Dense(10, activation = 'softmax')])
softmax_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
softmax_model.fit(x_train, y_train, epochs = 20, batch_size = 20, validation_data = (x_test, y_test))


new_image = x_test[10]
plt.imshow(new_image)
plt.axis("off")
img = np.expand_dims(new_image, axis = 0)
print(img.shape)

pred = softmax_model.predict(img)
prediction =  np.argmax(pred)
cifar_10_classes[prediction]

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 1. Evaluate on test set
loss, accuracy = softmax_model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# 2. Get predictions for all test images
y_pred = softmax_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 3. Detailed Metrics
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=cifar_10_classes))

# 4. Confusion Matrix Visualization
plt.figure(figsize=(10,8))
cm = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cifar_10_classes, yticklabels=cifar_10_classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

