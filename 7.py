# 7 Design and implement a deep learning model to classify underwater sonar signals into two categories (Rocks 'R' or Mines 'M') using the sonar_dataset.csv. Evaluate the performance of the model on unseen test data and demonstrate the impact of incorporating dropout layers to improve generalization. CO2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("sonar.csv", header= None)
df.sample(5)
df.shape
# check for non values
df.isna().sum()
df.columns
df[60].value_counts()
X = df.drop(60, axis =1)
y = df[60]
y.head()

y = pd.get_dummies(y, drop_first = True)
y.sample(5) # R--> and M --> 0
y.value_counts()
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split (X, y, test_size = 0.25, random_state = 1)
X_train.head()

#using Deep learning model without Dropout layer
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(60, input_dim = 60, activation = 'relu'),
    keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dense(15, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs= 100, batch_size=8)
model.evaluate(X_test, y_test)
y_pred = model.predict(X_test).reshape(-1)
print(y_pred[:10])
#round the values to neaerst integer ie 0 or 1
y_pred = np.round(y_pred)
print(y_pred[:10])
y_test[:10]

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Model with Dropout layer
modeld = keras.Sequential([
    keras.layers.Dense(60, input_dim = 60, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation = 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation = 'sigmoid')
])
modeld.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

modeld.fit(X_train, y_train, epochs= 100, batch_size=8)
modeld.evaluate(X_test, y_test)
y_pred = modeld.predict(X_test).reshape(-1)
print(y_pred[:10])
#round the values to neaerst integer ie 0 or 1
y_pred = np.round(y_pred)
print(y_pred[:10])
y_test[:10]

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
