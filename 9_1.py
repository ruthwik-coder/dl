
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# =====================================
# Load Dataset
# =====================================

train_data = ImageDataGenerator(
    rescale=1./255
)

test_data = ImageDataGenerator(
    rescale=1./255
)

train = train_data.flow_from_directory(

    "YOUR_TRAIN_FOLDER_PATH",

    target_size=(224,224),

    batch_size=8,

    class_mode='binary'

)

test = test_data.flow_from_directory(

    "YOUR_TEST_FOLDER_PATH",

    target_size=(224,224),

    batch_size=8,

    class_mode='binary'

)

# =====================================
# VGG16 Model
# =====================================

model = Sequential([

    # Block 1
    Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same',
        input_shape=(224,224,3)
    ),

    Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    ),

    MaxPooling2D((2,2)),

    # Block 2
    Conv2D(128,(3,3),activation='relu',padding='same'),

    Conv2D(128,(3,3),activation='relu',padding='same'),

    MaxPooling2D((2,2)),

    # Block 3
    Conv2D(256,(3,3),activation='relu',padding='same'),

    Conv2D(256,(3,3),activation='relu',padding='same'),

    Conv2D(256,(3,3),activation='relu',padding='same'),

    MaxPooling2D((2,2)),

    # Block 4
    Conv2D(512,(3,3),activation='relu',padding='same'),

    Conv2D(512,(3,3),activation='relu',padding='same'),

    Conv2D(512,(3,3),activation='relu',padding='same'),

    MaxPooling2D((2,2)),

    # Block 5
    Conv2D(512,(3,3),activation='relu',padding='same'),

    Conv2D(512,(3,3),activation='relu',padding='same'),

    Conv2D(512,(3,3),activation='relu',padding='same'),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(
        4096,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        4096,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        1,
        activation='sigmoid'
    )

])

# =====================================
# Compile Model
# =====================================

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)

# =====================================
# Model Summary
# =====================================

model.summary()

# =====================================
# Train Model
# =====================================

history = model.fit(

    train,

    epochs=10

)

# =====================================
# Evaluate Model
# =====================================

loss, accuracy = model.evaluate(test)

print("Accuracy:", accuracy * 100)

# =====================================
# Plot Accuracy
# =====================================

plt.plot(history.history['accuracy'])

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.show()

# =====================================
# Plot Loss
# =====================================

plt.plot(history.history['loss'])

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()
