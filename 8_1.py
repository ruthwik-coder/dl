# alexnet
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
# =====================================
# Create AlexNet Model
# =====================================
model = Sequential([
    # Block 1
    Conv2D(
        96,
        (11,11),
        activation='relu',
        input_shape=(227,227,3)
    ),

    BatchNormalization(),

    MaxPooling2D((3,3)),

    # Block 2
    Conv2D(
        256,
        (5,5),
        activation='relu'
    ),

    BatchNormalization(),

    MaxPooling2D((3,3)),

    # Block 3
    Conv2D(
        384,
        (3,3),
        activation='relu'
    ),

    # Block 4
    Conv2D(
        384,
        (3,3),
        activation='relu'
    ),

    # Block 5
    Conv2D(
        256,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((3,3)),

    # Fully Connected Layers
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

    # Output Layer
    Dense(
        10,
        activation='softmax'
    )

])

# =====================================
# Compile Model
# =====================================

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)

# =====================================
# Show Architecture
# =====================================

model.summary()
