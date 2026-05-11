# 9 Implement the standard VGG 16 CNN architecture model to classify cat and dog image dataset and check the accuracy CO3
import os
import shutil
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------------------------------
# 1. Original dataset folders
# -------------------------------------------------
# cats_source = r"C:\Users\ADMIN\VGG16\dataset\cats_set"
# dogs_source = r"C:\Users\ADMIN\VGG16\dataset\dogs_set"
cats_source = "/content/drive/MyDrive/vgg16/cats_set"
dogs_source = "/content/drive/MyDrive/vgg16/dogs_set"
# -------------------------------------------------
# 2. New train/test dataset folder
# -------------------------------------------------
base_dir = r"/content/drive/MyDrive/vgg16/dataset"

train_cats = os.path.join(base_dir, "train", "cats")
train_dogs = os.path.join(base_dir, "train", "dogs")
test_cats = os.path.join(base_dir, "test", "cats")
test_dogs = os.path.join(base_dir, "test", "dogs")

# Create folders
for folder in [train_cats, train_dogs, test_cats, test_dogs]:
    os.makedirs(folder, exist_ok=True)

# -------------------------------------------------
# 3. Split function
# -------------------------------------------------
def split_data(source_folder, train_folder, test_folder, split_ratio=0.8):
    files = [f for f in os.listdir(source_folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]

    random.shuffle(files)
    split_index = int(len(files) * split_ratio)

    train_files = files[:split_index]
    test_files = files[split_index:]
    for file in train_files:
        src = os.path.join(source_folder, file)
        dst = os.path.join(train_folder, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    for file in test_files:
        src = os.path.join(source_folder, file)
        dst = os.path.join(test_folder, file)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"{source_folder} -> {len(train_files)} train, {len(test_files)} test")
# -------------------------------------------------
# 4. Split cats and dogs images
# -------------------------------------------------
split_data(cats_source, train_cats, test_cats, split_ratio=0.8)
split_data(dogs_source, train_dogs, test_dogs, split_ratio=0.8)

# -------------------------------------------------
# 5. Final dataset paths
# -------------------------------------------------
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
print("Train folders:", os.listdir(train_dir))
print("Test folders:", os.listdir(test_dir))
# -------------------------------------------------
# 6. Image preprocessing
# -------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,rotation_range=20,zoom_range=0.2,
    width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2,  horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    train_dir,   target_size=(224, 224),  batch_size=8, class_mode='binary')

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

print("Train samples:", train_data.samples)
print("Test samples :", test_data.samples)
print("Class indices:", train_data.class_indices)

# -------------------------------------------------
# 7. Standard VGG16 architecture
# -------------------------------------------------
model = Sequential()

# Block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Block 4
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# -------------------------------------------------
# 8. Compile model
# -------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------
# 9. Callbacks
# -------------------------------------------------
checkpoint = ModelCheckpoint(
    "vgg16_cats_dogs_best.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)

earlystop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

# -------------------------------------------------
# 10. Train model
# -------------------------------------------------
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10,
    callbacks=[checkpoint, earlystop]
)

# -------------------------------------------------
# 11. Evaluate accuracy
# -------------------------------------------------
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# -------------------------------------------------
# 12. Plot results
# -------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
