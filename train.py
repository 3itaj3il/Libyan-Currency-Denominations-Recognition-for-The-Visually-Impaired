import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from google.colab import drive
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ====== 1. Mount Google Drive & Imports ======
drive.mount('/content/drive', force_remount=True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ====== 2. Set Paths ======
BASE_PROJECT = '/content/drive/My Drive/AI Project Dataset'

all_source_paths = [
    os.path.join(BASE_PROJECT, 'first_dataset/dataset_original'),
    os.path.join(BASE_PROJECT, 'first_dataset/dataset_augmented')
]

BATCH_SIZE = 32
IMG_SHAPE = (224, 224)

# ====== 3. Data Generators ======
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    all_source_paths[0],
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=123
)

# ====== 4. Model Architecture ======
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ====== 5. Model Training ======
EPOCHS = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS
)

# ====== 6. Plotting Results ======
acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', color='blue')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', color='red')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

# ====== 7. Save Model as .h5 ======
h5_save_path = os.path.join(BASE_PROJECT, 'my_currency_model.h5')
model.save(h5_save_path)
print(f"✅ تم حفظ الموديل بصيغة H5 في المسار التالي :\n {h5_save_path}")