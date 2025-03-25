# Add at the top of your file
import absl
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

# Configure GPU acceleration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("GPU acceleration enabled")
    except RuntimeError as e:
        print(e)

# Dataset parameters
DATASET_PATH = "dataset/CUB_200_2011/images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 200
EPOCHS = 50

# Optimized data pipeline
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='constant',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)



train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Fix steps calculation
train_steps = int(np.ceil(train_generator.samples / BATCH_SIZE))
val_steps = int(np.ceil(val_generator.samples / BATCH_SIZE))


# Verify class distribution
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# MobileNetV2 base model configuration
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Progressive unfreezing strategy
base_model.trainable = True
for layer in base_model.layers[:-40]:  # Unfreeze last 40 layers
    layer.trainable = False

# Enhanced model architecture
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(384, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.25),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

# Optimized learning configuration
optimizer = tf.keras.optimizers.Adam(
    learning_rate=2e-4,
    beta_1=0.9,
    beta_2=0.999,
    clipnorm=1.0
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')
    ]
)

# Training callbacks
callbacks = [
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_top5_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=12,
        min_delta=0.002,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )
]


# Start training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed. Best model saved as 'best_model.keras'")

# Save final model
model.save('final_model.keras')
print("Final model saved as 'final_model.keras'")