# recovery.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


DATASET_PATH = "/CUB_200_2011/images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


model = tf.keras.models.load_model('best_model.keras')  # From Epoch 26


print("\nLoaded Model Metrics:", model.metrics_names)
test_loss, test_acc, test_top5 = model.evaluate(val_generator)
print(f"Sanity Check - Should match Epoch 26:")
print(f"Val Accuracy: {test_acc:.1%} | Top-5: {test_top5:.1%}")


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Apply training fixes
model.optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-5,  # Start with lower rate
    clipvalue=1.0
)

# Freeze more layers for stability
for layer in model.layers[0].layers[:-30]:  # MobileNetV2 base
    layer.trainable = False

model.compile(optimizer=model.optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy',
                      tf.keras.metrics.TopKCategoricalAccuracy(k=5)])

# Resume training
model.fit(
    train_generator,
    validation_data=val_generator,
    initial_epoch=26,  # Continue from epoch 26
    epochs=50,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('recovery_model.keras',
                                          save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5)
    ]
)