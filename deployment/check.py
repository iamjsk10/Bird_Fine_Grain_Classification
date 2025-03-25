import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Data Pipeline Setup (Make sure it matches your training setup)
DATASET_PATH = "/CUB_200_2011/images"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Make sure this matches your training batch size

# Recreate validation generator
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2  # Must match the original split
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Critical for consistent evaluation
)

# Load the model
model = tf.keras.models.load_model('recovery_model.keras')  # or 'best_model.keras'

# Evaluate the model
test_loss, test_acc, test_top5 = model.evaluate(val_generator)
print(f"Validation Accuracy: {test_acc:.1%} | Top-5 Accuracy: {test_top5:.1%}")
