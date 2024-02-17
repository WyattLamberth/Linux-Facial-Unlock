# data_preprocessing.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_dir):
    # Setup image data generator with preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values
        validation_split=0.2  # Use 20% of images as a validation set
    )

    # Load training and validation data
    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(128, 128),  # Resize images to 128x128
        batch_size=20,
        class_mode='binary',  # For binary classification
        subset='training'  # Specify this is training data
    )

    validation_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(128, 128),
        batch_size=20,
        class_mode='binary',
        subset='validation'  # Specify this is validation data
    )

    return train_generator, validation_generator
