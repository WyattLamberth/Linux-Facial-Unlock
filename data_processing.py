import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_dir):
    # Setup image data generator with preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values
        validation_split=0.2  # Use 20% of images as a validation set
    )

    # Load regular images from the base directory
    regular_dir = os.path.join(base_dir, 'regular')
    regular_train_generator = datagen.flow_from_directory(
        regular_dir,
        target_size=(128, 128),  # Resize images to 128x128
        batch_size=20,
        class_mode='binary',  # For binary classification
        subset='training'  # Specify this is training data
    )

    regular_validation_generator = datagen.flow_from_directory(
        regular_dir,
        target_size=(128, 128),
        batch_size=20,
        class_mode='binary',
        subset='validation'  # Specify this is validation data
    )

    # Load IR images from the base directory
    ir_dir = os.path.join(base_dir, 'IR_Images')
    ir_train_generator = datagen.flow_from_directory(
        ir_dir,
        target_size=(128, 128),  # Resize images to 128x128
        batch_size=20,
        class_mode='binary',  # For binary classification
        subset='training'  # Specify this is training data
    )

    ir_validation_generator = datagen.flow_from_directory(
        ir_dir,
        target_size=(128, 128),
        batch_size=20,
        class_mode='binary',
        subset='validation'  # Specify this is validation data
    )

    # Merge regular and IR generators
    train_generator = tf.keras.preprocessing.image.Iterator(
        regular_train_generator.directory,
        regular_train_generator.image_data_generator,
        regular_train_generator.target_size,
        regular_train_generator.color_mode,
        regular_train_generator.classes,
        regular_train_generator.class_mode,
        regular_train_generator.batch_size,
        regular_train_generator.shuffle,
        regular_train_generator.seed,
        regular_train_generator.data_format,
        regular_train_generator.save_to_dir,
        regular_train_generator.save_prefix,
        regular_train_generator.save_format,
        subset='training'
    )

    validation_generator = tf.keras.preprocessing.image.Iterator(
        regular_validation_generator.directory,
        regular_validation_generator.image_data_generator,
        regular_validation_generator.target_size,
        regular_validation_generator.color_mode,
        regular_validation_generator.classes,
        regular_validation_generator.class_mode,
        regular_validation_generator.batch_size,
        regular_validation_generator.shuffle,
        regular_validation_generator.seed,
        regular_validation_generator.data_format,
        regular_validation_generator.save_to_dir,
        regular_validation_generator.save_prefix,
        regular_validation_generator.save_format,
        subset='validation'
    )

    # Concatenate IR generators to the regular ones
    for batch in ir_train_generator:
        train_generator._set_index_array()
        train_generator.filepaths.extend(batch[0])
        train_generator.classes = tf.concat([train_generator.classes, batch[1]], axis=0)
        break

    for batch in ir_validation_generator:
        validation_generator._set_index_array()
        validation_generator.filepaths.extend(batch[0])
        validation_generator.classes = tf.concat([validation_generator.classes, batch[1]], axis=0)
        break

    return train_generator, validation_generator
