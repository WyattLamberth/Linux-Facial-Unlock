import tensorflow as tf

def load_data():
    """Loads the MNIST dataset and returns train and test sets."""
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)

def preprocess_data(train_images, test_images):
    """Preprocesses the image data by normalizing the pixel values."""
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, test_images
