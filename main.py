import tensorflow as tf
from data_processing import load_data, preprocess_data

# Load and preprocess the data
(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, test_images = preprocess_data(train_images, test_images)

# Build your model (example)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile, train, and evaluate your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)
