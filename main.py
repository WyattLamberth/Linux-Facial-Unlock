import tensorflow as tf
from tensorflow.keras import layers, models
from data_processing import load_data  # Make sure this matches your file structure

def main():
    # Load and preprocess the data
    base_dir = '/home/wyatt/Developer/tf-learning/face_images'
    train_generator, validation_generator = load_data(base_dir)

    # Build your model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')  # Use 'sigmoid' for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Use 'binary_crossentropy' for binary classification
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=48,  # Adjust based on your dataset size
        epochs=5,
        validation_data=validation_generator,
        validation_steps=50  # Adjust based on your dataset size
    )

    # Evaluate the model (optional, if you want to print out the evaluation metrics)
    eval_result = model.evaluate(validation_generator)
    print(f"Validation Loss: {eval_result[0]}, Validation Accuracy: {eval_result[1]}")

if __name__ == "__main__":
    main()

