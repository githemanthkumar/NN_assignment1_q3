import tensorflow as tf
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Train model with a given optimizer
def train_model(optimizer, name):
    model = create_model()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=2)
    return history

# Train models with Adam and SGD
adam_history = train_model(tf.keras.optimizers.Adam(), "Adam")
sgd_history = train_model(tf.keras.optimizers.SGD(), "SGD")

# Plot accuracy comparison
plt.figure(figsize=(10, 5))
plt.plot(adam_history.history['val_accuracy'], label='Adam - Validation Accuracy', linestyle='dashed', color='blue')
plt.plot(adam_history.history['accuracy'], label='Adam - Training Accuracy', color='blue')
plt.plot(sgd_history.history['val_accuracy'], label='SGD - Validation Accuracy', linestyle='dashed', color='red')
plt.plot(sgd_history.history['accuracy'], label='SGD - Training Accuracy', color='red')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Adam vs SGD Optimizer Performance')
plt.legend()
plt.show()
