import tensorflow as tf


# Class for the model. In this case, we are using the MobileNetV2 model from Keras
class Model:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = None  # Initialize model without specifying input shape
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self, input_shape, output_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_shape)  # Adjust output_shape here
        ])
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=['accuracy'])

    def get_model(self):
        return self.model