import os
import tensorflow as tf
from model.base_model import Model

# TODO: Depreciated: See old commit to understand how to use this model
class MLP(Model):
    def __init__(self):
        super().__init__("MLP")

    def init_model(self, class_output):
        self.class_output = class_output

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=self.height*self.width))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.class_output, activation='softmax'))

    def train(self, x_train, y_train, x_val, y_val, optimizer, loss):
        if loss == 'categorical_crossentropy':
            # if we want to use categorical_crossentropy loss, we have to convert the "y" labels
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.class_output)
            y_val = tf.keras.utils.to_categorical(y_val, num_classes=self.class_output)

        # compile the model with optimizer and loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # train model
        history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=self.epoch)
        return history

    def predict(self, image):
        reshape_image = image.reshape(1, self.height*self.width)
        return self.model.predict(reshape_image)