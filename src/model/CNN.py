import os
import tensorflow as tf
from model.base_model import Model

class CNN(Model):
    def __init__(self):
        super().__init__("CNN")

        self.save_model_path = os.path.join(os.getcwd(), 'configuration/saved_model/cnn.keras')

    def init_model(self, class_output):
        self.class_output = class_output

        self.model = tf.keras.Sequential()
        # 1 Conv
        self.model.add(tf.keras.layers.Conv2D(6, kernel_size=5, strides=1, padding='valid', activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))
        # 2 Conv
        self.model.add(tf.keras.layers.Conv2D(6, kernel_size=5, strides=1, padding='valid', activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid'))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(class_output, activation='softmax'))

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
        reshape_image =  image.reshape(1, self.height, self.width, 1)
        return self.model.predict(reshape_image)