import os
import json

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

class MLP:
    def __init__(self):
        self.model = None
        self.optimiser = None
        self.loss = None
        self.epoch = None
        self.image_size = None
        self.class_output = None

        self.save_model_path = os.path.join(os.getcwd(), 'configuration/saved_model/mlp.keras')

        self.init_parameters()

    def init_parameters(self):
        parameters_path = os.path.join(os.getcwd(), 'configuration/parameters.json')
        with open(parameters_path, 'r') as file:
            data = json.load(file)
        self.optimiser = data['optimiser']
        self.loss = data['loss']
        self.epoch = data['epoch']
        self.image_size = data['image_size'] * data['image_size']

    def init_model(self, class_output):
        self.class_output = class_output

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(128, activation='relu', input_dim=self.image_size))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dense(32, activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.class_output, activation='softmax'))


    def save_model(self):
        self.model.save(self.save_model_path)

    def save_if_model_doesnt_exist(self):
        if not os.path.exists(self.save_model_path):
            self.model.save(self.save_model_path)
            print("model saved")
        else:
            print("model alredy saved")

    def load_model(self):
        self.model = tf.keras.models.load_model(self.save_model_path)

    def train(self, x_train, y_train, x_val, y_val, optimizer, loss):
        if loss == 'categorical_crossentropy':
            # if we want to use categorical_crossentropy loss, we have to convert the "y" labels
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
            y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)

        # compile the model with optimizer and loss
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # train model
        history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)
        return history

    def predict(self, image):
        reshape_image =  image.reshape(1, self.image_size)
        return self.model.predict(reshape_image)

    def predict_best_class(self, result):
        return np.argmax(result), np.amax(result) * 100


    def display_result(self, history):
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)  # Force y-axis to range from 0 to 1
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.ylim(0, 1)  # Force y-axis to range from 0 to 1
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper right')
        plt.show()