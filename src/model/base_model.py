import os
import json
import re
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class Model(ABC):
    def __init__(self, name):
        self.name = name
        self.model = None
        self.optimiser = None
        self.loss = None
        self.epoch = None
        self.height = None
        self.width = None
        self.class_output = None

        self.save_model_path = os.path.join(os.getcwd(), "configuration/saved_model")
        self.saved_model_name = ''

        self.init_parameters()


    @abstractmethod
    def init_model(self, class_output):
        pass


    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val, optimizer, loss):
        pass


    @abstractmethod
    def predict(self, image):
        pass


    def init_parameters(self):
        parameters_path = os.path.join(os.getcwd(), 'configuration/parameters.json')
        with open(parameters_path, 'r') as file:
            data = json.load(file)
        self.saved_model_name = f"{data['model'][self.name]['saved_model']}.keras"
        self.optimiser = data['model'][self.name]['optimiser']
        self.loss = data['model'][self.name]['loss']
        self.epoch = data['model'][self.name]['epoch']
        self.height = data['image']['height']
        self.width = data['image']['width']


    def save_model(self):
        saved_model = f"cnn_{self.get_number_model_to_save(self.save_model_path, self.name.lower())}.keras"
        path_model = os.path.join(self.save_model_path, saved_model)

        self.model.save(path_model)
        print(f'model {self.saved_model_name} saved')


    def get_number_model_to_save(self, base_path, name):
        files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
        numbers = []
        for file in files:
            match = re.search(rf"{name}_(\d+)", file)
            if match:
                numbers.append(int(match.group(1)))

        if len(numbers) == 0:
            return 0
        return max(numbers) + 1


    def load_model(self):
        self.model = tf.keras.models.load_model(os.path.join(self.save_model_path, self.saved_model_name))


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