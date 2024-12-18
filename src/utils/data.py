import json
import os
import numpy as np

def get_class_output(y):
    return len(np.unique(y))

def get_image_size():
    parameters_path = os.path.join(os.getcwd(), 'configuration/parameters.json')
    with open(parameters_path, 'r') as file:
        data = json.load(file)
    return data['image']['height'] , data['image']['width']

def preprocess_data_mlp(x_train, y_train, x_val, y_val):
    height, width = get_image_size()

    x_train = x_train.reshape(len(x_train), height*width)
    x_val = x_val.reshape(len(x_val), height*width)

    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255

    display_dataset(x_train, y_train, x_val, y_val)

    return x_train, y_train, x_val, y_val

def preprocess_data_cnn(x_train, y_train, x_val, y_val):
    height, width = get_image_size()

    x_train = x_train.reshape(len(x_train), height, width, 1)  # 1 canal, not RGB
    x_val = x_val.reshape(len(x_val), height, width, 1)

    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255

    display_dataset(x_train, y_train, x_val, y_val)

    return x_train, y_train, x_val, y_val

def display_dataset(x_train, y_train, x_val, y_val):
    print('train samples', x_train.shape)
    print('validation samples', x_val.shape)
    print('train label samples', y_train.shape)
    print('validation label samples', y_val.shape)
