import json
import os
import numpy as np

import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.utils.processing_data import preprocess_data, get_class_output, display_dataset
from src.model.MLP import MLP
from model.CNN import CNN

from src.ui.draw_digit_ui import DrawDigitUI

if __name__ == '__main__':
    parameters_path = os.path.join(os.getcwd(), 'configuration/parameters.json')
    with open(parameters_path, 'r') as file:
        data = json.load(file)
    height = data['image']['height']
    width = data['image']['width']
    dataset_name = data['dataset']['name']

    # load data
    path = os.path.join(os.getcwd(), "..", "data", dataset_name)
    x_data_loaded = np.load(os.path.join(path, "X", 'x_data.npy'))
    y_data_loaded = np.load(os.path.join(path, "y", 'y_label.npy'))

    # To see the dataset images before augmentation, uncomment this lines
    #display_dataset(x_data_loaded, y_data_loaded)

    x_train, x_val, y_train, y_val = train_test_split(x_data_loaded, y_data_loaded, test_size=0.2)    # pre process data

    # model = MLP()
    model = CNN()

    if model.name == "MLP":
        x_train, y_train, x_val, y_val = preprocess_data(x_train, y_train, x_val, y_val, "MLP", False)
    elif model.name == "CNN":
        x_train, y_train, x_val, y_val = preprocess_data(x_train, y_train, x_val, y_val,"CNN")

    # To see the entire dataset images after augmentation, uncomment this lines
    #display_dataset(x_train, y_train)

    # init the model
    class_output = get_class_output(y_train) # same if we use y_val
    model.init_model(class_output)

    # train
    history = model.train(x_train, y_train, x_val, y_val, model.optimiser, model.loss)
    model.display_result(history)

    print(model.model.summary())

    # one prediction with data from validation dataset
    i = 12
    image = x_val[i].reshape(width, height)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    vect_prob = model.predict(x_val[i])
    result, prob = model.predict_best_class(vect_prob)
    print('I am confident about {:.2f}% that this image corresponds to digit {}'.format(prob, result))

    # predict by creating the data yourself by drawing the digit
    ui = DrawDigitUI()
    ui.build_ui(model)
    ui.run()

    # save model if it doesn't exist
    # model.save_if_model_doesnt_exist()
    # or save the model even if it exists
    model.save_model()




