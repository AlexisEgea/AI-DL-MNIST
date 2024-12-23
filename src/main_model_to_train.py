import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

from src.ui.draw_digit_ui import DrawDigitUI
from src.utils.data import preprocess_data_mlp, preprocess_data_cnn, get_class_output, display_dataset_shape, \
    display_dataset
from src.model.MLP import MLP
from model.CNN import CNN

if __name__ == '__main__':
    # model = MLP()
    model = CNN()

    # load data
    mnist = tf.keras.datasets.mnist

    # the data, split between train and validation sets
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    # pre process data
    if model.name == "MLP":
        x_train, y_train, x_val, y_val = preprocess_data_mlp(x_train, y_train, x_val, y_val)
    elif model.name == "CNN":
        x_train, y_train, x_val, y_val = preprocess_data_cnn(x_train, y_train, x_val, y_val)

    # To see the entire dataset images, uncomment this lines
    # display_dataset(x_train, y_train)

    # init the model
    class_output = get_class_output(y_train) # same if we use y_val
    model.init_model(class_output)

    # train
    history = model.train(x_train, y_train, x_val, y_val, model.optimiser, model.loss)
    model.display_result(history)

    print(model.model.summary())

    # one prediction with data from validation dataset
    i = 12
    image = x_val[i].reshape(28, 28)
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




