import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

from src.ui.draw_digit_ui import DrawDigitUI
from src.utils.data import preprocess_data, get_class_output
from model.MLP import MLP

if __name__ == '__main__':
    # load data
    mnist = tf.keras.datasets.mnist

    # the data, split between train and validation sets
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    # pre process data
    x_train, y_train, x_val, y_val = preprocess_data(x_train, y_train, x_val, y_val)

    # init the model
    mlp = MLP()
    class_output = get_class_output(y_train) # same if we use y_val
    mlp.init_model(class_output)

    # train
    history = mlp.train(x_train, y_train, x_val, y_val, mlp.optimiser, mlp.loss)
    mlp.display_result(history)

    # one prediction with data from validation dataset
    image = x_val[6].reshape(28, 28)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    vect_prob = mlp.predict(x_val[12])
    result, prob = mlp.predict_best_class(vect_prob)
    print('I am confident about {:.2f}% that this image corresponds to digit {}'.format(prob, result))

    # predict by creating the data yourself by drawing the digit
    ui = DrawDigitUI()
    ui.build_ui(mlp)
    ui.run()

    # save model if it doesn't exist
    mlp.save_if_model_doesnt_exist()
    # or save the model even if it exists
    # mlp.save_model()




