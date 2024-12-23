import json
import os
import random

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

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

    display_dataset_shape(x_train, y_train, x_val, y_val)

    return x_train, y_train, x_val, y_val

def preprocess_data_cnn(x_train, y_train, x_val, y_val):
    height, width = get_image_size()

    x_train = x_train.reshape(len(x_train), height, width, 1)  # 1 canal, not RGB
    x_val = x_val.reshape(len(x_val), height, width, 1)

    x_train, y_train = reduce_dataset(x_train, y_train)
    x_train, y_train = augment_dataset(x_train, y_train)

    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255

    display_dataset_shape(x_train, y_train, x_val, y_val)

    return x_train, y_train, x_val, y_val

def reduce_dataset(x_train, y_train):
    x_train_sample = []
    y_train_sample = []
    num_image_per_class = 50
    class_output = get_class_output(y_train)
    # Select "num_image_per_class" random images for each class
    for i in range(class_output):
        indices = np.where(y_train == i)[0]
        selected_indices = np.random.choice(indices, num_image_per_class, replace=False)
        x_train_sample.append(x_train[selected_indices])
        y_train_sample.append(y_train[selected_indices])

    x_train_sample = np.concatenate(x_train_sample, axis=0)
    y_train_sample = np.concatenate(y_train_sample, axis=0)
    return x_train_sample, y_train_sample

def augment_dataset(x_train, y_train):
    x_dataset_shift = []
    y_dataset_shift = []
    x_dataset_zoom = []
    y_dataset_zoom = []

    zoom_factors = [0.25, 0.5, 0.75]
    j = 0
    for image, label in zip(x_train, y_train):
        # Shift
        # TODO: Add method to shift an image in 4 image corners

        # Random Shift
        for i in range(20):
            shifted_image = random_shift(image)
            x_dataset_shift.append(shifted_image)
            y_dataset_shift.append(label)

        # Zoom
        for zoom_factor in zoom_factors:
            zoomed_image = zoom(image, zoom_factor)
            x_dataset_zoom.append(zoomed_image)
            y_dataset_zoom.append(label)

        print(f"image {j} ...augmented")
        j += 1
    # Tranform the shift and zoom dataset in np.array
    x_dataset_shift = np.array(x_dataset_shift)
    y_dataset_shift = np.array(y_dataset_shift)
    x_dataset_zoom = np.array(x_dataset_zoom)
    y_dataset_zoom = np.array(y_dataset_zoom)
    # Concatenate all the dataset in one
    x_train = np.concatenate((x_train, x_dataset_shift, x_dataset_zoom), axis=0)
    y_train = np.concatenate((y_train, y_dataset_shift, y_dataset_zoom), axis=0)

    return x_train, y_train

# TODO: Add a method to add texture
#         image = tf.image.adjust_brightness()
#         image2 = tf.image.adjust_hue()
#         image3 = tf.image.adjust_gamma()
#         image4 = tf.image.adjust_contrast()
#         image5 = tf.image.adjust_saturation()
#         image6 = tf.image.adjust_brightness()
#         image7 = tf.image.adjust_jpeg_quality()

def random_shift(image):
    height, width, _ = image.shape

    zoom_factor = random.uniform(0.25, 0.75)
    zoom_height = int(height * zoom_factor)
    zoom_width = int(width * zoom_factor)
    # Get zoomed image to apply the  random shift
    zoomed_image = tf.image.resize(image, (zoom_height, zoom_width))

    # Get random x, y shift (1 for a minimal shift)
    offset_x = random.randint(1, width - zoom_width)
    offset_y = random.randint(1, height - zoom_height)

    # Create the new image with shift
    return tf.image.pad_to_bounding_box(zoomed_image, offset_y, offset_x, height, width)

def zoom(image, zoom):
    height, width, _ = image.shape

    zoom_height = int(height * zoom)
    zoom_width = int(width * zoom)
    zoomed_image = tf.image.resize(image, (zoom_height, zoom_width))
    # return image with the zoom
    return tf.image.resize_with_crop_or_pad(zoomed_image, height, width)


def display_dataset_shape(x_train, y_train, x_val, y_val):
    print('train samples', x_train.shape)
    print('validation samples', x_val.shape)
    print('train label samples', y_train.shape)
    print('validation label samples', y_val.shape)

def display_dataset(x_dataset, y_dataset):
    num_images = x_dataset.shape[0]
    batch_size = 100
    num_batches = num_images // batch_size

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        current_images = x_dataset[start_index:end_index]
        current_labels = y_dataset[start_index:end_index]

        fig, axes = plt.subplots(10, 10, figsize=(15, 15))
        for i, ax in enumerate(axes.flat):
            ax.imshow(current_images[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f"Label: {current_labels[i]}")

        plt.tight_layout()
        plt.show()
