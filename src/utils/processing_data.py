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


def preprocess_data(x_train, y_train, x_val, y_val, model_name, augmentation=True):
    height, width = get_image_size()

    if model_name == "MLP":
        x_train = x_train.reshape(len(x_train), height * width)
        x_val = x_val.reshape(len(x_val), height * width)
    elif model_name == "CNN":
        x_train = x_train.reshape(len(x_train), height, width, 1)  # 1 channel for grayscale
        x_val = x_val.reshape(len(x_val), height, width, 1)
    else:
        raise ValueError(f"Invalid model_type '{model_name}'. Choose 'mlp' or 'cnn'.")

    # Uncomment if dataset reduction is needed
    # x_train, y_train = reduce_dataset(x_train, y_train)

    if augmentation:
        x_train, y_train = augment_dataset(x_train, y_train)
        x_val, y_val = augment_dataset(x_val, y_val)

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
        j += 1
        # Corner Shift
        x_dataset_shift.append(corner_shift(image, "top_left"))
        y_dataset_shift.append(label)
        x_dataset_shift.append(corner_shift(image, "top_right"))
        y_dataset_shift.append(label)
        x_dataset_shift.append(corner_shift(image, "bottom_left"))
        y_dataset_shift.append(label)
        x_dataset_shift.append(corner_shift(image, "bottom_right"))
        y_dataset_shift.append(label)

        print(f"image {j} ...corner shift applied")

        # Random Shift
        for i in range(20):
            shifted_image = random_shift(image)
            x_dataset_shift.append(shifted_image)
            y_dataset_shift.append(label)

        print(f"image {j} ...random shift applied")

        # Zoom
        for zoom_factor in zoom_factors:
            zoomed_image = zoom(image, zoom_factor)
            x_dataset_zoom.append(zoomed_image)
            y_dataset_zoom.append(label)

        print(f"image {j} ...zoom applied")

    # Tranform the shift and zoom dataset in np.array
    x_dataset_shift = np.array(x_dataset_shift)
    y_dataset_shift = np.array(y_dataset_shift)
    x_dataset_zoom = np.array(x_dataset_zoom)
    y_dataset_zoom = np.array(y_dataset_zoom)
    # Concatenate all the dataset in one
    x_train = np.concatenate((x_train, x_dataset_shift, x_dataset_zoom), axis=0)
    y_train = np.concatenate((y_train, y_dataset_shift, y_dataset_zoom), axis=0)

    x_train, y_train = rotate_images(x_train, y_train)

    return x_train, y_train


def corner_shift(image, corner):
    height, width, _ = image.shape

    zoom_factor = 0.5
    zoom_height = int(height * zoom_factor)
    zoom_width = int(width * zoom_factor)
    # Get zoomed image to apply the  random shift
    zoomed_image = tf.image.resize(image, (zoom_height, zoom_width))

    if corner == 'top_left':
        offset_x, offset_y = 0, 0
    elif corner == 'top_right':
        offset_x, offset_y = width - zoom_width, 0
    elif corner == 'bottom_left':
        offset_x, offset_y = 0, height - zoom_height
    elif corner == 'bottom_right':
        offset_x, offset_y = width - zoom_width, height - zoom_height

    # Create the new image with shift
    shifted_image = tf.image.pad_to_bounding_box(zoomed_image, offset_y, offset_x, height, width)

    return shifted_image


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

def rotate_images(x_train, y_train, min_angle=-30, max_angle=30):
    x_rotated = []
    y_rotated = []

    j = 0
    for image, label in zip(x_train, y_train):
        for i in range(10):
            # Generate a random angle between min_angle and max_angle
            angle = random.uniform(min_angle, max_angle)

            rotated_image = rotate_image(image, angle)

            x_rotated.append(rotated_image)
            y_rotated.append(label)

        j+=1
        print(f"image {j} ...rotation applied")

    # Convert the rotated datasets to numpy arrays
    x_rotated = np.array(x_rotated)
    y_rotated = np.array(y_rotated)

    x_train = np.concatenate((x_train, x_rotated), axis=0)
    y_train = np.concatenate((y_train, y_rotated), axis=0)

    return x_train, y_train


def rotate_image(image, angle):
    # Convert angle from degrees to a fraction of a full rotation (1.0 represents 360 degrees)
    rotation_factor = angle / 360.0

    # Define the RandomRotation layer
    random_rotation_layer = tf.keras.layers.RandomRotation(factor=(rotation_factor, rotation_factor))

    # Apply the rotation
    rotated_image = random_rotation_layer(image)
    return rotated_image


# TODO: Add a method to add texture, can be upgrade the dataset ?
#         image = tf.image.adjust_brightness()
#         image2 = tf.image.adjust_hue()
#         image3 = tf.image.adjust_gamma()
#         image4 = tf.image.adjust_contrast()
#         image5 = tf.image.adjust_saturation()
#         image6 = tf.image.adjust_brightness()
#         image7 = tf.image.adjust_jpeg_quality()


def display_dataset_shape(x_train, y_train, x_val, y_val):
    print('train samples', x_train.shape)
    print('validation samples', x_val.shape)
    print('train label samples', y_train.shape)
    print('validation label samples', y_val.shape)

def display_dataset(x_dataset, y_dataset):
    num_images = x_dataset.shape[0]
    x = 15
    y = 15
    batch_size = x * y
    num_batches = (num_images + batch_size - 1) // batch_size
    if num_batches == 0 and num_images > 0:
        num_batches = 1

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min(start_index + batch_size, num_images)
        print(f"Batch {batch_index + 1}/{num_batches}: {end_index}")

        current_images = x_dataset[start_index:end_index]
        current_labels = y_dataset[start_index:end_index]

        num_current_images = current_images.shape[0]

        fig, axes = plt.subplots(x, y, figsize=(20, 20))
        axes = axes.flat

        for i, ax in enumerate(axes):
            if i < num_current_images:  # Assurez-vous que i est dans les limites
                ax.imshow(current_images[i], cmap='gray')
                ax.axis('off')
                ax.set_title(f"Label: {current_labels[i]}")
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.show()
