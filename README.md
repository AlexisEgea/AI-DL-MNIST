# AI-DrawPredict-MNIST

## Description

This project uses the technologies of Neural Networks especially Convolutional Neural Network (CNNs) 
in order to go beyond basic training and prediction on standard datasets by enabling freehand number drawing and
obtaining accurate predictions, along with the confidence level for each predicted digit (referred to as a class).

The project consists of four executable components:
1. An executable for creating a custom dataset.
2. An executable for training and use the model on the custom dataset.
3. An executable for training and use the model on the MNIST dataset.
4. An executable for drawing numbers and using a saved model for prediction.

---

### Key Concepts of the CNN in This Project

Convolutional Neural Networks (CNNs) are a powerful class of deep neural networks used for image recognition and processing tasks. 
They excel in analyzing visual data by leveraging their ability to learn and detect spatial hierarchies within images, such as edges, textures, and complex shapes, at multiple levels of granularity.

The CNN in this project is tailored to recognize and classify handwritten digits with high accuracy. 
Its architecture, described below, demonstrates the principles of convolutional networks 
and the importance of feature extraction, dimensionality reduction, and decision-making layers.

#### Model Summary 

``` python
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (32, 24, 24, 6)             │             156 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (32, 24, 24, 6)             │              24 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (32, 12, 12, 6)             │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (32, 8, 8, 6)               │             906 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (32, 8, 8, 6)               │              24 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (32, 4, 4, 6)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (32, 96)                    │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (32, 128)                   │          12,416 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (32, 10)                    │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 14,818 (57.89 KB)
 Trainable params: 14,792 (57.78 KB)
 Non-trainable params: 24 (96.00 B)
 Optimizer params: 2 (12.00 B)
```

The input images are of size `28x28` and are preprocessed to be compatible with the CNN.

1. **Convolutional Layers**:
   - Two convolutional layers are used to extract spatial features from input images. 
   Each layer applies 6 filters (kernels) of size 5x5, which scan the input data for patterns like edges and curves. 
   - The `ReLU` activation function introduces non-linearity, allowing the network to model complex patterns.

2. **Batch Normalization**:
   - Batch normalization is applied after each convolutional layer to stabilize and accelerate the training process. 
   It normalizes the input of each layer, reducing internal shifts and allowing the model to converge faster.

3. **Max Pooling Layers**:
   - Following each convolutional layer, a max pooling operation is performed with a pool size of 2x2 and stride of 2. 
   This reduces the spatial dimensions of the feature maps, retaining the most salient features while improving computational efficiency and mitigating overfitting.

4. **Flatten Layer**:
   - The 2D feature maps output by the pooling layers are flattened into a 1D vector to prepare them for the fully connected layers.

5. **Fully Connected Layers**:
   - A dense layer with 128 neurons and `ReLU` activation further processes the features extracted by the convolutional layers.
   - The final dense layer outputs predictions for each class (digits 0-9) using a `softmax` activation function. 
   This ensures the output is a probability distribution across the classes, enabling the network to determine the most likely digit.

#### Training Methodology:
- The network uses `categorical cross-entropy` as the loss function to measure the error between predicted probabilities and actual labels. 
- The optimizer used is `SGD`, which iteratively adjusts the network weights based on gradient descent, ensuring convergence towards minimizing the error function efficiently.
- The inclusion of batch normalization and pooling layers ensures robustness and efficiency during training, even with high-dimensional image data.

---

## Features

- Create custom datasets.
- Preprocess data effectively.
- Utilize CNNs for learning and predictions.
- Train and predict on either the custom dataset or the MNIST dataset.
- Test the model on the validation set.
- Draw a digit and observe real-time predictions with confidence scores.

---

## Requirements

- Python 3+
- Pip (Python package manager)

---

## Execution

To get the project running on your machine, follow these steps:

### General Steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/AlexisEgea/AI-DrawPredict-MNIST.git
   ```

2. Install the required dependencies:
   - Use the provided `installation-requirements.sh` script to set up a virtual environment (`venv`) and install the dependencies from `requirement.txt` file.

3. Run the project:
   - Use the `Draw-Predict.sh` script to execute the project.
   - Follow the project instructions and enjoy exploring its features!

---

### On Ubuntu:

--- 

#### To execute a `.sh` script:

1. Open a terminal.
2. Make the script executable:
   ```sh
   chmod +x script_name.sh
   ```
3. Run the script:
   ```sh
   ./script_name.sh
   ```
--- 

#### Commands to set up and run the project:

In a terminal, follow these steps:

1. Clone the project to your machine:
   ```sh
   git clone https://github.com/AlexisEgea/AI-DrawPredict-MNIST.git
   ```

2. Install prerequisites:
   ```sh
   chmod +x installation-requirements.sh
   ./installation-requirements.sh
   ```

3. Run the project:
   ```sh
   chmod +x Draw-Predict.sh
   ./launcher.sh
   ```

---

### On Windows:

1. Double-click on the `installation-requirements.sh` script:
   ```sh
   installation-requirements.sh
   ```
3. Double-click on the `Draw-Predict.sh` script:
   ```sh
   launcher.sh
   ```

Alternatively, you can use an IDE (e.g., PyCharm, VS Code) to create a `venv` environment manually and execute the relevant Python files.

---

### Note

This project has been tested on both Linux and Windows (using Git Bash). If you encounter any issues running the scripts, consider using an IDE or Python environment manager.

---

## Contact Information

For inquiries or feedback, feel free to contact me at [alexisegea@outlook.com](mailto:alexisegea@outlook.com).

---

## Copyright

© 2024 Alexis EGEA. All Rights Reserved.

