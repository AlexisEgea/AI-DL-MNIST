import os
import re

import tkinter as tk
from PIL import Image, ImageOps

import numpy as np
from matplotlib import pyplot as plt
from utils.processing_data import get_image_size


class CreateDatasetUI:
    def __init__(self, n_class, number_of_draw):
        self.root = tk.Tk()
        self.canvas = tk.Canvas()

        self.digit_goal = {"text": ""}

        self.height, self.width = get_image_size()

        self.class_output = self.get_number_to_draw(n_class)
        self.class_number = self.class_output.pop(0)
        self.number_of_draw = number_of_draw
        self.number_of_draw_counter = 0

        self.x_data = []
        self.y_label = []

        self.dataset_name = None
        self.path_dataset = None
        self.create_dataset()


    def create_dataset(self):
        self.path_dataset = os.path.join(os.getcwd(), "..", "data")
        self.dataset_name = f"mnist_{self.get_number_dataset(self.path_dataset)}"

        self.path_dataset += f"/{self.dataset_name}"

        os.makedirs(self.path_dataset)
        os.makedirs(os.path.join(self.path_dataset, "X"))
        os.makedirs(os.path.join(self.path_dataset, "y"))

        print(f"'{self.dataset_name}' created.")

    # Return the name of the dataset "mnist_n+1" available
    def get_number_dataset(self, base_path):
        directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        numbers = []
        for directory in directories:
            match = re.search(r"mnist_(\d+)", directory)
            if match:
                numbers.append(int(match.group(1)))

        if len(numbers) == 0:
            return 0
        return max(numbers) + 1

    # Return a table with all the class to draw
    def get_number_to_draw(self, number):
        array = []
        for i in range(number):
            array.append(i)
        return array


    def update_digit_to_draw(self):
        self.digit_goal['text'] = f"Draw {self.class_number} - Goal {self.number_of_draw_counter}/{self.number_of_draw}"


    def clear(self):
        self.canvas.delete("all")


    def save(self):
        self.canvas.update()
        self.canvas.postscript(colormode='mono', file='tmp.ps')

        # Convert to an image
        img = Image.open('tmp.ps')
        img = img.convert('L')
        img = ImageOps.invert(img)
        img = img.resize((self.height, self.width))

        # Save the image with a unique name
        filename = f"digit_{self.class_number}_{self.number_of_draw_counter}.png"
        img.save(os.path.join(self.path_dataset, "X", filename))

        self.x_data.append(img)
        self.y_label.append(self.class_number)
        print(self.y_label)
        print(f"Image saved as {filename}")

        self.number_of_draw_counter+=1
        if self.number_of_draw_counter == self.number_of_draw:
            if len(self.class_output) == 0:
              self.x_data = np.array(self.x_data)
              np.save(os.path.join(self.path_dataset, "X", "x_data.npy"), self.x_data)
              self.y_label = np.array(self.y_label)
              np.save(os.path.join(self.path_dataset, "y", "y_label.npy"), self.y_label)
              exit()
            self.number_of_draw_counter = 0
            self.class_number = self.class_output.pop(0)
        self.update_digit_to_draw()

        self.clear()
        os.remove(os.path.join(os.getcwd(), "tmp.ps"))


    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='black', width=5)


    def display_result(self):
        self.canvas.update()
        self.canvas.postscript(colormode='mono', file='tmp.ps')
        img = Image.open('tmp.ps')
        img = img.convert('L')
        img = ImageOps.invert(img)
        img = img.resize((28, 28))

        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show(block=False)

        os.remove(os.path.join(os.getcwd(), "tmp.ps"))


    def build_ui(self):
        self.update_digit_to_draw()
        self.digit_goal = tk.Label(self.root, text=self.digit_goal['text'], font=("Arial", 20), justify="left")
        self.digit_goal.pack()

        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        self.canvas.bind("<ButtonRelease-1>", lambda event: self.display_result())

        button_save = tk.Button(self.root, text="Save", command=self.save, font=("Arial", 20))
        button_save.pack()

        button_clear = tk.Button(self.root, text="Clear", command=self.clear, font=("Arial", 20))
        button_clear.pack()


    def run(self):
        self.root.mainloop()

if __name__ == '__main__':

    ui = CreateDatasetUI(10, 10)
    ui.build_ui()
    ui.run()
