import os
import tkinter as tk
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt

class DrawDigitUI:
    def __init__(self):
        self.root = tk.Tk()
        self.canvas = tk.Canvas()
        self.prob_digit = {"text": ""}

    def clear(self):
        self.canvas.delete("all")
        self.prob_digit['text'] = ""

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill='black', width=5)

    def predict_digit(self, model):
        self.canvas.update()
        self.canvas.postscript(colormode='mono', file='tmp.ps')
        # Get image
        img = Image.open('tmp.ps')
        img = img.convert('L')
        img = ImageOps.invert(img)
        img = img.resize((model.height, model.width))

        img = np.array(img)
        # Preprocess image
        if model.name == "CNN":
            img = img.reshape(1, model.height, model.width, 1)
        img = img.astype('float32') / 255
        # Display image
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
        plt.show(block=False)
        # Predict digit
        result = model.predict(img)
        digit, prob = model.predict_best_class(result)

        print(f"image: {img.shape}")
        print(result[0])
        print(f"{digit} -> {prob}%")

        self.prob_digit['text'] = ""
        probabilities = result[0]
        for i in range(len(probabilities)):
            self.prob_digit['text'] += f"{str(i).ljust(2)}: {probabilities[i] * 100:6.2f}%\n"

        os.remove(os.path.join(os.getcwd(), "tmp.ps"))

    def build_ui(self, model):
        self.canvas = tk.Canvas(self.root, width=200, height=200, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        self.canvas.bind("<ButtonRelease-1>", lambda event: self.predict_digit(model))

        button_clear = tk.Button(self.root, text="Clear", command=self.clear, font=("Arial", 20))
        button_clear.pack()

        self.prob_digit = tk.Label(self.root, text="", font=("Arial", 20), justify="left")
        self.prob_digit.pack()


    def run(self):
        self.root.mainloop()
