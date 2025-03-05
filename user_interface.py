import numpy as np
import cv2
import pickle 
import tkinter as tk
from tkinter import Canvas, Button
import os
from PIL import Image
from scipy.ndimage import gaussian_filter

import warnings
#suppress warnings
warnings.filterwarnings('ignore')


class DigitGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Generator")

        self.canvas = Canvas(root, width=850, height=270, bg="black")
        self.canvas.pack()

        # Add a red line at 100px
        self.canvas.create_line(135*2, 0, 135*2, 155*2, fill="red", width=2)
        self.canvas.create_line(290*2, 0, 290*2, 155*2, fill="red", width=2)


        self.label = tk.Label(root, text="Draw a digit, a math operation, and a digit:", fg="black")
        self.label.pack()

        self.save_button = Button(root, text="Save Expression", command=self.save_digit)
        self.save_button.pack()

        self.clear_button = Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

        self.image = np.zeros((270, 850), dtype=np.uint8)
        self.drawing = False
        self.last_x = 0
        self.last_y = 0

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw_digit(self, event):
        if self.drawing:
            x, y = event.x, event.y
            radius = 8
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white", outline="white", tags="user_drawing")

            self.image[y - radius:y + radius, x - radius:x + radius] = 255

    def stop_drawing(self, event):
        self.drawing = False

    def save_digit(self):
        # Save the image as a larger PNG file
        digit_filename = os.path.join("drawn_digit.png")
        image = Image.fromarray(self.image)
        resized_image = image.resize((850, 270), Image.Resampling.LANCZOS)  # Resize to original size
        resized_image.save(digit_filename)
        # print(f"Saved {digit_filename}")

        self.clear_canvas()

    def clear_canvas(self):
        # Delete only the user-drawn content (ovals) tagged as "user_drawing"
        self.canvas.delete("user_drawing")
        # Clear the image data for user-drawn content
        self.image.fill(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitGeneratorApp(root)
    root.mainloop()

with open('digit_training_4hidden_i6000_L25_a0.12.txt', 'rb') as d:
    digit_model = pickle.load(d)

weights_digit = digit_model[0]
bias_digit = digit_model[1]
n_hidden_digit = digit_model[2]

with open('symbols_training_3hidden_i2500_L25_a0.08.txt', 'rb') as s:
    symbols_model = pickle.load(s)

weights_symbol = symbols_model[0]
bias_symbol = symbols_model[1]
n_hidden_symbol = symbols_model[2]

image = Image.open('drawn_digit.png')
img = np.asarray(image)

digit_1 = img.T[:270]
digit_1 = digit_1.T
digit_1 = gaussian_filter(digit_1, sigma=3).astype(np.float32).astype(np.uint8)  # type: ignore
digit_1 = cv2.resize(digit_1, (28,28))
digit_1 = digit_1.reshape(1, 784)

symbol = img.T[270:580]
symbol = symbol.T
symbol = cv2.resize(symbol, (155, 135))
symbol = symbol.reshape(1, 20925)

digit_2 = img.T[580:]
digit_2 = digit_2.T
digit_2 = gaussian_filter(digit_2, sigma = 3).astype(np.float32).astype(np.uint8) # type: ignore
digit_2 = cv2.resize(digit_2, (28,28))
digit_2 = digit_2.reshape(1, 784)


def sigmoid(Z):
    # return np.maximum(Z, 0)
    return 1/(1 + np.exp(-Z))
def softmax(Z):
    # A = np.exp(Z) / sum(np.exp(Z))
    # return A
    return 1/(1 + np.exp(-Z))

def forward_prop(n_hidden, weights, bias, X):
    
    z_layer = []
    activations = []
    
    for i in range(n_hidden+1):
        z_layer.append('')
        activations.append('')
    
    z_layer[0] = weights[0].dot(X.T) + bias[0]
    activations[0] = sigmoid(z_layer[0])
    
    for j in range(n_hidden):
        z_layer[j + 1]  = weights[j + 1].dot(activations[j]) + bias[j + 1]
        activations[j + 1] = sigmoid(z_layer[j + 1])
    
    return z_layer, activations

def csv_to_image_d(csv_row):
    img = csv_row.reshape(28,28)
    image = np.zeros((28,28,1))
    image[:,:,0] = img
    image = cv2.resize(image, (560,560),20,20)   
    cv2.imshow("image",image)
    # wait untill window is closed
    cv2.waitKey(0)
   
# csv_to_image_d(digit_1)
   
# csv_to_image_d(digit_2)

z_layer_d1, activations_d1 = forward_prop(n_hidden_digit, weights_digit, bias_digit, digit_1)
z_layer_d2, activations_d2 = forward_prop(n_hidden_digit, weights_digit, bias_digit, digit_2)
z_layer_s1, activations_s1 = forward_prop(n_hidden_symbol, weights_symbol, bias_symbol, symbol)

first_digit = np.argmax(activations_d1[-1])
second_digit = np.argmax(activations_d2[-1])
final_symbol = np.argmax(activations_s1[-1])

symbols_list = ['+', '-', '*', '/']
operation = symbols_list[final_symbol]

if final_symbol == 0:
    result = first_digit + second_digit
elif final_symbol == 1:
    result = first_digit - second_digit
elif final_symbol == 2:
    result = first_digit * second_digit
elif final_symbol == 3:
        result = first_digit / second_digit


print(first_digit, operation, second_digit, '=', result)
