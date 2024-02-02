import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np    
import math
from tkinter import font as tkFont

# Activation functions - change this according to your model!
# Sigmoid for hidden layers - returns a value between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Softmax used in the output layer for probabilities
def softmax(z):
    # Subtracting the max of z for numerical stability
    z_exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return z_exp / np.sum(z_exp, axis=0, keepdims=True)

# Set up the drawing canvas
class DrawingApp:
    def __init__(self, root):
        self.root = root

        # Load the model from a numpy file
        self.load_model("weights-1.npz")

        # Set up the canvas
        self.canvas = tk.Canvas(root, bg='black', width=280, height=280)
        self.canvas.pack(padx=10, pady=10)

        # Create an image to draw on
        self.image = Image.new("L", (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)

        # Set up mouse movement bindings to draw on the canvas nicely 
        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x, self.last_y = None, None
        
        self.canvas.bind("<ButtonRelease-1>", self.reset_mouse_pos)

        self.prob_frame = tk.Frame(root)
        self.prob_frame.pack(fill='both', expand=True)
        self.prob_labels = [tk.Label(self.prob_frame, text=f"Digit {i}: 0%") for i in range(10)]
        for label in self.prob_labels:
            label.pack(side='top')
    
    # Load the trained weights - change this according to your model!
    def load_model(self, path):
        weights = np.load(path)
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']

    # Reset the mouse position - keeps the drawing from connecting lines
    def reset_mouse_pos(self, event):
        self.last_x, self.last_y = None, None

    # Draw on the canvas
    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # Calculate the distance
            distance = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
            # Set a minimal distance threshold (for example, 2 pixels)
            if distance > 2:
                self.canvas.create_line((self.last_x, self.last_y, x, y), width=12, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
                self.draw.line((self.last_x, self.last_y, x, y), fill='white', width=12)
        self.last_x, self.last_y = x, y
        self.predict_digit()

    # Reset the canvas
    def reset(self, event=None):
        self.last_x, self.last_y = None, None
        self.canvas.delete("all")
        self.draw.rectangle(((0, 0, 280, 280)), fill='black')
        self.predict_digit()

    def predict_digit(self):
        # Preprocessing
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
        img = np.array(img)
        img = img.reshape(1, 28*28).T  # Reshape to match training input
        img = img / 255.0

        # Forward pass using the loaded model - change this according to your model!
        Z1 = np.dot(self.W1, img) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = softmax(Z2)

        # Display probabilities and bold the highest one
        probabilities = A2.flatten() * 100  # Convert to percentage
        max_prob_index = np.argmax(probabilities)  # Get the highest probability

        bold_font = tkFont.Font(weight="bold")
        regular_font = tkFont.Font(weight="normal")

        for i, prob in enumerate(probabilities):
            if i == max_prob_index:
                self.prob_labels[i].config(text=f"Digit {i}: {prob:.2f}%", font=bold_font)
            else:
                self.prob_labels[i].config(text=f"Digit {i}: {prob:.2f}%", font=regular_font)


root = tk.Tk()
app = DrawingApp(root)
button_frame = tk.Frame(root)
button_frame.pack(fill='both', expand=True)
clear_button = tk.Button(button_frame, text='Clear', command=app.reset)
clear_button.pack(side='right', fill='both', expand=True)
root.mainloop()
