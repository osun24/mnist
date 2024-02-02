import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from tkinter import font as tkFont

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    # Subtracting the max of z for numerical stability
    z_exp = np.exp(z - np.max(z, axis=0, keepdims=True))
    return z_exp / np.sum(z_exp, axis=0, keepdims=True)

# Set up the drawing canvas
class MNISTGallery:
    def __init__(self, root, train_images):
        self.root = root
        self.train_images = train_images
        self.current_index = 0

        # Load the model weights from numpy file
        self.load_model("weights-1.npz")

        # Set up the UI components: canvas, buttons, and labels
        self.canvas = tk.Canvas(root, width=280, height=280)
        self.canvas.pack()

        frame_buttons = tk.Frame(root)
        frame_buttons.pack(side=tk.BOTTOM, fill=tk.X)

        btn_prev = tk.Button(frame_buttons, text="<< Prev", command=self.show_prev_image)
        btn_prev.pack(side=tk.LEFT)

        btn_next = tk.Button(frame_buttons, text="Next >>", command=self.show_next_image)
        btn_next.pack(side=tk.RIGHT)
        self.prob_frame = tk.Frame(root)
        self.prob_frame.pack(fill='both', expand=True)
        self.prob_labels = [tk.Label(self.prob_frame, text=f"Digit {i}: 0%") for i in range(10)]
        for label in self.prob_labels:
            label.pack(side='top')

        self.update_canvas(self.train_images[0])

    # Load the trained weights
    def load_model(self, path):
        weights = np.load(path)
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']

    # Update the canvas with the new image and predict the digit
    def update_canvas(self, image):
        photo = ImageTk.PhotoImage(image=Image.fromarray(image).resize((280, 280)))
        self.canvas.create_image(140, 140, image=photo)
        self.canvas.image = photo  # Keep a reference!
        self.predict_digit(Image.fromarray(image))

    def show_next_image(self):
        self.current_index = (self.current_index + 1) % len(self.train_images)
        self.update_canvas(self.train_images[self.current_index])

    def show_prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.train_images)
        self.update_canvas(self.train_images[self.current_index])

    def predict_digit(self, image):
        # Preprocess the image: resize to 28x28 and convert to grayscale, then reshape to 784x1
        img = image.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
        img = np.array(img)
        img = img.reshape(1, 28*28).T  
        img = img / 255.0

        # Forward pass using the loaded model
        Z1 = np.dot(self.W1, img) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = softmax(Z2) # Softmax for probabilities

        # Display probabilities and bold the highest one
        probabilities = A2.flatten() * 100  # Convert to percentage
        max_prob_index = np.argmax(probabilities)  # Get the index of the highest probability

        # Show probabilities in the UI, bolding the highest one
        bold_font = tkFont.Font(weight="bold")
        regular_font = tkFont.Font(weight="normal")

        for i, prob in enumerate(probabilities):
            if i == max_prob_index:
                self.prob_labels[i].config(text=f"Digit {i}: {prob:.2f}%", font=bold_font)
            else:
                self.prob_labels[i].config(text=f"Digit {i}: {prob:.2f}%", font=regular_font)

# Load MNIST data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Create the main window
root = tk.Tk()
root.title("MNIST Gallery")

# Initialize and run the gallery application
app = MNISTGallery(root, train_images)
root.mainloop()