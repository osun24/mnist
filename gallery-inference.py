import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

# Load MNIST data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Create the main window
root = tk.Tk()
root.title("MNIST Gallery")

# Frame for navigation buttons
frame_buttons = tk.Frame(root)
frame_buttons.pack(side=tk.BOTTOM, fill=tk.X)

# Canvas to display images
canvas = tk.Canvas(root, width=280, height=280)
canvas.pack()

# Function to update canvas with a new image
def update_canvas(image):
    photo = ImageTk.PhotoImage(image=Image.fromarray(image).resize((280, 280)))
    canvas.create_image(140, 140, image=photo)
    canvas.image = photo  # Keep a reference!

# Index to keep track of which image is being displayed
current_index = 0

# Function to show next image
def show_next_image():
    global current_index
    current_index = (current_index + 1) % len(train_images)
    update_canvas(train_images[current_index])

# Function to show previous image
def show_prev_image():
    global current_index
    current_index = (current_index - 1) % len(train_images)
    update_canvas(train_images[current_index])

# Buttons for navigation
btn_prev = tk.Button(frame_buttons, text="<< Prev", command=show_prev_image)
btn_prev.pack(side=tk.LEFT)
btn_next = tk.Button(frame_buttons, text="Next >>", command=show_next_image)
btn_next.pack(side=tk.RIGHT)

# Initialize with the first image
update_canvas(train_images[0])

# Start the GUI
root.mainloop()

# Function to predict the digit


# Add a prediction button
# btn_predict = tk.Button(frame_buttons, text="Predict Digit", command=predict_digit)
# btn_predict.pack(side=tk.TOP)
