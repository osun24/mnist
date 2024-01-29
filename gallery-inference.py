import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

class MNISTGallery:
    def __init__(self, root, train_images):
        self.root = root
        self.train_images = train_images
        self.current_index = 0

        # Set up the UI components
        self.canvas = tk.Canvas(root, width=280, height=280)
        self.canvas.pack()

        frame_buttons = tk.Frame(root)
        frame_buttons.pack(side=tk.BOTTOM, fill=tk.X)

        btn_prev = tk.Button(frame_buttons, text="<< Prev", command=self.show_prev_image)
        btn_prev.pack(side=tk.LEFT)

        btn_next = tk.Button(frame_buttons, text="Next >>", command=self.show_next_image)
        btn_next.pack(side=tk.RIGHT)

        # Uncomment to add a prediction button
        # btn_predict = tk.Button(frame_buttons, text="Predict Digit", command=self.predict_digit)
        # btn_predict.pack(side=tk.TOP)

        self.update_canvas(self.train_images[0])

    def update_canvas(self, image):
        photo = ImageTk.PhotoImage(image=Image.fromarray(image).resize((280, 280)))
        self.canvas.create_image(140, 140, image=photo)
        self.canvas.image = photo  # Keep a reference!

    def show_next_image(self):
        self.current_index = (self.current_index + 1) % len(self.train_images)
        self.update_canvas(self.train_images[self.current_index])

    def show_prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.train_images)
        self.update_canvas(self.train_images[self.current_index])

    # Uncomment to add the prediction function
    # def predict_digit(self):
    #     model = tf.keras.models.load_model('mnist_model.h5')  # Path to your model
    #     image = self.train_images[self.current_index].reshape(1, 28, 28, 1) / 255.0
    #     prediction = model.predict(image)
    #     predicted_digit = np.argmax(prediction)
    #     print(f"Predicted Digit: {predicted_digit}")

# Load MNIST data
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Create the main window
root = tk.Tk()
root.title("MNIST Gallery")

# Initialize and run the gallery application
app = MNISTGallery(root, train_images)
root.mainloop()