import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np    
import math

# Set up the drawing canvas
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, bg='black', width=280, height=280)
        self.canvas.pack(padx=10, pady=10)

        # Create an image to draw on
        self.image = Image.new("L", (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)

        # Set up mouse movement bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x, self.last_y = None, None
        
        self.canvas.bind("<ButtonRelease-1>", self.reset_mouse_pos)
        
    def reset_mouse_pos(self, event):
        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # Calculate the distance
            distance = math.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
            # Set a minimal distance threshold (for example, 2 pixels)
            if distance > 2:
                self.canvas.create_line((self.last_x, self.last_y, x, y), width=8, fill='white', capstyle=tk.ROUND, smooth=tk.TRUE, splinesteps=36)
                self.draw.line((self.last_x, self.last_y, x, y), fill='white', width=8)
        self.last_x, self.last_y = x, y

    def reset(self, event=None):
        self.last_x, self.last_y = None, None
        self.canvas.delete("all")
        self.draw.rectangle(((0, 0, 280, 280)), fill='black')

    '''def predict_digit(self):
        # Preprocess the image
        img = self.image.resize((28, 28), Image.ANTIALIAS).convert('L')
        img = np.array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img / 255.0

        # Load model and predict
        model = tf.keras.models.load_model('mnist_model.h5')  # Replace with your model path
        result = model.predict(img)
        digit = np.argmax(result)
        print(f"Predicted Digit: {digit}")'''

root = tk.Tk()
app = DrawingApp(root)
button_frame = tk.Frame(root)
button_frame.pack(fill='both', expand=True)
#predict_button = tk.Button(button_frame, text='Predict Digit', command=app.predict_digit)
#predict_button.pack(side='left', fill='both', expand=True)
clear_button = tk.Button(button_frame, text='Clear', command=app.reset)
clear_button.pack(side='right', fill='both', expand=True)
root.mainloop()
