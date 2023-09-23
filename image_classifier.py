import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the trained model
model = tf.keras.models.load_model('C:/Users/Raghav/OneDrive/Desktop/Python/datasets_dogs_cats/model.h5')

# Create Tkinter window
root = tk.Tk()
root.title("Cat or Dog Predictor")

# Function to open a file dialog and load an image
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        load_and_predict_image(file_path)

# Function to preprocess the image and make a prediction
def load_and_predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((150, 150))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array.reshape(1, 150, 150, 3))

    prediction = model.predict(image_array)

    if prediction[0][0] >= 0.5:
        result_label.config(text="It's a dog!")
    else:
        result_label.config(text="It's a cat!")

# Create UI components
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack()

# Run the Tkinter event loop
root.mainloop()