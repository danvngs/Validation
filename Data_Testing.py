import numpy as np
import cv2
import csv
import os
import h5py
import keras
from keras.models import load_model
from datetime import datetime as dt

path_to_images = '/Users/danvn/Desktop/Validation_Data/IMGs'

# f = open("Validation_CSV.csv", "a", newline="")
path_to_csv = '/Users/danvn/Desktop/Validation_Data/Validation_CSV.csv'

# Code for creating a file folder storing the images and CSV file.

model = load_model('Final_Model_Keras.keras')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = image[:, :, :] # Crops the image to cut out unnecessary training data. (Sky, trees, etc)
    # This one is not cutting out anything as the first metric is ":"
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV) # Recolors RGB pictures to YUV
    image = cv2.resize(image, (200, 66))  # Resize the image as per model input size
    img = cv2.GaussianBlur(image, (3, 3), 0) # Puts a blur on the image
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Iterate through images in the folder
with open(path_to_csv, 'w', newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Image', 'Predicted_Class'])  # Write header

    for img_file in os.listdir(path_to_images):
        img_path = os.path.join(path_to_images, img_file)
        img_array = preprocess_image(img_path)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        # Write to CSV
        csv_writer.writerow([img_file, predicted_class])