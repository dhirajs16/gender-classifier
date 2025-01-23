
# Gender Classification Web App

## Overview

This web application is built with Flask and utilizes a machine learning model powered by OpenCV to classify the gender of individuals based on uploaded images. The app is designed to process images efficiently by converting them into grayscale, cropping the face, and transforming them into eigen images before predicting the gender.

## Features

- **Image Upload**: Users can upload images directly through the web interface.
- **Face Detection**: The application automatically detects and crops faces from the uploaded images.
- **Image Processing**: Each image is converted to grayscale and transformed into eigen images for better classification accuracy.
- **Gender Prediction**: The application uses a pre-trained machine learning model to predict the gender of the detected face.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework for Python.
- **OpenCV**: An open-source computer vision library used for image processing.
- **NumPy**: A library for numerical computations in Python.
- **HTML/CSS**: For creating the web interface.

## Installation

### Prerequisites

Make sure you have Python 3.x installed on your machine. You will also need to install the required libraries.

### Steps to Set Up

1. Clone the repository:
   ```
   git clone https://github.com/dhirajs16/gender-classifier.git
   cd gender-classification-flask
   
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
   - On Windows:
     ```
     venv\Scripts\activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Run the Flask application:
   ```
   flask run
   ```

6. Open your web browser and go to `http://127.0.0.1:5000` to access the application.

## Usage

1. Navigate to the home page of the application.
2. Click on the "Upload Image" button to select an image file from your device.
3. After uploading, the application will process the image and display the predicted gender.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes.
