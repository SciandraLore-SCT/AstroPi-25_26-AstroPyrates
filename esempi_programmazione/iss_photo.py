# Import the Camera class from the picamera-zero module
from picamzero import Camera

# Create an instance of the Camera class
cam = Camera()

# Capture an image
cam.take_photo("image1.jpg")

#programma che sscatta una foto e la scarica come image1.jpg