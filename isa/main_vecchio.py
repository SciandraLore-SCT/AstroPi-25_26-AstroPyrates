from picamzero import Camera
from time import sleep
import cv2
import numpy as np
import math


class ISSSpeedCalculator:

    def __init__(self):
        self.file_risultati = "risultati.txt"

    def take_picture(self):
        cam = Camera()
        for cont in range(2):
            cam.take_photo(f"image{cont}.jpg")
            sleep(10)

    def distanza_tra_immagini(self, images):
        img1 = cv2.imread(images[0], 0)
        img2 = cv2.imread(images[1], 0)

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        distances = []
        for m in matches:
            p1 = kp1[m.queryIdx].pt
            p2 = kp2[m.trainIdx].pt
            distances.append(
                np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            )

        return np.mean(distances)

    def pixel_to_meters(self, pixel_distance, image_width):
        FOV = math.radians(62.2)
        ISS_HEIGHT = 408000  # metri

        width_earth = 2 * ISS_HEIGHT * math.tan(FOV / 2)
        meters_per_pixel = width_earth / image_width

        return pixel_distance * meters_per_pixel

    def scrivi_risultatiFinale(self, messaggio):
        with open(self.file_risultati, 'w') as f:
            f.write(str(messaggio))

    def esegui(self):
        self.take_picture()

        images = ["image0.jpg", "image1.jpg"]

        pixel_shift = self.distanza_tra_immagini(images)
        distanza_metri = self.pixel_to_meters(pixel_shift, image_width=4056)

        self.scrivi_risultatiFinale(
            f"Distanza stimata: {distanza_metri:.2f} metri"
        )


def main():
    calculator = ISSSpeedCalculator()
    calculator.esegui()


if __name__ == "__main__":
    main()
