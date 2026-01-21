from picamzero import Camera
from time import sleep
import cv2
import numpy as np
import math


class ISSSpeedCalculator:

    def __init__(self):
        self.file_risultati = "risultati.txt"
        self.camera = Camera()

    def take_picture(self, n):
        for cont in range(n):
            self.camera.take_photo(f"image{cont}.jpg")
            sleep(10)

    def distanza_tra_immagini(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path, 0)
        img2 = cv2.imread(img2_path, 0)

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) == 0:
            return 0

        distances = []
        for m in matches:
            p1 = kp1[m.queryIdx].pt
            p2 = kp2[m.trainIdx].pt
            distances.append(
                np.hypot(p1[0] - p2[0], p1[1] - p2[1])
            )

        return np.mean(distances)

    def pixel_to_meters(self, pixel_distance, image_width):
        FOV = math.radians(62.2)
        ISS_HEIGHT = 408000  # metri

        width_earth = 2 * ISS_HEIGHT * math.tan(FOV / 2)
        meters_per_pixel = width_earth / image_width

        return pixel_distance * meters_per_pixel

    def scrivi_risultatiFinale(self, messaggio):
        with open(self.file_risultati, 'a') as f:
            f.write(messaggio + "\n")

    def esegui(self):
        n = 4
        self.take_picture(n)

        images = [f"image{i}.jpg" for i in range(n)]

        for k in range(n - 1):
            pixel_shift = self.distanza_tra_immagini(images[k], images[k + 1])
            distanza_metri = self.pixel_to_meters(pixel_shift, image_width=4056)

            self.scrivi_risultatiFinale(
                f"Immagini {k}-{k+1}: {distanza_metri:.2f} metri"
            )


def main():
    calculator = ISSSpeedCalculator()
    calculator.esegui()


if __name__ == "__main__":
    main()
