from picamzero import Camera
from time import sleep
import cv2
import numpy as np
import math


class ISSSpeedCalculator:

    def __init__(self):
        self.file_risultati = "risultati.txt"
        self.camera = Camera()
        self.TIME_INTERVAL = 10          # secondi
        self.CORRECTION_FACTOR = 1.8     # correzione geometrica dell'angolo

    def take_picture(self, n):
        for cont in range(n):
            self.camera.take_photo(f"image{cont}.jpg")
            sleep(self.TIME_INTERVAL)

    def distanza_tra_immagini(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path, 0)
        img2 = cv2.imread(img2_path, 0)

        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des1, des2)

        if len(matches) < 10:
            return 0

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # RANSAC: elimina i match sbagliati
        M, mask = cv2.estimateAffinePartial2D(
            pts1, pts2, method=cv2.RANSAC#rimuove outlier
            #Un outlier è un dato anomalo, cioè un valore che non segue il comportamento generale degli altri dati.
        )

        if M is None:
            return 0

        dx = M[0, 2]
        dy = M[1, 2]

        return np.hypot(dx, dy)

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
        n = 10
        self.take_picture(n)

        images = [f"image{i}.jpg" for i in range(n)]
        velocita_list = []

        for k in range(n - 1):
            pixel_shift = self.distanza_tra_immagini(
                images[k], images[k + 1]
            )

            distanza_metri = self.pixel_to_meters(
                pixel_shift, image_width=4056
            )

            velocita = distanza_metri / self.TIME_INTERVAL

            # filtro fisico
            if velocita < 500 or velocita > 10000:
                continue

            velocita_list.append(velocita)

            self.scrivi_risultatiFinale(
                f"Immagini {k}-{k+1}: "
                f"{velocita:.2f} m/s ({velocita*3.6:.0f} km/h)"
            )

        if len(velocita_list) > 0:
            velocita_media = np.mean(velocita_list)

            self.scrivi_risultatiFinale(
                f"\nVelocità media stimata ISS: "
                f"{velocita_media:.2f} m/s "
                f"({velocita_media*3.6:.0f} km/h, "
                f"{velocita_media/1000:.2f} km/s)"
            )



def main():
    calculator = ISSSpeedCalculator()
    calculator.esegui()


if __name__ == "__main__":
    main()
