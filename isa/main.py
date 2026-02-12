from picamzero import Camera
from sense_hat import SenseHat
from time import sleep
import cv2
import numpy as np
import math
from datetime import datetime


class ISSSpeedCalculator:

    def __init__(self):
        self.camera = Camera()
        self.sense = SenseHat()

        # Parametri temporali
        self.TIME_INTERVAL = 10        # secondi
        self.DURATA_SEC = 400
        self.N = int(self.DURATA_SEC / self.TIME_INTERVAL)

        # File risultati
        self.file_risultati = "result.txt"

        # Costanti fisiche
        self.SPEED_ISS_REALE = 7665     # m/s
        self.RAGGIO_TERRA = 6371000     # m
        self.ALTEZZA_ISS = 408000       # m

        # Camera (AstroPi)
        self.FOV_X = math.radians(62.2)  # FOV orizzontale (Pi Camera)

        # Stabilità
        self.MAX_OMEGA = 0.01           # rad/s


    # -------------------------------
    # GIROSCOPIO (SOLO FILTRO)
    # -------------------------------
    def iss_stabile(self):
        gyro = self.sense.get_gyroscope_raw()
        omega = math.sqrt(
            gyro['x']**2 +
            gyro['y']**2 +
            gyro['z']**2
        )
        return omega < self.MAX_OMEGA


    # -------------------------------
    # FOTO
    # -------------------------------
    def take_picture(self, name):
        self.camera.take_photo(name)


    # -------------------------------
    # ORB + RANSAC
    # -------------------------------
    def pixel_shift(self, img1_path, img2_path):

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return None, None

        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des1, des2)

        if len(matches) < 10:
            return None, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Spostamento medio (robusto)
        shift = np.median(pts2 - pts1, axis=0)
        pixel_distance = np.linalg.norm(shift)

        h, w = img1.shape
        return pixel_distance, w


    # -------------------------------
    # PIXEL → METRI
    # -------------------------------
    def pixel_to_meters(self, pixel_distance, image_width):

        # Larghezza reale del terreno inquadrato
       
        width_ground = 2 * self.ALTEZZA_ISS * math.tan(self.FOV_X / 2)


        meters_per_pixel = width_ground / image_width
        return pixel_distance * meters_per_pixel


    # -------------------------------
    # MAIN
    # -------------------------------
    def esegui(self):

        velocita = []
        prev_img = None

        for i in range(self.N):

            if not self.iss_stabile():
                sleep(self.TIME_INTERVAL)
                continue

            img_name = f"image_{i}.jpg"
            self.take_picture(img_name)

            if prev_img is not None:

                pixel_dist, img_width = self.pixel_shift(prev_img, img_name)

                if pixel_dist is not None:
                    distanza_m = self.pixel_to_meters(pixel_dist, img_width)
                    v = distanza_m / self.TIME_INTERVAL

                    if 500 < v < 10000:
                        velocita.append(v)

            prev_img = img_name
            sleep(self.TIME_INTERVAL)

        if velocita:
            v_media = np.mean(velocita)
            v_km=v_media/1000
            errore = abs(v_media - self.SPEED_ISS_REALE) / self.SPEED_ISS_REALE * 100

            with open(self.file_risultati, "a") as f:
                f.write(
                    f"{v_km:.2f} km/s\n"
                    
                )


def main():
    calc = ISSSpeedCalculator()
    calc.esegui()


if __name__ == "__main__":
    main()
