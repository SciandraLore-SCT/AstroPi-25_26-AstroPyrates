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

        self.TIME_INTERVAL = 10
        self.DURATA_SEC = 400
        self.file_risultati = "risultati.txt"

        self.SPEED_ISS = 7665  # m/s
        self.MAX_OMEGA = 0.01

        self.RAGGIO_TERRA = 6371000       # m
        self.ALTEZZA_ISS = 408000         # m
        self.RAGGIO_ORBITA = self.RAGGIO_TERRA + self.ALTEZZA_ISS

    # -------------------------------
    # GIROSCOPIO
    # -------------------------------
    def velocita_da_giroscopio(self):
        gyro = self.sense.get_gyroscope_raw()

        omega = math.sqrt(
            gyro['x']**2 +
            gyro['y']**2 +
            gyro['z']**2
        )

        # v = ω * r
        velocita = omega * self.RAGGIO_ORBITA
        return velocita

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
    def distanza_tra_immagini(self, img1_path, img2_path):
        img1 = cv2.imread(img1_path, 0)
        img2 = cv2.imread(img2_path, 0)

        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des1, des2)

        if len(matches) < 10:
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

        if M is None:
            return None

        dx = M[0, 2]
        dy = M[1, 2]

        return np.hypot(dx, dy)

    # -------------------------------
    # PIXEL → METRI
    # -------------------------------
    def pixel_to_meters(self, pixel_distance, image_width):
        FOV = math.radians(62.2)
        ISS_HEIGHT = 408000

        width_ground = 2 * ISS_HEIGHT * math.tan(FOV / 2)
        meters_per_pixel = width_ground / image_width

        return pixel_distance * meters_per_pixel

    # -------------------------------
    # MAIN
    # -------------------------------
    def esegui(self):

        n = int(self.DURATA_SEC / self.TIME_INTERVAL)

        velocita_pixel = []
        velocita_gyro = []

        prev_img = None

        for i in range(n):

            if not self.iss_stabile():
                sleep(self.TIME_INTERVAL)
                continue

            img_name = f"image{i}.jpg"
            self.take_picture(img_name)

            v_g = self.velocita_da_giroscopio()
            if 500 < v_g < 10000:
                velocita_gyro.append(v_g)

            if prev_img is not None:
                pixel_shift = self.distanza_tra_immagini(prev_img, img_name)

                if pixel_shift is not None:
                    distanza = self.pixel_to_meters(pixel_shift, 4056)
                    v_p = distanza / self.TIME_INTERVAL

                    if 500 < v_p < 10000:
                        velocita_pixel.append(v_p)

            prev_img = img_name
            sleep(self.TIME_INTERVAL)

        if velocita_pixel and velocita_gyro:
            v_pixel_media = np.mean(velocita_pixel)
            v_gyro_media = np.mean(velocita_gyro)

            v_finale = (v_pixel_media + v_gyro_media) / 2
            errore = abs(v_finale - self.SPEED_ISS) / self.SPEED_ISS * 100

            with open(self.file_risultati, "a") as f:
                f.write(
                    f"\nVelocità pixel: {v_pixel_media:.2f} m/s"
                    f"\nVelocità giroscopio: {v_gyro_media:.2f} m/s"
                    f"\nVelocità media finale: {v_finale:.2f} m/s"
                    f"\nErrore: {errore:.2f}%\n"
                )


def main():
    calc = ISSSpeedCalculator()
    calc.esegui()


if __name__ == "__main__":
    main()
