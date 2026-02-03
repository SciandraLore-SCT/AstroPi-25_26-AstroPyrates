from picamzero import Camera
from time import sleep
import cv2
import numpy as np
import math
from datetime import datetime, timedelta
import os


class ISSSpeedCalculator:

    def __init__(self):
        # File risultati
        self.file_risultati = "risultati.txt"

        # Camera
        self.camera = Camera()

        # Parametri temporali
        self.TIME_INTERVAL = 10          # secondi tra uno scatto e l'altro
        self.DURATA_MINUTI = 10          # durata totale esperimento

        # Parametri fisici
        self.ISS_HEIGHT = 408000         # metri
        self.FOV = math.radians(62.2)    # campo visivo camera
        self.CORRECTION_FACTOR = 1.8     # correzione geometrica empirica

        # Velocità reale ISS (per confronto)
        self.SPEED_ISS = 7665            # m/s

        # Orario di inizio
        self.start_time = datetime.now()

        # Cartella immagini
        self.image_dir = "images"
        os.makedirs(self.image_dir, exist_ok=True)

    # ---------------------------------------------------

    def take_pictures(self):
        """
        Scatta immagini per tutta la durata dell'esperimento
        """
        n = (self.DURATA_MINUTI * 60) // self.TIME_INTERVAL

        image_paths = []

        for i in range(n):
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"img_{timestamp}_{i}.jpg"
            path = os.path.join(self.image_dir, filename)

            self.camera.take_photo(path)
            image_paths.append(path)

            sleep(self.TIME_INTERVAL)

        return image_paths

    # ---------------------------------------------------

    def distanza_tra_immagini(self, img1_path, img2_path):
        """
        Calcola lo spostamento medio in pixel tra due immagini
        usando ORB + RANSAC
        """
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0, None

        # ORB detector
        orb = cv2.ORB_create(nfeatures=3000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            return 0, None

        # Matcher con cross-check
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 15:
            return 0, None

        # Ordina per qualità
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # RANSAC: rimuove outlier
        M, mask = cv2.estimateAffinePartial2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3
        )

        if M is None:
            return 0, None

        dx = M[0, 2]
        dy = M[1, 2]

        pixel_shift = np.hypot(dx, dy)

        return pixel_shift, img1.shape[1]

    # ---------------------------------------------------

    def pixel_to_meters(self, pixel_distance, image_width):
        """
        Converte pixel in metri usando il FOV
        """
        width_ground = 2 * self.ISS_HEIGHT * math.tan(self.FOV / 2)
        meters_per_pixel = width_ground / image_width

        return pixel_distance * meters_per_pixel * self.CORRECTION_FACTOR

    # ---------------------------------------------------

    def scrivi_risultati(self, messaggio):
        with open(self.file_risultati, 'a') as f:
            f.write(messaggio + "\n")

    # ---------------------------------------------------

    def esegui(self):
        while datetime.now() < self.start_time + timedelta(minutes=1):

        # 1. Scatta immagini
            images = self.take_pictures()

            velocita_list = []

            # 2. Analizza immagini
            for i in range(len(images) - 1):
                pixel_shift, image_width = self.distanza_tra_immagini(
                    images[i], images[i + 1]
                )

                if pixel_shift == 0:
                    continue

                distanza_metri = self.pixel_to_meters(
                    pixel_shift, image_width
                )

                velocita = distanza_metri / self.TIME_INTERVAL

                # Filtro fisico realistico (largo)
                if 1000 < velocita < 12000:
                    velocita_list.append(velocita)

            # 3. Risultati
            if len(velocita_list) > 5:
                velocita_media = np.mean(velocita_list)
                velocita_mediana = np.median(velocita_list)
                std = np.std(velocita_list)

                errore = abs(velocita_media - self.SPEED_ISS) / self.SPEED_ISS * 100

                self.scrivi_risultati(
                    f"\nVelocità media stimata ISS: {velocita_media:.2f} m/s"
                    f" ({velocita_media * 3.6:.0f} km/h)"
                    f"\nVelocità mediana: {velocita_mediana:.2f} m/s"
                    f"\nDeviazione standard: {std:.2f} m/s"
                    f"\nErrore rispetto al valore reale: {errore:.2f} %"
                )
            else:
                self.scrivi_risultati(
                    "\nDati insufficienti per una stima affidabile."
                )


# ---------------------------------------------------

def main():
    calculator = ISSSpeedCalculator()
    calculator.esegui()


if __name__ == "__main__":
    main()
