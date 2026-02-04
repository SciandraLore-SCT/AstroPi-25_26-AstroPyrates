from picamzero import Camera
from time import sleep
import cv2
import numpy as np
import math
import sys
from datetime import datetime

ALTEZZA_ISS_KM = 408
RAGGIO_TERRA_KM = 6371


class ISS_SpeedCalculator:

    def __init__(self):
        self.file_risultati = "risultati.txt"
        self.state_estimate = 0.0
        self.state_variance = 1.0
        self.process_variance = 0.005
        self.measurement_variance_photo = 0.5
        self.measurement_variance_gyro = 0.3

    def take_picture(self):
        cam = Camera()
        name_files = []
        for cont in range(10):
            filename = f"image{cont}.jpg"
            cam.take_photo(filename)
            name_files.append(filename)
            sleep(10)
        return name_files

    def distances(self, name_img):
        des_list = []
        kp_list = []
        image_list = []

        for dati in name_img:
            image = cv2.imread(dati, 0)
            image_list.append(image)

        orb = cv2.ORB_create(1000)
        for image in image_list:
            kp, des = orb.detectAndCompute(image, None)
            kp_list.append(kp)
            des_list.append(des)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        distances_all_pairs = []

        for i in range(len(des_list) - 1):
            matches = bf.match(des_list[i], des_list[i + 1])
            distances = []

            for m in matches:
                p1 = kp_list[i][m.queryIdx].pt
                p2 = kp_list[i + 1][m.trainIdx].pt
                d = np.linalg.norm(np.array(p1) - np.array(p2))
                distances.append(d)

            if distances:
                distances_all_pairs.append(np.mean(distances))

        return np.mean(distances_all_pairs) if distances_all_pairs else None

    def kalman_filter_fusion(self, measurement_photo, measurement_gyro):
        predicted_state = self.state_estimate
        predicted_variance = self.state_variance + self.process_variance

        measurements = []
        variances = []

        if measurement_photo is not None:
            measurements.append(measurement_photo)
            variances.append(self.measurement_variance_photo)

        if measurement_gyro is not None:
            measurements.append(measurement_gyro)
            variances.append(self.measurement_variance_gyro)

        if not measurements:
            self.state_estimate = predicted_state
            self.state_variance = predicted_variance
            return predicted_state

        if len(measurements) == 1:
            z = measurements[0]
            R = variances[0]
            K = predicted_variance / (predicted_variance + R)
            self.state_estimate = predicted_state + K * (z - predicted_state)
            self.state_variance = (1 - K) * predicted_variance
            return self.state_estimate

        z = np.array(measurements)
        R = np.diag(variances)
        H = np.ones((2, 1))
        P = np.array([[predicted_variance]])

        y = z - (H @ np.array([[predicted_state]])).flatten()
        S = H @ P @ H.T + R
        K = (P @ H.T @ np.linalg.inv(S)).flatten()

        self.state_estimate = predicted_state + K @ y
        self.state_variance = predicted_variance - np.sum(K) * predicted_variance

        return float(self.state_estimate)

    def pixel_to_meters(self, pixel_distance, image_width):
        FOV = math.radians(62.2)
        ISS_HEIGHT = 408000
        width_earth = 2 * ISS_HEIGHT * math.tan(FOV / 2)
        return pixel_distance * (width_earth / image_width)

    def calcola_velocita_angolare(self):
        try:
            gyro = self.sense.get_gyroscope_raw()
            omega = math.sqrt(gyro['x']**2 + gyro['y']**2 + gyro['z']**2)
            raggio_orbita = RAGGIO_TERRA_KM + ALTEZZA_ISS_KM
            return omega * raggio_orbita
        except Exception as e:
            print(f"Errore giroscopio: {e}")
            return None

    def write_final_result(self, messaggio):
        with open(self.file_risultati, 'w') as f:
            f.write(messaggio)

    def esegui(self):
        images = self.take_picture()
        pixel_shift = self.distances(images)

        photo_velocity = None
        if pixel_shift is not None:
            meters = self.pixel_to_meters(pixel_shift, 4056)
            photo_velocity = meters / 10.0 / 1000.0

        gyro_velocity = self.calcola_velocita_angolare()

        velocity_estimate = self.kalman_filter_fusion(
            photo_velocity,
            gyro_velocity
        )

        self.write_final_result(
            f"Velocità stimata ISS: {velocity_estimate:.3f} km/s"
        )


def main():
    calc = ISS_SpeedCalculator()
    calc.esegui()


if __name__ == "__main__":
    main()
