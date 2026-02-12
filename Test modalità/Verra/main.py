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

        # Time parameters
        self.TIME_INTERVAL = 10        # seconds
        self.DURATION_SEC = 400
        self.N = int(self.DURATION_SEC / self.TIME_INTERVAL)

        # Results file
        self.results_file = "result.txt"

        # Physical constants
        self.REAL_ISS_SPEED = 7665     # m/s
        self.EARTH_RADIUS = 6371000    # meters
        self.ISS_HEIGHT = 408000       # meters

        # Camera (AstroPi)
        self.FOV_X = math.radians(62.2)  # Horizontal FOV (Pi Camera)

        # Stability
        self.MAX_OMEGA = 0.01           # rad/s


    # -------------------------------
    # GYROSCOPE (FILTER ONLY)
    # -------------------------------
    def iss_stable(self):
        gyro = self.sense.get_gyroscope_raw()
        omega = math.sqrt(
            gyro['x']**2 +
            gyro['y']**2 +
            gyro['z']**2
        )
        return omega < self.MAX_OMEGA


    # -------------------------------
    # PHOTO
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

        # Average shift (robust)
        shift = np.median(pts2 - pts1, axis=0)
        pixel_distance = np.linalg.norm(shift)

        h, w = img1.shape
        return pixel_distance, w


    # -------------------------------
    # PIXEL TO METERS
    # -------------------------------
    def pixel_to_meters(self, pixel_distance, image_width):

        # Real ground width captured
        ground_width = 2 * self.ISS_HEIGHT * math.tan(self.FOV_X / 2)

        meters_per_pixel = ground_width / image_width
        return pixel_distance * meters_per_pixel


    # -------------------------------
    # MAIN
    # -------------------------------
    def run(self):

        speeds = []
        prev_img = None

        for i in range(self.N):

            if not self.iss_stable():
                sleep(self.TIME_INTERVAL)
                continue

            img_name = f"image_{i}.jpg"
            self.take_picture(img_name)

            if prev_img is not None:

                pixel_dist, img_width = self.pixel_shift(prev_img, img_name)

                if pixel_dist is not None:
                    distance_m = self.pixel_to_meters(pixel_dist, img_width)
                    v = distance_m / self.TIME_INTERVAL

                    if 500 < v < 10000:
                        speeds.append(v)

            prev_img = img_name
            sleep(self.TIME_INTERVAL)

        if speeds:
            avg_speed = np.mean(speeds)
            avg_speed_km = avg_speed / 1000
            with open(self.results_file, "w") as f:
                f.write(f"{avg_speed_km:.2f}")


def main():
    calculator = ISSSpeedCalculator()
    calculator.run()


if __name__ == "__main__":
    main()