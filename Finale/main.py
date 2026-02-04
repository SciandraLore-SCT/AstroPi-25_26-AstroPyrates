# ==========================================================
# ISS SPEED ESTIMATION - FOTO + GIRO + KALMAN (SINGLE FILE)
# ==========================================================

from picamzero import Camera
from sense_hat import SenseHat
from time import sleep
from datetime import datetime
from datetime import timedelta
import cv2
import numpy as np
import math

# ===================== COSTANTI =====================

TIME_INTERVAL = 5          # secondi
DURATA_MINUTI = 10
IMAGE_WIDTH = 4056         # risoluzione camera
FOV = math.radians(62.2)

ISS_HEIGHT_M = 408000      # m
R_TERRA_KM = 6371
H_ISS_KM = 408

VELOCITA_REALE = 7.66      # km/s

# ===================== KALMAN =====================

def kalman_filter_fusion(z_photo, z_gyro,
                         x_prev, P_prev,
                         Q=0.01,
                         R_photo=0.4,
                         R_gyro=1.2):
    # --- Prediction ---
    x_pred = x_prev
    P_pred = P_prev + Q

    measurements = []
    variances = []

    if z_photo is not None:
        measurements.append(z_photo)
        variances.append(R_photo)

    if z_gyro is not None:
        measurements.append(z_gyro)
        variances.append(R_gyro)

    if len(measurements) == 0:
        return x_pred, P_pred

    # --- Update ---
    if len(measurements) == 1:
        z = measurements[0]
        R = variances[0]

        K = P_pred / (P_pred + R)
        x_upd = x_pred + K * (z - x_pred)
        P_upd = (1 - K) * P_pred

    else:
        z = np.array(measurements)
        R = np.diag(variances)

        H = np.ones((len(measurements), 1))
        y = z - (H @ np.array([[x_pred]])).flatten()
        S = H @ np.array([[P_pred]]) @ H.T + R
        K = (np.array([[P_pred]]) @ H.T @ np.linalg.inv(S)).flatten()

        x_upd = x_pred + K @ y
        KH = (K @ H)[0]
        P_upd = (1 - KH) * P_pred

        x_upd = float(x_upd)
        P_upd = float(P_upd)

    return x_upd, P_upd

# ===================== IMMAGINI =====================

def distanza_tra_immagini(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    if img1 is None or img2 is None:
        return None

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


def pixel_to_meters(pixel_dist):
    width_earth = 2 * ISS_HEIGHT_M * math.tan(FOV / 2)
    meters_per_pixel = width_earth / IMAGE_WIDTH
    return pixel_dist * meters_per_pixel

# ===================== GIROSCOPIO =====================

def velocita_da_giroscopio(sense):
    gyro = sense.get_gyroscope_raw()

    omega = math.sqrt(
        gyro['x']**2 +
        gyro['y']**2 +
        gyro['z']**2
    )

    r = R_TERRA_KM + H_ISS_KM
    return omega * r   # km/s (stima rumorosa)

# ===================== MAIN =====================

def main():

    camera = Camera()
    sense = SenseHat()

    immagini = []
    velocita_kalman = []

    x_est = 7.6
    P_est = 1.0

    start = datetime.now()
    fine = start + timedelta(minutes=DURATA_MINUTI)

    print("INIZIO ACQUISIZIONE\n")

    i = 0
    while datetime.now() < fine:

        nome = f"image{i}.jpg"
        camera.take_photo(nome)
        immagini.append(nome)

        z_photo = None
        if i > 0:
            pixel_shift = distanza_tra_immagini(immagini[i-1], immagini[i])
            if pixel_shift is not None:
                dist_m = pixel_to_meters(pixel_shift)
                z_photo = (dist_m / TIME_INTERVAL) / 1000  # km/s

        z_gyro = velocita_da_giroscopio(sense)

        x_est, P_est = kalman_filter_fusion(
            z_photo, z_gyro,
            x_est, P_est
        )

        velocita_kalman.append(x_est)

        print(f"[{i}] FOTO={z_photo}  GIRO={z_gyro:.3f}  KALMAN={x_est:.3f}")

        i += 1
        sleep(TIME_INTERVAL)

    # ===================== RISULTATI =====================

    media = np.mean(velocita_kalman)
    errore = abs(media - VELOCITA_REALE) / VELOCITA_REALE * 100

    print("\n================ RISULTATO FINALE ================")
    print(f"Velocità stimata ISS: {media:.3f} km/s")
    print(f"Velocità reale ISS : {VELOCITA_REALE:.2f} km/s")
    print(f"Errore percentuale : {errore:.2f}%")
    print("==================================================")

# ===================== AVVIO =====================

if __name__ == "__main__":
    main()
