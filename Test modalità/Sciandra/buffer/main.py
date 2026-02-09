#!/usr/bin/env python3

from picamzero import Camera
from sense_hat import SenseHat
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep
import math
import cv2
import numpy as np

BASE_FOLDER = Path(__file__).parent.resolve()

# Time settings
DURATION_MINUTES = 10
PHOTO_INTERVAL = 5

# ISS constants
ISS_ALTITUDE_KM = 408
EARTH_RADIUS_KM = 6371
CAMERA_FOV_DEGREES = 62

class ISSSpeedCalculator:    
    def __init__(self):
        self.camera = Camera()
        self.sense = SenseHat()
        
        self.results_file = BASE_FOLDER / "result.txt"
        
        self.photo_list = []
        self.photo_velocities = []
        self.gyro_velocities = []
        self.kalman_velocities = []
        
        # Raw measurements (for debugging)
        self.photo_raw = []
        self.gyro_raw = []
        
        # Kalman filter state
        self.kalman_state = 7.66  # ISS initial velocity
        self.kalman_variance = 1.0  # initial uncertainty
    
    
    def detect_photo_displacement(self, photo1, photo2):
        try:
            img1 = cv2.imread(str(photo1), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(photo2), cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return None
            
            scale = 0.5
            img1 = cv2.resize(img1, None, fx=scale, fy=scale)
            img2 = cv2.resize(img2, None, fx=scale, fy=scale)
            
            orb = cv2.ORB_create(nfeatures=3000)
            
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None or len(kp1) < 10:
                return None
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            matches = sorted(matches, key=lambda x: x.distance)
            
            num_matches = min(50, max(20, len(matches)))
            good_matches = matches[:num_matches]
            
            if len(good_matches) < 10:
                return None
            
            displacements = []
            for match in good_matches:
                pt1 = np.array(kp1[match.queryIdx].pt)
                pt2 = np.array(kp2[match.trainIdx].pt)
                displacement = np.linalg.norm(pt2 - pt1)
                displacements.append(displacement)
            
            median_displacement = np.median(displacements)
            real_displacement = median_displacement / scale
            
            return real_displacement
        
        except Exception as e:
            self.write_results(f"Error in feature detection: {e}")
            return None
    
    
    def pixels_to_km(self, pixels):
        fov_km = 2 * ISS_ALTITUDE_KM * math.tan(math.radians(CAMERA_FOV_DEGREES / 2))
        pixel_width = 4056
        km_per_pixel = fov_km / pixel_width
        distance_km = pixels * km_per_pixel
        
        return distance_km
    
    
    def calculate_photo_velocity(self, photo1, photo2, time_sec):
        pixel_displacement = self.detect_photo_displacement(photo1, photo2)
        
        if pixel_displacement is None:
            return None
        
        distance_km = self.pixels_to_km(pixel_displacement)
        velocity_km_s = distance_km / time_sec
        
        return velocity_km_s
    
    
    def calculate_gyro_velocity(self):
        """
        Calculate velocity from gyroscope rotation
        Uses raw gyroscope values directly (already in rad/s)
        """
        try:
            gyro = self.sense.get_gyroscope_raw()
            
            # The gyroscope ALREADY gives angular velocity in rad/s
            omega_x = abs(gyro['x'])
            omega_y = abs(gyro['y'])
            omega_z = abs(gyro['z'])
            
            # Total angular velocity (magnitude of vector)
            omega_total = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
            
            # Convert to linear velocity: v = omega * r
            orbital_radius_km = EARTH_RADIUS_KM + ISS_ALTITUDE_KM
            velocity_km_s = omega_total * orbital_radius_km
            
            return velocity_km_s
        
        except Exception as e:
            self.write_results(f"Error in gyro calculation: {e}")
            return None
    
    
    def estimate_measurement_variance(self, measurement, measurement_type):
        """
        Adaptive variance based on how far the measurement is from expected range
        This makes Kalman robust to outliers!
        """
        if measurement is None:
            return None
        
        EXPECTED_MIN = 7.0
        EXPECTED_MAX = 8.5
        EXPECTED_CENTER = 7.66
        
        # Base variance
        if measurement_type == "photo":
            base_var = 0.8  # Photos are generally noisier
        else:  # gyro
            base_var = 0.15  # Gyro should be more precise
        
        # If measurement is within expected range, use base variance
        if EXPECTED_MIN <= measurement <= EXPECTED_MAX:
            return base_var
        
        # If measurement is outside range, increase variance proportionally
        # The further from expected, the less we trust it
        distance_from_center = abs(measurement - EXPECTED_CENTER)
        
        # Exponential increase in variance for outliers
        # measurement at 20 km/s → very high variance → Kalman ignores it
        outlier_factor = 1 + (distance_from_center / EXPECTED_CENTER) ** 2
        adaptive_var = base_var * outlier_factor
        
        # Cap maximum variance to prevent numerical issues
        return min(adaptive_var, 100.0)
    
    
    def kalman_filter_fusion(self, measurement_photo, measurement_gyro,
                            process_variance=0.0005):
        """
        ROBUST Kalman filter that handles ALL measurements, even outliers
        Uses adaptive measurement variance based on how realistic the value is
        """
        
        # PREDICT
        self.kalman_variance += process_variance
        
        # UPDATE with photo (even if it seems "wrong")
        if measurement_photo is not None and not np.isnan(measurement_photo):
            # Adaptive variance - high for outliers
            meas_var = self.estimate_measurement_variance(measurement_photo, "photo")
            
            K = self.kalman_variance / (self.kalman_variance + meas_var)
            innovation = measurement_photo - self.kalman_state
            self.kalman_state += K * innovation
            self.kalman_variance *= (1 - K)
            
            # Debug info
            self.write_results(
                f"  PHOTO: {measurement_photo:.4f} km/s, var={meas_var:.2f}, K={K:.4f}, innov={innovation:.4f}",
                console=False
            )
        
        # UPDATE with gyroscope (even if it seems "wrong")
        if measurement_gyro is not None and not np.isnan(measurement_gyro):
            # Adaptive variance - high for outliers
            meas_var = self.estimate_measurement_variance(measurement_gyro, "gyro")
            
            K = self.kalman_variance / (self.kalman_variance + meas_var)
            innovation = measurement_gyro - self.kalman_state
            self.kalman_state += K * innovation
            self.kalman_variance *= (1 - K)
            
            # Debug info
            self.write_results(
                f"  GYRO:  {measurement_gyro:.4f} km/s, var={meas_var:.2f}, K={K:.4f}, innov={innovation:.4f}",
                console=False
            )
        
        # Prevent divergence
        self.kalman_variance = max(self.kalman_variance, 1e-6)
        
        # Soft clamp - don't hard limit, but push gently towards expected range
        if self.kalman_state < 6.0:
            self.kalman_state = 6.0 + (self.kalman_state - 6.0) * 0.1
        elif self.kalman_state > 9.0:
            self.kalman_state = 9.0 + (self.kalman_state - 9.0) * 0.1
        
        return self.kalman_state, self.kalman_variance
    
    
    def take_photo(self, number):
        timestamp = datetime.now()
        photo_name = BASE_FOLDER / f"photo_{number:04d}.jpg"
        
        try:
            self.camera.capture(str(photo_name))
        except AttributeError:
            try:
                self.camera.take_photo(str(photo_name))
            except:
                import io
                from PIL import Image
                img = self.camera.capture_image()
                img.save(str(photo_name))
        
        return photo_name, timestamp
    
    
    def write_results(self, message, console=True):
        with open(self.results_file, 'a') as f:
            f.write(message + '\n')
        
        if console:
            print(message)
    
    
    def validate_velocity(self, velocity, method):
        """Keep for statistics, but don't use for filtering anymore"""
        if velocity is None:
            return False
        
        min_vel = 7.0  # km/s
        max_vel = 8.5  # km/s
        
        if min_vel <= velocity <= max_vel:
            return True
        else:
            return False
    
    
    def run(self):
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=DURATION_MINUTES)
        
        photo_number = 0
        
        try:
            while datetime.now() < end_time:
                
                # 1. TAKE PHOTO
                photo_path, photo_time = self.take_photo(photo_number)
                self.write_results(f"\nPhoto {photo_number}: {photo_path.name}")
                
                self.photo_list.append({
                    'path': photo_path,
                    'time': photo_time,
                    'number': photo_number
                })
                
                # 2. CALCULATE GYROSCOPE VELOCITY
                vel_gyro = self.calculate_gyro_velocity()
                
                # Store raw value (for statistics)
                self.gyro_raw.append(vel_gyro if vel_gyro else None)
                
                # Keep validated list (for comparison)
                if vel_gyro and self.validate_velocity(vel_gyro, "GYRO"):
                    self.gyro_velocities.append(vel_gyro)
                    self.write_results(f"GYRO: {vel_gyro:.4f} km/s (validated)")
                elif vel_gyro:
                    self.write_results(f"GYRO: {vel_gyro:.4f} km/s (outlier - will be downweighted)")
                
                # 3. CALCULATE PHOTO VELOCITY (if possible)
                vel_photo = None
                if len(self.photo_list) >= 2:
                    photo1_data = self.photo_list[-2]
                    photo2_data = self.photo_list[-1]
                    
                    time_diff = (photo2_data['time'] - photo1_data['time']).total_seconds()
                    
                    vel_photo = self.calculate_photo_velocity(
                        photo1_data['path'],
                        photo2_data['path'],
                        time_diff
                    )
                    
                    # Store raw value
                    self.photo_raw.append(vel_photo if vel_photo else None)
                    
                    # Keep validated list (for comparison)
                    if vel_photo and self.validate_velocity(vel_photo, "PHOTO"):
                        self.photo_velocities.append(vel_photo)
                        self.write_results(f"PHOTO: {vel_photo:.4f} km/s (validated)")
                    elif vel_photo:
                        self.write_results(f"PHOTO: {vel_photo:.4f} km/s (outlier - will be downweighted)")
                
                # 4. APPLY KALMAN FILTER - RECEIVES ALL MEASUREMENTS!
                # Even outliers! The adaptive variance handles them.
                kalman_vel, kalman_var = self.kalman_filter_fusion(
                    measurement_photo=vel_photo,  # Can be outlier!
                    measurement_gyro=vel_gyro     # Can be outlier!
                )
                
                self.kalman_velocities.append(kalman_vel)
                self.write_results(
                    f"KALMAN: {kalman_vel:.4f} km/s (var={kalman_var:.4f})"
                )
                
                photo_number += 1
                sleep(PHOTO_INTERVAL)
        
        except Exception as e:
            self.write_results(f"\nERROR: {e}")
            import traceback
            self.write_results(traceback.format_exc())
        
        finally:
            self.analyze_results()
    
    
    def analyze_results(self):
        REAL_VELOCITY = 7.66
        
        methods = [
            ("PHOTO (validated only)", self.photo_velocities),
            ("GYRO (validated only)", self.gyro_velocities),
            ("KALMAN (uses all data)", self.kalman_velocities)
        ]
        
        final_results = []
        
        for method_name, velocity_list in methods:
            self.write_results(f"\n{'='*60}")
            self.write_results(f"METHOD: {method_name}")
            self.write_results('='*60)
            
            if velocity_list:
                mean = np.mean(velocity_list)
                median = np.median(velocity_list)
                std_dev = np.std(velocity_list)
                
                error = abs(mean - REAL_VELOCITY)
                error_pct = (error / REAL_VELOCITY) * 100
                
                self.write_results(f"Valid measurements: {len(velocity_list)}")
                self.write_results(f"Mean:     {mean:.4f} km/s ({mean*3600:.0f} km/h)")
                self.write_results(f"Median:   {median:.4f} km/s")
                self.write_results(f"Std dev:  {std_dev:.4f} km/s")
                self.write_results(f"Error:    {error:.4f} km/s ({error_pct:.2f}%)")
                
                final_results.append({
                    'method': method_name,
                    'mean': mean,
                    'error_pct': error_pct
                })
            else:
                self.write_results("No valid measurements")
        
        # FINAL RESULT - Use ONLY Kalman
        if len(self.kalman_velocities) > 0:
            self.write_results("\n" + "-" * 80)
            self.write_results("FINAL RESULT (Kalman filter)")
            self.write_results("-" * 80 + "\n")
            
            kalman_mean = np.mean(self.kalman_velocities)
            final_error = abs(kalman_mean - REAL_VELOCITY)
            final_error_pct = (final_error / REAL_VELOCITY) * 100
            
            self.write_results(f"Calculated ISS velocity: {kalman_mean:.4f} km/s")
            self.write_results(f"                       = {kalman_mean*3600:.0f} km/h")
            self.write_results(f"\nReal ISS velocity:      {REAL_VELOCITY} km/s (~27,600 km/h)")
            self.write_results(f"Final error:            {final_error:.4f} km/s ({final_error_pct:.2f}%)")
            
            # Evaluation
            if final_error_pct < 1:
                self.write_results("\nEXCELLENT - Error < 1%")
            elif final_error_pct < 3:
                self.write_results("\nGOOD - Error < 3%")
            elif final_error_pct < 5:
                self.write_results("\nACCEPTABLE - Error < 5%")
            else:
                self.write_results("\nNEEDS IMPROVEMENT - Error > 5%")
        
        else:
            self.write_results("\nNo valid data collected")


def main():
    try:
        calculator = ISSSpeedCalculator()
        calculator.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
