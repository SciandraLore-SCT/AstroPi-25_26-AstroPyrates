# AstroPyrates - Astro Pi Mission 2025/2026

Team AstroPyrates
ITIS Mario Del'Pozzo

## Project Overview

AstroPyrates is a project developed for the Astro Pi 2025/2026 Mission Space Lab competition. Our program calculates the orbital velocity of the International Space Station (ISS) using computer vision techniques applied to images captured by the Astro Pi Camera Module.

Unlike traditional methods that rely on GPS or gyroscopic data, our approach uses feature matching algorithms to detect ground movement between consecutive images, allowing us to determine the ISS speed with high accuracy.

## Team Members

- Lorenzo Sciandra
- Francesco Verra
- Isabel Ciartano
- Nicolas Elne

Institution: ITIS Mario Del'Pozzo

## Features

- Optical Flow Speed Calculation: Computes ISS velocity by analyzing ground displacement between sequential photos
- ORB Feature Detection: Implements Oriented FAST and Rotated BRIEF algorithm for robust keypoint matching
- RANSAC-based Filtering: Uses robust estimation to eliminate outlier matches
- Gyroscope Stability Check: Ensures camera stability before image capture to prevent motion blur
- Automatic Image Processing: Captures and processes images over a 400-second observation window
- Autonomous Operation: Runs completely unattended once deployed on the ISS

## Technology Stack

Hardware:
- Raspberry Pi (Astro Pi flight unit)
- Sense HAT - Gyroscope for stability detection
- Camera HQ Module - High-quality image capture

Software:
- Python 3.11+
- picamzero - Simplified camera interface
- sense_hat - Sense HAT sensor access
- OpenCV (cv2) - Computer vision and image processing
- NumPy - Numerical computations
- math - Mathematical functions
- datetime - Timestamp management
- pathlib - File path handling
- exif - Image metadata extraction
- time - Timing and sleep functions

## How It Works

1. Stability Verification:
   The gyroscope checks if the ISS is stable enough for clear imaging. If angular velocity exceeds 0.01 rad/s, the program waits and retries.

2. Image Acquisition:
   The program captures one photo every 10 seconds for a total duration of 400 seconds, creating a sequence of 40 images.

3. Feature Detection and Matching:
   Using ORB algorithm, the program:
   - Detects keypoints in consecutive images
   - Computes descriptors for each keypoint
   - Matches corresponding features between frames
   - Applies median filtering for robust shift estimation

4. Speed Calculation:
   Ground Width = 2 * ISS_Height * tan(FOV/2)
   Meters per Pixel = Ground Width / Image Width
   Distance Traveled = Pixel Shift * Meters per Pixel
   Velocity = Distance Traveled / Time Interval

5. Data Validation:
   - Filters speeds outside the plausible range (500-10000 m/s)
   - Requires minimum 10 feature matches for reliable calculation
   - Only processes images when ISS is stable

## Project Structure

AstroPyrates/
├── iss_speed_calculator.py    # Main program
├── result.txt                 # Output file with calculated speed
├── image_*.jpg               # Captured images (generated)
└── README.md                 # This documentation

## Installation and Setup

Prerequisites:
pip install picamzero sense-hat opencv-python numpy exif

Configuration Parameters:
TIME_INTERVAL = 10        # Seconds between captures
DURATION_SEC = 400        # Total observation time
MAX_OMEGA = 0.01         # Maximum angular velocity (rad/s)
FOV_X = 62.2°           # Camera horizontal field of view

## Output

The program creates a result.txt file containing a single floating-point number: the average ISS speed in km/s.

Example output:
7.67

## Scientific Background

ORB Algorithm:
ORB is a fast, rotation-invariant feature detector suitable for real-time applications on resource-constrained devices like the Astro Pi. It combines:
- FAST keypoint detector for finding corners
- BRIEF descriptor for efficient feature representation

Pixel-to-Meters Conversion:
The conversion from pixel displacement to real-world distance relies on:
- Known ISS orbital altitude (408 km)
- Camera field of view (62.2 degrees horizontal)
- Trigonometric relationships

## Testing and Validation

The algorithm has been tested using:
- Simulated ground motion sequences
- ISS video footage from previous missions
- Static camera tests for zero-velocity validation

Expected ISS orbital velocity: 7.66 km/s

## Limitations and Considerations

- Cloud cover: May reduce detectable features
- Night passes: Insufficient illumination for feature detection
- Ocean areas: Limited distinct features for matching
- ISS attitude changes: May affect effective FOV
- Image blur: Requires stable platform for clear captures

## License

This project is developed for educational purposes as part of the Astro Pi Challenge 2025/2026.

## Acknowledgments

- ESA Education - For organizing the Astro Pi Challenge
- Raspberry Pi Foundation - For providing the Astro Pi hardware
- OpenCV Community - For computer vision tools and documentation

---

Made with determination by AstroPyrates
