from picamzero import Camera
from exif import Image
from time import sleep


class ISSSpeedCalculator:

    def __init__(self):
        self.file_risultati = "risultati.txt"

    def take_picture(self):
        cam = Camera()
        for cont in range(5):
            filename = f"image{cont}.jpg"
            cam.take_photo(filename)
            sleep(10)

    def read_image(self):
        latitudes = []
        longitudes = []

        for cont in range(5):
            with open(f"image{cont}.jpg", "rb") as img_file:
                img = Image(img_file)

                if img.has_exif:
                    latitudes.append(img.gps_latitude)
                    longitudes.append(img.gps_longitude)

        return latitudes, longitudes

    def scrivi_risultatiFinale(self, messaggio):
        with open(self.file_risultati, 'w') as f:
            f.write(messaggio)

    def esegui(self):
        self.take_picture()
        lat, lon = self.read_image()
        self.scrivi_risultatiFinale(
            f"Latitudes: {lat}\nLongitudes: {lon}"
        )


def main():
    calculator = ISSSpeedCalculator()
    calculator.esegui()


if __name__ == "__main__":
    main()
