from picamzero import Camera
from exif import Image

def take_picture():
    #crea un istanza della classe camera
    cam = Camera()

    i = 5
    for cont in range(i):
        filename = f"image{cont}.jpg"
        cam.take_photo(filename)
        sleep(10)
    #cattura una sequenza di foto
    #cam.capture_sequence("sequence",num_images=3, interval=3)
    cam.take_photo("image1.jpg")

def read_image():
    i = 5
    for cont in range(i):
        img_file = open(f"image{i}.jpg","rb")#lettura in modalità binaria("rb")
        lat = img.gps_latitude
        lon = img.gps_longitude
    return lat, lon

def main():
    take_picture()
    latitude, longitude = read_image()
    print(f"latitude = {latitude}       longitude = {longitude}")
