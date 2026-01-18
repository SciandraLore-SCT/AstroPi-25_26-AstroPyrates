from picamzero import Camera
from exif import Image
class ISSSpeedCalculator:
    def take_picture(self):
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

    def read_image(self):
        i = 5
        for cont in range(i):
            img_file = open(f"image{i}.jpg","rb")#lettura in modalità binaria("rb")
            lat = img_file.gps_latitude
            lon = img_file.gps_longitude
        return lat, lon

    def scrivi_risultatiFinale(self,messaggio):#messaggio finale finale 
        with open(self.file_risultati,'w')as f:
            f.write(messaggio)

    def esegui(self):

        self.take_picture()
        latitude, longitude = self.read_image()
        self.scrivi_risultatiFinale(f"latitude = {latitude}       longitude = {longitude}")
         
def main():
    calculator = ISS_ SpeedCalculator()
    calculator.esegui()
   

if __name__  == "__main__": 
    main() 