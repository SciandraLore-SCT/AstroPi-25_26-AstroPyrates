from exif import Image
from datetime import datetime


def get_time(image):#funzione che restituisce il tempo in cui la foto è stata fatta 
    with open(image, 'rb') as image_file:
        img = Image(image_file)#immagine aperta e convertita 
        time_str = img.get("datetime_original")#converto la striga in un numero x i calcoli
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        return time


print(get_time('atlas_photo_012.jpg'))