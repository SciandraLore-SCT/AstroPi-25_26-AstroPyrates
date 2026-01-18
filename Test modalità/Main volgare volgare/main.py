#!/usr/bin/env python3

from picamzero import Camera
from sense_hat import SenseHat
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep
import math
import cv2
import numpy as np
from exif import Image as ExifImage

BASE_FOLDER = Path(__file__).parent.resolve()

# Variabili di tempo legate a minuti e a secondi
DURATA_MINUTI = 10
INTERVALLO_FOTO = 5

# Altezza media della ISS in km (varia tra 400-420 km)
ALTEZZA_ISS_KM = 408
# Raggio della Terra in km
RAGGIO_TERRA_KM = 6371

FOV_CAMERA_GRADI = 62

class ISSSpeedCalculator:    
    def __init__(self):
        self.camera = Camera()
        self.sense = SenseHat()
        
        self.file_risultati = BASE_FOLDER / "result.txt"
        
        self.foto_lista = []
        self.velocita_gps = []
        self.velocita_foto = []
        self.velocita_angolare = []
        
        self.gyro_prev = None
        self.tempo_prev = None
        
    
    def converti_coordinate_decimali(self, coordinate, riferimento):

        gradi_dec = coordinate[0] + coordinate[1] / 60 + coordinate[2] / 3600
        
        if riferimento in ['S', 'W']:
            gradi_dec = -gradi_dec
        
        return gradi_dec
    
    
    def estrai_gps_da_foto(self, percorso_foto):#non si può usare il gps con exif 
        try:
            with open(percorso_foto, 'rb') as f:
                img = ExifImage(f)
            
            if img.has_exif and hasattr(img, 'gps_latitude'):#da vedere
                lat = self.converti_coordinate_decimali(
                    img.gps_latitude,
                    img.gps_latitude_ref
                )
                lon = self.converti_coordinate_decimali(
                    img.gps_longitude,
                    img.gps_longitude_ref
                )
                return lat, lon
            else:
                return None, None
        
        except Exception as e:
            print(f"Errore lettura GPS: {e}")
            return None, None
    
    
    def calcola_distanza_haversine(self, lat1, lon1, lat2, lon2):#da vedere

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        distanza_km = RAGGIO_TERRA_KM * c
        
        return distanza_km
    
    
    def calcola_velocita_gps(self, foto1, foto2, tempo_sec):#da vedere => collegato alla costante di haversine

        lat1, lon1 = self.estrai_gps_da_foto(foto1)
        lat2, lon2 = self.estrai_gps_da_foto(foto2)
        
        if None in [lat1, lon1, lat2, lon2]:
            return None
        
        distanza_km = self.calcola_distanza_haversine(lat1, lon1, lat2, lon2)
        velocita_km_s = distanza_km / tempo_sec
        
        return velocita_km_s
    
    
    
    def rileva_spostamento_foto(self, foto1, foto2):#da vedere
        try:
            img1 = cv2.imread(str(foto1), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(foto2), cv2.IMREAD_GRAYSCALE)
            
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
            
            spostamenti = []
            for match in good_matches:
                pt1 = np.array(kp1[match.queryIdx].pt)
                pt2 = np.array(kp2[match.trainIdx].pt)
                spostamento = np.linalg.norm(pt2 - pt1)
                spostamenti.append(spostamento)
            
            spostamento_mediano = np.median(spostamenti)
            
            spostamento_reale = spostamento_mediano / scale
            
            return spostamento_reale
        
        except Exception as e:
            print(f"Errore rilevamento features: {e}")
            return None
    
    
    def pixel_a_km(self, pixel):

        fov_km = 2 * ALTEZZA_ISS_KM * math.tan(math.radians(FOV_CAMERA_GRADI / 2))
        
        larghezza_pixel = 4056
        
        km_per_pixel = fov_km / larghezza_pixel
        distanza_km = pixel * km_per_pixel
        
        return distanza_km
    
    
    def calcola_velocita_foto(self, foto1, foto2, tempo_sec):

        spostamento_pixel = self.rileva_spostamento_foto(foto1, foto2)
        
        if spostamento_pixel is None:
            return None
        
        distanza_km = self.pixel_a_km(spostamento_pixel)
        velocita_km_s = distanza_km / tempo_sec
        
        return velocita_km_s
    
    
    def calcola_velocita_angolare(self):#no cambiare giusto
        try:
            gyro = self.sense.get_gyroscope_raw()
            tempo_corrente = datetime.now()
            

            omega_x = gyro['x']  # rad/s
            omega_y = gyro['y']  # rad/s
            omega_z = gyro['z']  # rad/s
            
            omega_tot = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
            
            raggio_orbita_km = RAGGIO_TERRA_KM + ALTEZZA_ISS_KM
            
            velocita_km_s = omega_tot * raggio_orbita_km
            
            self.gyro_prev = gyro
            self.tempo_prev = tempo_corrente
            
            return velocita_km_s
        
        except Exception as e:
            print(f"Errore calcolo angolare: {e}")
            return None
    
    
    
    def scatta_foto(self, numero):
        timestamp = datetime.now()
        nome_foto = BASE_FOLDER / f"foto_{numero:04d}.jpg"
        
        try:
            self.camera.capture(str(nome_foto))
        except AttributeError:
            # Fallback per versioni diverse della libreria
            try:
                self.camera.take_photo(str(nome_foto))
            except:
                import io
                from PIL import Image
                img = self.camera.capture_image()
                img.save(str(nome_foto))
        
        return nome_foto, timestamp
    
    
    def scrivi_risultati(self, messaggio, console=True):
        with open(self.file_risultati, 'a') as f:
            f.write(messaggio + '\n')
        
        if console:
            print(messaggio)

    def scrivi_risultatiFinale(self,messaggio):#messaggio finale finale 
        with open(self.file_risultati,'w')as f:
            f.write(messaggio)

    
    
    def valida_velocita(self, velocita, metodo):
        if velocita is None:
            return False
        
        min_vel = 7.0  # km/s
        max_vel = 8.0  # km/s
        
        if min_vel <= velocita <= max_vel:
            return True
        else:
            self.scrivi_risultati(
                f"Errore {metodo}: Velocità fuori range ({velocita:.2f} km/s) - scartata"
            )
            return False
    
    
    
    def esegui(self):
        
        tempo_inizio = datetime.now()
        tempo_fine = tempo_inizio + timedelta(minutes=DURATA_MINUTI)
        
        
        foto_numero = 0
        
        try:
            while datetime.now() < tempo_fine:
                
                foto_path, foto_tempo = self.scatta_foto(foto_numero)
                self.scrivi_risultati(f"\nFoto {foto_numero}: {foto_path.name}")
                
                self.foto_lista.append({
                    'path': foto_path,
                    'tempo': foto_tempo,
                    'numero': foto_numero
                })
                
                vel_ang = self.calcola_velocita_angolare()
                if vel_ang and self.valida_velocita(vel_ang, "ANGOLARE"):
                    self.velocita_angolare.append(vel_ang)
                    self.scrivi_risultati(
                        f"ANGOLARE: {vel_ang:.4f} km/s "
                        f"({vel_ang*3600:.0f} km/h)"
                    )
                
                
                if len(self.foto_lista) >= 2:
                    foto1_data = self.foto_lista[-2]
                    foto2_data = self.foto_lista[-1]
                    
                    tempo_diff = (foto2_data['tempo'] - foto1_data['tempo']).total_seconds()
                    
                    vel_gps = self.calcola_velocita_gps(
                        foto1_data['path'],
                        foto2_data['path'],
                        tempo_diff
                    )
                    
                    if vel_gps and self.valida_velocita(vel_gps, "GPS"):
                        self.velocita_gps.append(vel_gps)
                        self.scrivi_risultati(
                            f"GPS: {vel_gps:.4f} km/s "
                            f"({vel_gps*3600:.0f} km/h)"
                        )
                    
                    vel_foto = self.calcola_velocita_foto(
                        foto1_data['path'],
                        foto2_data['path'],
                        tempo_diff
                    )
                    
                    if vel_foto and self.valida_velocita(vel_foto, "FOTO"):
                        self.velocita_foto.append(vel_foto)
                        self.scrivi_risultati(
                            f"FOTO: {vel_foto:.4f} km/s "
                            f"({vel_foto*3600:.0f} km/h)"
                        )
                
                foto_numero += 1
                
                sleep(INTERVALLO_FOTO)
        
        except Exception as e:
            self.scrivi_risultati(f"\nERRORE: {e}")
            import traceback
            self.scrivi_risultati(traceback.format_exc())
        
        finally:
            self.analizza_risultati()

    
    
    def analizza_risultati(self):
        

        # velocità media relativa della ISS
        VELOCITA_REALE = 7.66
        
        metodi = [
            ("GPS", self.velocita_gps),
            ("FOTO", self.velocita_foto),
            ("ANGOLARE", self.velocita_angolare)
        ]
        
        risultati_finali = []
        
        for nome_metodo, velocita_lista in metodi:
            self.scrivi_risultati(f"\n--- METODO {nome_metodo} ---")
            
            if velocita_lista:
                media = np.mean(velocita_lista)
                mediana = np.median(velocita_lista)
                std_dev = np.std(velocita_lista)
                
                errore = abs(media - VELOCITA_REALE)
                errore_perc = (errore / VELOCITA_REALE) * 100
                
                self.scrivi_risultati(f"Misurazioni valide: {len(velocita_lista)}")
                self.scrivi_risultati(f"Media: {media:.4f} km/s ({media*3600:.0f} km/h)")
                self.scrivi_risultati(f"Mediana: {mediana:.4f} km/s")
                self.scrivi_risultati(f"Dev. standard: {std_dev:.4f} km/s")
                self.scrivi_risultati(f"Errore: {errore:.4f} km/s ({errore_perc:.2f}%)")
                
                risultati_finali.append({
                    'metodo': nome_metodo,
                    'media': media,
                    'errore_perc': errore_perc
                })
            else:
                self.scrivi_risultati("Nessuna misurazione valida")
        
        if len(risultati_finali) > 0:
            self.scrivi_risultati("\n" + "-" * 80)
            self.scrivi_risultati("RISULTATO FINALE COMBINATO")
            self.scrivi_risultati("-" * 80 + "\n")
            
            pesi = [1.0 / (r['errore_perc'] + 0.1) for r in risultati_finali]
            peso_tot = sum(pesi)
            
            media_combinata = sum(
                r['media'] * pesi[i] / peso_tot 
                for i, r in enumerate(risultati_finali)
            )
            
            errore_finale = abs(media_combinata - VELOCITA_REALE)
            errore_finale_perc = (errore_finale / VELOCITA_REALE) * 100
            
            self.scrivi_risultati(f"Velocità ISS calcolata: {media_combinata:.4f} km/s")
            self.scrivi_risultati(f"                      = {media_combinata*3600:.0f} km/h")
            self.scrivi_risultati(f"\nVelocità reale ISS: {VELOCITA_REALE} km/s (~27,600 km/h)")
            self.scrivi_risultati(f"Errore finale: {errore_finale:.4f} km/s ({errore_finale_perc:.2f}%)")
            
            # Valutazione
            if errore_finale_perc < 1:
                self.scrivi_risultati("\nerrore < 1%")
            elif errore_finale_perc < 3:
                self.scrivi_risultati("\nerrore < 3%")
            elif errore_finale_perc < 5:
                self.scrivi_risultati("\nerrore < 5%")
            else:
                self.scrivi_risultati("\nRerrore > 5%")
        
        else:
            self.scrivi_risultati("\nNessun dato valido raccolto")




def main():
    try:
        calculator = ISSSpeedCalculator()
        calculator.esegui()
    except KeyboardInterrupt:
        print("\nProgramma interrotto dall'utente")
    except Exception as e:
        print(f"\nERRORE FATALE: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
