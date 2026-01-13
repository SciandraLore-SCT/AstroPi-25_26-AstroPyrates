#!/usr/bin/env python3
"""
=============================================================================
ASTRO PI MISSION SPACE LAB - CALCOLO VELOCITÀ ISS
=============================================================================
Team: [Il tuo team]
Progetto: Calcolo della velocità della ISS con 3 metodi

METODI UTILIZZATI:
1. GPS: Calcola la velocità usando le coordinate GPS delle foto
2. FOTO: Calcola la velocità tramite rilevamento features nelle immagini
3. ANGOLARE: Calcola la velocità usando il giroscopio del Sense HAT

Velocità attesa della ISS: ~7.66 km/s (~27,600 km/h)
=============================================================================
"""

from picamzero import Camera
from sense_hat import SenseHat
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep
import math
import cv2
import numpy as np
from exif import Image as ExifImage

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# Cartella base per salvare i dati
BASE_FOLDER = Path(__file__).parent.resolve()

# Durata del programma (max 3 ore = 180 minuti per Astro Pi)
DURATA_MINUTI = 10  # 10 minuti per test, cambia a 175 per la missione reale

# Intervallo tra le foto (in secondi)
INTERVALLO_FOTO = 5  # Cambia a 10-15 per la missione reale

# Altezza media della ISS in km (varia tra 400-420 km)
ALTEZZA_ISS_KM = 408

# Raggio della Terra in km
RAGGIO_TERRA_KM = 6371

# Campo visivo della camera (approssimato)
FOV_CAMERA_GRADI = 62

# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

class ISSSpeedCalculator:
    """Classe principale per calcolare la velocità della ISS"""
    
    def __init__(self):
        """Inizializza il calcolatore"""
        print("=" * 80)
        print(" " * 25 + "ASTRO PI - ISS SPEED CALCULATOR")
        print("=" * 80)
        print()
        
        # Inizializza hardware
        self.camera = Camera()
        self.sense = SenseHat()
        
        # File di output
        self.file_risultati = BASE_FOLDER / "result.txt"
        
        # Liste per salvare i dati
        self.foto_lista = []
        self.velocita_gps = []
        self.velocita_foto = []
        self.velocita_angolare = []
        
        # Dati giroscopio precedenti
        self.gyro_prev = None
        self.tempo_prev = None
        
        print("✓ Camera inizializzata")
        print("✓ Sense HAT inizializzato")
        print()
    
    
    # ========================================================================
    # METODO 1: CALCOLO VELOCITÀ CON GPS
    # ========================================================================
    
    def converti_coordinate_decimali(self, coordinate, riferimento):
        """
        Converte coordinate GPS da DMS a formato decimale
        coordinate: (gradi, minuti, secondi)
        riferimento: 'N', 'S', 'E', 'W'
        """
        gradi_dec = coordinate[0] + coordinate[1] / 60 + coordinate[2] / 3600
        
        if riferimento in ['S', 'W']:
            gradi_dec = -gradi_dec
        
        return gradi_dec
    
    
    def estrai_gps_da_foto(self, percorso_foto):
        """
        Estrae le coordinate GPS dai dati EXIF della foto
        Ritorna: (latitudine, longitudine) o (None, None)
        """
        try:
            with open(percorso_foto, 'rb') as f:
                img = ExifImage(f)
            
            if img.has_exif and hasattr(img, 'gps_latitude'):
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
    
    
    def calcola_distanza_haversine(self, lat1, lon1, lat2, lon2):
        """
        Calcola la distanza tra due punti sulla Terra
        usando la formula di Haversine
        """
        # Conversione in radianti
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Differenze
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Formula di Haversine
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        distanza_km = RAGGIO_TERRA_KM * c
        
        return distanza_km
    
    
    def calcola_velocita_gps(self, foto1, foto2, tempo_sec):
        """
        METODO 1: Calcola la velocità usando i dati GPS
        """
        lat1, lon1 = self.estrai_gps_da_foto(foto1)
        lat2, lon2 = self.estrai_gps_da_foto(foto2)
        
        if None in [lat1, lon1, lat2, lon2]:
            return None
        
        distanza_km = self.calcola_distanza_haversine(lat1, lon1, lat2, lon2)
        velocita_km_s = distanza_km / tempo_sec
        
        return velocita_km_s
    
    
    # ========================================================================
    # METODO 2: CALCOLO VELOCITÀ CON FEATURES FOTO
    # ========================================================================
    
    def rileva_spostamento_foto(self, foto1, foto2):
        """
        METODO 2: Calcola lo spostamento tra due foto
        usando rilevamento features con ORB
        Ritorna lo spostamento in pixel
        """
        try:
            # Carica le immagini in scala di grigi
            img1 = cv2.imread(str(foto1), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(foto2), cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return None
            
            # Ridimensiona per velocità (opzionale)
            scale = 0.5
            img1 = cv2.resize(img1, None, fx=scale, fy=scale)
            img2 = cv2.resize(img2, None, fx=scale, fy=scale)
            
            # Rilevatore ORB (Oriented FAST and Rotated BRIEF)
            orb = cv2.ORB_create(nfeatures=3000)
            
            # Trova keypoints e descrittori
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            if des1 is None or des2 is None or len(kp1) < 10:
                return None
            
            # Matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            # Ordina per qualità
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Prendi i migliori match (minimo 20)
            num_matches = min(50, max(20, len(matches)))
            good_matches = matches[:num_matches]
            
            if len(good_matches) < 10:
                return None
            
            # Calcola lo spostamento medio
            spostamenti = []
            for match in good_matches:
                pt1 = np.array(kp1[match.queryIdx].pt)
                pt2 = np.array(kp2[match.trainIdx].pt)
                spostamento = np.linalg.norm(pt2 - pt1)
                spostamenti.append(spostamento)
            
            # Usa la mediana per robustezza (esclude outliers)
            spostamento_mediano = np.median(spostamenti)
            
            # Scala di nuovo allo spostamento originale
            spostamento_reale = spostamento_mediano / scale
            
            return spostamento_reale
        
        except Exception as e:
            print(f"Errore rilevamento features: {e}")
            return None
    
    
    def pixel_a_km(self, pixel):
        """
        Converte lo spostamento in pixel in distanza in km
        """
        # Campo visivo alla superficie terrestre
        fov_km = 2 * ALTEZZA_ISS_KM * math.tan(math.radians(FOV_CAMERA_GRADI / 2))
        
        # Assumiamo una risoluzione di 4056 pixel (larghezza HQ Camera)
        larghezza_pixel = 4056
        
        km_per_pixel = fov_km / larghezza_pixel
        distanza_km = pixel * km_per_pixel
        
        return distanza_km
    
    
    def calcola_velocita_foto(self, foto1, foto2, tempo_sec):
        """
        METODO 2: Calcola la velocità usando features delle foto
        """
        spostamento_pixel = self.rileva_spostamento_foto(foto1, foto2)
        
        if spostamento_pixel is None:
            return None
        
        distanza_km = self.pixel_a_km(spostamento_pixel)
        velocita_km_s = distanza_km / tempo_sec
        
        return velocita_km_s
    
    
    # ========================================================================
    # METODO 3: CALCOLO VELOCITÀ ANGOLARE
    # ========================================================================
    
    def calcola_velocita_angolare(self):
        """
        METODO 3: Calcola la velocità usando il giroscopio
        
        Il giroscopio misura la velocità angolare (gradi/sec o rad/sec)
        della ISS. Conoscendo l'altezza dell'orbita, possiamo convertire
        la velocità angolare in velocità lineare.
        
        Formula: v = ω × r
        dove:
        - v = velocità lineare (km/s)
        - ω = velocità angolare (rad/s)
        - r = raggio dell'orbita = raggio_terra + altezza_iss
        """
        try:
            # Leggi il giroscopio (restituisce rad/s su x, y, z)
            gyro = self.sense.get_gyroscope_raw()
            tempo_corrente = datetime.now()
            
            # Calcola la velocità angolare totale
            # (magnitudine del vettore velocità angolare)
            omega_x = gyro['x']  # rad/s
            omega_y = gyro['y']  # rad/s
            omega_z = gyro['z']  # rad/s
            
            # Velocità angolare totale (magnitudine)
            omega_tot = math.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
            
            # Raggio dell'orbita
            raggio_orbita_km = RAGGIO_TERRA_KM + ALTEZZA_ISS_KM
            
            # Velocità lineare: v = ω × r
            velocita_km_s = omega_tot * raggio_orbita_km
            
            # Salva i dati per il prossimo calcolo
            self.gyro_prev = gyro
            self.tempo_prev = tempo_corrente
            
            return velocita_km_s
        
        except Exception as e:
            print(f"Errore calcolo angolare: {e}")
            return None
    
    
    
    def scatta_foto(self, numero):
        """Scatta una foto e ritorna il percorso"""
        timestamp = datetime.now()
        nome_foto = BASE_FOLDER / f"foto_{numero:04d}.jpg"
        
        # CORREZIONE: Usa capture() invece di capture_file() per Astro Pi Replay
        try:
            self.camera.capture(str(nome_foto))
        except AttributeError:
            # Fallback per versioni diverse della libreria
            try:
                self.camera.take_photo(str(nome_foto))
            except:
                # Ultima risorsa: salva direttamente l'immagine
                import io
                from PIL import Image
                img = self.camera.capture_image()
                img.save(str(nome_foto))
        
        return nome_foto, timestamp
    
    
    def scrivi_risultati(self, messaggio, console=True):
        """Scrive i risultati sia su file che su console"""
        with open(self.file_risultati, 'a') as f:
            f.write(messaggio + '\n')
        
        if console:
            print(messaggio)
    
    
    def valida_velocita(self, velocita, metodo):
        """
        Valida che la velocità sia ragionevole
        La ISS viaggia tra 7.5 e 7.8 km/s
        """
        if velocita is None:
            return False
        
        # Range di validità (con margine)
        min_vel = 6.0  # km/s
        max_vel = 9.0  # km/s
        
        if min_vel <= velocita <= max_vel:
            return True
        else:
            self.scrivi_risultati(
                f"Errore {metodo}: Velocità fuori range ({velocita:.2f} km/s) - scartata"
            )
            return False
    
    
    
    def esegui(self):
        """Funzione principale - esegue il programma"""
        
        # Inizializza il file dei risultati
        with open(self.file_risultati, 'w') as f:
            f.write("=" * 80 + '\n')
            f.write("ASTRO PI MISSION SPACE LAB - CALCOLO VELOCITÀ ISS\n")
            f.write("=" * 80 + '\n\n')
        
        # Tempo di esecuzione
        tempo_inizio = datetime.now()
        tempo_fine = tempo_inizio + timedelta(minutes=DURATA_MINUTI)
        
        self.scrivi_risultati(f"Inizio: {tempo_inizio.strftime('%Y-%m-%d %H:%M:%S')}")
        self.scrivi_risultati(f"Fine prevista: {tempo_fine.strftime('%Y-%m-%d %H:%M:%S')}")
        self.scrivi_risultati(f"Durata: {DURATA_MINUTI} minuti")
        self.scrivi_risultati(f"Intervallo foto: {INTERVALLO_FOTO} secondi\n")
        self.scrivi_risultati("-" * 80 + "\n")
        
        foto_numero = 0
        
        try:
            # Loop principale
            while datetime.now() < tempo_fine:
                
                # ===== ACQUISIZIONE DATI =====
                
                # Scatta foto
                foto_path, foto_tempo = self.scatta_foto(foto_numero)
                self.scrivi_risultati(f"\n📸 Foto {foto_numero}: {foto_path.name}")
                
                # Salva i dati della foto
                self.foto_lista.append({
                    'path': foto_path,
                    'tempo': foto_tempo,
                    'numero': foto_numero
                })
                
                # Calcola velocità angolare
                vel_ang = self.calcola_velocita_angolare()
                if vel_ang and self.valida_velocita(vel_ang, "ANGOLARE"):
                    self.velocita_angolare.append(vel_ang)
                    self.scrivi_risultati(
                        f"ANGOLARE: {vel_ang:.4f} km/s "
                        f"({vel_ang*3600:.0f} km/h)"
                    )
                
                # ===== CALCOLO VELOCITÀ (se abbiamo almeno 2 foto) =====
                
                if len(self.foto_lista) >= 2:
                    foto1_data = self.foto_lista[-2]
                    foto2_data = self.foto_lista[-1]
                    
                    # Tempo tra le foto
                    tempo_diff = (foto2_data['tempo'] - foto1_data['tempo']).total_seconds()
                    
                    # METODO 1: GPS
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
                    
                    # METODO 2: FOTO
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
                
                # Attendi prima della prossima foto
                sleep(INTERVALLO_FOTO)
        
        except Exception as e:
            self.scrivi_risultati(f"\nERRORE: {e}")
            import traceback
            self.scrivi_risultati(traceback.format_exc())
        
        finally:
            # ===== ANALISI FINALE =====
            self.analizza_risultati()
            
            self.scrivi_risultati("\n" + "=" * 80)
            self.scrivi_risultati("PROGRAMMA COMPLETATO")
            self.scrivi_risultati("=" * 80)
    
    
    def analizza_risultati(self):
        """Analizza e mostra i risultati finali"""
        
        self.scrivi_risultati("\n" + "=" * 80)
        self.scrivi_risultati("ANALISI FINALE")
        self.scrivi_risultati("=" * 80 + "\n")
        
        # Velocità reale ISS per confronto
        VELOCITA_REALE = 7.66
        
        # Analizza ogni metodo
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
        
        # Calcola la media combinata (se abbiamo dati da più metodi)
        if len(risultati_finali) > 0:
            self.scrivi_risultati("\n" + "-" * 80)
            self.scrivi_risultati("RISULTATO FINALE COMBINATO")
            self.scrivi_risultati("-" * 80 + "\n")
            
            # Media ponderata (pesando inversamente agli errori)
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
                self.scrivi_risultati("\nRISULTATO: ECCELLENTE! (errore < 1%)")
            elif errore_finale_perc < 3:
                self.scrivi_risultati("\nRISULTATO: MOLTO BUONO! (errore < 3%)")
            elif errore_finale_perc < 5:
                self.scrivi_risultati("\nRISULTATO: BUONO! (errore < 5%)")
            else:
                self.scrivi_risultati("\nRISULTATO: DA MIGLIORARE (errore > 5%)")
        
        else:
            self.scrivi_risultati("\nNessun dato valido raccolto")


# ============================================================================
#   MAIN
# ============================================================================

def main():
    """Punto di ingresso del programma"""
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
