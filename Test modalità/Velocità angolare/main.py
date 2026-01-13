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


# Cartella base per salvare i dati
BASE_FOLDER = Path(__file__).parent.resolve()

DURATA_MINUTI = 10  
INTERVALLO_FOTO = 5  

# Altezza media della ISS in km (varia tra 400-420 km)
ALTEZZA_ISS_KM = 408
# Raggio della Terra in km
RAGGIO_TERRA_KM = 6371


class ISSSpeedCalculator:

    def __init__(self):
        
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
    

    
    def calcola_velocita_angolare(self):
        """
        Calcola la velocità usando il giroscopio
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
        
        try:
            self.camera.capture(str(nome_foto))
        except AttributeError:
            # Per utilizzare altre librerie nel caso inutilizzabili
            try:
                self.camera.take_photo(str(nome_foto))
            except:
                # Salva direttamente l'immagine invece che fare il tutto 
                import io
                from PIL import Image
                img = self.camera.capture_image()
                img.save(str(nome_foto))
            
            return nome_foto, timestamp
    
    
    def scrivi_risultati(self, messaggio):
        """Scrive i risultati sia su file che su console"""
        with open(self.file_risultati, 'a') as f:
            f.write(messaggio + '\n')
        
    
    
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
                
                
                # Scatta foto
                foto_path, foto_tempo = self.scatta_foto(foto_numero)
                self.scrivi_risultati(f"\Foto {foto_numero}: {foto_path.name}")
                
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