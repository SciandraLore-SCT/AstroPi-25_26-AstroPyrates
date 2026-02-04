class ISSSpeedCalculator:
    def __init__(self):
        # ... parametri esistenti ...
        
        # Stato Kalman
        self.kalman_state = self.SPEED_ISS  # Stima iniziale
        self.kalman_variance = 100.0  # Incertezza iniziale
        
    def esegui(self):
        n = (self.DURATA_MINUTI * 60) / self.TIME_INTERVAL
        n = int(n)
        self.take_picture(n)

        images = [f"image{i}.jpg" for i in range(n)]
        velocita_list = []
        velocita_filtrate_kalman = []

        for k in range(n - 1):
            pixel_shift = self.distanza_tra_immagini(images[k], images[k + 1])
            
            if pixel_shift == 0:
                # Nessuna misura disponibile
                self.kalman_state, self.kalman_variance = kalman_filter_fusion(
                    None, None, 
                    self.kalman_state, 
                    self.kalman_variance
                )
                continue

            distanza_metri = self.pixel_to_meters(pixel_shift, image_width=4056)
            distanza_metri *= self.CORRECTION_FACTOR  # ← Applica correzione!
            
            velocita = distanza_metri / self.TIME_INTERVAL

            # Filtro fisico largo
            if velocita < 500 or velocita > 10000:
                continue

            # Applica Kalman PRIMA del filtro stretto
            self.kalman_state, self.kalman_variance = kalman_filter_fusion(
                velocita, None,  # Solo misura photo, gyro=None
                self.kalman_state,
                self.kalman_variance,
                process_variance=0.001,  # ISS ha velocità molto stabile
                measurement_variance_photo=500.0  # Adatta in base ai test
            )
            
            velocita_filtrate_kalman.append(self.kalman_state)

            # Filtro ISS realistico (opzionale, ora Kalman dovrebbe gestirlo)
            if 6000 <= self.kalman_state <= 8000:
                velocita_list.append(self.kalman_state)

        # Usa le velocità filtrate
        if len(velocita_list) > 0:
            velocita_media = np.mean(velocita_list)
            velocita_kalman_finale = self.kalman_state
            errore = abs(velocita_media - self.SPEED_ISS) / self.SPEED_ISS * 100

            self.scrivi_risultatiFinale(
                f"\nVelocità media stimata: {velocita_media:.2f} m/s "
                f"({velocita_media*3.6:.0f} km/h)\n"
                f"Velocità Kalman finale: {velocita_kalman_finale:.2f} m/s\n"
                f"Errore medio: {errore:.2f}%\n"
                f"Misure valide: {len(velocita_list)}/{n-1}"
            )