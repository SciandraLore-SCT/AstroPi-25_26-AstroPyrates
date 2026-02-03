# idea fare un filtro di kalman per python 3.11
# senza librerie infinite e per le foto e per il giroscopio 

import numpy as np

def kalman_filter_fusion(measurement_photo, measurement_gyro, 
                        state_estimate, state_variance,
                        process_variance=0.005, 
                        measurement_variance_photo=0.5,
                        measurement_variance_gyro=0.3):
    """
    Docstring for kalman_filter_fusion
    
    :param measurement_photo: Velocity from photo pixel calc
    :param measurement_gyro: Velocity from angular calc
    :param state_estimate: Current Velocity estimate - previous
    :param state_variance: Current estimate Error
    :param process_variance: The Rumor of the sensor
    :param measurement_variance_photo: Description
    :param measurement_variance_gyro: Description
    
    
    
    predicted_state = state_estimate 
    predicted_variance = state_variance + process_variance  
    
    measurements = []
    variances = []
    
    # Check which measurements are available
    if measurement_photo is not None:
        measurements.append(measurement_photo)
        variances.append(measurement_variance_photo)
    
    if measurement_gyro is not None:
        measurements.append(measurement_gyro)
        variances.append(measurement_variance_gyro)
    
    
    # Whit no measurements available, return prediction only
    if len(measurements) == 0:
        return predicted_state, predicted_variance
    
    
    # The Update - with at least one value

    if len(measurements) == 1:
        
        measurement = measurements[0]
        meas_variance = variances[0]
        # A single measurement to greatly simplify everything (the list is simply a single value)

        
        kalman_gain = predicted_variance / (predicted_variance + meas_variance)
        
        updated_state = predicted_state + kalman_gain * (measurement - predicted_state)        
        updated_variance = (1 - kalman_gain) * predicted_variance
    
    else:
        z = np.array(measurements)
        R = np.diag(variances)  
        
        H = np.ones((len(measurements), 1))
        y = z - (H @ np.array([[predicted_state]])).flatten()
        S = H @ np.array([[predicted_variance]]) @ H.T + R
        
        K = (np.array([[predicted_variance]]) @ H.T @ np.linalg.inv(S)).flatten()
        
        updated_state = predicted_state + K @ y
        
        updated_variance = predicted_variance - K @ H * predicted_variance
        
        updated_state = float(updated_state)
        updated_variance = float(updated_variance)
    
    
    return updated_state, updated_variance

