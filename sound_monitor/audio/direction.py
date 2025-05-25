# TODO from claude, review

import math
from typing import NamedTuple

import numpy as np
from scipy.signal import correlate

from sound_monitor.config import Config

_config = Config.get()


class Direction(NamedTuple):
    """
    Direction finding result as a unit vector with separate confidence estimates.
    
    The direction is stored as XYZ components of a unit vector, with
    azimuth/elevation available as properties. Confidence estimates
    are separated and calculated rigorously from fit residuals.
    
    x, y, z: Unit vector components pointing toward the sound source
    azimuth_confidence: Uncertainty in azimuth (degrees), from fit analysis
    elevation_confidence: Uncertainty in elevation (degrees), from fit analysis
    
    Notes:
    - Both confidence values are calculated from least-squares residuals
    - Large confidence values naturally indicate poor data quality
    - XY components come from whichever fit (2D or 3D) gives better accuracy
    """
    x: float
    y: float  
    z: float
    azimuth_confidence: float
    elevation_confidence: float

    @property
    def azimuth(self) -> float:
        """
        Azimuth in degrees (0° = north, 90° = east, etc.)
        Includes the azimuth offset from config.
        """
        # Convert XY to angle, apply config offset
        raw_angle = math.degrees(math.atan2(self.y, self.x))
        return (raw_angle + _config.direction_azimuth_offset) % 360

    @property 
    def elevation(self) -> float:
        """
        Elevation in degrees (-90° = straight down, +90° = straight up)
        Will be noisy with flat microphone array.
        """
        return math.degrees(math.asin(np.clip(self.z, -1.0, 1.0)))


def find_direction(audio_data: np.ndarray) -> Direction | None:
    """
    Find the direction of a sound using two-stage TDOA analysis.
    
    ALGORITHM:
    1. Find time delays between all microphone pairs using correlation
    2. Fit both 2D (azimuth-only) and 3D (full direction) using least-squares
    3. Use whichever gives better XY accuracy, with rigorous error propagation
    4. Let residual magnitude naturally indicate data quality
    
    Args:
        audio_data: shape (samples, 7) - filtered audio from all 7 mics
                   Should contain a single clear sound event
    
    Returns:
        Direction with unit vector and confidence estimates, or None if calculation fails
    
    Notes:
        - Large confidence values indicate poor data quality
        - Returns None only for matrix singularity (very rare)
        - No pre-filtering - let the math handle data quality assessment
    """
    
    if audio_data.shape[1] != 7:
        raise ValueError(f"Expected 7 channels, got {audio_data.shape[1]}")
    
    if len(audio_data) < 10:
        # Too short to get reliable correlations
        return None
    
    # Get microphone positions and constants
    mic_positions = np.array(_config.uma8_mic_positions)
    sample_rate = _config.uma8_sample_rate
    speed_of_sound = _config.speed_of_sound
    
    # STEP 1: Find time delays between all microphone pairs
    delays = []
    baselines = []  # mic position differences
    
    for i in range(7):
        for j in range(i + 1, 7):
            mic1_audio = audio_data[:, i]
            mic2_audio = audio_data[:, j]
            
            # Cross-correlation to find time delay
            correlation = correlate(mic1_audio, mic2_audio, mode='full')
            peak_idx = np.argmax(np.abs(correlation))
            
            # Convert to time delay (positive = sound reached mic2 first)
            delay_samples = peak_idx - (len(mic1_audio) - 1)
            delay_seconds = delay_samples / sample_rate
            
            # Physical constraint: delay can't exceed distance/speed_of_sound
            mic_distance = np.linalg.norm(mic_positions[j] - mic_positions[i])
            max_delay = mic_distance / speed_of_sound
            delay_seconds = np.clip(delay_seconds, -max_delay, max_delay)
            
            # Store results  
            baseline = mic_positions[j] - mic_positions[i]  # vector from mic1 to mic2
            delays.append(delay_seconds * speed_of_sound)   # convert to distance units
            baselines.append(baseline)
    
    if len(delays) < 3:
        # Need at least 3 constraints for meaningful fit
        return None
    
    delays = np.array(delays)
    baselines = np.array(baselines)  # shape: (n_pairs, 3)
    
    # STEP 2: 2D least-squares fit (z=0 constraint)  
    A_2d = baselines[:, :2]  # just x,y components
    
    try:
        direction_2d, residuals_2d, rank_2d, s_2d = np.linalg.lstsq(A_2d, delays, rcond=None)
        
        if rank_2d < 2:
            # Insufficient rank for 2D solution
            return None
            
        # Normalize to unit vector
        direction_2d_norm = direction_2d / np.linalg.norm(direction_2d)
        
        # Calculate covariance matrix and azimuth confidence
        dof_2d = len(delays) - 2  # degrees of freedom
        if len(residuals_2d) > 0 and dof_2d > 0:
            residual_variance_2d = residuals_2d[0] / dof_2d
            cov_matrix_2d = np.linalg.inv(A_2d.T @ A_2d) * residual_variance_2d
            
            # Error propagation for azimuth = atan2(y, x)
            x, y = direction_2d_norm
            r_squared = x*x + y*y
            d_az_dx = -y / r_squared
            d_az_dy = x / r_squared
            
            azimuth_variance_2d = (d_az_dx**2 * cov_matrix_2d[0,0] + 
                                  d_az_dy**2 * cov_matrix_2d[1,1] + 
                                  2 * d_az_dx * d_az_dy * cov_matrix_2d[0,1])
            
            azimuth_confidence_2d = math.degrees(math.sqrt(abs(azimuth_variance_2d)))
        else:
            azimuth_confidence_2d = float('inf')  # no residual info
            
    except np.linalg.LinAlgError:
        return None
    
    # STEP 3: 3D least-squares fit
    try:
        direction_3d, residuals_3d, rank_3d, s_3d = np.linalg.lstsq(baselines, delays, rcond=None)
        
        if rank_3d < 3:
            # Use 2D results only
            x_final, y_final = direction_2d_norm
            z_final = 0.0
            azimuth_confidence_final = azimuth_confidence_2d
            elevation_confidence_final = float('inf')
        else:
            # Normalize 3D solution
            direction_3d_norm = direction_3d / np.linalg.norm(direction_3d)
            
            # Calculate 3D covariance matrix
            dof_3d = len(delays) - 3
            if len(residuals_3d) > 0 and dof_3d > 0:
                residual_variance_3d = residuals_3d[0] / dof_3d
                cov_matrix_3d = np.linalg.inv(baselines.T @ baselines) * residual_variance_3d
                
                # Calculate XY uncertainty from 3D fit
                x3d, y3d, z3d = direction_3d_norm
                r_squared_3d = x3d*x3d + y3d*y3d
                d_az_dx_3d = -y3d / r_squared_3d
                d_az_dy_3d = x3d / r_squared_3d
                
                azimuth_variance_3d = (d_az_dx_3d**2 * cov_matrix_3d[0,0] + 
                                      d_az_dy_3d**2 * cov_matrix_3d[1,1] + 
                                      2 * d_az_dx_3d * d_az_dy_3d * cov_matrix_3d[0,1])
                
                azimuth_confidence_3d = math.degrees(math.sqrt(abs(azimuth_variance_3d)))
                
                # Calculate elevation confidence: elevation = asin(z)
                # d(elevation)/dz = 1/sqrt(1-z^2)
                z_clipped = np.clip(z3d, -0.99, 0.99)  # avoid singularity
                d_el_dz = 1.0 / math.sqrt(1 - z_clipped**2)
                elevation_variance = d_el_dz**2 * cov_matrix_3d[2,2]
                elevation_confidence_final = math.degrees(math.sqrt(abs(elevation_variance)))
                
                # Compare 2D vs 3D accuracy for XY components
                if azimuth_confidence_2d <= azimuth_confidence_3d:
                    # 2D fit gives better azimuth accuracy
                    x_final, y_final = direction_2d_norm
                    azimuth_confidence_final = azimuth_confidence_2d
                else:
                    # 3D fit actually helps azimuth accuracy  
                    x_final, y_final = x3d, y3d
                    azimuth_confidence_final = azimuth_confidence_3d
                
                z_final = z3d
                
            else:
                # No residual info from 3D - fall back to 2D
                x_final, y_final = direction_2d_norm
                z_final = 0.0
                azimuth_confidence_final = azimuth_confidence_2d
                elevation_confidence_final = float('inf')
                
    except np.linalg.LinAlgError:
        # 3D fit failed - use 2D results
        x_final, y_final = direction_2d_norm
        z_final = 0.0
        azimuth_confidence_final = azimuth_confidence_2d
        elevation_confidence_final = float('inf')
    
    return Direction(
        x=float(x_final),
        y=float(y_final), 
        z=float(z_final),
        azimuth_confidence=azimuth_confidence_final,
        elevation_confidence=elevation_confidence_final
    )


def _test_direction_finding():
    """Simple test function for development/debugging."""
    # Create fake audio data with a known direction
    samples = 1000
    sample_rate = _config.uma8_sample_rate
    
    # Simulate sound coming from 45° azimuth (northeast)
    true_direction = np.array([math.cos(math.radians(45)), math.sin(math.radians(45)), 0])
    
    fake_audio = np.random.randn(samples, 7) * 0.01  # background noise
    signal = np.sin(2 * np.pi * 1000 * np.linspace(0, samples/sample_rate, samples))
    
    # Add signal to each mic with appropriate delay
    mic_positions = np.array(_config.uma8_mic_positions)
    speed_of_sound = _config.speed_of_sound
    
    for mic_idx in range(7):
        # Calculate expected delay for this direction
        delay_distance = np.dot(true_direction, mic_positions[mic_idx])
        delay_seconds = delay_distance / speed_of_sound
        delay_samples = int(delay_seconds * sample_rate)
        
        # Add delayed signal
        if delay_samples >= 0 and delay_samples < samples:
            fake_audio[delay_samples:, mic_idx] += signal[:samples-delay_samples]
        elif delay_samples < 0 and -delay_samples < samples:
            fake_audio[:samples+delay_samples, mic_idx] += signal[-delay_samples:]
    
    result = find_direction(fake_audio)
    if result:
        print(f"True direction: 45.0°")
        print(f"Detected: {result.azimuth:.1f}° ± {result.azimuth_confidence:.1f}° (azimuth)")
        print(f"Elevation: {result.elevation:.1f}° ± {result.elevation_confidence:.1f}° (elevation)")
        print(f"Unit vector: ({result.x:.3f}, {result.y:.3f}, {result.z:.3f})")
    else:
        print("Direction calculation failed")


if __name__ == "__main__":
    _test_direction_finding()
