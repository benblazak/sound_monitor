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
    are separated because azimuth (from 2D fit) is much more reliable
    than elevation (from 3D fit) with this flat array.
    
    x, y, z: Unit vector components pointing toward the sound source
    azimuth_confidence: Uncertainty in azimuth (degrees), from rigorous 2D analysis
    elevation_confidence: Uncertainty in elevation (degrees), from geometric approximation
    
    Notes:
    - Azimuth confidence is mathematically rigorous (2D least-squares + error propagation)
    - Elevation confidence is approximate (will be large with flat array)
    - For most purposes, trust azimuth, be skeptical of elevation
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
        WARNING: Will be very noisy with flat microphone array.
        """
        return math.degrees(math.asin(np.clip(self.z, -1.0, 1.0)))


def find_direction(audio_data: np.ndarray) -> Direction:
    """
    Find the direction of a sound using two-stage TDOA analysis.
    
    ALGORITHM:
    1. Find time delays between all microphone pairs using correlation
    2. Stage 1: 2D least-squares fit (z=0) for clean azimuth estimate
    3. Stage 2: 3D fit for elevation estimate  
    4. Rigorous error propagation for azimuth, geometric approximation for elevation
    
    Args:
        audio_data: shape (samples, 7) - filtered audio from all 7 mics
                   Should contain a single clear sound event
    
    Returns:
        Direction with unit vector (x,y,z) and separate confidence estimates
    
    Notes:
        - Optimized for short clips containing single transient sounds (barks)
        - Azimuth confidence is rigorous and reliable
        - Elevation confidence is approximate and will be large
    """
    
    if audio_data.shape[1] != 7:
        raise ValueError(f"Expected 7 channels, got {audio_data.shape[1]}")
    
    if len(audio_data) < 10:
        # Too short to get reliable correlations
        return Direction(x=1.0, y=0.0, z=0.0, azimuth_confidence=180.0, elevation_confidence=180.0)
    
    # Get microphone positions and constants
    mic_positions = np.array(_config.uma8_mic_positions)
    sample_rate = _config.uma8_sample_rate
    speed_of_sound = _config.speed_of_sound
    
    # STEP 1: Find time delays between all microphone pairs
    delays = []
    baselines = []  # mic position differences
    weights = []    # correlation quality weights
    
    for i in range(7):
        for j in range(i + 1, 7):
            mic1_audio = audio_data[:, i]
            mic2_audio = audio_data[:, j]
            
            # Cross-correlation to find time delay
            correlation = correlate(mic1_audio, mic2_audio, mode='full')
            
            # Check correlation quality before trusting the result
            max_correlation = np.max(np.abs(correlation))
            noise_floor = np.std(correlation)
            correlation_snr = max_correlation / noise_floor if noise_floor > 0 else 0
            
            # Only use correlations with good signal-to-noise ratio
            if correlation_snr < 2.0:  # Require at least 2:1 SNR
                continue
                
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
            weights.append(correlation_snr)  # weight by correlation quality
    
    if len(delays) < 3:
        # Not enough good correlations
        return Direction(x=1.0, y=0.0, z=0.0, azimuth_confidence=180.0, elevation_confidence=180.0)
    
    delays = np.array(delays)
    baselines = np.array(baselines)  # shape: (n_pairs, 3)
    weights = np.array(weights)
    
    # STEP 2: 2D least-squares fit for azimuth (z=0 constraint)
    # Equation: baseline @ direction = delay
    # For 2D: [baseline_x, baseline_y] @ [x, y] = delay
    
    A_2d = baselines[:, :2]  # just x,y components
    b_2d = delays
    W = np.diag(weights)     # weight matrix
    
    try:
        # Weighted least squares: (A^T W A) x = A^T W b
        AtWA = A_2d.T @ W @ A_2d
        AtWb = A_2d.T @ W @ b_2d
        
        direction_2d = np.linalg.solve(AtWA, AtWb)
        
        # Normalize to unit vector (azimuth only)
        direction_2d_norm = direction_2d / np.linalg.norm(direction_2d)
        x, y = direction_2d_norm
        
        # Calculate azimuth confidence using error propagation
        residuals_2d = A_2d @ direction_2d - b_2d
        residual_variance = np.sum(weights * residuals_2d**2) / (len(delays) - 2)  # weighted variance
        
        # Covariance matrix of direction estimate
        try:
            cov_matrix = np.linalg.inv(AtWA) * residual_variance
            var_x, var_y = cov_matrix[0,0], cov_matrix[1,1]
            cov_xy = cov_matrix[0,1]
            
            # Propagate to azimuth uncertainty: azimuth = atan2(y, x)
            # d(azimuth)/dx = -y/(x^2 + y^2), d(azimuth)/dy = x/(x^2 + y^2)
            r_squared = x*x + y*y
            d_az_dx = -y / r_squared
            d_az_dy = x / r_squared
            
            azimuth_variance = (d_az_dx**2 * var_x + 
                              d_az_dy**2 * var_y + 
                              2 * d_az_dx * d_az_dy * cov_xy)
            
            azimuth_confidence = math.degrees(math.sqrt(abs(azimuth_variance)))
            azimuth_confidence = max(1.0, min(azimuth_confidence, 180.0))  # reasonable bounds
            
        except np.linalg.LinAlgError:
            azimuth_confidence = 30.0  # fallback
            
    except np.linalg.LinAlgError:
        # Fallback to simple estimate
        x, y = 1.0, 0.0
        azimuth_confidence = 180.0
    
    # STEP 3: 3D fit for elevation estimate
    try:
        # Full 3D weighted least squares
        AtWA_3d = baselines.T @ W @ baselines
        AtWb_3d = baselines.T @ W @ b_2d
        
        direction_3d = np.linalg.solve(AtWA_3d, AtWb_3d)
        direction_3d_norm = direction_3d / np.linalg.norm(direction_3d)
        z = direction_3d_norm[2]
        
        # Simple elevation confidence estimate (geometric approximation)
        residuals_3d = baselines @ direction_3d - b_2d
        rms_residual = math.sqrt(np.mean(residuals_3d**2))
        
        # Convert residual to angular error (rough approximation)
        array_radius = 0.04  # meters (approximate)
        angular_error_rad = rms_residual / array_radius
        elevation_confidence = math.degrees(angular_error_rad)
        
        # For flat arrays, elevation uncertainty is inherently large
        elevation_confidence = max(elevation_confidence, 30.0)  # minimum reasonable value
        elevation_confidence = min(elevation_confidence, 180.0)  # maximum bound
        
    except np.linalg.LinAlgError:
        z = 0.0
        elevation_confidence = 180.0
    
    return Direction(
        x=float(x),
        y=float(y), 
        z=float(z),
        azimuth_confidence=azimuth_confidence,
        elevation_confidence=elevation_confidence
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
    print(f"True direction: 45.0°")
    print(f"Detected: {result.azimuth:.1f}° ± {result.azimuth_confidence:.1f}° (azimuth)")
    print(f"Elevation: {result.elevation:.1f}° ± {result.elevation_confidence:.1f}° (elevation)")
    print(f"Unit vector: ({result.x:.3f}, {result.y:.3f}, {result.z:.3f})")


if __name__ == "__main__":
    _test_direction_finding()
