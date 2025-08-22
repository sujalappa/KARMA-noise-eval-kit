#!/usr/bin/env python3
"""
Configurable Noise Intensity System

Defines specific step values for each noise type to enable
systematic and reproducible noise robustness evaluation.
"""

import logging
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

# === Predefined Noise Intensity Steps ===

NOISE_INTENSITY_STEPS = {
    "background_noise": [5, 10, 20, 40],      # dB SNR values
    "gaussian": [0.001, 0.005, 0.009, 0.013], # amplitude values
    "color_noise": [5, 10, 20, 40],           # dB SNR values
    "short_noise": [1, 4, 7, 10],             # dB SNR values
    "clip": [0.5, 0.6, 0.8, 0.9]              # threshold values
}

# === Intensity Level Names ===
INTENSITY_LEVELS = ["low", "medium", "high", "extreme"]

def get_noise_intensity_value(noise_type: str, intensity: Union[str, int, float]) -> Union[float, int]:
    """
    Get the actual noise intensity value for a given noise type and intensity specification.
    
    Args:
        noise_type: Type of noise (background_noise, gaussian, etc.)
        intensity: Can be:
                  - String: "low", "medium", "high", "extreme"
                  - Int: Index (0, 1, 2, 3)
                  - Float: Direct value
    
    Returns:
        Actual intensity value to use
    """
    if noise_type not in NOISE_INTENSITY_STEPS:
        logger.warning(f"Unknown noise type: {noise_type}")
        return None
    
    steps = NOISE_INTENSITY_STEPS[noise_type]
    
    # Handle string intensity levels
    if isinstance(intensity, str):
        if intensity.lower() in INTENSITY_LEVELS:
            index = INTENSITY_LEVELS.index(intensity.lower())
            if index < len(steps):
                value = steps[index]
                logger.info(f"🎚️ {noise_type} intensity '{intensity}' = {value}")
                return value
            else:
                logger.warning(f"Intensity level '{intensity}' not available for {noise_type}")
                return steps[-1]  # Return highest available
        else:
            logger.warning(f"Unknown intensity level: {intensity}")
            return steps[1]  # Default to medium
    
    # Handle numeric index
    elif isinstance(intensity, int):
        if 0 <= intensity < len(steps):
            value = steps[intensity]
            logger.info(f"{noise_type} intensity index {intensity} = {value}")
            return value
        else:
            logger.warning(f"Intensity index {intensity} out of range for {noise_type}")
            return steps[1]  # Default to medium
    
    # Handle direct numeric value
    elif isinstance(intensity, (float, int)):
        if intensity in steps:
            logger.info(f"{noise_type} using exact intensity = {intensity}")
            return intensity
        else:
            # Find closest value
            closest = min(steps, key=lambda x: abs(x - intensity))
            logger.info(f"{noise_type} intensity {intensity} rounded to closest step = {closest}")
            return closest
    
    else:
        logger.warning(f"Invalid intensity format: {intensity}")
        return steps[1]  # Default to medium

def list_available_intensities(noise_type: str = None) -> Dict[str, List]:
    """
    List available intensity values for noise types.
    
    Args:
        noise_type: Specific noise type to list, or None for all
        
    Returns:
        Dictionary of noise types and their available intensities
    """
    if noise_type:
        if noise_type in NOISE_INTENSITY_STEPS:
            return {noise_type: NOISE_INTENSITY_STEPS[noise_type]}
        else:
            return {}
    else:
        return NOISE_INTENSITY_STEPS.copy()

def parse_noise_specification(noise_spec: str) -> tuple:
    """
    Parse noise specification string into noise type and intensity.
    
    Args:
        noise_spec: String like "background_noise:20" or "gaussian:medium"
        
    Returns:
        Tuple of (noise_type, intensity)
    """
    if ":" in noise_spec:
        noise_type, intensity_str = noise_spec.split(":", 1)
        
        # Try to convert to number
        try:
            intensity = float(intensity_str)
            if intensity.is_integer():
                intensity = int(intensity)
        except ValueError:
            # Keep as string (level name)
            intensity = intensity_str
        
        logger.debug(f"Parsed '{noise_spec}' -> type: {noise_type}, intensity: {intensity}")
        return noise_type.strip(), intensity
    else:
        # No intensity specified, use default (medium)
        logger.debug(f"No intensity specified for '{noise_spec}', using medium")
        return noise_spec.strip(), "medium"

def validate_noise_intensity(noise_type: str, intensity: Union[str, int, float]) -> bool:
    """
    Validate if the specified intensity is valid for the noise type.
    
    Args:
        noise_type: Type of noise
        intensity: Intensity specification
        
    Returns:
        True if valid, False otherwise
    """
    if noise_type not in NOISE_INTENSITY_STEPS:
        return False
    
    steps = NOISE_INTENSITY_STEPS[noise_type]
    
    if isinstance(intensity, str):
        return intensity.lower() in INTENSITY_LEVELS
    elif isinstance(intensity, int):
        return 0 <= intensity < len(steps)
    elif isinstance(intensity, (float, int)):
        return True  # We can find closest value
    else:
        return False
