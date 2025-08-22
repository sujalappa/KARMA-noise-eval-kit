import numpy as np
from audiomentations import Compose, AddBackgroundNoise, AddGaussianNoise, AddShortNoises, AddColorNoise, Clip
import logging
import os
from typing import Optional
from pathlib import Path
import os
def create_clip_augmenter(intensity=None):
    """Create clip augmenter with configurable intensity."""
    if intensity is not None:
        threshold = get_noise_intensity_value("clip", intensity)
    else:
        threshold = 0.5  # Default medium value
    
    # Clip values between -threshold and +threshold
    return Clip(
        a_min=-threshold,
        a_max=threshold,
        p=1.0
    )


logger = logging.getLogger(__name__)

# Import noise selection logger and file selector
try:
    from karma.utils.noise.noise_selection_logger import noise_logger
    from karma.utils.noise.noise_file_selector import noise_selector
    from karma.utils.noise.noise_intensity_config import get_noise_intensity_value, NOISE_INTENSITY_STEPS
except ImportError:
    logger.warning("Could not import noise selection components")
    noise_logger = None
    noise_selector = None

def get_deterministic_noise_file(audio_file_id: str, noise_type: str, dataset_name: str = "eka_med_asr") -> Optional[str]:
    """
    Get deterministic noise file for a given audio file and noise type.
    Ensures same audio file always gets same noise file across all model evaluations.
    """
    if noise_type not in ["background_noise", "short_noise"]:
        return None
        
    # Load or create mapping
    mapping_file = f"noise_mappings/{dataset_name}_noise_mapping.json"
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
                
            # Get specific noise file for this audio + noise type
            if audio_file_id in mapping_data.get('mappings', {}):
                specific_file = mapping_data['mappings'][audio_file_id].get(noise_type)
                if specific_file and os.path.exists(specific_file):
                    logger.info(f"Using deterministic noise: {audio_file_id} + {noise_type} = {os.path.basename(specific_file)}")
                    return specific_file
                    
        except Exception as e:
            logger.warning(f"Could not load noise mapping: {e}")
    
    logger.info(f"No deterministic mapping found for {audio_file_id} + {noise_type}, using random selection")
    return None

def create_gaussian_noise_augmenter(intensity=None):
    """Create gaussian noise augmenter with configurable intensity."""
    if intensity is not None:
        amplitude = get_noise_intensity_value("gaussian", intensity)
    else:
        amplitude = 0.005  # Default medium value
    
    return AddGaussianNoise(
        min_amplitude=amplitude,
        max_amplitude=amplitude,  # Use same value for consistent intensity
        p=1.0
    )

def create_short_noise_augmenter(specific_file: str = None, audio_file_id: str = None, dataset_name: str = None, intensity=None):
    """Create short noise augmenter with configurable intensity."""
    if intensity is not None:
        snr_db = get_noise_intensity_value("short_noise", intensity)
    else:
        snr_db = 4  # Default medium value
    
    if specific_file:
        # Use specific file for deterministic behavior
        augmenter = AddShortNoises(
            sounds_path=specific_file,
            min_snr_db=snr_db,
            max_snr_db=snr_db,  # Use same value for consistent intensity
            noise_rms="relative_to_whole_input",
            min_time_between_sounds=1.0,
            max_time_between_sounds=4.0,
            p=1.0
        )
        
        # Log the selection if we have the info
        if noise_logger and audio_file_id and dataset_name:
            noise_logger.log_noise_selection(dataset_name, "short_noise", audio_file_id, specific_file)
            
        return augmenter
    else:
        # Use repo-relative noises directory for random selection
        
        base_dir = os.path.dirname(__file__)
        noises_dir = os.path.join(base_dir, "noises", "Short noises")
        return AddShortNoises(
            sounds_path=noises_dir,
            min_snr_db=snr_db,
            max_snr_db=snr_db,  # Use same value for consistent intensity
            noise_rms="relative_to_whole_input",
            min_time_between_sounds=1.0,
            max_time_between_sounds=4.0,
            p=1.0
        )

def create_color_noise_augmenter(intensity=None):
    """Create color noise augmenter with configurable intensity."""
    if intensity is not None:
        snr_db = get_noise_intensity_value("color_noise", intensity)
    else:
        snr_db = 10  # Default medium value
    
    return AddColorNoise(
        p=1.0,
        min_snr_db=snr_db,
        max_snr_db=snr_db,  # Use same value for consistent intensity
        min_f_decay=-3,     # Keep decay range moderate
        max_f_decay=3
    )

def create_background_noise_augmenter(specific_file: str = None, audio_file_id: str = None, dataset_name: str = None, intensity=None):
    """Create background noise augmenter with configurable intensity."""
    if intensity is not None:
        snr_db = get_noise_intensity_value("background_noise", intensity)
    else:
        snr_db = 10  # Default medium value
    
    if specific_file:
        # Use specific file for deterministic behavior
        augmenter = AddBackgroundNoise(
            sounds_path=specific_file,
            min_snr_db=snr_db,
            max_snr_db=snr_db,  # Use same value for consistent intensity
            noise_rms="relative_to_whole_input",
            p=1.0
        )
        
        # Log the selection if we have the info
        if noise_logger and audio_file_id and dataset_name:
            noise_logger.log_noise_selection(dataset_name, "background_noise", audio_file_id, specific_file)
            
        return augmenter
    else:
        # Use repo-relative noises directory for random selection
        import os
        base_dir = os.path.dirname(__file__)
        noises_dir = os.path.join(base_dir, "noises", "Background noises")
        return AddBackgroundNoise(
            sounds_path=noises_dir,
            min_snr_db=snr_db,
            max_snr_db=snr_db,  # Use same value for consistent intensity
            noise_rms="relative_to_whole_input",
            p=1.0
        )

# === Available noise types ===
NOISE_TYPES = {
    "gaussian": create_gaussian_noise_augmenter,
    "short_noise": create_short_noise_augmenter,
    "color_noise": create_color_noise_augmenter,
    "background_noise": create_background_noise_augmenter,
    "clip": create_clip_augmenter,
    "clean": lambda specific_file=None: None  # No noise
}

def apply_augmentations(waveform: np.ndarray, sample_rate: int, noise_type: str, 
                       audio_file_id: str = None, specific_noise_file: str = None, 
                       dataset_name: str = "ekacare/eka-medical-asr-evaluation-dataset",
                       intensity=None) -> np.ndarray:
    """
    Apply noise augmentations with per-sample deterministic selection and configurable intensity.
    
    For each sample:
    1. Check if noise file is already logged for this sample
    2. If not logged, select deterministically and log it
    3. All models will use the same noise file for the same sample
    4. Apply noise with specified intensity level
    
    Args:
        waveform: Audio waveform as numpy array
        sample_rate: Sample rate of the audio
        noise_type: Type of noise to apply
        audio_file_id: ID of the audio file (e.g., sample_001, sample_002)
        specific_noise_file: Optional specific noise file to use
        dataset_name: Name of the dataset for logging
        intensity: Noise intensity (string: "low"/"medium"/"high"/"extreme", number, or None for default)
    
    Returns:
        Augmented waveform
    """
    if noise_type == "clean" or noise_type not in NOISE_TYPES:
        if noise_type not in NOISE_TYPES and noise_type != "clean":
            logger.warning(f"Unknown noise type '{noise_type}', returning clean audio")
        return waveform.astype(np.float32)
    
    try:
        # For background_noise and short_noise, implement per-sample deterministic selection
        if noise_type in ["background_noise", "short_noise"] and audio_file_id:
            # Step 1: Check if we already have a logged noise file for this sample
            logged_noise_file = None
            if noise_logger:
                logged_noise_file = noise_logger.get_logged_noise_file(dataset_name, noise_type, audio_file_id)
            
            # Step 2: If no logged file and no specific file provided, select deterministically
            if not logged_noise_file and not specific_noise_file and noise_selector:
                selected_file = noise_selector.select_noise_file(noise_type, audio_file_id)
                if selected_file and noise_logger:
                    # Log this selection for future use
                    noise_logger.log_noise_selection(dataset_name, noise_type, audio_file_id, selected_file)
                    logged_noise_file = selected_file
            
            # Step 3: Use the deterministic file (logged or newly selected)
            noise_file_to_use = logged_noise_file or specific_noise_file
            
            if noise_file_to_use:
                # Create augmenter with the specific file and intensity
                if noise_type == "background_noise":
                    augmenter = create_background_noise_augmenter(
                        specific_file=noise_file_to_use,
                        audio_file_id=audio_file_id,
                        dataset_name=dataset_name,
                        intensity=intensity
                    )
                else:  # short_noise
                    augmenter = create_short_noise_augmenter(
                        specific_file=noise_file_to_use,
                        audio_file_id=audio_file_id,
                        dataset_name=dataset_name,
                        intensity=intensity
                    )
                
                intensity_str = f" (intensity: {intensity})" if intensity else ""
                logger.info(f"Applied {noise_type} to {audio_file_id}: {os.path.basename(noise_file_to_use)}{intensity_str}")
            else:
                # Fallback to random selection with intensity
                if noise_type == "background_noise":
                    augmenter = create_background_noise_augmenter(intensity=intensity)
                else:  # short_noise
                    augmenter = create_short_noise_augmenter(intensity=intensity)
                intensity_str = f" (intensity: {intensity})" if intensity else ""
                logger.warning(f"Fallback to random {noise_type} for {audio_file_id}{intensity_str}")
        else:
            # For other noise types (gaussian, color_noise, clip) - use intensity
            if noise_type == "gaussian":
                augmenter = create_gaussian_noise_augmenter(intensity=intensity)
            elif noise_type == "color_noise":
                augmenter = create_color_noise_augmenter(intensity=intensity)
            elif noise_type == "clip":
                augmenter = create_clip_augmenter(intensity=intensity)
            else:
                augmenter = NOISE_TYPES[noise_type]()
            
            intensity_str = f" (intensity: {intensity})" if intensity else ""
            if audio_file_id:
                logger.info(f"Applied {noise_type} to {audio_file_id}{intensity_str}")
            else:
                logger.info(f"Applied {noise_type}{intensity_str}")
        
        # Apply the augmentation
        augmented = augmenter(samples=waveform, sample_rate=sample_rate)
        
        if augmented.ndim > 1:
            augmented = augmented.T
            
        return augmented.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Error applying {noise_type} noise: {e}")
        if audio_file_id:
            logger.error(f"Failed for audio {audio_file_id}")
        logger.info("Returning clean audio")
        return waveform.astype(np.float32)
 
