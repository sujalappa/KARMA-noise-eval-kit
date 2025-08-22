#!/usr/bin/env python3
"""
Noise File Selector for Deterministic Per-Sample Selection

This handles the random selection and logging of noise files
to ensure deterministic behavior across models.
"""

import os
import random
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class NoiseFileSelector:
    """Handles selection and logging of noise files for deterministic behavior."""
    
    def __init__(self, noise_base_path: str = None):
        # Default to repo-relative path: karma/utils/noise/noise_data
        if noise_base_path is None:
            this_dir = Path(__file__).parent
            self.noise_base_path = this_dir / "noises"
        else:
            self.noise_base_path = Path(noise_base_path)
        self.background_noise_path = self.noise_base_path / "Background noises"
        self.short_noise_path = self.noise_base_path / "Short noises"
        
        # Cache discovered files
        self._background_files: Optional[List[str]] = None
        self._short_noise_files: Optional[List[str]] = None
    
    def get_background_noise_files(self) -> List[str]:
        """Get list of all background noise files."""
        if self._background_files is None:
            if self.background_noise_path.exists():
                self._background_files = [
                    str(self.background_noise_path / f)
                    for f in os.listdir(self.background_noise_path)
                    if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))
                ]
            else:
                self._background_files = []
        return self._background_files
    
    def get_short_noise_files(self) -> List[str]:
        """Get list of all short noise files."""
        if self._short_noise_files is None:
            if self.short_noise_path.exists():
                self._short_noise_files = [
                    str(self.short_noise_path / f)
                    for f in os.listdir(self.short_noise_path)
                    if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))
                ]
            else:
                self._short_noise_files = []
        return self._short_noise_files
    
    def select_noise_file(self, noise_type: str, audio_file_id: str) -> Optional[str]:
        """
        Select a deterministic noise file for a given audio sample.
        Uses the audio_file_id as seed for reproducible random selection.
        
        Args:
            noise_type: "background_noise" or "short_noise"
            audio_file_id: ID of the audio file (e.g., "sample_001")
            
        Returns:
            Path to selected noise file
        """
        if noise_type == "background_noise":
            files = self.get_background_noise_files()
        elif noise_type == "short_noise":
            files = self.get_short_noise_files()
        else:
            return None
        
        if not files:
            logger.warning(f"No {noise_type} files found")
            return None
        
        # Use audio_file_id + noise_type as seed for deterministic selection
        seed_string = f"{audio_file_id}_{noise_type}"
        seed = hash(seed_string) % (2**32)
        
        # Create deterministic random selection
        random.seed(seed)
        selected_file = random.choice(files)
        
        logger.info(f"🎲 Selected {noise_type} for {audio_file_id}: {os.path.basename(selected_file)}")
        return selected_file

# Global instance
noise_selector = NoiseFileSelector()
