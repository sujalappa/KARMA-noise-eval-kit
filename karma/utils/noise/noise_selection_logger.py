#!/usr/bin/env python3
"""
Noise Selection Logger and Replay System

This system logs which specific noise files were used for each audio file,
then replays the exact same noise files for subsequent model evaluations.
"""

import os
import json
import logging
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class NoiseSelectionLogger:
    """Logs and replays noise file selections for deterministic evaluation."""
    
    def __init__(self, log_dir: str = "noise_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
    def get_log_file_path(self, dataset_name: str, noise_type: str) -> Path:
        """Get the path for a specific noise log file."""
        safe_dataset = dataset_name.replace("/", "_").replace("-", "_")
        return self.log_dir / f"{safe_dataset}_{noise_type}_selections.json"
    
    def log_noise_selection(self, dataset_name: str, noise_type: str, 
                          audio_file_id: str, selected_noise_file: str) -> None:
        """
        Log which noise file was selected for a specific audio file.
        This creates deterministic per-sample noise selection across all models.
        
        Args:
            dataset_name: Name of the dataset
            noise_type: Type of noise (background_noise, short_noise)
            audio_file_id: ID of the audio file (e.g., sample_001, sample_002)
            selected_noise_file: Path to the noise file that was selected
        """
        if noise_type not in ["background_noise", "short_noise"]:
            return
            
        log_file = self.get_log_file_path(dataset_name, noise_type)
        
        # Load existing log or create new one
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing log: {e}")
                log_data = {"metadata": {}, "selections": {}}
        else:
            log_data = {
                "metadata": {
                    "dataset": dataset_name,
                    "noise_type": noise_type,
                    "created": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "description": "Per-sample noise mapping for fair model comparison"
                },
                "selections": {}
            }
        
        # Update the selection for this specific sample
        log_data["selections"][audio_file_id] = selected_noise_file
        log_data["metadata"]["last_updated"] = datetime.now().isoformat()
        log_data["metadata"]["total_samples"] = len(log_data["selections"])
        
        # Save the updated log
        try:
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            logger.info(f"Logged deterministic noise for {audio_file_id}: {os.path.basename(selected_noise_file)}")
        except Exception as e:
            logger.error(f"Failed to save noise log: {e}")
    
    def get_logged_noise_file(self, dataset_name: str, noise_type: str, 
                            audio_file_id: str) -> Optional[str]:
        """
        Get the previously logged noise file for a specific audio sample.
        This ensures all models use the same noise file for the same sample.
        
        Args:
            dataset_name: Name of the dataset
            noise_type: Type of noise (background_noise, short_noise)
            audio_file_id: ID of the audio file (e.g., sample_001, sample_002)
            
        Returns:
            Path to the noise file if found, None if not logged yet
        """
        if noise_type not in ["background_noise", "short_noise"]:
            return None
            
        log_file = self.get_log_file_path(dataset_name, noise_type)
        
        if not log_file.exists():
            logger.debug(f" No noise log found for {noise_type}, will create on first selection")
            return None
        
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            if audio_file_id in log_data.get("selections", {}):
                noise_file = log_data["selections"][audio_file_id]
                if os.path.exists(noise_file):
                    logger.info(f"Using logged {noise_type} for {audio_file_id}: {os.path.basename(noise_file)}")
                    return noise_file
                else:
                    logger.warning(f"Logged noise file not found: {noise_file}")
            else:
                logger.debug(f"No logged {noise_type} for {audio_file_id}, will select and log")
                
        except Exception as e:
            logger.error(f"Error reading noise log: {e}")
        
        return None
    
    def clear_logs(self, dataset_name: str = None, noise_type: str = None) -> None:
        """
        Clear noise selection logs.
        
        Args:
            dataset_name: Specific dataset to clear (None = all)
            noise_type: Specific noise type to clear (None = all)
        """
        if dataset_name and noise_type:
            # Clear specific log file
            log_file = self.get_log_file_path(dataset_name, noise_type)
            if log_file.exists():
                log_file.unlink()
                logger.info(f"Cleared noise log: {log_file}")
        else:
            # Clear all log files
            for log_file in self.log_dir.glob("*.json"):
                log_file.unlink()
                logger.info(f"Cleared noise log: {log_file}")
    
    def list_logs(self) -> Dict[str, Dict]:
        """List all available noise logs with metadata."""
        logs = {}
        
        for log_file in self.log_dir.glob("*.json"):
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                
                metadata = log_data.get("metadata", {})
                logs[log_file.name] = {
                    "dataset": metadata.get("dataset", "unknown"),
                    "noise_type": metadata.get("noise_type", "unknown"),
                    "total_files": metadata.get("total_files", 0),
                    "last_updated": metadata.get("last_updated", "unknown")
                }
            except Exception as e:
                logger.warning(f"Could not read log {log_file}: {e}")
        
        return logs

# Global instance
noise_logger = NoiseSelectionLogger()
