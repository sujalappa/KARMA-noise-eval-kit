from tkinter.constants import FALSE
import torch
from typing import Dict, Any, Generator, Optional, List
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from karma.utils.audio import resample_audio
import numpy as np
from datasets import Audio
import soundfile as sf
import io
from karma.utils.noise.noise_utils import apply_augmentations
DATASET_NAME = "ai4bharat/indicvoices_r"
SPLIT = "test"
COMMIT_HASH = "5f4495c91d500742a58d1be2ab07d77f73c0acf8"


@register_dataset(
    DATASET_NAME,
    #metrics=["asr_semantic_metric"], 
    #metrics=["wer", "cer", "asr_semantic_metric"],  # Original HF aggregated metrics
    metrics=["wer_jiw", "cer_jiw","asr_semantic_jiw"],  # Per-sample jiwer metrics
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    task_type="transcription",
    required_args=["language"],
    default_args={"language": "hindi"},
    processors=["multilingual_text_processor"],
)
class IndicVoicesRDataset(BaseMultimodalDataset):
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        language: str = "hindi",
        processors=None,
        **kwargs,
    ):
        """
        Initialize the IndicVoicesR dataset.

        """
        super().__init__(
            dataset_name=dataset_name,
            split=split,
            commit_hash=commit_hash,
            config=language,
            processors=processors,
            **kwargs,
        )
        self.language = language
        self.dataset_name = f"{DATASET_NAME}-{self.language}"
        self.dataset = self.dataset.cast_column(
            "audio", Audio(sampling_rate=16000, decode=False)
        )

    def format_item(self, sample: Dict[str, Any]) -> DataLoaderIterable:
        print(">>>> ENTERING IndicVoicesR.format_item")
        audio_info = sample.get("audio", {})
        audio_data = audio_info.get("bytes")
        waveform, sr = sf.read(io.BytesIO(audio_data))
        augmented_waveform = apply_augmentations(waveform, sr)
        buf = io.BytesIO()
        sf.write(buf, augmented_waveform, sr, format="WAV")
        buf.seek(0)
        augmented_bytes = buf.read()
        debug_out = "/tmp/debug_augmented.wav"
        with open(debug_out, "wb") as f:
            
                
            f.write(augmented_bytes)
        print(f"[DEBUG] wrote augmented audio to {debug_out}")

        return DataLoaderIterable(
             audio=augmented_bytes,
             expected_output=sample.get("text", ""),
        )
        