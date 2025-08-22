from typing import Dict, Any
from karma.data_models.dataloader_iterable import DataLoaderIterable
from karma.eval_datasets.base_dataset import BaseMultimodalDataset
from karma.registries.dataset_registry import register_dataset
from datasets import Audio
import soundfile as sf
import io
from karma.utils.noise.noise_utils import apply_augmentations
from karma.utils.noise.noise_intensity_config import parse_noise_specification
import os
DATASET_NAME = "ekacare/eka-medical-asr-evaluation-dataset"
SPLIT = "test"
COMMIT_HASH = "991bc807cab1f323f0283c836c634796bbf1ed3e"

@register_dataset(
    DATASET_NAME,
    #metrics=["wer", "cer", "asr_semantic_metric"],
    #metrics=["wer_jiw", "cer_jiw"],
    metrics=["wer_jiw", "cer_jiw","asr_semantic_jiw"],
    commit_hash=COMMIT_HASH,
    split=SPLIT,
    task_type="transcription",
    required_args=["language"],
    default_args={"language": "hi"},
)
class EkaMedicalAsrDataset(BaseMultimodalDataset):
    def set_eval_context(self, model_name: str, noise_types, config: str):
        """
        Set context for evaluation (model, list of noise_types, config) to be used in __iter__.
        Now supports noise_types as list of 'type:intensity' strings, e.g., ['gaussian:0.009', 'clip:0.5']
        """
        self._eval_model_name = model_name
        if isinstance(noise_types, str):
            self._eval_noise_types = [noise_types]
        else:
            self._eval_noise_types = list(noise_types)
        self._eval_config = config

    def __iter__(self):
        """For each sample, save all requested noise variants. Yield only the first for evaluation."""
        for idx, sample in enumerate(self.dataset):
            if idx >= self.max_samples:
                break
            first = True
            for noise_type in getattr(self, '_eval_noise_types', ['unknown_noise']):
                item = self.format_item(
                    sample,
                    getattr(self, '_eval_model_name', 'unknown_model'),
                    noise_type,
                    getattr(self, '_eval_config', 'unknown_config')
                )
                if first:
                    yield item
                    first = False
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = SPLIT,
        commit_hash: str = COMMIT_HASH,
        language: str = "hi",
        noise_type=None,
        processors=None,
        noisy_audio_dir: str = "saved_noisy_audio",
        **kwargs,
    ):
        """
        Initialize the EkaMedicalAsrDataset dataset.
        Args:
            noise_type: Type(s) of noise to apply (str or list), e.g., 'gaussian:0.009', 'clip:0.5'
        """
        super().__init__(
            dataset_name=DATASET_NAME,
            split=SPLIT,
            config=language,
            processors=processors,
            **kwargs,
        )
        self.language = language
        self.noisy_audio_dir = noisy_audio_dir
        # Accept a list of noise types with explicit intensity, e.g., ['gaussian:0.009', 'clip:0.5']
        if noise_type is None:
            self.noise_types = ["clean"]
        elif isinstance(noise_type, str):
            self.noise_types = [noise_type]
        else:
            self.noise_types = list(noise_type)
        self.noise_spec = self.noise_types  # For logging
        self.dataset_name = f"{DATASET_NAME}-{self.language}"
        self.dataset = self.dataset.cast_column(
            "audio", Audio(sampling_rate=16000, decode=False)
        )

    def format_item(self, sample: Dict[str, Any], model_name: str, noise_type: str, config: str) -> DataLoaderIterable:
        """
        Accepts noise_type as 'type:intensity' (e.g., 'gaussian:0.009').
        If no intensity is provided, defaults to previous behavior.
        """
        audio_info = sample.get("audio", {})
        audio_data = audio_info.get("bytes")
        waveform, sr = sf.read(io.BytesIO(audio_data))
        audio_file_id = sample.get("file_name", "unknown")
        # Parse noise_type and intensity from argument (e.g., 'gaussian:0.009')
        if ":" in noise_type:
            parsed_noise_type, intensity = noise_type.split(":", 1)
            try:
                intensity = float(intensity)
            except Exception:
                intensity = None
        else:
            parsed_noise_type = noise_type
            intensity = None
        augmented_waveform = apply_augmentations(
            waveform=waveform,
            sample_rate=sr,
            noise_type=parsed_noise_type,
            audio_file_id=audio_file_id,
            dataset_name=DATASET_NAME,
            intensity=intensity
        )
        buf = io.BytesIO()
        sf.write(buf, augmented_waveform, sr, format="WAV")
        buf.seek(0)
        augmented_bytes = buf.read()
        # Save each augmented audio to a unique file in nested folders
        safe_model = model_name.replace("/", "_").replace("-", "_")
        safe_noise = noise_type.replace(":", "_").replace("/", "_")
        safe_config = str(config).replace("/", "_")
        file_id_str = str(audio_file_id).replace("/", "_")
        save_dir = os.path.join(self.noisy_audio_dir, safe_model, safe_noise, safe_config)
        os.makedirs(save_dir, exist_ok=True)
        out_filename = f"{file_id_str}.wav"
        out_path = os.path.join(save_dir, out_filename)
        with open(out_path, "wb") as f:
            f.write(augmented_bytes)
        print(f"[AUDIO SAVE] Saved augmented audio to {out_path}")
        return DataLoaderIterable(
            audio=augmented_bytes,
            expected_output=sample.get("text", ""),
        )