from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np, librosa, pvrecorder, pathlib


class SpeechToText:
    def __init__(self, cache_dir: pathlib.Path = pathlib.Path("cache-dir")):
        cache_dir.mkdir(exist_ok=True, parents=True)
        model_name = "openai/whisper-small"
        self.proc = WhisperProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.config.forced_decoder_ids = None

    def __call__(self, speech: np.ndarray) -> str:
        input_model = self.proc(speech, sampling_rate=16000,
                                return_tensors="pt",
                                return_attention_mask=True) 

        output_model = self.model.generate(input_model.input_features, 
                                task='transcribe',
                                attention_mask=input_model["attention_mask"])

        transcription = self.proc.batch_decode(output_model, skip_special_tokens=True)
        return transcription[0] if transcription else ""