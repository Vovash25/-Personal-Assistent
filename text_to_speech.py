import sounddevice, time, numpy as np
from piper.voice import PiperVoice
from pathlib import Path
from huggingface_hub import hf_hub_download
from langdetect import detect

class TextToSpeech:
    def __init__(self, cache_dir: Path = Path("cache-dir")):
        self.cache_dir = cache_dir
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        self.en_model = "en_US-lessac-medium.onnx"
        self.pl_model = "pl_PL-gosia-medium.onnx"

        self._ensure_model("en/en_US/lessac/medium/en_US-lessac-medium.onnx", self.en_model)
        self._ensure_model("pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx", self.pl_model)

        self.voice_en = PiperVoice.load(self.cache_dir / self.en_model)
        self.voice_pl = PiperVoice.load(self.cache_dir / self.pl_model)

    def _ensure_model(self, repo_path, local_name):
        target_path = self.cache_dir / local_name
        if not target_path.exists():
            for ext in ["", ".json"]:
                full_filename = repo_path + ext
                hf_hub_download(
                    repo_id="rhasspy/piper-voices", 
                    filename=full_filename,
                    local_dir=self.cache_dir
                )
                
                downloaded_file = self.cache_dir / full_filename
                final_destination = self.cache_dir / (local_name + ext)
                
                if downloaded_file.exists():
                    downloaded_file.replace(final_destination)

    def __call__(self, text: str):
        try:
            lang = detect(text)
        except:
            lang = "en"

        model = self.voice_pl if lang == "pl" else self.voice_en
        
        audio = model.synthesize(text)
        out = np.concatenate([i.audio_float_array for i in audio])
        
        sounddevice.play(out, 22050)
        sounddevice.wait()