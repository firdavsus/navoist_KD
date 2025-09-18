# pip install faster-whisper
import re, time, subprocess, logging
import torch, torchaudio
import numpy as np
from faster_whisper import WhisperModel
from typing import Union

logging.getLogger("faster_whisper").setLevel(logging.ERROR)

class NavaiFastSTT:
    def __init__(
        self,
        model_name: str = "./model/faster-go",
        target_sample_rate: int = 16000,
        language: str = "auto",
        is_capitalize: bool = True,
    ):
        self.target_sr = target_sample_rate
        if language!="auto":
            self.lang      = language
        else:
            self.lang      = None
        self.is_cap    = is_capitalize

        # load faster-whisper model (handles CPU/GPU + compute_type)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type="float16" if device.startswith("cuda") else "int8"
        )

    def _load_and_resample(self, path: str) -> np.ndarray:
        wav, sr = torchaudio.load(path)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).numpy()

    def capitalize_sentences(self, text: str) -> str:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return " ".join(p[:1].upper() + p[1:] if p else "" for p in parts)

    def transcribe(self, audio_path: Union[str, np.ndarray]) -> str:
        if isinstance(audio_path, str):
            wave = self._load_and_resample(audio_path)
        else:
            wave =  audio_path.astype(np.float32)

        chunk_secs = 30
        chunk_len  = self.target_sr * chunk_secs
        texts = []

        # process in chunks
        for start in range(0, len(wave), chunk_len):
            segment = wave[start : start + chunk_len]
            # faster-whisper does feature extraction + decode in one call
            if self.lang:
                segments, _ = self.model.transcribe(
                    segment,
                    beam_size=5,
                    language=self.lang,
                    word_timestamps=False,
                )
            else:
                segments, _ = self.model.transcribe(
                    segment,
                    beam_size=5,
                    word_timestamps=False,
                )
            seg_text = " ".join(s.text for s in segments)
            if self.is_cap:
                seg_text = self.capitalize_sentences(seg_text)
            texts.append(seg_text)

        return " ".join(texts)


if __name__ == "__main__":
    t0 = time.time()
    path       = "./Usefull/rus1.wav"
    #input_spx  = path + ".spx"
    output_wav = path #+ ".wav"

    # convert speex to wav@16k
    #subprocess.run(["ffmpeg", "-y", "-i", input_spx, "-ar", "16000", output_wav])

    transcriber   = NavaiFastSTT()
    transcription = transcriber.transcribe(output_wav)
    # with open("./navoist_test/5M.txt", "w") as file:
    #     file.write(transcription)

    print(f"Transcription: {transcription}")
    print(f"Time taken: {time.time() - t0:.2f} seconds")
