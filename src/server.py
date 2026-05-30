import os
import tempfile
from contextlib import asynccontextmanager

import librosa
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.model import VGG

CLASS_NAMES = {0: "flute", 1: "trumpet", 2: "violin", 3: "acoustic guitar", 4: "piano"}
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".aiff", ".m4a"}
MODEL_PATH = os.getenv("MODEL_PATH", "vgg_model.ckpt")

_model: VGG | None = None
_device: torch.device | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = VGG(num_classes=5)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device, weights_only=True))
    _model.to(_device)
    _model.eval()
    yield


app = FastAPI(title="Soundi Instrument Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


def _compute_spectrogram(audio_path: str, sr: int = 16000, n_mels: int = 64) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    target_length = 10 * sr
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # Match training: power mel spectrogram, then log10 (mirrors Essentia pipeline)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels, fmax=8000
    )
    log_mel = np.log10(mel + 1e-10)  # (n_mels, n_frames)

    # Trim or pad time axis to 64 frames
    if log_mel.shape[1] >= 64:
        log_mel = log_mel[:, :64]
    else:
        log_mel = np.pad(log_mel, ((0, 0), (0, 64 - log_mel.shape[1])))

    return log_mel  # (64, 64)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Use: {', '.join(SUPPORTED_EXTENSIONS)}")

    suffix = ext if ext else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        spec = _compute_spectrogram(tmp_path)                          # (64, 64)
        tensor = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(_device)  # (1, 1, 64, 64)

        with torch.no_grad():
            probs = _model(tensor).squeeze().cpu().numpy()             # (5,)

        pred_idx = int(np.argmax(probs))
        return {
            "instrument": CLASS_NAMES[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
