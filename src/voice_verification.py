"""
voice_verification.py – Voiceprint Verification Model (Formative 2, Member 3)

Usage from another script (e.g. main.py):
    from src.voice_verification import verify_voice
    result = verify_voice("datasets/audio/Nazira_approve.wav")
    # {'identity': 'nazira', 'verified': True, 'confidence': 0.97}

Usage from the command line:
    python src/voice_verification.py path/to/recording.wav

Requires models/voice_model.pkl (created by notebooks/audio_preprocessing.ipynb).
"""

import os
import sys

import joblib
import librosa
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "voice_model.pkl")
N_MFCC = 13
_bundle = None  # the model is loaded once and reused


def load_audio(path, sr=22050):
    """Load as mono, trim silence and normalize volume (same as in the notebook)."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def extract_features(y, sr=22050):
    """MFCCs, spectral roll-off/centroid, RMS energy and zero-crossing rate (same as in the notebook)."""
    feats = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc_{i+1}_std"] = float(mfcc[i].std())
    feats["spectral_rolloff"] = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    feats["spectral_centroid"] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    rms = librosa.feature.rms(y=y)
    feats["rms_energy_mean"] = float(rms.mean())
    feats["rms_energy_std"] = float(rms.std())
    feats["zero_crossing_rate"] = float(librosa.feature.zero_crossing_rate(y).mean())
    return feats


def _load_bundle(model_path):
    global _bundle
    if _bundle is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Voice model not found: {model_path}")
        _bundle = joblib.load(model_path)
    return _bundle


def verify_voice(audio_path, model_path=DEFAULT_MODEL_PATH, threshold=None):
    """
    Check whether a recording belongs to an authorized group member.

    Returns a dict:
        identity   – member name, or "unknown" if not verified
        verified   – True/False
        confidence – the model's probability for the most likely member (0–1)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    bundle = _load_bundle(model_path)
    model, feature_cols, sr = bundle["model"], bundle["feature_cols"], bundle["sr"]
    threshold = bundle["threshold"] if threshold is None else threshold

    y = load_audio(audio_path, sr=sr)
    if len(y) == 0:
        return {"identity": "unknown", "verified": False, "confidence": 0.0}

    feats = pd.DataFrame([extract_features(y, sr=sr)])[feature_cols]
    probs = model.predict_proba(feats.values)[0]
    best = int(np.argmax(probs))
    confidence = float(probs[best])
    verified = confidence >= threshold

    return {
        "identity": str(model.classes_[best]) if verified else "unknown",
        "verified": bool(verified),
        "confidence": round(confidence, 3),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/voice_verification.py <audio_file>")
        sys.exit(1)
    try:
        result = verify_voice(sys.argv[1])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if result["verified"]:
        print(f"✅ Voice verified: {result['identity']} (confidence {result['confidence']:.2f})")
    else:
        print(f"❌ Access denied: voice not recognized (confidence {result['confidence']:.2f})")