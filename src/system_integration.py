"""This module is the integration layer: it chains the three models into one secure workflow, 
where a face check comes first, the product model runs but its result is held back, 
and a voice check must pass before the predicted product is shown. 
It expects `verify_face`, `verify_voice` and `predict_product` from teammates, and 
it accepts several result formats (a dict, a tuple, a bool or a string) and 
converts them all into one standard form.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
IDENTITY_MAP_PATH = BASE_DIR / "identity_map.json"

# One spelling for every member, lowercase, used by the face AND voice models.
AUTHORIZED_IDENTITIES = ("allan", "david", "nazira", "shamila")

# Extra safety net on a model's reported confidence. The voice model decides by confidence, so a
# floor is applied. The face model decides by cosine similarity (its own threshold); the "confidence"
# it reports is only a classifier probability, so no extra floor is applied to it.
MIN_CONFIDENCE = {"voice": 0.60}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

UNAUTHORIZED_MARKERS = ("unauthorized", "unauthorised", "unknown", "intruder", "stranger", "outsider", "impostor")
_TRUE_WORDS = {"verified", "authorized", "authorised", "success", "approved", "granted", "true", "yes", "pass", "passed", "ok"}

# Human-readable versions of the reason codes returned by the face model.
REASON_TEXT = {
    "face_not_detected": "no face could be detected in the image",
    "similarity_below_threshold": "the face is not similar enough to any authorized member",
    "confidence_below_threshold": "the face does not match any authorized member with enough confidence",
    "verified": "",
}

# Final statuses of a transaction
GRANTED = "GRANTED"
DENIED_FACE = "DENIED_FACE"
DENIED_VOICE = "DENIED_VOICE"
ERROR = "ERROR"


# --------------------------------------------------------------------------- errors
class IntegrationError(Exception):
    """Problem with setup or input (not an authentication failure)."""


class ModelNotAvailableError(IntegrationError):
    """A teammate's module or function is missing or cannot be imported."""


class InputFileError(IntegrationError):
    """The user supplied a missing, unreadable or wrongly typed file."""


class ModelRuntimeError(IntegrationError):
    """A model raised an exception while running."""


class ProductError(IntegrationError):
    """The product model could not produce a prediction for this customer."""


# --------------------------------------------------------------------------- data types
@dataclass
class VerificationResult:
    identity: Optional[str]
    verified: bool
    confidence: Optional[float] = None
    reason: str = ""
    similarity: Optional[float] = None  # cosine similarity, reported by the face model

    def describe(self) -> str:
        who = self.identity or "no identity"
        parts = []
        if self.similarity is not None:
            parts.append(f"similarity {self.similarity:.2f}")
        if self.confidence is not None:
            parts.append(f"confidence {self.confidence:.2f}")
        return f"{who}" + (f" ({', '.join(parts)})" if parts else "")


@dataclass
class TransactionResult:
    status: str
    granted: bool
    message: str
    identity: Optional[str] = None
    customer_id: Optional[Union[int, str]] = None
    product: Optional[str] = None  # only filled when access is granted
    face: Optional[VerificationResult] = None
    voice: Optional[VerificationResult] = None
    steps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        def vr(v: Optional[VerificationResult]):
            return None if v is None else {"identity": v.identity, "verified": v.verified,
                                           "confidence": v.confidence, "similarity": v.similarity,
                                           "reason": v.reason}
        return {"status": self.status, "granted": self.granted, "message": self.message,
                "identity": self.identity, "customer_id": self.customer_id, "product": self.product,
                "face": vr(self.face), "voice": vr(self.voice), "steps": self.steps}


# --------------------------------------------------------------------------- helpers
def normalize_identity(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_WORDS
    return bool(value)


def normalize_verification(raw: Any, modality: str) -> VerificationResult:
    """Turn whatever a teammate's verify_* function returns into a VerificationResult."""
    identity = status = confidence = similarity = None
    reason = ""
    if isinstance(raw, dict):
        identity = raw.get("identity", raw.get("label", raw.get("name")))
        status = raw.get("verified", raw.get("status", raw.get("authorized")))
        confidence = raw.get("confidence", raw.get("probability", raw.get("score")))
        similarity = raw.get("similarity")
        reason = str(raw.get("reason") or "")
    elif isinstance(raw, (tuple, list)) and 2 <= len(raw) <= 3:
        identity, status = raw[0], raw[1]
        confidence = raw[2] if len(raw) == 3 else None
    elif isinstance(raw, str):
        identity = raw
    elif isinstance(raw, bool) or type(raw).__name__ == "bool_":
        status = raw
    else:
        raise ModelRuntimeError(f"The {modality} model returned an unsupported value: {raw!r}")

    identity = normalize_identity(identity)
    reason = REASON_TEXT.get(reason, reason)
    if status is None:
        if identity is None:
            raise ModelRuntimeError(f"The {modality} model returned neither an identity nor a status: {raw!r}")
        status = identity in AUTHORIZED_IDENTITIES
    verified = _to_bool(status)
    confidence = None if confidence is None else float(confidence)
    similarity = None if similarity is None else float(similarity)
    min_confidence = MIN_CONFIDENCE.get(modality)

    if verified and identity is None:
        verified, reason = False, reason or f"{modality} model did not report who the person is"
    elif verified and identity not in AUTHORIZED_IDENTITIES:
        verified, reason = False, reason or f"'{identity}' is not an authorized user"
    elif verified and min_confidence is not None and confidence is not None and confidence < min_confidence:
        verified = False
        reason = reason or f"confidence {confidence:.2f} is below the {min_confidence:.2f} threshold"
    elif not verified and not reason:
        reason = f"{modality} not recognised as an authorized user"
    return VerificationResult(identity, verified, confidence, reason, similarity)


def validate_file(path: Any, kind: str) -> Path:
    """Check a user-supplied path. kind is 'image' or 'audio'."""
    if path is None or not str(path).strip():
        raise InputFileError(f"No {kind} file was provided.")
    p = Path(str(path).strip().strip('"').strip("'")).expanduser()
    if not p.exists():
        raise InputFileError(f"{kind.capitalize()} file not found: {p}")
    if not p.is_file():
        raise InputFileError(f"Not a file: {p}")
    allowed = IMAGE_EXTENSIONS if kind == "image" else AUDIO_EXTENSIONS
    if p.suffix.lower() not in allowed:
        raise InputFileError(f"Unsupported {kind} type '{p.suffix}'. Use one of: {', '.join(sorted(allowed))}")
    if p.stat().st_size == 0:
        raise InputFileError(f"{kind.capitalize()} file is empty: {p}")
    return p


def load_identity_map(path: Path = IDENTITY_MAP_PATH) -> dict:
    """identity -> customer_id (from the merged dataset). Keys starting with '_' are comments."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise IntegrationError(f"identity_map.json not found at {path}")
    except json.JSONDecodeError as err:
        raise IntegrationError(f"identity_map.json is not valid JSON: {err}")
    return {normalize_identity(k): v for k, v in data.items() if not str(k).startswith("_")}


# --------------------------------------------------------------------------- model loading
def _load_callable(module_name: str, func_name: str) -> Callable:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        module = importlib.import_module(module_name)
    except Exception as err:  # ImportError, or a crash while the module runs its top-level code
        raise ModelNotAvailableError(f"src/{module_name}.py could not be loaded ({type(err).__name__}: {err})")
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise ModelNotAvailableError(f"src/{module_name}.py does not define {func_name}() yet")
    return fn


def _guess_identity(path: Any) -> Optional[str]:
    p = Path(str(path))
    text = f"{p.parent.name} {p.stem}".lower()
    if any(marker in text for marker in UNAUTHORIZED_MARKERS):
        return None
    for name in AUTHORIZED_IDENTITIES:
        if name in text:
            return name
    return None


def mock_verify_face(image_path):
    """STAND-IN for facial_recognition.verify_face. Decides from the file name only."""
    who = _guess_identity(image_path)
    return {"identity": who or "unknown", "verified": who is not None, "confidence": 0.93 if who else 0.12}


def mock_verify_voice(audio_path):
    """STAND-IN for voice_verification.verify_voice. Decides from the file name only."""
    who = _guess_identity(audio_path)
    return {"identity": who or "unknown", "verified": who is not None, "confidence": 0.91 if who else 0.10}


def mock_predict_product(customer_id):
    """STAND-IN for product_recommendation.predict_product."""
    products = ["Books", "Clothing", "Electronics", "Groceries", "Sports"]
    return products[int(str(customer_id).lstrip("Aa")) % len(products)]


@dataclass
class Components:
    face_fn: Callable
    voice_fn: Callable
    product_fn: Callable
    mocked: tuple = ()  # names of components that are stand-ins


def ensure_ffmpeg() -> bool:
    """Make an `ffmpeg` executable available on PATH. Returns True if one is available.

    The team's voice recordings are AAC/M4A/Opus files saved with a .wav extension, so librosa
    must decode them through ffmpeg. If the machine has none, use the one bundled with the
    `imageio-ffmpeg` package (listed in requirements.txt).
    """
    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg
        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
        folder = Path(tempfile.gettempdir()) / "multimodal_ffmpeg"
        folder.mkdir(exist_ok=True)
        target = folder / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not target.exists():
            shutil.copy2(exe, target)
            target.chmod(0o755)
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
        return shutil.which("ffmpeg") is not None
    except Exception:
        return False


def component_status() -> dict:
    """Report, per component, whether the real implementation can be loaded."""
    specs = {"face": ("facial_recognition", "verify_face"),
             "voice": ("voice_verification", "verify_voice"),
             "product": ("product_recommendation", "predict_product")}
    status = {}
    for key, (module, func) in specs.items():
        try:
            _load_callable(module, func)
            status[key] = "READY"
        except ModelNotAvailableError as err:
            status[key] = f"MISSING - {err}"
    return status


def load_components(allow_mock: bool = False) -> Components:
    ensure_ffmpeg()
    """Load the real models. With allow_mock, any model that is not ready is replaced by a stand-in."""
    specs = [("face", "facial_recognition", "verify_face", mock_verify_face),
             ("voice", "voice_verification", "verify_voice", mock_verify_voice),
             ("product", "product_recommendation", "predict_product", mock_predict_product)]
    fns, mocked, missing = {}, [], []
    for key, module, func, mock in specs:
        try:
            fns[key] = _load_callable(module, func)
        except ModelNotAvailableError as err:
            if allow_mock:
                fns[key] = mock
                mocked.append(key)
            else:
                missing.append(str(err))
    if missing:
        raise ModelNotAvailableError("; ".join(missing) + ". Run with --mock to use stand-ins until teammates push their models.")
    return Components(fns["face"], fns["voice"], fns["product"], tuple(mocked))


# --------------------------------------------------------------------------- the system
class SecureRecommendationSystem:
    """Runs the face -> product model -> voice -> display workflow."""

    def __init__(self, face_fn: Callable, voice_fn: Callable, product_fn: Callable,
                 identity_map: Optional[dict] = None,
                 reporter: Optional[Callable[[str, str], None]] = None):
        self._face_fn, self._voice_fn, self._product_fn = face_fn, voice_fn, product_fn
        self._identity_map = identity_map if identity_map is not None else load_identity_map()
        self._reporter = reporter
        self._steps: list = []

    def _emit(self, kind: str, text: str) -> None:
        """kind: info | ok | fail. Recorded for the report and optionally printed."""
        self._steps.append(f"[{kind}] {text}")
        if self._reporter:
            self._reporter(kind, text)

    def _call_model(self, name: str, fn: Callable, arg: Any):
        try:
            return fn(arg)
        except IntegrationError:
            raise
        except Exception as err:
            hint = " Check that installed package versions match requirements.txt." if isinstance(err, (AttributeError, ModuleNotFoundError, ImportError)) else ""
            raise ModelRuntimeError(f"The {name} model failed on this input ({type(err).__name__}: {err}).{hint}")

    def verify_face_step(self, image_path: Path) -> VerificationResult:
        return normalize_verification(self._call_model("face", self._face_fn, str(image_path)), "face")

    def run_prediction_step(self, identity: str):
        customer_id = self._identity_map.get(identity)
        if customer_id is None:
            raise ProductError(f"No customer ID is mapped to '{identity}' in identity_map.json")
        try:
            product = self._product_fn(customer_id)
        except ValueError as err:
            raise ProductError(str(err))
        except Exception as err:
            hint = " Check that installed package versions match requirements.txt (the .pkl needs the same scikit-learn)."
            raise ModelRuntimeError(f"The product model failed ({type(err).__name__}: {err}).{hint}")
        return customer_id, str(product)

    def verify_voice_step(self, audio_path: Path, expected_identity: str) -> VerificationResult:
        result = normalize_verification(self._call_model("voice", self._voice_fn, str(audio_path)), "voice")
        if result.verified and result.identity != expected_identity:
            result = VerificationResult(result.identity, False, result.confidence,
                                        f"voice belongs to '{result.identity}', but the face was verified as '{expected_identity}'")
        return result

    def run_transaction(self, image_path: Any, audio: Union[str, Path, Callable[[], Any]]) -> TransactionResult:
        """audio may be a path or a zero-argument callable returning a path (asked only after the face passes)."""
        self._steps = []
        face = voice = None
        identity = customer_id = None
        try:
            image = validate_file(image_path, "image")
            audio_path = None if callable(audio) else validate_file(audio, "audio")

            self._emit("info", f"Step 1/3  Facial recognition on {image.name}")
            face = self.verify_face_step(image)
            if not face.verified:
                self._emit("fail", f"Face rejected: {face.reason}")
                return self._finish(DENIED_FACE, False, "ACCESS DENIED - face not recognised.", face=face)
            identity = face.identity
            self._emit("ok", f"Face verified: {face.describe()}")

            self._emit("info", "Step 2/3  Running product recommendation model (result is held until voice is verified)")
            customer_id, product = self.run_prediction_step(identity)
            self._emit("ok", f"Prediction ready for customer {customer_id} (hidden)")

            if audio_path is None:
                audio_path = validate_file(audio(), "audio")
            self._emit("info", f"Step 3/3  Voice verification on {audio_path.name}")
            voice = self.verify_voice_step(audio_path, identity)
            if not voice.verified:
                self._emit("fail", f"Voice rejected: {voice.reason}")
                return self._finish(DENIED_VOICE, False, "ACCESS DENIED - voice not verified.",
                                    identity=identity, customer_id=customer_id, face=face, voice=voice)
            self._emit("ok", f"Voice verified: {voice.describe()}")
            return self._finish(GRANTED, True, f"ACCESS GRANTED - recommended product for {identity}: {product}",
                                identity=identity, customer_id=customer_id, product=product, face=face, voice=voice)
        except IntegrationError as err:
            self._emit("fail", str(err))
            return self._finish(ERROR, False, f"ERROR - {err}", identity=identity, customer_id=customer_id, face=face, voice=voice)

    def _finish(self, status, granted, message, **kwargs) -> TransactionResult:
        return TransactionResult(status=status, granted=granted, message=message, steps=list(self._steps), **kwargs)
