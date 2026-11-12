"""Facial recognition for the multimodal authentication system.

Pipeline
--------
1. Detect the largest face with OpenCV's pretrained YuNet detector.
2. Align and crop it to 112x112 using the detected eye/nose/mouth landmarks.
3. Augment each aligned crop (rotation, horizontal flip, grayscale, brightness).
4. Embed every crop with OpenCV's pretrained SFace network (128-d, L2-normalised).
5. Train a Logistic Regression classifier on the embeddings to name the member.
6. Verify open-set access with cosine similarity: a face is authorized only when
   it is close enough to that member's enrolled embeddings, so a stranger is
   rejected even though the classifier always picks one of the four members.

Public interface used by the integration layer: ``verify_face(image_path)``.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from pillow_heif import register_heif_opener
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

register_heif_opener()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "datasets" / "images"
UNAUTHORIZED_DIR = PROJECT_ROOT / "datasets" / "unauthorized_images"
FEATURES_PATH = PROJECT_ROOT / "datasets" / "image_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "face_model.pkl"
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"

# Official OpenCV model zoo weights (downloaded automatically when missing).
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
DETECTOR_PATH = PRETRAINED_DIR / "face_detection_yunet_2023mar.onnx"
EMBEDDER_PATH = PRETRAINED_DIR / "face_recognition_sface_2021dec.onnx"
PRETRAINED_URLS = {
    DETECTOR_PATH: f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    EMBEDDER_PATH: f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# Cosine similarity a probe must reach against the claimed member's enrolled
# faces. Leave-one-out on our data gives genuine scores >= 0.57 and impostor
# scores <= 0.40 (SFace's published reference threshold is 0.363), so 0.45
# sits in the gap between the two distributions.
DEFAULT_SIMILARITY_THRESHOLD = 0.45
MAX_IMAGE_SIDE = 1024
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
EXPRESSIONS = ("neutral", "smiling", "surprised")


class FaceNotDetectedError(ValueError):
    """Raised when an image does not contain a detectable face."""


# --------------------------------------------------------------------------- #
# Pretrained models
# --------------------------------------------------------------------------- #
def ensure_pretrained_models() -> None:
    """Download the YuNet detector and SFace embedder if they are missing."""
    for path, url in PRETRAINED_URLS.items():
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name} from the OpenCV model zoo...")
        urllib.request.urlretrieve(url, path)


_detector = None
_embedder = None


def _get_detector():
    global _detector
    if _detector is None:
        ensure_pretrained_models()
        _detector = cv2.FaceDetectorYN.create(str(DETECTOR_PATH), "", (320, 320), 0.6, 0.3, 5000)
    return _detector


def _get_embedder():
    global _embedder
    if _embedder is None:
        ensure_pretrained_models()
        _embedder = cv2.FaceRecognizerSF.create(str(EMBEDDER_PATH), "")
    return _embedder


# --------------------------------------------------------------------------- #
# Data loading and preprocessing
# --------------------------------------------------------------------------- #
def discover_images(image_dir: Path = IMAGE_DIR) -> list[tuple[str, Path]]:
    """Return supported image paths labeled by their immediate identity folder."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    images = [
        (identity_dir.name, path)
        for identity_dir in sorted(image_dir.iterdir())
        if identity_dir.is_dir() and not identity_dir.name.startswith(".")
        for path in sorted(identity_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No supported images found in {image_dir}")

    counts = pd.Series([identity for identity, _ in images]).value_counts()
    insufficient = sorted(identity for identity, count in counts.items() if count < 3)
    if insufficient:
        raise ValueError(
            "At least three original images are required per identity; "
            f"not enough images for: {', '.join(insufficient)}"
        )
    return images


def expression_from_name(image_path: Path) -> str:
    """Infer the facial expression from a file name such as ``David_smiling.JPG``."""
    stem = Path(image_path).stem.lower()
    return next((expression for expression in EXPRESSIONS if expression in stem), "unknown")


def load_image(image_path: str | Path) -> Image.Image:
    """Open JPEG/PNG/HEIC images upright (EXIF-corrected) in RGB."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image


def detect_face(image: Image.Image) -> np.ndarray:
    """Return the YuNet detection row (box, 5 landmarks, score) of the largest face."""
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    detector = _get_detector()
    detector.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) == 0:
        raise FaceNotDetectedError("No face detected")
    return max(faces, key=lambda face: face[2] * face[3])


def load_face_crop(image_path: str | Path) -> Image.Image:
    """Load an image, detect its largest face and return the aligned 112x112 RGB crop."""
    image = load_image(image_path)
    try:
        face = detect_face(image)
    except FaceNotDetectedError as error:
        raise FaceNotDetectedError(f"No face detected in {image_path}") from error
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    aligned = _get_embedder().alignCrop(bgr, face)
    return Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))


def augment_face_images(face: Image.Image) -> dict[str, Image.Image]:
    """Return the original crop plus rotation, flip, grayscale and brightness variants."""
    face = face.convert("RGB")
    return {
        "original": face,
        "rotated": face.rotate(10, resample=Image.Resampling.BICUBIC, fillcolor=(127, 127, 127)),
        "flipped": ImageOps.mirror(face),
        "grayscale": ImageOps.grayscale(face).convert("RGB"),
        "brightness": ImageEnhance.Brightness(face).enhance(1.25),
    }


def extract_embedding(face: Image.Image) -> np.ndarray:
    """Embed an aligned 112x112 face crop into an L2-normalised 128-d SFace vector."""
    bgr = cv2.cvtColor(np.asarray(face.convert("RGB").resize((112, 112))), cv2.COLOR_RGB2BGR)
    embedding = _get_embedder().feature(bgr).flatten().astype(np.float32)
    return embedding / np.linalg.norm(embedding)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_feature_table(images: list[tuple[str, Path]]) -> tuple[pd.DataFrame, list[str]]:
    """Create one row per (original image, augmentation) with its embedding."""
    rows: list[dict[str, Any]] = []
    feature_names: list[str] = []
    for identity, image_path in images:
        crop = load_face_crop(image_path)
        for augmentation, augmented in augment_face_images(crop).items():
            embedding = extract_embedding(augmented)
            if not feature_names:
                feature_names = [f"emb_{index:03d}" for index in range(len(embedding))]
            rows.append(
                {
                    "identity": identity,
                    "source_image": _relative(image_path),
                    "expression": expression_from_name(image_path),
                    "augmentation": augmentation,
                    **dict(zip(feature_names, embedding.tolist())),
                }
            )
    return pd.DataFrame(rows), feature_names


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _build_classifier() -> LogisticRegression:
    return LogisticRegression(C=10.0, max_iter=5000)


def _best_similarity(embedding: np.ndarray, gallery: np.ndarray) -> float:
    return float(np.max(gallery @ embedding))


def _decide(
    embedding: np.ndarray,
    classifier: LogisticRegression,
    galleries: dict[str, np.ndarray],
    feature_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    probabilities = classifier.predict_proba(pd.DataFrame([embedding], columns=feature_names))[0]
    best = int(np.argmax(probabilities))
    predicted = str(classifier.classes_[best])
    similarity = _best_similarity(embedding, galleries[predicted])
    authorized = similarity >= threshold
    return {
        "identity": predicted if authorized else None,
        "predicted_identity": predicted,
        "authorized": bool(authorized),
        "confidence": float(probabilities[best]),
        "similarity": similarity,
        "reason": "verified" if authorized else "similarity_below_threshold",
        "probabilities": probabilities,
    }


def _fit(table: pd.DataFrame, feature_names: list[str]):
    classifier = _build_classifier().fit(table[feature_names], table["identity"])
    galleries = {
        identity: group[feature_names].to_numpy(dtype=np.float32)
        for identity, group in table.groupby("identity")
    }
    return classifier, galleries


def evaluate_face_model(
    feature_table: pd.DataFrame,
    feature_names: list[str],
    unauthorized_paths: list[Path] | None = None,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate without leakage.

    * Identification - leave-one-image-out: each original photo is predicted by a
      model trained on the *other* photos (and their augmentations only).
    * Impostor test - leave-one-person-out: each member is removed from training
      and their photos are presented as strangers, plus any images in
      ``datasets/unauthorized_images``. All of these must be rejected.
    """
    sources = feature_table["source_image"].unique()
    classes = sorted(feature_table["identity"].unique())
    y_true, y_pred, probabilities, genuine_scores, genuine_accepted = [], [], [], [], []

    for source in sources:
        train = feature_table[feature_table["source_image"] != source]
        probe = feature_table[
            (feature_table["source_image"] == source) & (feature_table["augmentation"] == "original")
        ].iloc[0]
        classifier, galleries = _fit(train, feature_names)
        decision = _decide(probe[feature_names].to_numpy(np.float32), classifier, galleries, feature_names, threshold)
        y_true.append(probe["identity"])
        y_pred.append(decision["predicted_identity"])
        probabilities.append(decision["probabilities"])
        genuine_scores.append(_best_similarity(probe[feature_names].to_numpy(np.float32), galleries[probe["identity"]]))
        genuine_accepted.append(decision["authorized"] and decision["predicted_identity"] == probe["identity"])

    impostor_attempts: list[dict[str, Any]] = []
    for identity in classes:
        train = feature_table[feature_table["identity"] != identity]
        classifier, galleries = _fit(train, feature_names)
        probes = feature_table[(feature_table["identity"] == identity) & (feature_table["augmentation"] == "original")]
        for _, probe in probes.iterrows():
            decision = _decide(probe[feature_names].to_numpy(np.float32), classifier, galleries, feature_names, threshold)
            impostor_attempts.append(
                {"probe": probe["source_image"], "type": "held-out member", **_public(decision)}
            )

    classifier, galleries = _fit(feature_table, feature_names)
    for path in unauthorized_paths or []:
        embedding = extract_embedding(load_face_crop(path))
        decision = _decide(embedding, classifier, galleries, feature_names, threshold)
        impostor_attempts.append({"probe": _relative(path), "type": "unknown person", **_public(decision)})

    impostors = pd.DataFrame(impostor_attempts)
    false_accepts = int(impostors["authorized"].sum()) if len(impostors) else 0
    genuine_total = len(genuine_accepted)
    false_rejects = genuine_total - int(sum(genuine_accepted))

    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y_true, np.array(probabilities), labels=classes)),
            "false_reject_rate": false_rejects / genuine_total,
            "false_accept_rate": false_accepts / len(impostors) if len(impostors) else 0.0,
            "verification_accuracy": (genuine_total - false_rejects + len(impostors) - false_accepts)
            / (genuine_total + len(impostors)),
        },
        "classes": classes,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes),
        "genuine_scores": genuine_scores,
        "impostor_attempts": impostors,
        "threshold": threshold,
    }


def _public(decision: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in decision.items() if key != "probabilities"}


def discover_unauthorized_images(directory: Path = UNAUTHORIZED_DIR) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)


def train_face_model(
    image_dir: Path = IMAGE_DIR,
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    return_details: bool = False,
) -> dict[str, Any]:
    """Extract features, evaluate without leakage, then train and save the final model."""
    images = discover_images(image_dir)
    feature_table, feature_names = build_feature_table(images)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(features_path, index=False)

    evaluation = evaluate_face_model(
        feature_table, feature_names, discover_unauthorized_images(), threshold
    )

    classifier, galleries = _fit(feature_table, feature_names)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": classifier,
            "galleries": galleries,
            "feature_names": feature_names,
            "classes": list(classifier.classes_),
            "similarity_threshold": threshold,
            "embedder": EMBEDDER_PATH.name,
            "detector": DETECTOR_PATH.name,
        },
        model_path,
    )
    return evaluation if return_details else evaluation["metrics"]


def verify_face(
    image_path: str | Path,
    model_path: Path = MODEL_PATH,
    similarity_threshold: float | None = None,
) -> dict[str, str | float | bool | None]:
    """Identify the face in ``image_path`` and decide whether it is an authorized member.

    Returns ``identity`` (None when rejected), ``authorized``, ``confidence``
    (classifier probability), ``similarity`` (cosine score) and ``reason``.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Face model does not exist: {model_path}. Run train_face_model() first.")
    if similarity_threshold is not None and not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between -1 and 1")

    artifact = joblib.load(model_path)
    try:
        embedding = extract_embedding(load_face_crop(image_path))
    except FaceNotDetectedError:
        return {
            "identity": None,
            "authorized": False,
            "confidence": 0.0,
            "similarity": 0.0,
            "reason": "face_not_detected",
        }

    threshold = artifact["similarity_threshold"] if similarity_threshold is None else similarity_threshold
    decision = _decide(embedding, artifact["model"], artifact["galleries"], artifact["feature_names"], threshold)
    decision.pop("probabilities")
    decision.pop("predicted_identity")
    return decision


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(json.dumps(verify_face(sys.argv[1]), indent=2))
    else:
        print(json.dumps(train_face_model(), indent=2))
