"""Preprocess team photos and train a small facial identity classifier."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from pillow_heif import register_heif_opener
from skimage.feature import hog
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

register_heif_opener()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "datasets" / "images"
FEATURES_PATH = PROJECT_ROOT / "datasets" / "image_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "face_model.pkl"
IMAGE_SIZE = (128, 128)
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


class FaceNotDetectedError(ValueError):
    """Raised when an image does not contain a detectable frontal face."""


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


def load_face_crop(image_path: Path) -> Image.Image:
    """Load an image, detect its largest face, and return a square RGB face crop."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load the OpenCV face detector: {cascade_path}")

    normalized = cv2.equalizeHist(
        cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    )
    detections = detector.detectMultiScale(
        normalized,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(30, 30),
    )
    image_width, image_height = image.size
    candidates: list[tuple[float, int, int, int, int]] = []
    for detection in detections:
        x, y, width, height = (int(value / 2) for value in detection)
        center_x = (x + width / 2) / image_width
        center_y = (y + height / 2) / image_height
        if not 0.12 <= center_x <= 0.88 or not 0.08 <= center_y <= 0.82:
            continue
        distance = ((center_x - 0.5) ** 2 + ((center_y - 0.42) * 0.8) ** 2) ** 0.5
        score = (width * height) ** 0.5 * float(np.exp(-5 * distance))
        candidates.append((score, x, y, width, height))

    if not candidates:
        raise FaceNotDetectedError(f"No frontal face detected in {image_path}")

    _, x, y, width, height = max(candidates, key=lambda candidate: candidate[0])
    center_x = x + width / 2
    center_y = y + height / 2 + height * 0.08
    crop_size = max(width, height) * 1.4
    left = max(0, int(center_x - crop_size / 2))
    top = max(0, int(center_y - crop_size / 2))
    right = min(image.width, int(center_x + crop_size / 2))
    bottom = min(image.height, int(center_y + crop_size / 2))

    face = image.crop((left, top, right, bottom))
    return face.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)


def augment_face_images(face: Image.Image) -> dict[str, Image.Image]:
    """Return the original and rotation, flip, grayscale, and brightness variants."""
    face = face.convert("RGB")
    return {
        "original": face,
        "rotated": face.rotate(
            8,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(127, 127, 127),
        ),
        "flipped": ImageOps.mirror(face),
        "grayscale": ImageOps.grayscale(face).convert("RGB"),
        "brightness": ImageEnhance.Brightness(face).enhance(1.15),
    }


def extract_hog_features(face: Image.Image) -> np.ndarray:
    """Convert a face crop to a fixed-length HOG feature vector."""
    grayscale = np.asarray(face.convert("L").resize(IMAGE_SIZE, Image.Resampling.LANCZOS))
    return hog(
        grayscale,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)


def _build_estimator(random_state: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    )


def _choose_evaluation_images(
    images: list[tuple[str, Path]],
    random_state: int,
) -> set[Path]:
    random_generator = random.Random(random_state)
    by_identity: dict[str, list[Path]] = {}
    for identity, image_path in images:
        by_identity.setdefault(identity, []).append(image_path)
    return {
        random_generator.choice(sorted(paths))
        for paths in by_identity.values()
    }


def _feature_rows(
    images: list[tuple[str, Path]],
    evaluation_images: set[Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    feature_names: list[str] = []

    for identity, image_path in images:
        face = load_face_crop(image_path)
        split = "test" if image_path in evaluation_images else "train"
        for augmentation, augmented_face in augment_face_images(face).items():
            features = extract_hog_features(augmented_face)
            if not feature_names:
                feature_names = [f"hog_{index:04d}" for index in range(len(features))]
            try:
                source_image = image_path.resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                source_image = image_path.resolve().as_posix()
            rows.append(
                {
                    "identity": identity,
                    "source_image": source_image,
                    "augmentation": augmentation,
                    "split": split,
                    **dict(zip(feature_names, features.tolist())),
                }
            )
    return rows, feature_names


def train_face_model(
    image_dir: Path = IMAGE_DIR,
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    random_state: int = 42,
) -> dict[str, float]:
    """Evaluate on one held-out original per identity, then train and save the full model."""
    images = discover_images(image_dir)
    evaluation_images = _choose_evaluation_images(images, random_state)
    rows, feature_names = _feature_rows(images, evaluation_images)
    feature_table = pd.DataFrame(rows)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(features_path, index=False)

    columns = feature_names
    training_rows = feature_table[feature_table["split"] == "train"]
    testing_rows = feature_table[
        (feature_table["split"] == "test")
        & (feature_table["augmentation"] == "original")
    ]
    evaluation_model = _build_estimator(random_state)
    evaluation_model.fit(training_rows[columns], training_rows["identity"])
    probabilities = evaluation_model.predict_proba(testing_rows[columns])
    predictions = evaluation_model.classes_[np.argmax(probabilities, axis=1)]
    metrics = {
        "accuracy": float(accuracy_score(testing_rows["identity"], predictions)),
        "precision_macro": float(
            precision_score(testing_rows["identity"], predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(testing_rows["identity"], predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(testing_rows["identity"], predictions, average="macro", zero_division=0)
        ),
        "log_loss": float(
            log_loss(
                testing_rows["identity"],
                probabilities,
                labels=evaluation_model.classes_,
            )
        ),
    }

    final_model = _build_estimator(random_state)
    final_model.fit(feature_table[columns], feature_table["identity"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_names": columns,
            "classes": list(final_model.classes_),
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "image_size": IMAGE_SIZE,
        },
        model_path,
    )
    return metrics


def verify_face(
    image_path: str | Path,
    model_path: Path = MODEL_PATH,
    confidence_threshold: float | None = None,
) -> dict[str, str | float | bool | None]:
    """Predict an identity and reject images below the model confidence threshold."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Face model does not exist: {model_path}. Run train_face_model() first."
        )
    if confidence_threshold is not None and not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    artifact = joblib.load(model_path)
    try:
        face = load_face_crop(Path(image_path))
    except FaceNotDetectedError:
        return {
            "identity": None,
            "authorized": False,
            "confidence": 0.0,
            "reason": "face_not_detected",
        }
    feature_vector = extract_hog_features(face)
    if len(feature_vector) != len(artifact["feature_names"]):
        raise ValueError("The face model and extracted feature dimensions do not match")
    features = pd.DataFrame([feature_vector], columns=artifact["feature_names"])

    probabilities = artifact["model"].predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    threshold = (
        artifact["confidence_threshold"]
        if confidence_threshold is None
        else confidence_threshold
    )
    authorized = confidence >= threshold
    return {
        "identity": str(artifact["classes"][best_index]) if authorized else None,
        "authorized": authorized,
        "confidence": confidence,
        "reason": "verified" if authorized else "confidence_below_threshold",
    }


if __name__ == "__main__":
    print(json.dumps(train_face_model(), indent=2))