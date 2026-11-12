# Multimodal User Identity and Product Recommendation System

ALU Formative 2 - Multimodal Data Preprocessing. A user must pass **face recognition** and **voice verification** before the system shows a personalised product recommendation.

```
Start -> Face check --fail--> ACCESS DENIED
            | ok
            v
     Product model runs (result held, not shown)
            |
            v
     Voice check --fail--> ACCESS DENIED
            | ok
            v
     Display predicted product
```

The voice must also belong to the **same person** whose face was verified (face of A + voice of B is denied).

## Team

| Member | Component | Files |
|--------|-----------|-------|
| Shamila | Data merge, EDA, product recommendation model | `src/product_recommendation.py`, `models/product_model.pkl`, `datasets/merged_dataset.csv`, `notebooks/data_merge.ipynb` |
| David | Image processing, facial recognition model | `src/facial_recognition.py`, `models/face_model.pkl`, `datasets/image_features.csv`, `notebooks/image_preprocessing.ipynb` |
| Nazira | Audio processing, voiceprint verification model | `src/voice_verification.py`, `models/voice_model.pkl`, `datasets/audio_features.csv`, `notebooks/audio_preprocessing.ipynb` |
| Allan | System integration, command-line app, system testing, evaluation notebook | `main.py`, `src/system_integration.py`, `identity_map.json`, `demo_scenarios.json`, `tests/`, `results/`, `notebooks/model_evaluation.ipynb` |

## Setup

Requires Python 3.10+ (tested on 3.13).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> All three `.pkl` models were saved with **scikit-learn 1.6.1**. Other versions fail to load them, so keep the pinned version.

## Run

```bash
python main.py                                      # interactive
python main.py --face PATH.jpg --voice PATH.wav     # one transaction
python main.py --demo                               # authorized + unauthorized scenarios
python main.py --check                              # which models are ready?
python main.py --face F --voice V --json            # machine-readable result
```

Example:

```bash
python main.py --face datasets/images/Allan/Allan_neutral.jpg --voice datasets/audio/Allan_confirm.wav
```

Exit codes: `0` access granted, `1` access denied, `2` error (missing file, bad type, model problem).

Accepted inputs: images `.jpg .jpeg .png .bmp .heic .heif`; audio `.wav .mp3 .m4a .flac .ogg`. Each attempt is appended to `logs/access_log.csv` 
## System demonstration and testing

`python main.py --demo` runs the scenarios in `demo_scenarios.json` and writes the table in `results/system_test_results.md`:

1. Authorized transaction, Allan (face + voice) - access granted, product shown
2. Authorized transaction, Nazira (HEIC photo + voice) - access granted
3. Unauthorized: photo of a stranger (`datasets/unauthorized_images/unknown_astronaut.png`) - denied at step 1
4. Unauthorized: valid face, unknown voice (`datasets/audio/unknown_confirm.wav`) - denied at step 3
5. Identity mismatch: Allan's face with David's voice - denied at step 3

Unit and end-to-end tests (standard library `unittest`; the end-to-end tests run the three real models):

```bash
python -m unittest discover tests
```

## Evaluation

`notebooks/model_evaluation.ipynb` recomputes metrics for the face, voice and product models, measures how well strangers are rejected, tests the face model's robustness to photo changes, and evaluates the combined logic over 20 face/voice combinations (4 genuine pairs granted, 16 impostor pairs denied, false-accept rate 0%).

## Integration contract (what each model provides)

| Module | Function | Input | Returns |
|--------|----------|-------|---------|
| `src/facial_recognition.py` | `verify_face(image_path)` | path to a raw image | `{"identity", "authorized", "confidence", "similarity", "reason"}` |
| `src/voice_verification.py` | `verify_voice(audio_path)` | path to a raw audio file | `{"identity", "verified", "confidence"}` |
| `src/product_recommendation.py` | `predict_product(customer_id)` | customer id (int or `A151`) | product name `str`; `ValueError` if the id is unknown |

The face model (YuNet detector + SFace embeddings + logistic regression, ONNX weights in `models/pretrained/`) accepts a face only when its cosine similarity to the enrolled photos reaches 0.45, so strangers are rejected even though the classifier always names a member. The voice model uses a 0.60 confidence threshold, which the integration layer also enforces.

The integration layer accepts either `verified` or `authorized` as the status key, lowercases identities (`allan`, `david`, `nazira`, `shamila`), and denies access when no identity is named, the identity is not a group member, or the voice confidence is below 0.60. `identity_map.json` links each member to a `customer_id` in `datasets/merged_dataset.csv`; that id is passed to `predict_product` after the face is verified.


## Repository layout

```
main.py                     CLI entry point
identity_map.json           member -> customer_id
demo_scenarios.json         scripted demo cases
requirements.txt            dependencies (scikit-learn pinned)
src/                        system_integration.py + the three model modules + data_processing.py
models/                     face_model.pkl, voice_model.pkl, product_model.pkl, pretrained/ (YuNet + SFace ONNX)
datasets/                   merged_dataset.csv, image_features.csv, audio_features.csv, images/, audio/, unauthorized_images/
notebooks/                  data_merge, image_preprocessing, audio_preprocessing, model_evaluation
results/                    system_test_results.md (from --demo)
tests/                      unit and end-to-end tests
```

