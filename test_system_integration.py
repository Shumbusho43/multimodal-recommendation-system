"""Tests for the integration layer. Standard library only:  python -m unittest discover tests"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import system_integration as si  # noqa: E402

IDENTITY_MAP = {"allan": 151, "david": 192, "nazira": 114, "shamila": 100}


class Fakes:
    """Configurable stand-ins that record how they were called."""

    def __init__(self, face=None, voice=None, product="Electronics"):
        self.face = face or {"identity": "allan", "verified": True, "confidence": 0.95}
        self.voice = voice or {"identity": "allan", "verified": True, "confidence": 0.95}
        self.product = product
        self.product_calls = []

    def face_fn(self, path):
        return self.face

    def voice_fn(self, path):
        return self.voice

    def product_fn(self, customer_id):
        self.product_calls.append(customer_id)
        if isinstance(self.product, Exception):
            raise self.product
        return self.product

    def system(self):
        return si.SecureRecommendationSystem(self.face_fn, self.voice_fn, self.product_fn, identity_map=IDENTITY_MAP)


class TransactionFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.img = Path(self.tmp.name) / "face.jpg"
        self.aud = Path(self.tmp.name) / "voice.wav"
        self.img.write_bytes(b"x")
        self.aud.write_bytes(b"x")

    def test_authorized_user_gets_product(self):
        f = Fakes()
        r = f.system().run_transaction(self.img, self.aud)
        self.assertEqual((r.status, r.granted, r.product, r.customer_id), (si.GRANTED, True, "Electronics", 151))

    def test_face_failure_denies_and_never_runs_product_model(self):
        f = Fakes(face={"identity": "unknown", "verified": False, "confidence": 0.1})
        r = f.system().run_transaction(self.img, self.aud)
        self.assertEqual(r.status, si.DENIED_FACE)
        self.assertIsNone(r.product)
        self.assertEqual(f.product_calls, [])

    def test_voice_failure_denies_and_hides_product(self):
        f = Fakes(voice={"identity": "unknown", "verified": False, "confidence": 0.1})
        r = f.system().run_transaction(self.img, self.aud)
        self.assertEqual(r.status, si.DENIED_VOICE)
        self.assertIsNone(r.product)
        self.assertEqual(len(f.product_calls), 1)  # prediction ran after the face step, but was never shown

    def test_voice_of_a_different_member_is_denied(self):
        f = Fakes(voice={"identity": "david", "verified": True, "confidence": 0.99})
        r = f.system().run_transaction(self.img, self.aud)
        self.assertEqual(r.status, si.DENIED_VOICE)
        self.assertIn("david", r.voice.reason)

    def test_low_voice_confidence_is_denied(self):
        f = Fakes(voice={"identity": "allan", "verified": True, "confidence": 0.30})
        self.assertEqual(f.system().run_transaction(self.img, self.aud).status, si.DENIED_VOICE)

    def test_face_is_decided_by_its_own_similarity_verdict(self):
        # The face model decides by cosine similarity; its classifier probability is not a gate.
        f = Fakes(face={"identity": "allan", "authorized": True, "confidence": 0.45, "similarity": 0.82})
        r = f.system().run_transaction(self.img, self.aud)
        self.assertEqual(r.status, si.GRANTED)
        self.assertAlmostEqual(r.face.similarity, 0.82)

    def test_face_rejected_by_similarity_gives_readable_reason(self):
        f = Fakes(face={"identity": None, "authorized": False, "confidence": 0.45, "similarity": 0.2,
                        "reason": "similarity_below_threshold"})
        r = f.system().run_transaction(self.img, self.aud)
        self.assertEqual(r.status, si.DENIED_FACE)
        self.assertIn("not similar enough", r.face.reason)

    def test_identity_outside_group_is_denied(self):
        f = Fakes(face={"identity": "mallory", "verified": True, "confidence": 0.99})
        self.assertEqual(f.system().run_transaction(self.img, self.aud).status, si.DENIED_FACE)

    def test_audio_is_requested_only_after_face_passes(self):
        asked = []
        f = Fakes(face={"identity": "unknown", "verified": False})
        r = f.system().run_transaction(self.img, lambda: asked.append(1) or self.aud)
        self.assertEqual((r.status, asked), (si.DENIED_FACE, []))
        f = Fakes()
        r = f.system().run_transaction(self.img, lambda: asked.append(1) or self.aud)
        self.assertEqual((r.status, asked), (si.GRANTED, [1]))

    def test_missing_file_is_a_clean_error(self):
        r = Fakes().system().run_transaction("/no/such/face.jpg", self.aud)
        self.assertEqual(r.status, si.ERROR)
        self.assertIn("not found", r.message)

    def test_wrong_extension_and_empty_file_are_errors(self):
        txt = Path(self.tmp.name) / "face.txt"
        txt.write_bytes(b"x")
        self.assertEqual(Fakes().system().run_transaction(txt, self.aud).status, si.ERROR)
        empty = Path(self.tmp.name) / "empty.jpg"
        empty.write_bytes(b"")
        self.assertEqual(Fakes().system().run_transaction(empty, self.aud).status, si.ERROR)

    def test_unknown_customer_is_an_error_not_a_crash(self):
        f = Fakes(product=ValueError("Customer 151 was not found"))
        r = f.system().run_transaction(self.img, self.aud)
        self.assertEqual(r.status, si.ERROR)
        self.assertIsNone(r.product)

    def test_model_crash_becomes_error(self):
        f = Fakes()
        f.face_fn = lambda path: 1 / 0
        r = f.system().run_transaction(self.img, self.aud)
        self.assertEqual(r.status, si.ERROR)

    def test_member_without_customer_id_is_an_error(self):
        f = Fakes(face={"identity": "nazira", "verified": True, "confidence": 0.9},
                  voice={"identity": "nazira", "verified": True, "confidence": 0.9})
        system = si.SecureRecommendationSystem(f.face_fn, f.voice_fn, f.product_fn, identity_map={"allan": 151})
        self.assertEqual(system.run_transaction(self.img, self.aud).status, si.ERROR)


class NormalizationTests(unittest.TestCase):
    def test_accepted_return_shapes(self):
        n = si.normalize_verification
        self.assertTrue(n(("Allan", True, 0.9), "face").verified)
        self.assertEqual(n(("Allan", True), "face").identity, "allan")
        self.assertTrue(n("David", "voice").verified)
        self.assertFalse(n("unknown", "voice").verified)
        self.assertFalse(n(False, "face").verified)
        self.assertFalse(n(True, "face").verified)  # verified but nobody named -> cannot map to a customer
        self.assertTrue(n({"label": "shamila", "status": "authorized", "probability": 0.8}, "face").verified)

    def test_unsupported_value_raises(self):
        with self.assertRaises(si.ModelRuntimeError):
            si.normalize_verification(3.14, "face")


class RepoConsistencyTests(unittest.TestCase):
    def test_identity_map_covers_members_and_ids_exist_in_dataset(self):
        mapping = si.load_identity_map()
        self.assertEqual(set(mapping), set(si.AUTHORIZED_IDENTITIES))
        with (ROOT / "datasets" / "merged_dataset.csv").open(newline="", encoding="utf-8") as fh:
            ids = {int(row["customer_id"]) for row in csv.DictReader(fh)}
        for who, cid in mapping.items():
            self.assertIn(int(cid), ids, f"{who} -> {cid} is not in merged_dataset.csv")

    def test_demo_scenarios_with_stand_in_models(self):
        scenarios = json.loads((ROOT / "demo_scenarios.json").read_text(encoding="utf-8"))
        system = si.SecureRecommendationSystem(si.mock_verify_face, si.mock_verify_voice,
                                               si.mock_predict_product, identity_map=IDENTITY_MAP)
        for sc in scenarios:
            if sc.get("optional") and not ((ROOT / sc["face"]).exists() and (ROOT / sc["voice"]).exists()):
                continue
            r = system.run_transaction(ROOT / sc["face"], ROOT / sc["voice"])
            self.assertEqual(r.status, sc["expected"], sc["name"])


def _real_product_model_loads() -> bool:
    try:
        import product_recommendation as pr
        pr.predict_product(151)
        return True
    except Exception:
        return False


@unittest.skipUnless(_real_product_model_loads(), "real product model needs the pinned scikit-learn (requirements.txt)")
class CliWithStandInTests(unittest.TestCase):
    def test_cli_demo_end_to_end(self):
        proc = subprocess.run([sys.executable, str(ROOT / "main.py"), "--demo", "--mock", "--delay", "0", "--no-results-file"],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("5/5 scenarios behaved as expected", proc.stdout)

    def test_cli_single_transaction_json(self):
        proc = subprocess.run([sys.executable, str(ROOT / "main.py"), "--mock", "--json", "--no-log",
                               "--face", "datasets/images/Allan/Allan_neutral.jpg",
                               "--voice", "datasets/audio/Allan_confirm.wav"],
                              capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual((data["status"], data["identity"]), ("GRANTED", "allan"))
        self.assertIn(data["product"], {"Books", "Clothing", "Electronics", "Groceries", "Sports"})


def _real_models_ready() -> bool:
    try:
        return all(v == "READY" for v in si.component_status().values())
    except Exception:
        return False


@unittest.skipUnless(_real_models_ready(), "real face/voice/product models need the dependencies in requirements.txt")
class RealModelEndToEndTests(unittest.TestCase):
    """Runs the three real models (no stand-ins) through the whole workflow."""

    @classmethod
    def setUpClass(cls):
        import warnings
        warnings.filterwarnings("ignore")
        comps = si.load_components(allow_mock=False)
        cls.system = si.SecureRecommendationSystem(comps.face_fn, comps.voice_fn, comps.product_fn)

    def run_tx(self, face, voice):
        return self.system.run_transaction(ROOT / face, ROOT / voice)

    def test_every_member_is_granted_with_own_face_and_voice(self):
        faces = {"allan": "datasets/images/Allan/Allan_neutral.jpg", "david": "datasets/images/David/David_neutral.JPG",
                 "nazira": "datasets/images/Nazira/Nazira_neutral.HEIC", "shamila": "datasets/images/Shamila/Shamila_neutral.jpeg"}
        for who, face in faces.items():
            r = self.run_tx(face, f"datasets/audio/{who.capitalize()}_confirm.wav")
            self.assertEqual((r.status, r.identity), (si.GRANTED, who), who)
            self.assertIn(r.product, {"Books", "Clothing", "Electronics", "Groceries", "Sports"})

    def test_unknown_voice_is_denied_after_valid_face(self):
        for phrase in ("approve", "confirm"):
            r = self.run_tx("datasets/images/Allan/Allan_neutral.jpg", f"datasets/audio/unknown_{phrase}.wav")
            self.assertEqual(r.status, si.DENIED_VOICE)
            self.assertIsNone(r.product)

    def test_every_cross_member_voice_is_denied(self):
        faces = {"allan": "datasets/images/Allan/Allan_neutral.jpg", "david": "datasets/images/David/David_neutral.JPG"}
        for face_owner, face in faces.items():
            for voice_owner in ("allan", "david", "nazira", "shamila"):
                if voice_owner == face_owner:
                    continue
                r = self.run_tx(face, f"datasets/audio/{voice_owner.capitalize()}_approve.wav")
                self.assertEqual(r.status, si.DENIED_VOICE, f"{face_owner} face + {voice_owner} voice")

    def test_stranger_photo_is_denied(self):
        photos = sorted((ROOT / "datasets" / "unauthorized_images").glob("*"))
        self.assertTrue(photos, "datasets/unauthorized_images has no photo to test with")
        for photo in photos:
            r = self.system.run_transaction(photo, ROOT / "datasets/audio/Allan_confirm.wav")
            self.assertEqual(r.status, si.DENIED_FACE, photo.name)
            self.assertIsNone(r.product)

    def test_image_without_a_face_is_denied(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            blank = Path(tmp) / "blank.png"
            Image.new("RGB", (200, 200), (128, 128, 128)).save(blank)
            r = self.system.run_transaction(blank, ROOT / "datasets/audio/Allan_confirm.wav")
        self.assertEqual(r.status, si.DENIED_FACE)


if __name__ == "__main__":
    unittest.main()
