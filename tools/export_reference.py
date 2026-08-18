# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==1.26.4",
#   "onnx==1.17.0",
#   "onnxruntime==1.20.1",
#   "scikit-learn==1.4.0",
#   "soundfile==0.12.1",
#   "speechbrain==1.0.1",
#   "torch==2.4.1",
#   "torchaudio==2.4.1",
# ]
# ///

"""Export the trusted HeightCeleb reference into browser-safe artifacts.

This script is development tooling only. It uses pinned source revisions,
verifies every executable source artifact by SHA-256 before deserializing it,
and writes no Python environment into the repository when run with
``uv run --isolated --no-project``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


SPACE_ID = "stachu86/HeightCeleb-estimator-demo"
SPACE_REVISION = "cd5feff37a9b8f77a69380dcc90fc55756f4330f"
ECAPA_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
EMBEDDING_DIMENSION = 192

SOURCE_HASHES = {
    "classifier.ckpt": (
        "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535"
    ),
    "gender_classifier.pickle": (
        "cfbaeda5674736fa530ddb341e1e102c84fdbc54b57d5d0f47beac088a689a7a"
    ),
    "height_estimator_0.pickle": (
        "6bb6798d2f3b0fb772d1c67a736843022474f5860e2757dab01d7bfc0df2a02f"
    ),
    "height_estimator_1.pickle": (
        "4ed101b447d415e34c0407b2e4d5521f651cef83f7bf6ad9d022a09442b186c2"
    ),
    "embedding_model.ckpt": (
        "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
    ),
    "example1.wav": (
        "bf2dde5cb516939ff619d62fc07d4f4bec5b5d521aee3d07ae51828c9d93be0b"
    ),
    "hyperparams.yaml": (
        "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41"
    ),
    "label_encoder.txt": (
        "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9"
    ),
    "mean_var_norm_emb.ckpt": (
        "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33"
    ),
}

ECAPA_SOURCE_FILES = (
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
    "example1.wav",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
        )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_verified_pickle(path: Path, expected_hash: str) -> Any:
    verify_hash(path, expected_hash)
    with path.open("rb") as source:
        return pickle.load(source)  # noqa: S301 - hash-pinned, reviewed upstream files


@dataclass(frozen=True)
class FoldedLinearModel:
    weights: np.ndarray
    bias: float


def fold_pls(model: Any) -> FoldedLinearModel:
    if model.__class__.__name__ != "PLSRegression":
        raise TypeError(f"Expected PLSRegression, got {type(model)!r}")

    mean = np.asarray(model._x_mean, dtype=np.float64).reshape(-1)
    scale = np.asarray(model._x_std, dtype=np.float64).reshape(-1)
    coefficients = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])

    if mean.size != EMBEDDING_DIMENSION:
        raise ValueError(f"Unexpected PLS input size: {mean.size}")
    if scale.size != mean.size or coefficients.size != mean.size:
        raise ValueError("PLS parameter shapes do not match")
    if np.any(scale == 0):
        raise ValueError("PLS scaler contains a zero scale")

    weights = coefficients / scale
    bias = intercept - float(mean @ weights)
    return FoldedLinearModel(weights=weights, bias=bias)


def verify_folded_pls(model: Any, folded: FoldedLinearModel) -> None:
    random = np.random.default_rng(20260819)
    probes = random.normal(size=(8, EMBEDDING_DIMENSION))
    expected = np.asarray(model.predict(probes), dtype=np.float64).reshape(-1)
    actual = probes @ folded.weights + folded.bias
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)


def portable_regressors(gender_model: Any, female_model: Any, male_model: Any) -> dict:
    if gender_model.__class__.__name__ != "LogisticRegression":
        raise TypeError(
            f"Expected LogisticRegression, got {type(gender_model)!r}"
        )

    gender_weights = np.asarray(gender_model.coef_, dtype=np.float64).reshape(-1)
    gender_bias = float(
        np.asarray(gender_model.intercept_, dtype=np.float64).reshape(-1)[0]
    )
    gender_classes = np.asarray(gender_model.classes_).reshape(-1).tolist()
    if gender_weights.size != EMBEDDING_DIMENSION or gender_classes != [0, 1]:
        raise ValueError("Unexpected gender classifier shape or class order")

    female = fold_pls(female_model)
    male = fold_pls(male_model)
    verify_folded_pls(female_model, female)
    verify_folded_pls(male_model, male)

    return {
        "formatVersion": 1,
        "embeddingDimension": EMBEDDING_DIMENSION,
        "source": {
            "space": SPACE_ID,
            "revision": SPACE_REVISION,
            "sha256": {
                name: SOURCE_HASHES[name]
                for name in (
                    "gender_classifier.pickle",
                    "height_estimator_0.pickle",
                    "height_estimator_1.pickle",
                )
            },
            "license": "CC-BY-4.0",
            "modified": "Executable sklearn pickles converted to linear JSON parameters.",
        },
        "routing": {
            "kind": "binary-logistic-regression",
            "weights": gender_weights.tolist(),
            "bias": gender_bias,
            "classes": gender_classes,
            "classLabels": {"0": "female-model", "1": "male-model"},
        },
        "height": {
            "female-model": {
                "kind": "folded-pls-regression",
                "sourceComponents": int(female_model.n_components),
                "weights": female.weights.tolist(),
                "bias": female.bias,
            },
            "male-model": {
                "kind": "folded-pls-regression",
                "sourceComponents": int(male_model.n_components),
                "weights": male.weights.tolist(),
                "bias": male.bias,
            },
        },
    }


class EmbeddingOnly(torch.nn.Module):
    def __init__(self, embedding_model: torch.nn.Module) -> None:
        super().__init__()
        self.embedding_model = embedding_model

    def forward(
        self, features: torch.Tensor, relative_lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.embedding_model(features, relative_lengths)


def compute_reference(
    classifier: EncoderClassifier,
    waveform: torch.Tensor,
    gender_model: Any,
    female_model: Any,
    male_model: Any,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    relative_lengths = torch.ones(1, dtype=torch.float32)
    with torch.inference_mode():
        features = classifier.mods.compute_features(waveform)
        normalized = classifier.mods.mean_var_norm(features, relative_lengths)
        embedding_tensor = classifier.encode_batch(waveform, relative_lengths)

    embedding = embedding_tensor.detach().cpu().numpy().reshape(1, -1)
    if embedding.shape != (1, EMBEDDING_DIMENSION):
        raise ValueError(f"Unexpected embedding shape: {embedding.shape}")

    gender_score = float(gender_model.decision_function(embedding)[0])
    route = int(gender_model.predict(embedding)[0])
    female_height = float(np.asarray(female_model.predict(embedding)).reshape(-1)[0])
    male_height = float(np.asarray(male_model.predict(embedding)).reshape(-1)[0])
    automatic_height = male_height if route == 1 else female_height

    result = {
        "sampleRate": 16000,
        "waveformSamples": int(waveform.shape[-1]),
        "features": {
            "shape": list(normalized.shape),
            "dtype": "little-endian-float32",
        },
        "embedding": embedding.reshape(-1).astype(np.float64).tolist(),
        "routing": {
            "decision": gender_score,
            "class": route,
        },
        "heightCm": {
            "female-model": female_height,
            "male-model": male_height,
            "automatic": automatic_height,
        },
    }
    return normalized, relative_lengths, embedding, result


def export_onnx(
    classifier: EncoderClassifier,
    example_features: torch.Tensor,
    relative_lengths: torch.Tensor,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper = EmbeddingOnly(classifier.mods.embedding_model).eval()

    torch.onnx.export(
        wrapper,
        (example_features, relative_lengths),
        destination,
        input_names=["features", "relative_lengths"],
        output_names=["embedding"],
        dynamic_axes={"features": {1: "frames"}},
        opset_version=17,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )

    model = onnx.load(destination)
    onnx.checker.check_model(model)


def validate_onnx(
    classifier: EncoderClassifier,
    onnx_path: Path,
    waveforms: list[torch.Tensor],
) -> list[dict]:
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    results = []
    for waveform in waveforms:
        lengths = torch.ones(1, dtype=torch.float32)
        with torch.inference_mode():
            features = classifier.mods.compute_features(waveform)
            normalized = classifier.mods.mean_var_norm(features, lengths)
            expected = classifier.mods.embedding_model(normalized, lengths)

        actual = session.run(
            ["embedding"],
            {
                "features": normalized.detach().cpu().numpy().astype(np.float32),
                "relative_lengths": lengths.numpy(),
            },
        )[0]
        expected_array = expected.detach().cpu().numpy()
        actual_array = np.asarray(actual)
        max_abs = float(np.max(np.abs(expected_array - actual_array)))
        cosine = float(
            np.dot(expected_array.reshape(-1), actual_array.reshape(-1))
            / (
                np.linalg.norm(expected_array.reshape(-1))
                * np.linalg.norm(actual_array.reshape(-1))
            )
        )
        if cosine <= 0.99999 or max_abs >= 1e-3:
            raise RuntimeError(
                f"ONNX parity failed for {waveform.shape[-1]} samples: "
                f"cosine={cosine}, max_abs={max_abs}"
            )
        results.append(
            {
                "samples": int(waveform.shape[-1]),
                "frames": int(normalized.shape[1]),
                "cosineSimilarity": cosine,
                "maxAbsoluteDifference": max_abs,
            }
        )
    return results


def repeated_waveform(waveform: torch.Tensor, samples: int) -> torch.Tensor:
    repeats = math.ceil(samples / waveform.shape[-1])
    return waveform.repeat(1, repeats)[:, :samples]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing hash-pinned source files in height/ and "
            "ecapa/. Downloads are deliberately kept outside this script."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    source_dir = args.source_dir.resolve()
    public_models = repo_root / "public" / "models"
    fixtures = repo_root / "tests" / "fixtures"
    public_models.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="voice-height-export-") as temporary:
        pickle_paths = {
            name: source_dir / "height" / name
            for name in (
                "gender_classifier.pickle",
                "height_estimator_0.pickle",
                "height_estimator_1.pickle",
            )
        }
        gender_model = load_verified_pickle(
            pickle_paths["gender_classifier.pickle"],
            SOURCE_HASHES["gender_classifier.pickle"],
        )
        female_model = load_verified_pickle(
            pickle_paths["height_estimator_0.pickle"],
            SOURCE_HASHES["height_estimator_0.pickle"],
        )
        male_model = load_verified_pickle(
            pickle_paths["height_estimator_1.pickle"],
            SOURCE_HASHES["height_estimator_1.pickle"],
        )

        regressors = portable_regressors(
            gender_model, female_model, male_model
        )
        write_json(public_models / "height-regressors.json", regressors)

        snapshot = source_dir / "ecapa"
        for source_name in ECAPA_SOURCE_FILES:
            verify_hash(snapshot / source_name, SOURCE_HASHES[source_name])
        classifier = EncoderClassifier.from_hparams(
            source=str(snapshot),
            savedir=str(Path(temporary) / "loaded-model"),
            run_opts={"device": "cpu"},
            overrides={"pretrained_path": str(snapshot)},
        )
        classifier.eval()

        waveform, sample_rate = torchaudio.load(snapshot / "example1.wav")
        if sample_rate != 16000 or waveform.shape[0] != 1:
            raise ValueError(
                f"Unexpected reference audio: {sample_rate} Hz, {waveform.shape}"
            )
        waveform = waveform.to(dtype=torch.float32)

        normalized, lengths, _, reference = compute_reference(
            classifier,
            waveform,
            gender_model,
            female_model,
            male_model,
        )
        feature_path = fixtures / "speechbrain-example1.features.f32"
        feature_bytes = (
            normalized.detach().cpu().numpy().astype("<f4").tobytes(order="C")
        )
        feature_path.write_bytes(feature_bytes)
        reference["features"]["sha256"] = hashlib.sha256(feature_bytes).hexdigest()
        reference["features"]["file"] = feature_path.name

        audio_path = fixtures / "speechbrain-example1.wav"
        shutil.copy2(snapshot / "example1.wav", audio_path)
        reference["audioSha256"] = sha256(audio_path)

        onnx_path = artifacts_dir / "ecapa-embedding-fp32.onnx"
        export_onnx(classifier, normalized, lengths, onnx_path)
        parity = validate_onnx(
            classifier,
            onnx_path,
            [
                repeated_waveform(waveform, 16000),
                repeated_waveform(waveform, 80000),
                repeated_waveform(waveform, 160000),
            ],
        )

        reference["source"] = {
            "model": ECAPA_ID,
            "revision": ECAPA_REVISION,
            "checkpointSha256": SOURCE_HASHES["embedding_model.ckpt"],
            "license": "Apache-2.0",
        }
        reference["onnxFp32"] = {
            "file": onnx_path.name,
            "bytes": onnx_path.stat().st_size,
            "sha256": sha256(onnx_path),
            "parity": parity,
        }
        write_json(fixtures / "speechbrain-example1.reference.json", reference)

        write_json(
            artifacts_dir / "export-summary.json",
            {
                "fp32Model": reference["onnxFp32"],
                "portableRegressorBytes": (
                    public_models / "height-regressors.json"
                ).stat().st_size,
                "fixture": str(
                    fixtures / "speechbrain-example1.reference.json"
                ),
            },
        )

    print(json.dumps(reference["onnxFp32"], indent=2))


if __name__ == "__main__":
    main()
