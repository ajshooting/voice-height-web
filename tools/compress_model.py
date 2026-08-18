# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==1.26.4",
#   "onnx==1.17.0",
#   "onnxruntime==1.20.1",
# ]
# ///

"""Create and validate a weight-compressed ONNX model.

Only FLOAT initializers used exclusively as Conv weights are compressed. INT8
mode gives each output channel a symmetric scale and inserts DequantizeLinear;
FP16 storage mode inserts Cast back to FLOAT. Both modes therefore keep FLOAT
activations and graph inputs/outputs while reducing the serialized weight size.

Run this development tool with ``uv run --isolated --no-project`` so its pinned
Python dependencies do not create an environment inside the repository.
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


INT8_FORMAT = "weight-only-symmetric-int8-per-output-channel-qdq"
FP16_CAST_FORMAT = "weight-only-float16-with-float32-cast"
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
ECAPA_CHECKPOINT_SHA256 = (
    "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
)
OFFICIAL_FP32_ONNX_SHA256 = (
    "9cfdd98f669153ea126d65562663e2e7104a1cb08828315c2d46133a30c2db6f"
)
VALIDATION_CASES = (
    ("1-second", 101),
    ("official-example", 327),
    ("5-seconds", 501),
    ("10-seconds", 1001),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def graph_names(model: onnx.ModelProto) -> set[str]:
    names = {
        value.name
        for values in (
            model.graph.input,
            model.graph.output,
            model.graph.value_info,
            model.graph.initializer,
        )
        for value in values
        if value.name
    }
    for node in model.graph.node:
        names.update(name for name in node.input if name)
        names.update(name for name in node.output if name)
        if node.name:
            names.add(node.name)
    return names


def unique_name(base: str, used_names: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def compress_conv_weights(
    input_path: Path,
    output_path: Path,
    compression: str,
    exclude_weight_patterns: tuple[str, ...],
    input_sha256: str,
    verified_official_source: bool,
) -> dict[str, Any]:
    model = onnx.load(input_path, load_external_data=True)
    onnx.checker.check_model(model)

    default_opset = next(
        (entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")),
        None,
    )
    if default_opset is None or default_opset < 13:
        raise ValueError(
            "Per-axis DequantizeLinear requires the default ONNX opset to be 13 or newer"
        )

    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    graph_inputs = {value.name for value in model.graph.input}
    graph_outputs = {value.name for value in model.graph.output}
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = defaultdict(list)
    for node in model.graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name:
                consumers[input_name].append((node, input_index))

    eligible: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    skipped: list[dict[str, Any]] = []
    matched_exclude_patterns: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight_name = node.input[1]
        initializer = initializers.get(weight_name)
        if initializer is None:
            skipped.append(
                {
                    "node": node.name or "<unnamed>",
                    "weight": weight_name,
                    "reason": "Conv weight is not an initializer",
                }
            )
            continue
        if initializer.data_type != TensorProto.FLOAT:
            skipped.append(
                {
                    "node": node.name or "<unnamed>",
                    "weight": weight_name,
                    "reason": "Conv weight initializer is not FLOAT",
                }
            )
            continue
        matching_patterns = {
            pattern
            for pattern in exclude_weight_patterns
            if fnmatch.fnmatchcase(weight_name, pattern)
        }
        if matching_patterns:
            matched_exclude_patterns.update(matching_patterns)
            skipped.append(
                {
                    "node": node.name or "<unnamed>",
                    "weight": weight_name,
                    "reason": "Excluded by --exclude-weight",
                }
            )
            continue
        if weight_name in graph_inputs or weight_name in graph_outputs:
            skipped.append(
                {
                    "node": node.name or "<unnamed>",
                    "weight": weight_name,
                    "reason": "Conv weight is also a graph input or output",
                }
            )
            continue
        weight_consumers = consumers[weight_name]
        if any(
            consumer.op_type != "Conv" or input_index != 1
            for consumer, input_index in weight_consumers
        ):
            skipped.append(
                {
                    "node": node.name or "<unnamed>",
                    "weight": weight_name,
                    "reason": "Initializer has a consumer other than a Conv weight input",
                }
            )
            continue
        eligible[weight_name].append(node)

    unmatched_patterns = set(exclude_weight_patterns) - matched_exclude_patterns
    if unmatched_patterns:
        raise ValueError(
            "--exclude-weight did not match an eligible FLOAT Conv weight: "
            + ", ".join(sorted(unmatched_patterns))
        )
    if not eligible:
        raise RuntimeError("No eligible FLOAT Conv weight initializers were found")

    used_names = graph_names(model)
    replacement_initializers: list[onnx.TensorProto] = []
    restore_nodes: list[onnx.NodeProto] = []
    weight_stats: list[dict[str, Any]] = []

    for weight_name in sorted(eligible):
        initializer = initializers[weight_name]
        weights = np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)
        if weights.ndim < 2 or weights.shape[0] == 0:
            raise ValueError(
                f"Unexpected Conv weight shape for {weight_name}: {weights.shape}"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"Conv weights contain NaN or Inf: {weight_name}")

        restored_name = unique_name(f"{weight_name}.restored_float32", used_names)
        if compression == "int8-qdq":
            reduction_axes = tuple(range(1, weights.ndim))
            max_absolute = np.max(np.abs(weights), axis=reduction_axes)
            scales = (max_absolute / np.float32(127.0)).astype(np.float32)
            scales[max_absolute == 0] = np.float32(1.0)
            broadcast_shape = (weights.shape[0],) + (1,) * (weights.ndim - 1)
            stored = np.clip(
                np.rint(weights / scales.reshape(broadcast_shape)),
                -127,
                127,
            ).astype(np.int8)
            reconstructed = stored.astype(np.float32) * scales.reshape(broadcast_shape)

            stored_name = unique_name(f"{weight_name}.int8", used_names)
            scale_name = unique_name(f"{weight_name}.scale", used_names)
            zero_point_name = unique_name(f"{weight_name}.zero_point", used_names)
            node_name = unique_name(f"{weight_name}.DequantizeLinear", used_names)
            zero_points = np.zeros(weights.shape[0], dtype=np.int8)
            replacement_initializers.extend(
                (
                    numpy_helper.from_array(stored, name=stored_name),
                    numpy_helper.from_array(scales, name=scale_name),
                    numpy_helper.from_array(zero_points, name=zero_point_name),
                )
            )
            restore_nodes.append(
                helper.make_node(
                    "DequantizeLinear",
                    inputs=[stored_name, scale_name, zero_point_name],
                    outputs=[restored_name],
                    name=node_name,
                    axis=0,
                )
            )
            stored_bytes = stored.nbytes + scales.nbytes + zero_points.nbytes
            format_specific_stats = {
                "zeroChannels": int(np.count_nonzero(max_absolute == 0)),
                "minimumScale": float(np.min(scales)),
                "maximumScale": float(np.max(scales)),
            }
        elif compression == "fp16-cast":
            stored = weights.astype(np.float16)
            reconstructed = stored.astype(np.float32)
            stored_name = unique_name(f"{weight_name}.float16", used_names)
            node_name = unique_name(f"{weight_name}.CastFloat32", used_names)
            replacement_initializers.append(
                numpy_helper.from_array(stored, name=stored_name)
            )
            restore_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[stored_name],
                    outputs=[restored_name],
                    name=node_name,
                    to=TensorProto.FLOAT,
                )
            )
            stored_bytes = stored.nbytes
            format_specific_stats = {}
        else:
            raise ValueError(f"Unsupported compression: {compression}")

        for node in eligible[weight_name]:
            node.input[1] = restored_name

        absolute_error = np.abs(weights - reconstructed)
        weight_stats.append(
            {
                "name": weight_name,
                "shape": list(weights.shape),
                "elements": int(weights.size),
                "convConsumers": len(consumers[weight_name]),
                "originalWeightBytes": int(weights.nbytes),
                "storedWeightBytes": int(stored_bytes),
                **format_specific_stats,
                "meanAbsoluteReconstructionError": float(np.mean(absolute_error)),
                "maxAbsoluteReconstructionError": float(np.max(absolute_error)),
            }
        )

    replaced_names = set(eligible)
    retained_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in replaced_names
    ]
    original_nodes = list(model.graph.node)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)
    model.graph.initializer.extend(replacement_initializers)
    del model.graph.node[:]
    model.graph.node.extend(restore_nodes)
    model.graph.node.extend(original_nodes)

    metadata = {entry.key: entry.value for entry in model.metadata_props}
    official_source_metadata = {
        "voice-height.source-model": ECAPA_MODEL_ID,
        "voice-height.source-revision": ECAPA_REVISION,
        "voice-height.source-checkpoint-sha256": ECAPA_CHECKPOINT_SHA256,
        "voice-height.license": "Apache-2.0",
    }
    if verified_official_source:
        metadata.update(official_source_metadata)
        metadata.pop("voice-height.source-verification", None)
    else:
        for key in official_source_metadata:
            metadata.pop(key, None)
        metadata["voice-height.source-verification"] = "unverified-input"

    compression_format = INT8_FORMAT if compression == "int8-qdq" else FP16_CAST_FORMAT
    modified = (
        "Conv FLOAT initializers stored as symmetric per-output-channel INT8 "
        "and restored by DequantizeLinear."
        if compression == "int8-qdq"
        else "Conv FLOAT initializers stored as FLOAT16 and restored by Cast to FLOAT."
    )
    metadata.update(
        {
            "voice-height.compression": compression_format,
            "voice-height.compression.source-sha256": input_sha256,
            "voice-height.compression.modified": modified,
        }
    )
    if compression == "int8-qdq":
        metadata["voice-height.quantization.axis"] = "0"
    else:
        metadata.pop("voice-height.quantization.axis", None)
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        onnx.save_model(model, temporary_path)
        saved_model = onnx.load(temporary_path, load_external_data=True)
        onnx.checker.check_model(saved_model)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "format": compression_format,
        **({"axis": 0} if compression == "int8-qdq" else {}),
        "excludedWeightPatterns": list(exclude_weight_patterns),
        "compressedInitializers": len(weight_stats),
        "compressedConvNodes": sum(item["convConsumers"] for item in weight_stats),
        "skippedConvNodes": skipped,
        "originalConvWeightBytes": sum(
            item["originalWeightBytes"] for item in weight_stats
        ),
        "storedCompressedWeightBytes": sum(
            item["storedWeightBytes"] for item in weight_stats
        ),
        "weights": weight_stats,
    }


def load_fixture(feature_path: Path, reference_path: Path) -> np.ndarray:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    feature_metadata = reference.get("features")
    if not isinstance(feature_metadata, dict):
        raise ValueError("Fixture reference has no features object")
    shape = feature_metadata.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(not isinstance(dimension, int) or dimension <= 0 for dimension in shape)
    ):
        raise ValueError(f"Invalid fixture feature shape: {shape!r}")
    if shape[0] != 1 or shape[2] != 80:
        raise ValueError(f"Expected fixture shape [1, frames, 80], got {shape}")

    expected_hash = feature_metadata.get("sha256")
    actual_hash = sha256(feature_path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise RuntimeError(
            f"Fixture SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )

    raw = feature_path.read_bytes()
    expected_bytes = math.prod(shape) * np.dtype("<f4").itemsize
    if len(raw) != expected_bytes:
        raise ValueError(
            f"Fixture byte size does not match shape {shape}: "
            f"expected {expected_bytes}, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)


def fixture_case(base_features: np.ndarray, frames: int) -> np.ndarray:
    repeats = math.ceil(frames / base_features.shape[1])
    return np.tile(base_features, (1, repeats, 1))[:, :frames, :].copy()


def load_regressors(path: Path) -> dict[str, Any]:
    regressors = json.loads(path.read_text(encoding="utf-8"))
    if regressors.get("embeddingDimension") != 192:
        raise ValueError("Expected regressors for a 192-dimensional embedding")
    routing = regressors.get("routing")
    height = regressors.get("height")
    if not isinstance(routing, dict) or not isinstance(height, dict):
        raise ValueError("Regressor JSON is missing routing or height parameters")
    if len(routing.get("weights", [])) != 192:
        raise ValueError("Routing weights do not have 192 elements")
    for route in ("female-model", "male-model"):
        model = height.get(route)
        if not isinstance(model, dict) or len(model.get("weights", [])) != 192:
            raise ValueError(f"Height weights for {route} do not have 192 elements")
    return regressors


def downstream_prediction(
    embedding: np.ndarray,
    regressors: dict[str, Any],
) -> dict[str, Any]:
    flattened = embedding.astype(np.float64).reshape(-1)
    if flattened.size != 192:
        raise ValueError(f"Expected 192 embedding values, got {flattened.size}")

    routing = regressors["routing"]
    routing_score = float(
        flattened @ np.asarray(routing["weights"], dtype=np.float64)
        + float(routing["bias"])
    )
    route = "male-model" if routing_score > 0 else "female-model"
    heights = {}
    for model_name in ("female-model", "male-model"):
        model = regressors["height"][model_name]
        heights[model_name] = float(
            flattened @ np.asarray(model["weights"], dtype=np.float64)
            + float(model["bias"])
        )
    return {
        "routingScore": routing_score,
        "route": route,
        "femaleHeightCm": heights["female-model"],
        "maleHeightCm": heights["male-model"],
        "automaticHeightCm": heights[route],
    }


def make_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def run_validation(
    source_path: Path,
    candidate_path: Path,
    base_features: np.ndarray,
    minimum_cosine: float,
    maximum_absolute_difference: float | None,
    regressors: dict[str, Any] | None,
    maximum_height_difference: float,
    height_gate_all_cases: bool,
) -> dict[str, Any]:
    inputs = [
        (
            case_name,
            fixture_case(base_features, frames),
            np.ones(1, dtype=np.float32),
        )
        for case_name, frames in VALIDATION_CASES
    ]

    source_session = make_session(source_path)
    source_input_names = {value.name for value in source_session.get_inputs()}
    if source_input_names != {"features", "relative_lengths"}:
        raise ValueError(f"Unexpected source model inputs: {sorted(source_input_names)}")
    source_outputs = [
        np.asarray(
            source_session.run(
                ["embedding"],
                {"features": features, "relative_lengths": lengths},
            )[0],
            dtype=np.float32,
        )
        for _, features, lengths in inputs
    ]
    del source_session
    gc.collect()

    candidate_session = make_session(candidate_path)
    candidate_input_names = {value.name for value in candidate_session.get_inputs()}
    if candidate_input_names != source_input_names:
        raise ValueError(
            "Candidate model inputs differ from the source model: "
            f"{sorted(candidate_input_names)} != {sorted(source_input_names)}"
        )

    cases: list[dict[str, Any]] = []
    for (case_name, features, lengths), expected in zip(
        inputs, source_outputs, strict=True
    ):
        actual = np.asarray(
            candidate_session.run(
                ["embedding"],
                {"features": features, "relative_lengths": lengths},
            )[0],
            dtype=np.float32,
        )
        if actual.shape != expected.shape:
            raise ValueError(
                f"Candidate output shape {actual.shape} differs from {expected.shape}"
            )
        if not np.all(np.isfinite(actual)):
            raise RuntimeError(f"Candidate produced NaN or Inf for {case_name}")

        expected_flat = expected.astype(np.float64).reshape(-1)
        actual_flat = actual.astype(np.float64).reshape(-1)
        denominator = np.linalg.norm(expected_flat) * np.linalg.norm(actual_flat)
        if denominator == 0:
            raise RuntimeError(f"Zero-norm embedding for {case_name}")
        cosine = float(np.dot(expected_flat, actual_flat) / denominator)
        max_absolute = float(np.max(np.abs(expected_flat - actual_flat)))
        case: dict[str, Any] = {
            "case": case_name,
            "frames": int(features.shape[1]),
            "cosineSimilarity": cosine,
            "maxAbsoluteDifference": max_absolute,
            "passed": (
                cosine >= minimum_cosine
                and (
                    maximum_absolute_difference is None
                    or max_absolute <= maximum_absolute_difference
                )
            ),
        }
        if regressors is not None:
            expected_prediction = downstream_prediction(expected, regressors)
            actual_prediction = downstream_prediction(actual, regressors)
            height_difference = abs(
                expected_prediction["automaticHeightCm"]
                - actual_prediction["automaticHeightCm"]
            )
            downstream_passed = (
                expected_prediction["route"] == actual_prediction["route"]
                and height_difference <= maximum_height_difference
            )
            downstream_required = height_gate_all_cases or case_name == "official-example"
            case["downstream"] = {
                "source": expected_prediction,
                "candidate": actual_prediction,
                "routeMatch": (
                    expected_prediction["route"] == actual_prediction["route"]
                ),
                "automaticHeightAbsoluteDifferenceCm": height_difference,
                "requiredForPass": downstream_required,
                "passed": downstream_passed,
            }
            case["passed"] = case["passed"] and (
                downstream_passed or not downstream_required
            )
        cases.append(case)

    return {
        "provider": "CPUExecutionProvider",
        "fixtureConstruction": (
            "Use the hash-verified 327-frame SpeechBrain fixture unchanged for "
            "the official example; truncate or repeat it for the other cases."
        ),
        "thresholds": {
            "minimumCosineSimilarity": minimum_cosine,
            **(
                {"maximumAbsoluteDifference": maximum_absolute_difference}
                if maximum_absolute_difference is not None
                else {}
            ),
            **(
                {"maximumAutomaticHeightDifferenceCm": maximum_height_difference}
                if regressors is not None
                else {}
            ),
            **(
                {
                    "heightGateCases": (
                        "all" if height_gate_all_cases else ["official-example"]
                    )
                }
                if regressors is not None
                else {}
            ),
        },
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Source FP32 ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX model")
    parser.add_argument(
        "--compression",
        choices=("int8-qdq", "fp16-cast"),
        default="int8-qdq",
    )
    parser.add_argument(
        "--exclude-weight",
        action="append",
        default=[],
        metavar="GLOB",
        help="Keep matching Conv weight initializers as FP32; may be repeated",
    )
    parser.add_argument(
        "--allow-unverified-input",
        action="store_true",
        help=(
            "Allow an input whose SHA-256 differs from the pinned official FP32 "
            "ONNX; strips official provenance and license claims from the output"
        ),
    )
    parser.add_argument("--fixture-features", type=Path, required=True)
    parser.add_argument("--fixture-reference", type=Path, required=True)
    parser.add_argument(
        "--regressors",
        type=Path,
        help="Optional portable regressor JSON for end-to-end height comparison",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to <output stem>.manifest.json next to the output model",
    )
    parser.add_argument("--minimum-cosine", type=float, default=0.999)
    parser.add_argument(
        "--maximum-absolute-difference",
        type=float,
        help="Optional diagnostic gate; max abs is always recorded",
    )
    parser.add_argument("--maximum-height-difference", type=float, default=0.1)
    parser.add_argument(
        "--height-gate-all-cases",
        action="store_true",
        help="Require the height threshold for derived-duration cases too",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output model or manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    fixture_path = args.fixture_features.resolve()
    reference_path = args.fixture_reference.resolve()
    regressors_path = args.regressors.resolve() if args.regressors is not None else None
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else output_path.with_name(f"{output_path.stem}.manifest.json")
    )

    for path, description in (
        (input_path, "input model"),
        (fixture_path, "fixture features"),
        (reference_path, "fixture reference"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if input_path == output_path:
        raise ValueError("Input and output model paths must differ")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("Output model must use the .onnx extension")
    if not args.force:
        for path in (output_path, manifest_path):
            if path.exists():
                raise FileExistsError(f"Refusing to replace existing file: {path}")
    if not -1.0 <= args.minimum_cosine <= 1.0:
        raise ValueError("--minimum-cosine must be between -1 and 1")
    if (
        args.maximum_absolute_difference is not None
        and args.maximum_absolute_difference < 0
    ):
        raise ValueError("--maximum-absolute-difference must be non-negative")
    if args.maximum_height_difference < 0:
        raise ValueError("--maximum-height-difference must be non-negative")
    if regressors_path is not None and not regressors_path.is_file():
        raise FileNotFoundError(f"Missing portable regressors: {regressors_path}")

    input_sha = sha256(input_path)
    verified_official_source = input_sha == OFFICIAL_FP32_ONNX_SHA256
    if not verified_official_source and not args.allow_unverified_input:
        raise RuntimeError(
            "Source FP32 ONNX SHA-256 mismatch: expected "
            f"{OFFICIAL_FP32_ONNX_SHA256}, got {input_sha}. "
            "Use --allow-unverified-input only for a generic input whose "
            "provenance and license will be reviewed separately."
        )

    compression = compress_conv_weights(
        input_path,
        output_path,
        args.compression,
        tuple(args.exclude_weight),
        input_sha,
        verified_official_source,
    )
    gc.collect()
    fixture = load_fixture(fixture_path, reference_path)
    regressors = load_regressors(regressors_path) if regressors_path is not None else None
    validation = run_validation(
        input_path,
        output_path,
        fixture,
        args.minimum_cosine,
        args.maximum_absolute_difference,
        regressors,
        args.maximum_height_difference,
        args.height_gate_all_cases,
    )

    if verified_official_source:
        source_model = {
            "upstream": ECAPA_MODEL_ID,
            "revision": ECAPA_REVISION,
            "checkpointSha256": ECAPA_CHECKPOINT_SHA256,
            "license": "Apache-2.0",
            "file": input_path.name,
            "bytes": input_path.stat().st_size,
            "sha256": input_sha,
        }
    else:
        source_model = {
            "verification": "unverified-input",
            "file": input_path.name,
            "bytes": input_path.stat().st_size,
            "sha256": input_sha,
        }

    modified = (
        "stored selected Conv weights as per-output-channel symmetric INT8 "
        "with DequantizeLinear back to FLOAT."
        if args.compression == "int8-qdq"
        else "stored selected Conv weights as FLOAT16 with Cast back to FLOAT."
    )
    candidate_model = {
        "file": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256(output_path),
        **({"license": "Apache-2.0"} if verified_official_source else {}),
        "modified": (
            "Converted from the pinned PyTorch checkpoint to ONNX, then "
            f"{modified}"
            if verified_official_source
            else f"Converted from an unverified input ONNX model, then {modified}"
        ),
    }

    manifest = {
        "formatVersion": 1,
        "sourceModel": source_model,
        "candidateModel": candidate_model,
        "sizeReduction": {
            "bytes": input_path.stat().st_size - output_path.stat().st_size,
            "fraction": 1.0 - output_path.stat().st_size / input_path.stat().st_size,
        },
        "compression": compression,
        "validation": validation,
        "fixtures": {
            "features": {
                "file": fixture_path.name,
                "bytes": fixture_path.stat().st_size,
                "sha256": sha256(fixture_path),
            },
            "reference": {
                "file": reference_path.name,
                "bytes": reference_path.stat().st_size,
                "sha256": sha256(reference_path),
            },
        },
        **(
            {
                "regressors": {
                    "file": regressors_path.name,
                    "bytes": regressors_path.stat().st_size,
                    "sha256": sha256(regressors_path),
                }
            }
            if regressors_path is not None
            else {}
        ),
        "toolchain": {
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "candidateModel": manifest["candidateModel"],
                "sizeReduction": manifest["sizeReduction"],
                "compression": compression["format"],
                "compressedInitializers": compression["compressedInitializers"],
                "compressedConvNodes": compression["compressedConvNodes"],
                "skippedConvNodes": compression["skippedConvNodes"],
                "validation": validation,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if not validation["passed"]:
        raise SystemExit("Candidate failed one or more parity thresholds")


if __name__ == "__main__":
    main()
