# Third-party notices

This project combines original application code with converted model artifacts and permissively licensed runtime libraries. The project `LICENSE` applies only to the original application code unless stated otherwise below.

## SpeechBrain ECAPA-TDNN model

- Work: `speechbrain/spkrec-ecapa-voxceleb`
- Author/project: SpeechBrain
- Source: <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb>
- Pinned revision: `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`
- Source checkpoint SHA-256: `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2`
- License: Apache License 2.0; see [`licenses/Apache-2.0.txt`](./licenses/Apache-2.0.txt)

Modification notice: the pinned PyTorch checkpoint was exported to an embedding-only ONNX graph. Selected Conv weights were then stored as per-output-channel symmetric INT8 and restored to FLOAT with `DequantizeLinear`; three accuracy-sensitive Conv weights remain FLOAT. The distributed artifact and complete conversion report are:

- `public/models/ecapa-voxceleb.onnx`
- `public/models/ecapa-voxceleb.manifest.json`
- generation tools under `tools/`

The bundled example WAV and its derived feature fixture originate from the same SpeechBrain model repository revision and are retained solely for numerical compatibility tests.

## HeightCeleb estimator coefficients

- Work: `HeightCeleb-estimator-demo`
- Space author: `stachu86` (Stanisław Kacprzak)
- Source: <https://huggingface.co/spaces/stachu86/HeightCeleb-estimator-demo>
- Pinned revision: `cd5feff37a9b8f77a69380dcc90fc55756f4330f`
- License metadata: Creative Commons Attribution 4.0 International; see [`licenses/CC-BY-4.0.txt`](./licenses/CC-BY-4.0.txt)
- Related paper: “HeightCeleb - an enrichment of VoxCeleb dataset with speaker height information”, Stanisław Kacprzak and Konrad Kowalczyk, 2024, <https://arxiv.org/abs/2410.12668>

Modification notice: the executable scikit-learn pickle artifacts were hash-verified, loaded only by the isolated export tool, and converted to non-executable linear JSON parameters in `public/models/height-regressors.json`. The original classifier and regressors are not distributed here. Their source hashes and transformation are recorded in that JSON file.

The Space declares CC BY 4.0 in its repository metadata but does not contain a separate license file for each pickle. Attribution and modification details are supplied here in reliance on that metadata. The binary router was trained using TIMIT; no TIMIT audio is distributed by this project.

## ONNX Runtime Web

- Package: `onnxruntime-web` and `onnxruntime-common` 1.27.0
- Copyright: Microsoft Corporation
- Source: <https://github.com/microsoft/onnxruntime>
- License: MIT; see [`licenses/ONNX-Runtime-MIT.txt`](./licenses/ONNX-Runtime-MIT.txt)

The production dependency tree also contains the following packages used by ONNX Runtime Web. Exact resolved versions are recorded in `pnpm-lock.yaml` and can be checked with `pnpm licenses list --prod`.

- protobuf.js 7.6.5 and its `@protobufjs/*` components, copyright Daniel Wirtz, BSD-3-Clause; see [`licenses/protobufjs-BSD-3-Clause.txt`](./licenses/protobufjs-BSD-3-Clause.txt)
- FlatBuffers 25.9.23 and `long` 5.3.2, Apache-2.0; see [`licenses/Apache-2.0.txt`](./licenses/Apache-2.0.txt)
- `guid-typescript` 1.0.9, author listed as nicolas, ISC; see [`licenses/guid-typescript-ISC.txt`](./licenses/guid-typescript-ISC.txt)
- `platform` 1.3.6, copyright Benjamin Tan and John-David Dalton, MIT; see [`licenses/platform-MIT.txt`](./licenses/platform-MIT.txt)

Vite emits this notice and every linked license file into the static production output, and the application footer links back to this notice.

## Development-only conversion dependencies

The browser application does not ship Python or these packages. Reproduction tools use pinned isolated environments containing SpeechBrain (Apache-2.0), PyTorch (BSD-style), ONNX (Apache-2.0), ONNX Runtime (MIT), NumPy (BSD-3-Clause), scikit-learn (BSD-3-Clause), torchaudio (BSD-2-Clause), SoundFile (BSD-3-Clause), and their dependencies. Their upstream distributions provide the applicable notices when installed by `uv`.

## Training-data provenance

The speaker encoder was trained on VoxCeleb data and the HeightCeleb work enriches VoxCeleb identities with height information. VoxCeleb audio originated from online videos; dataset/model license metadata does not erase third-party copyright, personality, or privacy interests in source recordings. This repository distributes no training audio, identity list, height dataset, or TIMIT corpus. The model must not be treated as a legal, biometric-identification, medical, or fairness-certified system.
