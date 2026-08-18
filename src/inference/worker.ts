/// <reference lib="webworker" />

import * as ort from "onnxruntime-web/webgpu";

import { inspectAudio } from "../audio/quality";
import { resampleTo16k } from "../audio/resample";
import { extractSpeechBrainFeatures } from "../audio/speechbrain-features";
import {
  parseHeightRegressorBundle,
  predictHeight,
  type HeightRegressorBundle,
} from "./height-regressor";
import type {
  AudioQuality,
  InferenceAssetUrls,
  InferenceProvider,
  InferenceWorkerRequest,
  InferenceWorkerResponse,
} from "./protocol";

const worker = self as DedicatedWorkerGlobalScope;
const MINIMUM_SECONDS = 3;
const QUIET_RMS_THRESHOLD = 0.003;
const MAXIMUM_CLIPPED_FRACTION = 0.05;

let session: ort.InferenceSession | undefined;
let regressors: HeightRegressorBundle | undefined;
let provider: InferenceProvider | undefined;
let preparePromise: Promise<void> | undefined;
let modelDownloadMs = 0;
let modelInitializationMs = 0;

ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 1)
  : 1;
ort.env.wasm.proxy = false;

worker.addEventListener("message", (event: MessageEvent<InferenceWorkerRequest>) => {
  const request = event.data;
  if (request.type === "prepare") {
    preparePromise ??= prepare(request.assets).catch((error: unknown) => {
      preparePromise = undefined;
      post({
        type: "error",
        code: "MODEL_LOAD_FAILED",
        message: readableError(error, "モデルを読み込めませんでした。"),
      });
      throw error;
    });
    void preparePromise.catch(() => undefined);
    return;
  }

  void infer(request.id, request.samples, request.sampleRate);
});

async function prepare(assets: InferenceAssetUrls): Promise<void> {
  const downloadStarted = performance.now();
  const [modelBytes, regressorResponse] = await Promise.all([
    fetchModel(assets.model),
    fetch(assets.regressors, { cache: "force-cache", credentials: "same-origin" }),
  ]);
  if (!regressorResponse.ok) {
    throw new Error(`Regressor request failed (${regressorResponse.status})`);
  }
  regressors = parseHeightRegressorBundle(await regressorResponse.json());
  modelDownloadMs = performance.now() - downloadStarted;

  post({ type: "initializing" });
  const initializationStarted = performance.now();
  const initialized = await createSession(modelBytes, assets.preferredProvider);
  session = initialized.session;
  provider = initialized.provider;
  modelInitializationMs = performance.now() - initializationStarted;

  post({
    type: "ready",
    provider,
    modelDownloadMs,
    modelInitializationMs,
  });
}

async function createSession(
  modelBytes: Uint8Array,
  preferredProvider?: InferenceProvider,
): Promise<{
  session: ort.InferenceSession;
  provider: InferenceProvider;
}> {
  if (preferredProvider !== "wasm" && "gpu" in navigator) {
    try {
      return {
        session: await ort.InferenceSession.create(modelBytes.slice(), {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "all",
        }),
        provider: "webgpu",
      };
    } catch (error) {
      console.info("WebGPU initialization failed; falling back to WASM.", error);
    }
  }

  return {
    session: await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      extra: { session: { disable_quant_qdq: "1" } },
    }),
    provider: "wasm",
  };
}

async function fetchModel(url: string): Promise<Uint8Array> {
  const response = await fetch(url, {
    cache: "force-cache",
    credentials: "same-origin",
  });
  if (!response.ok) {
    throw new Error(`Model request failed (${response.status})`);
  }

  const totalHeader = response.headers.get("content-length");
  const totalBytes = totalHeader === null ? undefined : Number(totalHeader);
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    post({ type: "download-progress", loadedBytes: bytes.byteLength, totalBytes });
    return bytes;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loadedBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loadedBytes += value.byteLength;
    post({ type: "download-progress", loadedBytes, totalBytes });
  }

  const bytes = new Uint8Array(loadedBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function infer(
  id: number,
  samplesBuffer: ArrayBuffer,
  sampleRate: number,
): Promise<void> {
  try {
    await preparePromise;
    if (!session || !regressors || !provider) {
      throw new Error("Model has not been prepared");
    }

    const totalStarted = performance.now();
    const inputSamples = new Float32Array(samplesBuffer);
    const quality = inspectAudio(inputSamples, sampleRate);
    validateAudio(quality);

    post({ type: "stage", id, stage: "preprocessing" });
    const preprocessingStarted = performance.now();
    const waveform = resampleTo16k(inputSamples, sampleRate);
    const features = extractSpeechBrainFeatures(waveform);
    const preprocessingMs = performance.now() - preprocessingStarted;

    post({ type: "stage", id, stage: "inference" });
    const inferenceStarted = performance.now();
    const outputs = await session.run({
      features: new ort.Tensor("float32", features.data, [
        1,
        features.frames,
        features.bins,
      ]),
      relative_lengths: new ort.Tensor(
        "float32",
        new Float32Array([1]),
        [1],
      ),
    });
    const output = outputs.embedding;
    if (!output || !(output.data instanceof Float32Array)) {
      throw new Error("ECAPA model returned an unexpected output");
    }
    const prediction = predictHeight(output.data, regressors);
    const inferenceMs = performance.now() - inferenceStarted;

    post({
      type: "result",
      id,
      heightCm: prediction.heightCm,
      quality,
      metrics: {
        modelDownloadMs,
        modelInitializationMs,
        preprocessingMs,
        inferenceMs,
        totalMs: performance.now() - totalStarted,
        provider,
      },
    });
  } catch (error) {
    const qualityError = error instanceof AudioValidationError ? error : undefined;
    post({
      type: "error",
      id,
      code: qualityError?.code ?? "INFERENCE_FAILED",
      message: readableError(error, "推定処理に失敗しました。"),
    });
  }
}

function validateAudio(quality: AudioQuality): void {
  if (quality.durationSeconds < MINIMUM_SECONDS) {
    throw new AudioValidationError(
      "AUDIO_TOO_SHORT",
      "録音が短すぎます。3〜5秒ほど、普段の声で話してください。",
    );
  }
  if (quality.rms < QUIET_RMS_THRESHOLD) {
    throw new AudioValidationError(
      "AUDIO_TOO_QUIET",
      "声を十分に検出できませんでした。マイクに少し近づいて再録音してください。",
    );
  }
  if (quality.clippedFraction > MAXIMUM_CLIPPED_FRACTION) {
    throw new AudioValidationError(
      "AUDIO_CLIPPED",
      "音が大きすぎて歪んでいます。マイクから少し離れて再録音してください。",
    );
  }
}

class AudioValidationError extends Error {
  constructor(
    readonly code:
      | "AUDIO_TOO_SHORT"
      | "AUDIO_TOO_QUIET"
      | "AUDIO_CLIPPED",
    message: string,
  ) {
    super(message);
  }
}

function readableError(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function post(response: InferenceWorkerResponse): void {
  worker.postMessage(response);
}
