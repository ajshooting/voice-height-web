export interface InferenceAssetUrls {
  readonly model: string;
  readonly regressors: string;
  /** Development diagnostics can request the production fallback explicitly. */
  readonly preferredProvider?: InferenceProvider;
}

export type InferenceProvider = "webgpu" | "wasm";

export interface InferenceMetrics {
  readonly modelDownloadMs: number;
  readonly modelInitializationMs: number;
  readonly preprocessingMs: number;
  readonly inferenceMs: number;
  readonly totalMs: number;
  readonly provider: InferenceProvider;
}

export interface AudioQuality {
  readonly durationSeconds: number;
  readonly rms: number;
  readonly peak: number;
  readonly clippedFraction: number;
}

export type InferenceWorkerRequest =
  | {
      readonly type: "prepare";
      readonly assets: InferenceAssetUrls;
    }
  | {
      readonly type: "infer";
      readonly id: number;
      readonly samples: ArrayBuffer;
      readonly sampleRate: number;
    };

export type InferenceWorkerResponse =
  | {
      readonly type: "download-progress";
      readonly loadedBytes: number;
      readonly totalBytes?: number;
    }
  | {
      readonly type: "initializing";
    }
  | {
      readonly type: "ready";
      readonly provider: InferenceProvider;
      readonly modelDownloadMs: number;
      readonly modelInitializationMs: number;
    }
  | {
      readonly type: "stage";
      readonly id: number;
      readonly stage: "preprocessing" | "inference";
    }
  | {
      readonly type: "result";
      readonly id: number;
      readonly heightCm: number;
      readonly quality: AudioQuality;
      readonly metrics: InferenceMetrics;
    }
  | {
      readonly type: "error";
      readonly id?: number;
      readonly code:
        | "MODEL_LOAD_FAILED"
        | "AUDIO_TOO_SHORT"
        | "AUDIO_TOO_QUIET"
        | "AUDIO_CLIPPED"
        | "INFERENCE_FAILED";
      readonly message: string;
    };
