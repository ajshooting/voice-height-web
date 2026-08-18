import type {
  InferenceAssetUrls,
  InferenceWorkerResponse,
} from "./protocol";

export type InferenceEventListener = (event: InferenceWorkerResponse) => void;

export class InferenceClient {
  readonly #worker: Worker;
  readonly #listeners = new Set<InferenceEventListener>();
  readonly #pending = new Map<
    number,
    {
      resolve: (value: Extract<InferenceWorkerResponse, { type: "result" }>) => void;
      reject: (error: InferenceClientError) => void;
    }
  >();
  #nextId = 1;
  #failed = false;

  constructor() {
    this.#worker = new Worker(new URL("./worker.ts", import.meta.url), {
      type: "module",
      name: "voice-height-inference",
    });
    this.#worker.addEventListener(
      "message",
      (message: MessageEvent<InferenceWorkerResponse>) => {
        this.#handleMessage(message.data);
      },
    );
    this.#worker.addEventListener("error", () => this.#handleFatalError());
    this.#worker.addEventListener("messageerror", () => this.#handleFatalError());
  }

  prepare(assets: InferenceAssetUrls): void {
    this.#worker.postMessage({ type: "prepare", assets });
  }

  infer(samples: Float32Array, sampleRate: number): Promise<
    Extract<InferenceWorkerResponse, { type: "result" }>
  > {
    const id = this.#nextId;
    this.#nextId += 1;
    const transferable = samples.buffer.slice(
      samples.byteOffset,
      samples.byteOffset + samples.byteLength,
    );

    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage(
        { type: "infer", id, samples: transferable, sampleRate },
        [transferable],
      );
    });
  }

  subscribe(listener: InferenceEventListener): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  dispose(): void {
    this.#worker.terminate();
    this.#rejectAll(
      new InferenceClientError("INFERENCE_FAILED", "推定処理を終了しました。"),
    );
    this.#listeners.clear();
  }

  #handleMessage(event: InferenceWorkerResponse): void {
    for (const listener of this.#listeners) {
      listener(event);
    }

    if (event.type === "result") {
      const pending = this.#pending.get(event.id);
      if (pending) {
        this.#pending.delete(event.id);
        pending.resolve(event);
      }
    } else if (event.type === "error" && event.id !== undefined) {
      const pending = this.#pending.get(event.id);
      if (pending) {
        this.#pending.delete(event.id);
        pending.reject(new InferenceClientError(event.code, event.message));
      }
    }
  }

  #rejectAll(error: InferenceClientError): void {
    for (const pending of this.#pending.values()) {
      pending.reject(error);
    }
    this.#pending.clear();
  }

  #handleFatalError(): void {
    if (this.#failed) {
      return;
    }
    this.#failed = true;
    const error = new InferenceClientError(
      "INFERENCE_FAILED",
      "推定用のバックグラウンド処理が停止しました。モデルを再準備してください。",
    );
    this.#rejectAll(error);
    this.#handleMessage({
      type: "error",
      code: error.code,
      message: error.message,
    });
    this.#worker.terminate();
  }
}

export class InferenceClientError extends Error {
  constructor(
    readonly code: Extract<InferenceWorkerResponse, { type: "error" }>["code"],
    message: string,
  ) {
    super(message);
  }
}
