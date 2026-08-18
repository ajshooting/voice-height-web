import workletUrl from "./pcm-capture.worklet.ts?worker&url";

export interface RecordingSnapshot {
  readonly elapsedSeconds: number;
  readonly level: number;
}

export interface RecordedAudio {
  readonly samples: Float32Array;
  readonly sampleRate: number;
  readonly durationSeconds: number;
}

export type RecordingListener = (snapshot: RecordingSnapshot) => void;

export class PcmRecorder {
  readonly #listeners = new Set<RecordingListener>();
  #stream: MediaStream | undefined;
  #context: AudioContext | undefined;
  #source: MediaStreamAudioSourceNode | undefined;
  #capture: AudioWorkletNode | undefined;
  #silentGain: GainNode | undefined;
  #startPromise: Promise<void> | undefined;
  #chunks: Float32Array[] = [];
  #sampleCount = 0;
  #level = 0;
  #ticker: number | undefined;

  get recording(): boolean {
    return this.#context !== undefined;
  }

  subscribe(listener: RecordingListener): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  start(): Promise<void> {
    if (this.recording || this.#startPromise) {
      throw new Error("Recording is already active");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("このブラウザはマイク録音に対応していません。");
    }

    const operation = this.#startCapture();
    this.#startPromise = operation;
    return operation.finally(() => {
      if (this.#startPromise === operation) {
        this.#startPromise = undefined;
      }
    });
  }

  async #startCapture(): Promise<void> {
    // Start both browser-gated operations while the click's user activation is
    // still live. In particular, Safari may refuse a delayed AudioContext resume.
    const context = new AudioContext({ latencyHint: "interactive" });
    let streamPromise: Promise<MediaStream> | undefined;
    let stream: MediaStream | undefined;
    try {
      streamPromise = navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: { ideal: 1 },
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });
      const resumePromise = context.resume();
      const workletPromise = context.audioWorklet.addModule(workletUrl);
      [stream] = await Promise.all([
        streamPromise,
        resumePromise,
        workletPromise,
      ]);
      const source = context.createMediaStreamSource(stream);
      const capture = new AudioWorkletNode(context, "pcm-capture", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: "explicit",
        channelInterpretation: "speakers",
      });
      const silentGain = context.createGain();
      silentGain.gain.value = 0;

      this.#chunks = [];
      this.#sampleCount = 0;
      this.#level = 0;
      capture.port.addEventListener(
        "message",
        (event: MessageEvent<Float32Array>) => this.#acceptChunk(event.data),
      );
      capture.port.start();
      source.connect(capture);
      capture.connect(silentGain);
      silentGain.connect(context.destination);

      this.#stream = stream;
      this.#context = context;
      this.#source = source;
      this.#capture = capture;
      this.#silentGain = silentGain;
      this.#ticker = window.setInterval(() => this.#emit(), 50);
      this.#emit();
    } catch (error) {
      if (!stream && streamPromise) {
        stream = await streamPromise.catch(() => undefined);
      }
      if (context.state !== "closed") {
        await context.close().catch(() => undefined);
      }
      for (const track of stream?.getTracks() ?? []) {
        track.stop();
      }
      throw error;
    }
  }

  async stop(): Promise<RecordedAudio> {
    const context = this.#context;
    if (!context) {
      throw new Error("Recording is not active");
    }

    if (this.#ticker !== undefined) {
      window.clearInterval(this.#ticker);
      this.#ticker = undefined;
    }
    this.#source?.disconnect();
    this.#capture?.disconnect();
    this.#silentGain?.disconnect();
    for (const track of this.#stream?.getTracks() ?? []) {
      track.stop();
    }

    this.#capture?.port.close();
    const sampleRate = context.sampleRate;
    let samples: Float32Array = new Float32Array();
    try {
      await context.close();
    } finally {
      samples = mergeChunks(this.#chunks, this.#sampleCount);
      this.#stream = undefined;
      this.#context = undefined;
      this.#source = undefined;
      this.#capture = undefined;
      this.#silentGain = undefined;
      this.#chunks = [];
      this.#sampleCount = 0;
      this.#level = 0;
    }

    return {
      samples,
      sampleRate,
      durationSeconds: samples.length / sampleRate,
    };
  }

  async cancel(): Promise<void> {
    await this.#startPromise?.catch(() => undefined);
    if (this.recording) {
      await this.stop();
    }
  }

  #acceptChunk(chunk: Float32Array): void {
    if (!this.recording || !(chunk instanceof Float32Array)) {
      return;
    }
    this.#chunks.push(chunk);
    this.#sampleCount += chunk.length;

    let sumSquares = 0;
    for (const sample of chunk) {
      sumSquares += sample * sample;
    }
    const rms = Math.sqrt(sumSquares / chunk.length);
    this.#level = Math.max(rms, this.#level * 0.85);
  }

  #emit(): void {
    const sampleRate = this.#context?.sampleRate;
    const snapshot = {
      elapsedSeconds: sampleRate ? this.#sampleCount / sampleRate : 0,
      level: Math.min(1, this.#level * 8),
    };
    for (const listener of this.#listeners) {
      listener(snapshot);
    }
  }
}

function mergeChunks(chunks: readonly Float32Array[], length: number): Float32Array {
  const samples = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    samples.set(chunk, offset);
    offset += chunk.length;
  }
  return samples;
}
