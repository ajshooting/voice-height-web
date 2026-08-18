declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
  abstract process(inputs: Float32Array[][]): boolean;
}

declare function registerProcessor(
  name: string,
  processorCtor: new () => AudioWorkletProcessor,
): void;

class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs: Float32Array[][]): boolean {
    const channels = inputs[0];
    const firstChannel = channels?.[0];
    if (!firstChannel || firstChannel.length === 0) {
      return true;
    }

    const mono = new Float32Array(firstChannel.length);
    for (const channel of channels) {
      for (let index = 0; index < mono.length; index += 1) {
        mono[index] = mono[index]! + (channel[index] ?? 0) / channels.length;
      }
    }
    this.port.postMessage(mono, [mono.buffer]);
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
