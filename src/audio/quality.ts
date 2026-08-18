import type { AudioQuality } from "../inference/protocol";

const CLIPPING_LEVEL = 0.999;

export function inspectAudio(
  samples: Float32Array,
  sampleRate: number,
): AudioQuality {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError("Invalid audio sample rate");
  }

  let sumSquares = 0;
  let peak = 0;
  let clipped = 0;
  for (const rawValue of samples) {
    const value = Number.isFinite(rawValue) ? rawValue : 0;
    const absolute = Math.abs(value);
    sumSquares += value * value;
    peak = Math.max(peak, absolute);
    clipped += absolute >= CLIPPING_LEVEL ? 1 : 0;
  }
  return {
    durationSeconds: samples.length / sampleRate,
    rms: samples.length === 0 ? 0 : Math.sqrt(sumSquares / samples.length),
    peak,
    clippedFraction: samples.length === 0 ? 0 : clipped / samples.length,
  };
}
