/** Fixed sample rate expected by the SpeechBrain ECAPA frontend. */
export const RESAMPLE_TARGET_RATE = 16_000;

// 24 zero crossings with a Blackman window gives useful stop-band rejection
// without making a five-second 48 kHz recording unnecessarily expensive on a
// phone. Rolloff leaves a narrow transition band below the target Nyquist.
const ZERO_CROSSINGS = 24;
const ROLLOFF = 0.97;
const MINIMUM_WEIGHT_SUM = 1e-12;
const MAX_TYPED_ARRAY_LENGTH = 0xffff_ffff;

/**
 * Deterministically resamples mono PCM to 16 kHz with a windowed-sinc filter.
 *
 * The browser's implicit sample-rate conversion differs by implementation, so
 * inference uses this dependency-free converter instead. Non-finite samples
 * are replaced with zero and all-silence input takes a fast, finite path.
 */
export function resampleTo16k(
  input: Float32Array,
  inputSampleRate: number,
): Float32Array {
  assertValidSampleRate(inputSampleRate);

  const { samples, hasSignal } = finiteCopy(input);
  if (inputSampleRate === RESAMPLE_TARGET_RATE) {
    return samples;
  }

  const ratio = RESAMPLE_TARGET_RATE / inputSampleRate;
  const outputLength = Math.round(input.length * ratio);
  if (
    !Number.isSafeInteger(outputLength) ||
    outputLength < 0 ||
    outputLength > MAX_TYPED_ARRAY_LENGTH
  ) {
    throw new RangeError("Resampled audio is too large for a Float32Array");
  }

  const output = new Float32Array(outputLength);
  if (!hasSignal || outputLength === 0) {
    return output;
  }

  // For downsampling, reduce the cutoff in source-sample frequency and widen
  // the kernel by the reciprocal factor. For upsampling, retain the source
  // bandwidth while still applying the small anti-imaging rolloff.
  const cutoff = Math.min(1, ratio) * ROLLOFF;
  const radius = ZERO_CROSSINGS / cutoff;

  for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
    const sourcePosition = outputIndex / ratio;
    const firstSourceIndex = Math.max(0, Math.ceil(sourcePosition - radius));
    const lastSourceIndex = Math.min(
      samples.length - 1,
      Math.floor(sourcePosition + radius),
    );

    let weightedSum = 0;
    let weightSum = 0;

    for (
      let sourceIndex = firstSourceIndex;
      sourceIndex <= lastSourceIndex;
      sourceIndex += 1
    ) {
      const distance = sourcePosition - sourceIndex;
      const normalizedDistance = Math.abs(distance) / radius;
      if (normalizedDistance >= 1) {
        continue;
      }

      const window = blackmanWindow(normalizedDistance);
      const weight = cutoff * normalizedSinc(cutoff * distance) * window;
      weightedSum += samples[sourceIndex]! * weight;
      weightSum += weight;
    }

    if (Math.abs(weightSum) > MINIMUM_WEIGHT_SUM) {
      const value = weightedSum / weightSum;
      output[outputIndex] = Number.isFinite(value) ? value : 0;
    } else {
      // This is only a numerical fallback for extremely unusual rate/length
      // combinations. It keeps malformed inputs from producing a NaN.
      const nearestIndex = Math.max(
        0,
        Math.min(samples.length - 1, Math.round(sourcePosition)),
      );
      output[outputIndex] = samples[nearestIndex] ?? 0;
    }
  }

  return output;
}

function assertValidSampleRate(sampleRate: number): void {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RangeError("Input sample rate must be a positive finite number");
  }
}

function finiteCopy(input: Float32Array): {
  samples: Float32Array;
  hasSignal: boolean;
} {
  const samples = new Float32Array(input.length);
  let hasSignal = false;

  for (let index = 0; index < input.length; index += 1) {
    const value = input[index]!;
    if (Number.isFinite(value)) {
      samples[index] = value;
      hasSignal ||= value !== 0;
    }
  }

  return { samples, hasSignal };
}

function normalizedSinc(value: number): number {
  if (Math.abs(value) < Number.EPSILON) {
    return 1;
  }

  const radians = Math.PI * value;
  return Math.sin(radians) / radians;
}

function blackmanWindow(normalizedDistance: number): number {
  return (
    0.42 +
    0.5 * Math.cos(Math.PI * normalizedDistance) +
    0.08 * Math.cos(2 * Math.PI * normalizedDistance)
  );
}
