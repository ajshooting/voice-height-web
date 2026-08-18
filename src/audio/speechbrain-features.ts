/**
 * SpeechBrain aa018540-compatible frontend for spkrec-ecapa-voxceleb.
 *
 * The implementation deliberately keeps this DSP outside the ONNX graph so the
 * browser pipeline has an explicit, testable feature contract. Input samples
 * must already be mono, 16 kHz PCM in a Float32Array. Returned data is a flat,
 * frame-major [frames, 80] matrix suitable for transferring to a Web Worker.
 *
 * Reference:
 * https://github.com/speechbrain/speechbrain/tree/aa018540
 */

export const SPEECHBRAIN_SAMPLE_RATE = 16_000;
export const SPEECHBRAIN_FFT_SIZE = 400;
export const SPEECHBRAIN_WINDOW_SIZE = 400;
export const SPEECHBRAIN_HOP_SIZE = 160;
export const SPEECHBRAIN_MEL_BINS = 80;

const SPECTRUM_BINS = SPEECHBRAIN_FFT_SIZE / 2 + 1;
const CENTER_PADDING = SPEECHBRAIN_FFT_SIZE / 2;
const MIN_MEL_ENERGY = 1e-10;
const TOP_DB = 80;

export interface SpeechBrainFeatures {
  /** Flat, frame-major values with shape [frames, bins]. */
  readonly data: Float32Array;
  readonly frames: number;
  readonly bins: typeof SPEECHBRAIN_MEL_BINS;
}

const HAMMING_WINDOW = createPeriodicHammingWindow();
const DFT_BASIS = createRealDftBasis();
const MEL_FILTERS = createSpeechBrainMelFilters();

/**
 * Produces log-mel features followed by per-mel-bin sentence mean subtraction.
 *
 * Non-finite PCM values are treated as silence. A silent waveform therefore
 * produces a finite all-zero normalized feature matrix instead of NaNs.
 */
export function extractSpeechBrainFeatures(
  input: Float32Array,
): SpeechBrainFeatures {
  const frames = Math.floor(input.length / SPEECHBRAIN_HOP_SIZE) + 1;
  const data = new Float32Array(frames * SPEECHBRAIN_MEL_BINS);
  const { samples, hasSignal } = sanitizeSamples(input);

  // SpeechBrain maps silence to -100 dB in every cell, then sentence mean
  // subtraction maps it to exactly zero. Avoiding the DFT is both equivalent
  // and important for responsive validation of silent recordings.
  if (!hasSignal) {
    return { data, frames, bins: SPEECHBRAIN_MEL_BINS };
  }

  const windowedFrame = new Float32Array(SPEECHBRAIN_WINDOW_SIZE);
  const powerSpectrum = new Float64Array(SPECTRUM_BINS);
  let maximumDb = -Infinity;

  for (let frame = 0; frame < frames; frame += 1) {
    const sourceStart = frame * SPEECHBRAIN_HOP_SIZE - CENTER_PADDING;

    for (let windowIndex = 0; windowIndex < SPEECHBRAIN_WINDOW_SIZE; windowIndex += 1) {
      const sourceIndex = sourceStart + windowIndex;
      const sample =
        sourceIndex >= 0 && sourceIndex < samples.length
          ? samples[sourceIndex]!
          : 0;
      windowedFrame[windowIndex] = sample * HAMMING_WINDOW[windowIndex]!;
    }

    computePowerSpectrum(windowedFrame, powerSpectrum);

    const outputOffset = frame * SPEECHBRAIN_MEL_BINS;
    for (let mel = 0; mel < SPEECHBRAIN_MEL_BINS; mel += 1) {
      const filterOffset = mel * SPECTRUM_BINS;
      let energy = 0;

      for (let frequencyBin = 0; frequencyBin < SPECTRUM_BINS; frequencyBin += 1) {
        energy +=
          powerSpectrum[frequencyBin]! *
          MEL_FILTERS[filterOffset + frequencyBin]!;
      }

      // The matrix multiplication in SpeechBrain produces float32 before the
      // log operation. Round at this boundary to follow that behavior closely.
      const floatEnergy = Math.fround(energy);
      const decibels = Math.fround(
        10 * Math.log10(Math.max(MIN_MEL_ENERGY, floatEnergy)),
      );
      data[outputOffset + mel] = decibels;
      maximumDb = Math.max(maximumDb, decibels);
    }
  }

  // SpeechBrain applies one global top-dB floor to the whole utterance before
  // sentence normalization (not a separate maximum for each frame or band).
  const decibelFloor = Math.fround(maximumDb - TOP_DB);
  for (let index = 0; index < data.length; index += 1) {
    if (data[index]! < decibelFloor) {
      data[index] = decibelFloor;
    }
  }

  subtractSentenceMean(data, frames);

  return { data, frames, bins: SPEECHBRAIN_MEL_BINS };
}

function sanitizeSamples(input: Float32Array): {
  samples: Float32Array;
  hasSignal: boolean;
} {
  let hasNonFinite = false;
  let hasSignal = false;

  for (let index = 0; index < input.length; index += 1) {
    const value = input[index]!;
    if (!Number.isFinite(value)) {
      hasNonFinite = true;
    } else if (value !== 0) {
      hasSignal = true;
    }
  }

  if (!hasNonFinite) {
    return { samples: input, hasSignal };
  }

  const sanitized = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    const value = input[index]!;
    sanitized[index] = Number.isFinite(value) ? value : 0;
  }

  return { samples: sanitized, hasSignal };
}

function createPeriodicHammingWindow(): Float32Array {
  const window = new Float32Array(SPEECHBRAIN_WINDOW_SIZE);

  // torch.hamming_window defaults to periodic=True. Its denominator is N,
  // rather than N - 1 as used by a symmetric Hamming window.
  for (let index = 0; index < window.length; index += 1) {
    window[index] = Math.fround(
      0.54 - 0.46 * Math.cos((2 * Math.PI * index) / window.length),
    );
  }

  return window;
}

function createRealDftBasis(): {
  cosine: Float64Array;
  sine: Float64Array;
} {
  const cosine = new Float64Array(SPECTRUM_BINS * SPEECHBRAIN_FFT_SIZE);
  const sine = new Float64Array(SPECTRUM_BINS * SPEECHBRAIN_FFT_SIZE);

  for (let frequencyBin = 0; frequencyBin < SPECTRUM_BINS; frequencyBin += 1) {
    const offset = frequencyBin * SPEECHBRAIN_FFT_SIZE;
    for (let sample = 0; sample < SPEECHBRAIN_FFT_SIZE; sample += 1) {
      const phase =
        (2 * Math.PI * frequencyBin * sample) / SPEECHBRAIN_FFT_SIZE;
      cosine[offset + sample] = Math.cos(phase);
      sine[offset + sample] = Math.sin(phase);
    }
  }

  return { cosine, sine };
}

function computePowerSpectrum(
  frame: Float32Array,
  destination: Float64Array,
): void {
  for (let frequencyBin = 0; frequencyBin < SPECTRUM_BINS; frequencyBin += 1) {
    const basisOffset = frequencyBin * SPEECHBRAIN_FFT_SIZE;
    let real = 0;
    let imaginary = 0;

    for (let sample = 0; sample < SPEECHBRAIN_FFT_SIZE; sample += 1) {
      const value = frame[sample]!;
      real += value * DFT_BASIS.cosine[basisOffset + sample]!;
      imaginary -= value * DFT_BASIS.sine[basisOffset + sample]!;
    }

    destination[frequencyBin] = real * real + imaginary * imaginary;
  }
}

/**
 * Reproduces the non-standard triangular-band definition used by SpeechBrain
 * aa018540: each filter uses the distance from its center to the lower mel
 * point as its bandwidth on both sides. The upper side therefore does not, in
 * general, end at the next mel-spaced point.
 */
function createSpeechBrainMelFilters(): Float32Array {
  const pointCount = SPEECHBRAIN_MEL_BINS + 2;
  const melMaximum = hzToHtkMel(SPEECHBRAIN_SAMPLE_RATE / 2);
  const hzPoints = new Float32Array(pointCount);

  for (let point = 0; point < pointCount; point += 1) {
    const mel = Math.fround((melMaximum * point) / (pointCount - 1));
    hzPoints[point] = Math.fround(htkMelToHz(mel));
  }

  const filters = new Float32Array(SPEECHBRAIN_MEL_BINS * SPECTRUM_BINS);
  for (let mel = 0; mel < SPEECHBRAIN_MEL_BINS; mel += 1) {
    const center = hzPoints[mel + 1]!;
    const band = Math.fround(center - hzPoints[mel]!);
    const filterOffset = mel * SPECTRUM_BINS;

    for (let frequencyBin = 0; frequencyBin < SPECTRUM_BINS; frequencyBin += 1) {
      const frequency =
        (frequencyBin * (SPEECHBRAIN_SAMPLE_RATE / 2)) /
        (SPECTRUM_BINS - 1);
      const slope = (frequency - center) / band;
      filters[filterOffset + frequencyBin] = Math.fround(
        Math.max(0, Math.min(slope + 1, 1 - slope)),
      );
    }
  }

  return filters;
}

function hzToHtkMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function htkMelToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

function subtractSentenceMean(data: Float32Array, frames: number): void {
  for (let mel = 0; mel < SPEECHBRAIN_MEL_BINS; mel += 1) {
    let sum = 0;
    for (let frame = 0; frame < frames; frame += 1) {
      sum += data[frame * SPEECHBRAIN_MEL_BINS + mel]!;
    }

    const mean = Math.fround(sum / frames);
    for (let frame = 0; frame < frames; frame += 1) {
      const index = frame * SPEECHBRAIN_MEL_BINS + mel;
      data[index] = Math.fround(data[index]! - mean);
    }
  }
}
