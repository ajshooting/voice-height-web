import { describe, expect, it } from "vitest";

import { RESAMPLE_TARGET_RATE, resampleTo16k } from "./resample";

describe("resampleTo16k", () => {
  it("copies identity-rate input and replaces non-finite samples", () => {
    const input = new Float32Array([
      -0.5,
      Number.NaN,
      0.25,
      Number.POSITIVE_INFINITY,
    ]);

    const output = resampleTo16k(input, RESAMPLE_TARGET_RATE);

    expect(output).not.toBe(input);
    expect(Array.from(output)).toEqual([-0.5, 0, 0.25, 0]);
  });

  it("preserves duration for common browser sample rates", () => {
    expect(resampleTo16k(new Float32Array(48_000), 48_000)).toHaveLength(
      16_000,
    );
    expect(resampleTo16k(new Float32Array(44_100), 44_100)).toHaveLength(
      16_000,
    );
  });

  it("keeps resampled silence finite", () => {
    const output = resampleTo16k(new Float32Array(4_800), 48_000);

    for (const value of output) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBe(0);
    }
  });

  it("retains an in-band tone when downsampling", () => {
    const input = sineWave(1_000, 0.1, 48_000);
    const output = resampleTo16k(input, 48_000);
    const expected = sineWave(1_000, 0.1, RESAMPLE_TARGET_RATE);

    // Ignore filter transients at the two recording boundaries.
    expect(rootMeanSquareError(output, expected, 100)).toBeLessThan(0.002);
  });

  it("attenuates frequencies above the target Nyquist limit", () => {
    const input = sineWave(12_000, 0.1, 48_000);
    const output = resampleTo16k(input, 48_000);

    expect(rootMeanSquare(output, 100)).toBeLessThan(0.01);
  });

  it("rejects invalid sample rates", () => {
    const input = new Float32Array(1);

    expect(() => resampleTo16k(input, 0)).toThrow(RangeError);
    expect(() => resampleTo16k(input, Number.NaN)).toThrow(RangeError);
  });
});

function sineWave(
  frequency: number,
  durationSeconds: number,
  sampleRate: number,
): Float32Array {
  const waveform = new Float32Array(Math.round(durationSeconds * sampleRate));
  for (let index = 0; index < waveform.length; index += 1) {
    waveform[index] = Math.sin((2 * Math.PI * frequency * index) / sampleRate);
  }
  return waveform;
}

function rootMeanSquareError(
  actual: Float32Array,
  expected: Float32Array,
  trim: number,
): number {
  expect(actual).toHaveLength(expected.length);
  let sum = 0;
  let count = 0;
  for (let index = trim; index < actual.length - trim; index += 1) {
    const difference = actual[index]! - expected[index]!;
    sum += difference * difference;
    count += 1;
  }
  return Math.sqrt(sum / count);
}

function rootMeanSquare(values: Float32Array, trim: number): number {
  let sum = 0;
  let count = 0;
  for (let index = trim; index < values.length - trim; index += 1) {
    sum += values[index]! * values[index]!;
    count += 1;
  }
  return Math.sqrt(sum / count);
}
