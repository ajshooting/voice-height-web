import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  SPEECHBRAIN_MEL_BINS,
  SPEECHBRAIN_SAMPLE_RATE,
  extractSpeechBrainFeatures,
} from "./speechbrain-features";

describe("extractSpeechBrainFeatures", () => {
  it("returns the SpeechBrain frame shape for exactly five seconds", () => {
    const features = extractSpeechBrainFeatures(
      new Float32Array(5 * SPEECHBRAIN_SAMPLE_RATE),
    );

    expect(features.frames).toBe(501);
    expect(features.bins).toBe(80);
    expect(features.data).toHaveLength(501 * SPEECHBRAIN_MEL_BINS);
  });

  it("keeps silence finite and zero after sentence mean subtraction", () => {
    const features = extractSpeechBrainFeatures(
      new Float32Array(SPEECHBRAIN_SAMPLE_RATE),
    );

    for (const value of features.data) {
      expect(Number.isFinite(value)).toBe(true);
      expect(Math.abs(value)).toBeLessThan(1e-7);
    }
  });

  it("uses centered framing and produces one frame per 160 samples plus one", () => {
    const waveform = new Float32Array(160);
    waveform[0] = 1;

    const features = extractSpeechBrainFeatures(waveform);

    expect(features.frames).toBe(2);
    expect(maximumAbsoluteValue(features.data)).toBeGreaterThan(0);
  });

  it("subtracts each mel bin mean across the complete utterance", () => {
    const waveform = sineWave(440, 0.1, SPEECHBRAIN_SAMPLE_RATE);
    const features = extractSpeechBrainFeatures(waveform);

    expect(maximumAbsoluteValue(features.data)).toBeGreaterThan(1);
    for (let mel = 0; mel < features.bins; mel += 1) {
      let sum = 0;
      for (let frame = 0; frame < features.frames; frame += 1) {
        const value = features.data[frame * features.bins + mel]!;
        expect(Number.isFinite(value)).toBe(true);
        sum += value;
      }
      expect(Math.abs(sum / features.frames)).toBeLessThan(1e-4);
    }
  });

  it("treats non-finite PCM values as silence", () => {
    const waveform = new Float32Array([Number.NaN, Number.POSITIVE_INFINITY]);
    const features = extractSpeechBrainFeatures(waveform);

    expect(features.frames).toBe(1);
    expect(maximumAbsoluteValue(features.data)).toBe(0);
  });

  it("matches the pinned SpeechBrain reference features", () => {
    const fixtureRoot = new URL("../../tests/fixtures/", import.meta.url);
    const reference = JSON.parse(
      readFileSync(
        new URL("speechbrain-example1.reference.json", fixtureRoot),
        "utf8",
      ),
    ) as SpeechBrainReference;
    const wav = readPcm16MonoWav(
      new URL("speechbrain-example1.wav", fixtureRoot),
    );
    const expected = readLittleEndianFloat32(
      new URL("speechbrain-example1.features.f32", fixtureRoot),
    );

    expect(wav.sampleRate).toBe(reference.sampleRate);
    expect(wav.samples).toHaveLength(reference.waveformSamples);

    const actual = extractSpeechBrainFeatures(wav.samples);
    expect([1, actual.frames, actual.bins]).toEqual(reference.features.shape);
    expect(actual.data).toHaveLength(expected.length);

    let maximumDifference = 0;
    let squaredDifferenceSum = 0;
    for (let index = 0; index < expected.length; index += 1) {
      const difference = Math.abs(actual.data[index]! - expected[index]!);
      maximumDifference = Math.max(maximumDifference, difference);
      squaredDifferenceSum += difference * difference;
    }

    expect(maximumDifference).toBeLessThan(0.002);
    expect(Math.sqrt(squaredDifferenceSum / expected.length)).toBeLessThan(
      0.0002,
    );
  });
});

interface SpeechBrainReference {
  readonly sampleRate: number;
  readonly waveformSamples: number;
  readonly features: {
    readonly shape: readonly number[];
  };
}

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

function maximumAbsoluteValue(values: Float32Array): number {
  let maximum = 0;
  for (const value of values) {
    maximum = Math.max(maximum, Math.abs(value));
  }
  return maximum;
}

function readPcm16MonoWav(url: URL): {
  sampleRate: number;
  samples: Float32Array;
} {
  const bytes = readFileSync(url);
  expect(bytes.toString("ascii", 0, 4)).toBe("RIFF");
  expect(bytes.toString("ascii", 8, 12)).toBe("WAVE");

  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataLength = 0;

  for (let offset = 12; offset + 8 <= bytes.length; ) {
    const chunkId = bytes.toString("ascii", offset, offset + 4);
    const chunkLength = bytes.readUInt32LE(offset + 4);
    const payloadOffset = offset + 8;
    if (payloadOffset + chunkLength > bytes.length) {
      throw new Error(`Invalid WAV chunk length for ${chunkId}`);
    }

    if (chunkId === "fmt ") {
      format = bytes.readUInt16LE(payloadOffset);
      channels = bytes.readUInt16LE(payloadOffset + 2);
      sampleRate = bytes.readUInt32LE(payloadOffset + 4);
      bitsPerSample = bytes.readUInt16LE(payloadOffset + 14);
    } else if (chunkId === "data") {
      dataOffset = payloadOffset;
      dataLength = chunkLength;
    }

    offset = payloadOffset + chunkLength + (chunkLength % 2);
  }

  expect(format).toBe(1);
  expect(channels).toBe(1);
  expect(bitsPerSample).toBe(16);
  expect(dataOffset).toBeGreaterThanOrEqual(0);
  expect(dataLength % 2).toBe(0);

  const samples = new Float32Array(dataLength / 2);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = bytes.readInt16LE(dataOffset + index * 2) / 32_768;
  }

  return { sampleRate, samples };
}

function readLittleEndianFloat32(url: URL): Float32Array {
  const bytes = readFileSync(url);
  expect(bytes.length % Float32Array.BYTES_PER_ELEMENT).toBe(0);

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = new Float32Array(
    bytes.length / Float32Array.BYTES_PER_ELEMENT,
  );
  for (let index = 0; index < values.length; index += 1) {
    values[index] = view.getFloat32(
      index * Float32Array.BYTES_PER_ELEMENT,
      true,
    );
  }
  return values;
}
