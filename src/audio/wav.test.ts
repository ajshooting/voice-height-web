import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { decodePcm16MonoWav } from "./wav";

describe("decodePcm16MonoWav", () => {
  it("decodes the pinned SpeechBrain example", () => {
    const bytes = readFileSync(
      new URL("../../tests/fixtures/speechbrain-example1.wav", import.meta.url),
    );
    const buffer = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    ) as ArrayBuffer;
    const decoded = decodePcm16MonoWav(buffer);

    expect(decoded.sampleRate).toBe(16_000);
    expect(decoded.samples).toHaveLength(52_173);
    expect(decoded.samples.some((sample) => sample !== 0)).toBe(true);
  });

  it("rejects unsupported data", () => {
    expect(() => decodePcm16MonoWav(new ArrayBuffer(12))).toThrow(TypeError);
  });
});
