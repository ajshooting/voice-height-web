import { describe, expect, it } from "vitest";

import { inspectAudio } from "./quality";

describe("inspectAudio", () => {
  it("reports duration, level, peak, and clipping", () => {
    const quality = inspectAudio(new Float32Array([0, 0.5, -1, 1]), 2);

    expect(quality.durationSeconds).toBe(2);
    expect(quality.rms).toBeCloseTo(0.75);
    expect(quality.peak).toBe(1);
    expect(quality.clippedFraction).toBe(0.5);
  });

  it("keeps an empty or non-finite recording finite", () => {
    expect(inspectAudio(new Float32Array(), 16_000)).toEqual({
      durationSeconds: 0,
      rms: 0,
      peak: 0,
      clippedFraction: 0,
    });
    expect(inspectAudio(new Float32Array([Number.NaN]), 1).rms).toBe(0);
  });
});
