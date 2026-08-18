import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  parseHeightRegressorBundle,
  predictHeight,
} from "./height-regressor";

interface Reference {
  readonly embedding: readonly number[];
  readonly routing: { readonly decision: number; readonly class: number };
  readonly heightCm: { readonly automatic: number };
}

describe("HeightCeleb portable regressors", () => {
  const root = new URL("../../", import.meta.url);
  const bundle = parseHeightRegressorBundle(
    JSON.parse(
      readFileSync(new URL("public/models/height-regressors.json", root), "utf8"),
    ),
  );
  const reference = JSON.parse(
    readFileSync(
      new URL("tests/fixtures/speechbrain-example1.reference.json", root),
      "utf8",
    ),
  ) as Reference;

  it("matches the pinned sklearn routing and height", () => {
    const result = predictHeight(reference.embedding, bundle);

    expect(result.routingDecision).toBeCloseTo(reference.routing.decision, 10);
    expect(result.route).toBe(
      reference.routing.class === 1 ? "male-model" : "female-model",
    );
    expect(Math.abs(result.heightCm - reference.heightCm.automatic)).toBeLessThan(
      1e-6,
    );
  });

  it("rejects malformed embeddings", () => {
    expect(() => predictHeight(new Float32Array(191), bundle)).toThrow(
      RangeError,
    );
    const invalid = new Float32Array(192);
    invalid[12] = Number.NaN;
    expect(() => predictHeight(invalid, bundle)).toThrow(TypeError);
  });
});
