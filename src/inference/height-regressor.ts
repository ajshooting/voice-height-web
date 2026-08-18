export type HeightRoute = "female-model" | "male-model";

interface LinearModel {
  readonly kind: string;
  readonly weights: readonly number[];
  readonly bias: number;
}

export interface HeightRegressorBundle {
  readonly formatVersion: number;
  readonly embeddingDimension: number;
  readonly routing: LinearModel & {
    readonly classes: readonly number[];
    readonly classLabels: Readonly<Record<string, HeightRoute>>;
  };
  readonly height: Readonly<Record<HeightRoute, LinearModel>>;
}

export interface HeightPrediction {
  readonly heightCm: number;
  /** Internal model route. Do not present this as a user attribute. */
  readonly route: HeightRoute;
  readonly routingDecision: number;
}

export function parseHeightRegressorBundle(
  value: unknown,
): HeightRegressorBundle {
  if (!isRecord(value)) {
    throw new TypeError("Height regressor must be a JSON object");
  }

  const dimension = value.embeddingDimension;
  const routing = value.routing;
  const height = value.height;
  if (
    value.formatVersion !== 1 ||
    !Number.isInteger(dimension) ||
    typeof dimension !== "number" ||
    dimension <= 0 ||
    !isRecord(routing) ||
    !isRecord(height)
  ) {
    throw new TypeError("Unsupported height regressor format");
  }

  validateLinearModel(routing, dimension, "routing");
  const classes = routing.classes;
  const classLabels = routing.classLabels;
  if (
    !Array.isArray(classes) ||
    classes.length !== 2 ||
    classes[0] !== 0 ||
    classes[1] !== 1 ||
    !isRecord(classLabels) ||
    classLabels["0"] !== "female-model" ||
    classLabels["1"] !== "male-model"
  ) {
    throw new TypeError("Unexpected routing classes");
  }

  validateLinearModel(height["female-model"], dimension, "female-model");
  validateLinearModel(height["male-model"], dimension, "male-model");
  return value as unknown as HeightRegressorBundle;
}

export function predictHeight(
  embedding: ArrayLike<number>,
  bundle: HeightRegressorBundle,
): HeightPrediction {
  if (embedding.length !== bundle.embeddingDimension) {
    throw new RangeError(
      `Expected ${bundle.embeddingDimension} embedding values, got ${embedding.length}`,
    );
  }

  const routingDecision = evaluateLinear(embedding, bundle.routing);
  const route: HeightRoute =
    routingDecision > 0 ? "male-model" : "female-model";
  const heightCm = evaluateLinear(embedding, bundle.height[route]);
  if (!Number.isFinite(heightCm)) {
    throw new Error("Height model produced a non-finite result");
  }

  return { heightCm, route, routingDecision };
}

function evaluateLinear(
  values: ArrayLike<number>,
  model: LinearModel,
): number {
  let result = model.bias;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    const weight = model.weights[index];
    if (value === undefined || weight === undefined || !Number.isFinite(value)) {
      throw new TypeError("Embedding contains an invalid value");
    }
    result += value * weight;
  }
  return result;
}

function validateLinearModel(
  value: unknown,
  dimension: number,
  name: string,
): asserts value is LinearModel {
  if (
    !isRecord(value) ||
    !Array.isArray(value.weights) ||
    value.weights.length !== dimension ||
    !value.weights.every(Number.isFinite) ||
    typeof value.bias !== "number" ||
    !Number.isFinite(value.bias)
  ) {
    throw new TypeError(`Invalid ${name} parameters`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
