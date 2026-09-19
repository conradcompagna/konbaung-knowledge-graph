export interface ThematicLayoutInput {
  ids: string[];
  entityCount: number;
  frequencies: Float64Array;
  connectivity: Float64Array;
  radii: Float64Array;
}

export interface ThematicLayoutResult {
  x: Float64Array;
  y: Float64Array;
  bounds: [number, number, number, number];
  overlapCount: number;
  coreMedianHigh: number;
  coreMedianLow: number;
  layoutHash: string;
  durationMs: number;
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

function stableNumber(text: string): number {
  let hash = 2166136261 >>> 0;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  return hash / 0xffffffff;
}

function layoutHash(x: Float64Array, y: Float64Array): string {
  let hash = 2166136261 >>> 0;
  const add = (value: number): void => {
    hash ^= value >>> 0;
    hash = Math.imul(hash, 16777619) >>> 0;
  };
  for (let index = 0; index < x.length; index += 1) {
    add(Math.round(x[index] * 1_000));
    add(Math.round(y[index] * 1_000));
  }
  return hash.toString(16).padStart(8, "0");
}

function auditOverlaps(
  x: Float64Array,
  y: Float64Array,
  radii: Float64Array,
): number {
  let overlaps = 0;
  for (let left = 0; left < x.length; left += 1) {
    for (let right = left + 1; right < x.length; right += 1) {
      const minimum = radii[left] + radii[right] + 4;
      const dx = x[right] - x[left];
      const dy = y[right] - y[left];
      if (dx * dx + dy * dy < minimum * minimum - 0.001) overlaps += 1;
    }
  }
  return overlaps;
}

/**
 * Deterministic centrality layout for the 52-node thematic entity network.
 * Relation categories label direct edges and never participate in placement.
 */
export function calculateThematicLayout(
  input: ThematicLayoutInput,
): ThematicLayoutResult {
  const startedAt = performance.now();
  const count = input.ids.length;
  if (
    input.frequencies.length !== count ||
    input.connectivity.length !== count ||
    input.radii.length !== count
  ) {
    throw new Error("Thematic layout received mismatched node arrays.");
  }

  const x = new Float64Array(count);
  const y = new Float64Array(count);
  const maximumConnectivity = Math.max(
    1,
    ...Array.from(input.connectivity.slice(0, input.entityCount)),
  );
  const entityOrder = Array.from(
    { length: input.entityCount },
    (_, index) => index,
  ).sort(
    (left, right) =>
      input.connectivity[right] - input.connectivity[left] ||
      input.ids[left].localeCompare(input.ids[right]),
  );

  // Entity placement is determined only by composite connectivity. The most
  // central category is fixed at the origin; all others receive a deterministic
  // golden-angle bearing and a monotonic connectivity radius.
  for (let rank = 0; rank < entityOrder.length; rank += 1) {
    const entity = entityOrder[rank];
    const connectedness = input.connectivity[entity] / maximumConnectivity;
    const radius =
      rank === 0 ? 0 : Math.pow(1 - connectedness, 0.72) * 470;
    const angle =
      (rank - 1) * GOLDEN_ANGLE +
      stableNumber(input.ids[entity]) * Math.PI * 0.16;
    x[entity] = Math.cos(angle) * radius;
    y[entity] = Math.sin(angle) * radius;
  }

  // Preserve every entity bearing and relative radius. If their size-aware
  // circles need more room, expand the entire entity layout uniformly.
  let entityScale = 1;
  for (let left = 0; left < input.entityCount; left += 1) {
    for (let right = left + 1; right < input.entityCount; right += 1) {
      const distance = Math.hypot(x[right] - x[left], y[right] - y[left]);
      if (distance <= 0.001) continue;
      entityScale = Math.max(
        entityScale,
        (input.radii[left] + input.radii[right] + 10) / distance,
      );
    }
  }
  if (entityScale > 1) {
    for (let entity = 0; entity < input.entityCount; entity += 1) {
      x[entity] *= entityScale;
      y[entity] *= entityScale;
    }
  }

  let extentX = 1;
  let extentY = 1;
  for (let node = 0; node < count; node += 1) {
    extentX = Math.max(extentX, Math.abs(x[node]) + input.radii[node]);
    extentY = Math.max(extentY, Math.abs(y[node]) + input.radii[node]);
  }
  const connectivityOrder = Array.from(
    { length: input.entityCount },
    (_, index) => index,
  ).sort(
    (left, right) =>
      input.connectivity[right] - input.connectivity[left] ||
      input.ids[left].localeCompare(input.ids[right]),
  );
  const cohortSize = Math.max(1, Math.floor(input.entityCount / 3));
  const medianDistance = (nodes: number[]): number => {
    const distances = nodes
      .map((node) => Math.hypot(x[node], y[node]))
      .sort((left, right) => left - right);
    return distances[Math.floor(distances.length / 2)] || 0;
  };
  const coreMedianHigh = medianDistance(
    connectivityOrder.slice(0, cohortSize),
  );
  const coreMedianLow = medianDistance(
    connectivityOrder.slice(-cohortSize),
  );

  return {
    x,
    y,
    bounds: [-extentX, -extentY, extentX, extentY],
    overlapCount: auditOverlaps(x, y, input.radii),
    coreMedianHigh,
    coreMedianLow,
    layoutHash: layoutHash(x, y),
    durationMs: performance.now() - startedAt,
  };
}
