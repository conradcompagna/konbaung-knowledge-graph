export const CORPUS_BUBBLE_RADIUS = 36;
export const CORPUS_BUBBLE_CLEARANCE = 8;
export const CORPUS_MIN_CENTER_DISTANCE =
  CORPUS_BUBBLE_RADIUS * 2 + CORPUS_BUBBLE_CLEARANCE;
export const CORPUS_LEVEL_GAP = 96;
export const CORPUS_COMPONENT_GAP = 320;

const CORPUS_COMPONENT_PACKING_DENSITY = 0.12;
const CORPUS_COMPONENT_PROBES_BEFORE_EXPANSION = 20_000;

export interface CorpusRadialLayoutInput {
  ids: string[];
  frequencies: Float64Array;
  sources: Int32Array;
  targets: Int32Array;
}

export interface CorpusRadialLayoutResult {
  x: Float64Array;
  y: Float64Array;
  parent: Int32Array;
  parentEdge: Int32Array;
  depth: Int32Array;
  degree: Int32Array;
  component: Int32Array;
  subtreeSize: Int32Array;
  ranked: Uint32Array;
  roots: Uint32Array;
  bounds: [number, number, number, number];
  medianTreeDistance: number;
  overlapCount: number;
  forestEdgeCount: number;
  maxDepth: number;
  dominantRoot: number;
  minimumComponentGap: number;
  layoutHash: string;
  durationMs: number;
}

interface Neighbor {
  node: number;
  edge: number;
}

interface ComponentLayout {
  nodes: number[];
  root: number;
  order: number[];
  radius: number;
  offsetX: number;
  offsetY: number;
}

const TWO_PI = Math.PI * 2;

function finiteOrZero(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function halton(index: number, base: number): number {
  let result = 0;
  let fraction = 1 / base;
  let value = index;
  while (value > 0) {
    result += fraction * (value % base);
    value = Math.floor(value / base);
    fraction /= base;
  }
  return result;
}

function stableHash(
  parent: Int32Array,
  depth: Int32Array,
  x: Float64Array,
  y: Float64Array,
): string {
  let hash = 2166136261 >>> 0;
  const add = (value: number): void => {
    hash ^= value >>> 0;
    hash = Math.imul(hash, 16777619) >>> 0;
  };
  for (let index = 0; index < parent.length; index += 1) {
    add(parent[index] + 1);
    add(depth[index]);
    add(Math.round(x[index] * 1_000));
    add(Math.round(y[index] * 1_000));
  }
  return hash.toString(16).padStart(8, "0");
}

function auditOverlaps(x: Float64Array, y: Float64Array): number {
  const cellSize = CORPUS_MIN_CENTER_DISTANCE;
  const minimumSquared =
    CORPUS_MIN_CENTER_DISTANCE * CORPUS_MIN_CENTER_DISTANCE;
  const grid = new Map<string, number[]>();
  let overlaps = 0;
  for (let index = 0; index < x.length; index += 1) {
    const cellX = Math.floor(x[index] / cellSize);
    const cellY = Math.floor(y[index] / cellSize);
    for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        const occupants = grid.get(`${cellX + offsetX}:${cellY + offsetY}`);
        if (!occupants) continue;
        for (const other of occupants) {
          const dx = x[index] - x[other];
          const dy = y[index] - y[other];
          if (dx * dx + dy * dy < minimumSquared - 0.0001) overlaps += 1;
        }
      }
    }
    const key = `${cellX}:${cellY}`;
    const occupants = grid.get(key);
    if (occupants) occupants.push(index);
    else grid.set(key, [index]);
  }
  return overlaps;
}

export function calculateCorpusRadialLayout(
  input: CorpusRadialLayoutInput,
): CorpusRadialLayoutResult {
  const startedAt = performance.now();
  const nodeCount = input.ids.length;
  if (input.frequencies.length !== nodeCount) {
    throw new Error("Corpus radial layout received mismatched node arrays.");
  }
  if (input.sources.length !== input.targets.length) {
    throw new Error("Corpus radial layout received mismatched edge arrays.");
  }

  const adjacency = Array.from(
    { length: nodeCount },
    (): Neighbor[] => [],
  );
  for (let edge = 0; edge < input.sources.length; edge += 1) {
    const source = input.sources[edge];
    const target = input.targets[edge];
    if (
      source < 0 ||
      target < 0 ||
      source >= nodeCount ||
      target >= nodeCount ||
      source === target
    ) continue;
    adjacency[source].push({ node: target, edge });
    adjacency[target].push({ node: source, edge });
  }

  const degree = new Int32Array(nodeCount);
  for (let node = 0; node < nodeCount; node += 1) {
    const neighbors = adjacency[node];
    neighbors.sort(
      (left, right) => left.node - right.node || left.edge - right.edge,
    );
    let write = 0;
    for (let read = 0; read < neighbors.length; read += 1) {
      if (write > 0 && neighbors[read].node === neighbors[write - 1].node) {
        continue;
      }
      neighbors[write] = neighbors[read];
      write += 1;
    }
    neighbors.length = write;
    degree[node] = write;
  }

  const compareNodes = (left: number, right: number): number =>
    degree[right] - degree[left] ||
    finiteOrZero(input.frequencies[right]) -
      finiteOrZero(input.frequencies[left]) ||
    input.ids[left].localeCompare(input.ids[right]);

  const rankedArray = Array.from(
    { length: nodeCount },
    (_, index) => index,
  ).sort(compareNodes);
  for (const neighbors of adjacency) {
    neighbors.sort(
      (left, right) =>
        compareNodes(left.node, right.node) || left.edge - right.edge,
    );
  }

  const seen = new Uint8Array(nodeCount);
  const rawComponents: Array<{ nodes: number[]; root: number }> = [];
  for (const seed of rankedArray) {
    if (seen[seed]) continue;
    const nodes: number[] = [];
    const stack = [seed];
    seen[seed] = 1;
    while (stack.length) {
      const node = stack.pop() as number;
      nodes.push(node);
      for (const neighbor of adjacency[node]) {
        if (seen[neighbor.node]) continue;
        seen[neighbor.node] = 1;
        stack.push(neighbor.node);
      }
    }
    rawComponents.push({ nodes, root: seed });
  }
  rawComponents.sort(
    (left, right) =>
      right.nodes.length - left.nodes.length ||
      compareNodes(left.root, right.root),
  );

  const parent = new Int32Array(nodeCount);
  const parentEdge = new Int32Array(nodeCount);
  const depth = new Int32Array(nodeCount);
  const component = new Int32Array(nodeCount);
  const subtreeSize = new Int32Array(nodeCount);
  parent.fill(-1);
  parentEdge.fill(-1);
  component.fill(-1);
  const children = Array.from({ length: nodeCount }, (): number[] => []);
  const components: ComponentLayout[] = [];
  let forestEdgeCount = 0;
  let maxDepth = 0;

  rawComponents.forEach((rawComponent, componentIndex) => {
    const { nodes, root } = rawComponent;
    const order: number[] = [];
    const queue = [root];
    let head = 0;
    component[root] = componentIndex;
    while (head < queue.length) {
      const node = queue[head];
      head += 1;
      order.push(node);
      for (const neighbor of adjacency[node]) {
        if (component[neighbor.node] !== -1) continue;
        component[neighbor.node] = componentIndex;
        parent[neighbor.node] = node;
        parentEdge[neighbor.node] = neighbor.edge;
        depth[neighbor.node] = depth[node] + 1;
        maxDepth = Math.max(maxDepth, depth[neighbor.node]);
        children[node].push(neighbor.node);
        queue.push(neighbor.node);
        forestEdgeCount += 1;
      }
    }
    components.push({
      nodes,
      root,
      order,
      radius: CORPUS_BUBBLE_RADIUS,
      offsetX: 0,
      offsetY: 0,
    });
  });

  const x = new Float64Array(nodeCount);
  const y = new Float64Array(nodeCount);
  const startAngle = new Float64Array(nodeCount);
  const endAngle = new Float64Array(nodeCount);
  const angle = new Float64Array(nodeCount);
  const geometricDemand = new Float64Array(nodeCount);
  const contours = new Array<Float64Array>(nodeCount);
  const rootDistance = new Float64Array(nodeCount);

  for (const item of components) {
    for (let orderIndex = item.order.length - 1; orderIndex >= 0; orderIndex -= 1) {
      const node = item.order[orderIndex];
      let size = 1;
      let contourDepth = 1;
      for (const child of children[node]) {
        size += subtreeSize[child];
        contourDepth = Math.max(contourDepth, contours[child].length + 1);
      }
      subtreeSize[node] = size;
      const contour = new Float64Array(contourDepth);
      contour[0] = CORPUS_MIN_CENTER_DISTANCE;
      for (const child of children[node]) {
        const childContour = contours[child];
        for (
          let relativeDepth = 0;
          relativeDepth < childContour.length;
          relativeDepth += 1
        ) {
          contour[relativeDepth + 1] += childContour[relativeDepth];
        }
      }
      contours[node] = contour;
      let demand = CORPUS_MIN_CENTER_DISTANCE;
      for (let relativeDepth = 0; relativeDepth < contour.length; relativeDepth += 1) {
        demand = Math.max(demand, contour[relativeDepth]);
      }
      geometricDemand[node] = demand;
    }

    startAngle[item.root] = -Math.PI / 2;
    endAngle[item.root] = startAngle[item.root] + TWO_PI;
    angle[item.root] = -Math.PI / 2;
    for (const node of item.order) {
      const nodeChildren = children[node];
      if (!nodeChildren.length) continue;
      const width = endAngle[node] - startAngle[node];
      let totalDemand = 0;
      for (const child of nodeChildren) totalDemand += geometricDemand[child];
      let cursor = startAngle[node];
      for (let index = 0; index < nodeChildren.length; index += 1) {
        const child = nodeChildren[index];
        const childWidth =
          index === nodeChildren.length - 1
            ? endAngle[node] - cursor
            : width * (geometricDemand[child] / totalDemand);
        startAngle[child] = cursor;
        endAngle[child] = cursor + childWidth;
        angle[child] = cursor + childWidth / 2;
        cursor += childWidth;
      }
    }

    const cellSize = CORPUS_MIN_CENTER_DISTANCE;
    const minimumSquared =
      CORPUS_MIN_CENTER_DISTANCE * CORPUS_MIN_CENTER_DISTANCE;
    const placed = new Map<string, number[]>();
    const addPlaced = (node: number): void => {
      const cellX = Math.floor(x[node] / cellSize);
      const cellY = Math.floor(y[node] / cellSize);
      const key = `${cellX}:${cellY}`;
      const values = placed.get(key);
      if (values) values.push(node);
      else placed.set(key, [node]);
    };
    const blockedUntilRadius = (
      candidateX: number,
      candidateY: number,
      rayX: number,
      rayY: number,
      radius: number,
    ): number | null => {
      const cellX = Math.floor(candidateX / cellSize);
      const cellY = Math.floor(candidateY / cellSize);
      let blockedUntil: number | null = null;
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
          const occupants = placed.get(
            `${cellX + offsetX}:${cellY + offsetY}`,
          );
          if (!occupants) continue;
          for (const other of occupants) {
            const dx = candidateX - x[other];
            const dy = candidateY - y[other];
            if (dx * dx + dy * dy < minimumSquared - 0.0001) {
              const projection = x[other] * rayX + y[other] * rayY;
              const perpendicularSquared = Math.max(
                0,
                x[other] * x[other] +
                  y[other] * y[other] -
                  projection * projection,
              );
              const exitRadius =
                projection +
                Math.sqrt(
                  Math.max(0, minimumSquared - perpendicularSquared),
                ) +
                0.001;
              blockedUntil = Math.max(
                blockedUntil ?? radius + 0.001,
                exitRadius,
              );
            }
          }
        }
      }
      return blockedUntil;
    };
    x[item.root] = 0;
    y[item.root] = 0;
    rootDistance[item.root] = 0;
    addPlaced(item.root);
    for (let orderIndex = 1; orderIndex < item.order.length; orderIndex += 1) {
      const node = item.order[orderIndex];
      const visualParent = parent[node];
      let radius =
        rootDistance[visualParent] + CORPUS_LEVEL_GAP;
      let attempts = 0;
      while (true) {
        const rayX = Math.cos(angle[node]);
        const rayY = Math.sin(angle[node]);
        const candidateX = rayX * radius;
        const candidateY = rayY * radius;
        const blockedUntil = blockedUntilRadius(
          candidateX,
          candidateY,
          rayX,
          rayY,
          radius,
        );
        if (blockedUntil === null) {
          rootDistance[node] = radius;
          x[node] = candidateX;
          y[node] = candidateY;
          addPlaced(node);
          item.radius = Math.max(
            item.radius,
            radius + CORPUS_BUBBLE_RADIUS,
          );
          break;
        }
        radius = blockedUntil;
        attempts += 1;
        if (attempts > nodeCount + 1) {
          throw new Error(
            `Unable to place corpus node ${input.ids[node]} constructively.`,
          );
        }
      }
    }
  }

  let minimumComponentGap = Number.POSITIVE_INFINITY;
  if (components.length) {
    const packedX = new Float64Array(nodeCount);
    const packedY = new Float64Array(nodeCount);
    const addComponent = (
      item: ComponentLayout,
      offsetX: number,
      offsetY: number,
    ): void => {
      item.offsetX = offsetX;
      item.offsetY = offsetY;
      for (const node of item.nodes) {
        packedX[node] = x[node] + offsetX;
        packedY[node] = y[node] + offsetY;
      }
    };

    addComponent(components[0], 0, 0);
    const packOrder = components
      .slice(1)
      .sort(
        (left, right) =>
          right.radius - left.radius ||
          right.nodes.length - left.nodes.length ||
          compareNodes(left.root, right.root),
      );

    // Pack whole connected components as padded islands. The previous packer
    // tested individual nodes against one golden-angle spiral, allowing
    // unrelated components to occupy holes inside one another. Here every
    // component owns a circular envelope. A deterministic two-dimensional
    // Halton field distributes those envelopes organically across the plane:
    // no randomness, no single spiral, and no unrelated component inserted
    // into the dominant graph's interior.
    const envelopeRadius = (item: ComponentLayout): number =>
      item.radius + CORPUS_COMPONENT_GAP / 2;
    const dominantEnvelope = envelopeRadius(components[0]);
    let satelliteEnvelopeArea = 0;
    let largestSatelliteEnvelope = 0;
    for (const item of packOrder) {
      const envelope = envelopeRadius(item);
      satelliteEnvelopeArea += envelope * envelope;
      largestSatelliteEnvelope = Math.max(
        largestSatelliteEnvelope,
        envelope,
      );
    }
    let fieldOuterRadius =
      Math.sqrt(
        dominantEnvelope * dominantEnvelope +
          satelliteEnvelopeArea / CORPUS_COMPONENT_PACKING_DENSITY,
      ) + largestSatelliteEnvelope;
    const placedComponents = [components[0]];
    let sample = 1;

    for (const item of packOrder) {
      const envelope = envelopeRadius(item);
      let attempts = 0;
      while (true) {
        const innerRadius = dominantEnvelope + envelope;
        const outerRadius = Math.max(
          innerRadius + CORPUS_COMPONENT_GAP,
          fieldOuterRadius - envelope,
        );
        const theta = halton(sample, 2) * TWO_PI;
        const radialSample = halton(sample, 3);
        sample += 1;
        const radius = Math.sqrt(
          innerRadius * innerRadius +
            radialSample *
              (outerRadius * outerRadius - innerRadius * innerRadius),
        );
        const offsetX = Math.cos(theta) * radius;
        const offsetY = Math.sin(theta) * radius;
        let clear = true;
        for (const other of placedComponents) {
          const dx = offsetX - other.offsetX;
          const dy = offsetY - other.offsetY;
          const required = envelope + envelopeRadius(other);
          if (dx * dx + dy * dy < required * required - 0.0001) {
            clear = false;
            break;
          }
        }
        if (clear) {
          addComponent(item, offsetX, offsetY);
          placedComponents.push(item);
          break;
        }
        attempts += 1;
        if (
          attempts % CORPUS_COMPONENT_PROBES_BEFORE_EXPANSION ===
          0
        ) {
          fieldOuterRadius *= 1.08;
        }
      }
    }

    x.set(packedX);
    y.set(packedY);

    for (let left = 0; left < components.length; left += 1) {
      for (let right = left + 1; right < components.length; right += 1) {
        const dx = components[left].offsetX - components[right].offsetX;
        const dy = components[left].offsetY - components[right].offsetY;
        minimumComponentGap = Math.min(
          minimumComponentGap,
          Math.hypot(dx, dy) -
            components[left].radius -
            components[right].radius,
        );
      }
    }
  }
  if (!Number.isFinite(minimumComponentGap)) minimumComponentGap = 0;

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  const treeDistances: number[] = [];
  for (const item of components) {
    for (const node of item.nodes) {
      minX = Math.min(minX, x[node] - CORPUS_BUBBLE_RADIUS);
      minY = Math.min(minY, y[node] - CORPUS_BUBBLE_RADIUS);
      maxX = Math.max(maxX, x[node] + CORPUS_BUBBLE_RADIUS);
      maxY = Math.max(maxY, y[node] + CORPUS_BUBBLE_RADIUS);
      if (parent[node] !== -1) {
        treeDistances.push(
          Math.hypot(x[node] - x[parent[node]], y[node] - y[parent[node]]),
        );
      }
    }
  }
  treeDistances.sort((left, right) => left - right);
  const medianTreeDistance = treeDistances.length
    ? treeDistances[Math.floor(treeDistances.length / 2)]
    : CORPUS_LEVEL_GAP;
  const overlapCount = auditOverlaps(x, y);
  if (overlapCount) {
    throw new Error(
      `Corpus radial layout audit found ${overlapCount} overlapping node pairs.`,
    );
  }

  return {
    x,
    y,
    parent,
    parentEdge,
    depth,
    degree,
    component,
    subtreeSize,
    ranked: Uint32Array.from(rankedArray),
    roots: Uint32Array.from(components.map((item) => item.root)),
    bounds: nodeCount
      ? [minX, minY, maxX, maxY]
      : [-1, -1, 1, 1],
    medianTreeDistance,
    overlapCount,
    forestEdgeCount,
    maxDepth,
    dominantRoot: components[0]?.root ?? -1,
    minimumComponentGap,
    layoutHash: stableHash(parent, depth, x, y),
    durationMs: performance.now() - startedAt,
  };
}
