import type { MultiDirectedGraph } from "graphology";
import type Sigma from "sigma";

interface Point {
  x: number;
  y: number;
}

interface Segment {
  edge: string;
  source: string;
  target: string;
  start: Point;
  end: Point;
  control: Point;
  points: Point[];
  loop?: boolean;
}

interface EdgeLabelPlacement {
  edge: string;
  label: string;
  fontSize: number;
  width: number;
  height: number;
  fraction: number;
}

interface SceneCollisions {
  nodeOverlaps: number;
  edgeNodeOverlaps: number;
  edgeCrossings: number;
  labelNodeOverlaps: number;
  labelEndpointOverlaps: number;
  labelEdgeOverlaps: number;
  labelOverlaps: number;
  nodeConflictPairs: string[];
  edgeNodeConflictEdges: string[];
  labelNodeConflictEdges: string[];
  labelEndpointConflictEdges: string[];
  labelEdgeConflictEdges: string[];
  labelEdgeConflictPairs: string[];
  labelConflictEdges: string[];
}

interface CachedLayout {
  graph: MultiDirectedGraph;
  width: number;
  height: number;
  revision: number;
  edgeLabels: Map<string, EdgeLabelPlacement>;
  parallelOffsets: Map<string, number>;
  collisionKey: string | null;
  collisions: SceneCollisions | null;
}

interface CollisionLabelOptions {
  renderer: Sigma;
  graph: MultiDirectedGraph;
  canvas: HTMLCanvasElement;
  fullyLabeledNodes: Set<string>;
  edgeLabeledNodes: Set<string>;
  selectedNode: string | null;
  hoveredNode: string | null;
  selectedEdge: string | null;
  selectedEdges: ReadonlySet<string> | null;
  hoveredEdge: string | null;
}


export interface CollisionLabelMetrics {
  edgeLabels: number;
  requiredEdgeLabels: number;
  nodeOverlaps: number;
  edgeNodeOverlaps: number;
  edgeCrossings: number;
  labelNodeOverlaps: number;
  labelEndpointOverlaps: number;
  labelEdgeOverlaps: number;
  labelOverlaps: number;
  nodeConflictPairs: string[];
  edgeNodeConflictEdges: string[];
  labelNodeConflictEdges: string[];
  labelEndpointConflictEdges: string[];
  labelEdgeConflictEdges: string[];
  labelEdgeConflictPairs: string[];
  labelConflictEdges: string[];
  forbiddenCollisions: number;
  totalCollisions: number;
}

const caches = new WeakMap<HTMLCanvasElement, CachedLayout>();
const edgeLabelHitRegions = new WeakMap<
  HTMLCanvasElement,
  Array<{ edge: string; polygon: Point[] }>
>();
let nextLayoutRevision = 1;

function rectPolygon(
  center: Point,
  width: number,
  height: number,
  angle = 0,
): Point[] {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const halfWidth = width / 2;
  const halfHeight = height / 2;
  return [
    { x: -halfWidth, y: -halfHeight },
    { x: halfWidth, y: -halfHeight },
    { x: halfWidth, y: halfHeight },
    { x: -halfWidth, y: halfHeight },
  ].map((point) => ({
    x: center.x + point.x * cosine - point.y * sine,
    y: center.y + point.x * sine + point.y * cosine,
  }));
}

function nodePolygon(point: Point, radius: number): Point[] {
  const extent = radius + 3;
  return rectPolygon(point, extent * 2, extent * 2);
}

function projection(polygon: Point[], axis: Point): [number, number] {
  const values = polygon.map((point) => point.x * axis.x + point.y * axis.y);
  return [Math.min(...values), Math.max(...values)];
}

function polygonsOverlap(left: Point[], right: Point[], gap = 2): boolean {
  const polygons = [left, right];
  for (const polygon of polygons) {
    for (let index = 0; index < polygon.length; index += 1) {
      const current = polygon[index];
      const next = polygon[(index + 1) % polygon.length];
      const dx = next.x - current.x;
      const dy = next.y - current.y;
      const length = Math.hypot(dx, dy) || 1;
      const axis = { x: -dy / length, y: dx / length };
      const [leftMin, leftMax] = projection(left, axis);
      const [rightMin, rightMax] = projection(right, axis);
      if (leftMax + gap <= rightMin || rightMax + gap <= leftMin) {
        return false;
      }
    }
  }
  return true;
}

function orientation(left: Point, middle: Point, right: Point): number {
  const value =
    (middle.y - left.y) * (right.x - middle.x) -
    (middle.x - left.x) * (right.y - middle.y);
  if (Math.abs(value) < 0.0001) return 0;
  return value > 0 ? 1 : 2;
}

function pointOnSegment(left: Point, middle: Point, right: Point): boolean {
  return (
    middle.x <= Math.max(left.x, right.x) + 0.001 &&
    middle.x >= Math.min(left.x, right.x) - 0.001 &&
    middle.y <= Math.max(left.y, right.y) + 0.001 &&
    middle.y >= Math.min(left.y, right.y) - 0.001
  );
}

function segmentsIntersect(
  firstStart: Point,
  firstEnd: Point,
  secondStart: Point,
  secondEnd: Point,
): boolean {
  const first = orientation(firstStart, firstEnd, secondStart);
  const second = orientation(firstStart, firstEnd, secondEnd);
  const third = orientation(secondStart, secondEnd, firstStart);
  const fourth = orientation(secondStart, secondEnd, firstEnd);
  if (first !== second && third !== fourth) return true;
  if (first === 0 && pointOnSegment(firstStart, secondStart, firstEnd)) {
    return true;
  }
  if (second === 0 && pointOnSegment(firstStart, secondEnd, firstEnd)) {
    return true;
  }
  if (third === 0 && pointOnSegment(secondStart, firstStart, secondEnd)) {
    return true;
  }
  return fourth === 0 && pointOnSegment(secondStart, firstEnd, secondEnd);
}

function pointInsidePolygon(point: Point, polygon: Point[]): boolean {
  let inside = false;
  for (
    let index = 0, previous = polygon.length - 1;
    index < polygon.length;
    previous = index, index += 1
  ) {
    const currentPoint = polygon[index];
    const previousPoint = polygon[previous];
    const crosses =
      currentPoint.y > point.y !== previousPoint.y > point.y &&
      point.x <
        ((previousPoint.x - currentPoint.x) *
          (point.y - currentPoint.y)) /
          (previousPoint.y - currentPoint.y || 1) +
          currentPoint.x;
    if (crosses) inside = !inside;
  }
  return inside;
}

function segmentIntersectsPolygon(
  start: Point,
  end: Point,
  polygon: Point[],
): boolean {
  if (pointInsidePolygon(start, polygon) || pointInsidePolygon(end, polygon)) {
    return true;
  }
  for (let index = 0; index < polygon.length; index += 1) {
    if (
      segmentsIntersect(
        start,
        end,
        polygon[index],
        polygon[(index + 1) % polygon.length],
      )
    ) return true;
  }
  return false;
}

function distanceToSegment(point: Point, start: Point, end: Point): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (!lengthSquared) return Math.hypot(point.x - start.x, point.y - start.y);
  const fraction = Math.max(
    0,
    Math.min(
      1,
      ((point.x - start.x) * dx + (point.y - start.y) * dy) /
        lengthSquared,
    ),
  );
  return Math.hypot(
    point.x - (start.x + fraction * dx),
    point.y - (start.y + fraction * dy),
  );
}

function polygonOverlapsCircle(
  polygon: Point[],
  center: Point,
  radius: number,
  gap = 2,
): boolean {
  const effectiveRadius = radius + gap;
  if (pointInsidePolygon(center, polygon)) return true;
  if (
    polygon.some(
      (point) =>
        Math.hypot(point.x - center.x, point.y - center.y) < effectiveRadius,
    )
  ) return true;
  for (let index = 0; index < polygon.length; index += 1) {
    if (
      distanceToSegment(
        center,
        polygon[index],
        polygon[(index + 1) % polygon.length],
      ) < effectiveRadius
    ) return true;
  }
  return false;
}

function polygonTouchesViewport(
  polygon: Point[],
  width: number,
  height: number,
): boolean {
  const xs = polygon.map((point) => point.x);
  const ys = polygon.map((point) => point.y);
  return (
    Math.max(...xs) >= 0 &&
    Math.min(...xs) <= width &&
    Math.max(...ys) >= 0 &&
    Math.min(...ys) <= height
  );
}

// A self-loop is a circle rather than a curve between two points, so it is
// sampled from its own polyline. Everything downstream already works off
// segment.points; only the two parametric helpers need to know the
// difference.
function segmentPoint(segment: Segment, fraction: number): Point {
  if (!segment.loop) return quadraticPoint(segment, fraction);
  const points = segment.points;
  const position = Math.max(
    0,
    Math.min(points.length - 1, fraction * (points.length - 1)),
  );
  const index = Math.min(points.length - 2, Math.floor(position));
  const offset = position - index;
  return {
    x: points[index].x + (points[index + 1].x - points[index].x) * offset,
    y: points[index].y + (points[index + 1].y - points[index].y) * offset,
  };
}

function segmentTangent(segment: Segment, fraction: number): Point {
  if (!segment.loop) return quadraticTangent(segment, fraction);
  const points = segment.points;
  const position = Math.max(
    0,
    Math.min(points.length - 1, fraction * (points.length - 1)),
  );
  const index = Math.min(points.length - 2, Math.floor(position));
  return {
    x: points[index + 1].x - points[index].x,
    y: points[index + 1].y - points[index].y,
  };
}

function edgeGeometry(
  placement: EdgeLabelPlacement,
  segment: Segment,
  padding = 0,
): { center: Point; angle: number; polygon: Point[] } {
  const center = segmentPoint(segment, placement.fraction);
  const tangent = segmentTangent(segment, placement.fraction);
  let angle = Math.atan2(tangent.y, tangent.x);
  if (angle > Math.PI / 2) angle -= Math.PI;
  if (angle < -Math.PI / 2) angle += Math.PI;
  return {
    center,
    angle,
    polygon: rectPolygon(
      center,
      placement.width + padding * 2,
      placement.height + padding * 2,
      angle,
    ),
  };
}

function quadraticPoint(segment: Segment, fraction: number): Point {
  const inverse = 1 - fraction;
  return {
    x:
      inverse * inverse * segment.start.x +
      2 * inverse * fraction * segment.control.x +
      fraction * fraction * segment.end.x,
    y:
      inverse * inverse * segment.start.y +
      2 * inverse * fraction * segment.control.y +
      fraction * fraction * segment.end.y,
  };
}

function quadraticTangent(segment: Segment, fraction: number): Point {
  return {
    x:
      2 * (1 - fraction) * (segment.control.x - segment.start.x) +
      2 * fraction * (segment.end.x - segment.control.x),
    y:
      2 * (1 - fraction) * (segment.control.y - segment.start.y) +
      2 * fraction * (segment.end.y - segment.control.y),
  };
}

function graphGeometry(
  renderer: Sigma,
  graph: MultiDirectedGraph,
  parallelOffsets: Map<string, number>,
  selectedNode: string | null,
  selectedEdge: string | null,
  selectedEdges: ReadonlySet<string> | null,
  visualScale: number,
  _auditAll: boolean,
): {
  nodes: Map<string, { point: Point; radius: number; polygon: Point[] }>;
  segments: Segment[];
} {
  const nodes = new Map<
    string,
    { point: Point; radius: number; polygon: Point[] }
  >();
  for (const id of graph.nodes()) {
    const display = renderer.getNodeDisplayData(id);
    if (!display || display.hidden) continue;
    const point = renderer.framedGraphToViewport(display);
    const radius = renderer.scaleSize(Number(display.size));
    nodes.set(id, {
      point,
      radius,
      polygon: nodePolygon(point, radius),
    });
  }

  const customRendered = Boolean(graph.getAttribute("customRenderedEdges"));
  const segments: Segment[] = [];
  for (const edge of graph.edges()) {
    const display = renderer.getEdgeDisplayData(edge);
    if (!customRendered && (!display || display.hidden)) continue;
    if (customRendered && selectedEdges && !selectedEdges.has(edge)) continue;
    if (
      customRendered &&
      !selectedEdges &&
      selectedEdge &&
      edge !== selectedEdge
    ) continue;
    const [source, target] = graph.extremities(edge);
    if (
      customRendered &&
      selectedNode &&
      source !== selectedNode &&
      target !== selectedNode
    ) continue;
    const sourceNode = nodes.get(source);
    const targetNode = nodes.get(target);
    if (!sourceNode || !targetNode) continue;

    if (source === target) {
      // Self-loop: a circle tangent to the node, reaching out to the anchor
      // the layout reserved for it, with the tag riding the far side. Drawn
      // from the node's rim so the arrowhead lands back on the node.
      const anchorX = graph.getEdgeAttribute(edge, "loopAnchorX");
      const anchorY = graph.getEdgeAttribute(edge, "loopAnchorY");
      if (anchorX === undefined || anchorY === undefined) continue;
      const anchor = renderer.graphToViewport({
        x: Number(anchorX),
        y: Number(anchorY),
      });
      const reachX = anchor.x - sourceNode.point.x;
      const reachY = anchor.y - sourceNode.point.y;
      const reach = Math.hypot(reachX, reachY) || 1;
      const loopRadius = reach / 2;
      const loopCenter = {
        x: (sourceNode.point.x + anchor.x) / 2,
        y: (sourceNode.point.y + anchor.y) / 2,
      };
      const base = Math.atan2(
        sourceNode.point.y - loopCenter.y,
        sourceNode.point.x - loopCenter.x,
      );
      const steps = 32;
      const points = Array.from({ length: steps + 1 }, (_, index) => {
        const turn = base + (index / steps) * Math.PI * 2;
        return {
          x: loopCenter.x + Math.cos(turn) * loopRadius,
          y: loopCenter.y + Math.sin(turn) * loopRadius,
        };
      });
      segments.push({
        edge,
        source,
        target,
        start: points[0],
        end: points[points.length - 1],
        control: anchor,
        points,
        loop: true,
      });
      continue;
    }

    const dx = targetNode.point.x - sourceNode.point.x;
    const dy = targetNode.point.y - sourceNode.point.y;
    const length = Math.hypot(dx, dy) || 1;
    const offset = (parallelOffsets.get(edge) || 0) * visualScale;
    const layoutCurve = Number(
      graph.getEdgeAttribute(edge, "layoutCurve") || 0,
    );
    let baseControl = {
      x: (sourceNode.point.x + targetNode.point.x) / 2,
      y: (sourceNode.point.y + targetNode.point.y) / 2,
    };
    if (layoutCurve) {
      const sourceAttributes = graph.getNodeAttributes(source);
      const targetAttributes = graph.getNodeAttributes(target);
      const graphDx =
        Number(targetAttributes.x) - Number(sourceAttributes.x);
      const graphDy =
        Number(targetAttributes.y) - Number(sourceAttributes.y);
      const graphLength = Math.hypot(graphDx, graphDy) || 1;
      baseControl = renderer.graphToViewport({
        x:
          (Number(sourceAttributes.x) + Number(targetAttributes.x)) / 2 -
          (graphDy / graphLength) * layoutCurve,
        y:
          (Number(sourceAttributes.y) + Number(targetAttributes.y)) / 2 +
          (graphDx / graphLength) * layoutCurve,
      });
    }
    const control = {
      x: baseControl.x - (dy / length) * offset,
      y: baseControl.y + (dx / length) * offset,
    };
    const segment: Segment = {
      edge,
      source,
      target,
      start: sourceNode.point,
      end: targetNode.point,
      control,
      points: [],
    };
    segment.points = offset === 0 && layoutCurve === 0
      ? [segment.start, segment.end]
      : Array.from(
          { length: 17 },
          (_, index) => quadraticPoint(segment, index / 16),
        );
    segments.push({
      ...segment,
    });
  }
  return { nodes, segments };
}

function parallelEdgeOffsets(graph: MultiDirectedGraph): Map<string, number> {
  const groupedEdges = new Map<string, string[]>();
  for (const edge of graph.edges()) {
    const [source, target] = graph.extremities(edge);
    // Self-loops are already separated by their own anchors; bowing them
    // here as if they were parallel edges would move them off those anchors.
    if (source === target) continue;
    const key = source < target
      ? `${source}\u0000${target}`
      : `${target}\u0000${source}`;
    const group = groupedEdges.get(key) || [];
    group.push(edge);
    groupedEdges.set(key, group);
  }
  const offsets = new Map<string, number>();
  const gap = Number(graph.getAttribute("parallelEdgeGap") || 48);
  for (const group of groupedEdges.values()) {
    group.sort();
    group.forEach((edge, index) => {
      const [source, target] = graph.extremities(edge);
      const slot = index - (group.length - 1) / 2;
      const canonicalDirection = source < target ? 1 : -1;
      offsets.set(edge, slot * gap * canonicalDirection);
    });
  }
  return offsets;
}

function ensurePlacements(
  graph: MultiDirectedGraph,
  context: CanvasRenderingContext2D,
  layout: CachedLayout,
  edgeLabeledNodes: Set<string>,
  selectedEdge: string | null,
  selectedEdges: ReadonlySet<string> | null,
  hoveredEdge: string | null,
): void {
  const labelEveryEdge = Boolean(graph.getAttribute("showAllLabels"));
  for (const edge of graph.edges()) {
    const existing = layout.edgeLabels.get(edge);
    if (existing) {
      existing.fraction = Number(
        graph.getEdgeAttribute(edge, "labelFraction") || 0.5,
      );
      continue;
    }
    const [source, target] = graph.extremities(edge);
    if (
      !labelEveryEdge &&
      edge !== selectedEdge &&
      !selectedEdges?.has(edge) &&
      edge !== hoveredEdge &&
      (!edgeLabeledNodes.has(source) || !edgeLabeledNodes.has(target))
    ) continue;
    const label = String(graph.getEdgeAttribute(edge, "relationLabel"));
    const fontSize = Boolean(graph.getAttribute("categoryMode")) ? 12 : 8;
    context.font = `600 ${fontSize}px Inter, Segoe UI, sans-serif`;
    layout.edgeLabels.set(edge, {
      edge,
      label,
      fontSize,
      width: context.measureText(label).width + 8,
      height: fontSize + 7,
      fraction: Number(
        graph.getEdgeAttribute(edge, "labelFraction") || 0.5,
      ),
    });
  }
}

function scaledPlacement(
  placement: EdgeLabelPlacement,
  visualScale: number,
): EdgeLabelPlacement {
  if (visualScale === 1) return placement;
  return {
    ...placement,
    fontSize: placement.fontSize * visualScale,
    width: placement.width * visualScale,
    height: placement.height * visualScale,
  };
}

function entityLines(
  context: CanvasRenderingContext2D,
  label: string,
  radius: number,
  visualScale: number,
): string[] {
  context.font =
    `650 ${8 * visualScale}px Inter, Segoe UI, sans-serif`;
  const maxWidth = Math.max(24 * visualScale, radius * 1.45);
  const tokens = label.split(/(?<=[ _])/);
  const lines: string[] = [];
  let line = "";
  for (const token of tokens) {
    if (line && context.measureText(line + token).width > maxWidth) {
      lines.push(line.trimEnd());
      line = token.trimStart();
    } else {
      line += token;
    }
    while (line && context.measureText(line).width > maxWidth) {
      let splitAt = line.length - 1;
      while (
        splitAt > 1 &&
        context.measureText(line.slice(0, splitAt)).width > maxWidth
      ) {
        splitAt -= 1;
      }
      lines.push(line.slice(0, splitAt));
      line = line.slice(splitAt);
    }
  }
  if (line) lines.push(line);
  return lines;
}

function drawEntityLabel(
  context: CanvasRenderingContext2D,
  label: string,
  point: Point,
  radius: number,
  visualScale: number,
): void {
  const lines = entityLines(context, label, radius, visualScale);
  const lineHeight = 8.5 * visualScale;
  const startY = point.y - ((lines.length - 1) * lineHeight) / 2;
  context.font =
    `650 ${8 * visualScale}px Inter, Segoe UI, sans-serif`;
  context.fillStyle = "#ffffff";
  context.textAlign = "center";
  context.textBaseline = "middle";
  lines.forEach((line, index) => {
    context.fillText(line, point.x, startY + index * lineHeight);
  });
}

function drawNodeHover(
  context: CanvasRenderingContext2D,
  point: Point,
  radius: number,
  visualScale: number,
): void {
  context.beginPath();
  context.arc(
    point.x,
    point.y,
    radius + 3 * visualScale,
    0,
    Math.PI * 2,
  );
  context.strokeStyle = "#66348c";
  context.lineWidth = 3 * visualScale;
  context.stroke();
}

function trimmedPoints(
  segment: Segment,
  sourceRadius: number,
  targetRadius: number,
  visualScale: number,
): Point[] {
  const sourceClearance = sourceRadius + 2 * visualScale;
  const targetClearance = targetRadius + 3 * visualScale;
  const distances = [0];
  for (let index = 1; index < segment.points.length; index += 1) {
    distances.push(
      distances[index - 1] +
        Math.hypot(
          segment.points[index].x - segment.points[index - 1].x,
          segment.points[index].y - segment.points[index - 1].y,
        ),
    );
  }
  const total = distances[distances.length - 1];
  if (total <= sourceClearance + targetClearance) return [];
  const pointAtDistance = (distance: number): Point => {
    let index = 1;
    while (index < distances.length - 1 && distances[index] < distance) {
      index += 1;
    }
    const previousDistance = distances[index - 1];
    const span = distances[index] - previousDistance || 1;
    const fraction = (distance - previousDistance) / span;
    const previous = segment.points[index - 1];
    const next = segment.points[index];
    return {
      x: previous.x + (next.x - previous.x) * fraction,
      y: previous.y + (next.y - previous.y) * fraction,
    };
  };
  const startDistance = sourceClearance;
  const endDistance = total - targetClearance;
  const trimmed = [pointAtDistance(startDistance)];
  for (let index = 1; index < segment.points.length - 1; index += 1) {
    if (distances[index] > startDistance && distances[index] < endDistance) {
      trimmed.push(segment.points[index]);
    }
  }
  trimmed.push(pointAtDistance(endDistance));
  return trimmed;
}

function drawCustomEdge(
  context: CanvasRenderingContext2D,
  segment: Segment,
  sourceRadius: number,
  targetRadius: number,
  active: boolean,
  visualScale: number,
  size: number,
  categoryMode: boolean,
): void {
  const points = trimmedPoints(
    segment,
    sourceRadius,
    targetRadius,
    visualScale,
  );
  if (points.length < 2) return;
  context.save();
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  for (let index = 1; index < points.length; index += 1) {
    context.lineTo(points[index].x, points[index].y);
  }
  const edgeWidth = Math.max(
    categoryMode ? 0.08 : 0.35,
    (active ? Math.max(2.2, size + 1.2) : size) * visualScale,
  );
  const alpha = categoryMode
    ? Math.min(0.72, 0.08 + (Math.max(0.65, size) / 8) * 0.64)
    : 1;
  context.strokeStyle = categoryMode
    ? `rgba(247,249,250,${Math.min(0.38, alpha)})`
    : "rgba(247,249,250,0.96)";
  context.lineWidth =
    edgeWidth + (categoryMode ? 0.7 : 2.6) * visualScale;
  context.stroke();
  context.strokeStyle = active
    ? "#66348c"
    : categoryMode
      ? `rgba(117,65,154,${alpha})`
      : "#8761a8";
  context.lineWidth = edgeWidth;
  context.stroke();

  const tip = points[points.length - 1];
  const previous = points[points.length - 2];
  const angle = Math.atan2(tip.y - previous.y, tip.x - previous.x);
  const arrowLength = (active ? 12 : 10) * visualScale;
  const arrowHalfWidth = (active ? 5.5 : 4.5) * visualScale;
  const base = {
    x: tip.x - Math.cos(angle) * arrowLength,
    y: tip.y - Math.sin(angle) * arrowLength,
  };
  const normal = { x: -Math.sin(angle), y: Math.cos(angle) };
  context.beginPath();
  context.moveTo(tip.x, tip.y);
  context.lineTo(
    base.x + normal.x * arrowHalfWidth,
    base.y + normal.y * arrowHalfWidth,
  );
  context.lineTo(
    base.x - normal.x * arrowHalfWidth,
    base.y - normal.y * arrowHalfWidth,
  );
  context.closePath();
  context.fillStyle = active
    ? "#66348c"
    : categoryMode
      ? `rgba(117,65,154,${alpha})`
      : "#8761a8";
  context.fill();
  context.restore();
}

function drawEdgeLabel(
  context: CanvasRenderingContext2D,
  placement: EdgeLabelPlacement,
  segment: Segment,
  active: boolean,
): void {
  const { center, angle } = edgeGeometry(placement, segment);
  context.save();
  context.translate(center.x, center.y);
  context.rotate(angle);
  context.beginPath();
  context.roundRect(
    -placement.width / 2,
    -placement.height / 2,
    placement.width,
    placement.height,
    3,
  );
  context.fillStyle = active
    ? "rgba(255,248,230,0.99)"
    : "rgba(250,247,252,0.98)";
  context.fill();
  context.strokeStyle = active ? "#66348c" : "#d4c1df";
  context.lineWidth = active ? 1.4 : 0.7;
  context.stroke();
  context.font =
    `600 ${placement.fontSize}px Inter, Segoe UI, sans-serif`;
  context.fillStyle = "#5c2b5d";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(placement.label, 0, 0.4);
  context.restore();
}

export function edgeLabelAtViewportPoint(
  canvas: HTMLCanvasElement,
  x: number,
  y: number,
): string | null {
  const regions = edgeLabelHitRegions.get(canvas) || [];
  for (let index = regions.length - 1; index >= 0; index -= 1) {
    if (pointInsidePolygon({ x, y }, regions[index].polygon)) {
      return regions[index].edge;
    }
  }
  return null;
}

function distanceToPolyline(point: Point, points: Point[]): number {
  let distance = Number.POSITIVE_INFINITY;
  for (let index = 1; index < points.length; index += 1) {
    distance = Math.min(
      distance,
      distanceToSegment(point, points[index - 1], points[index]),
    );
  }
  return distance;
}

function polylineIntersectsPolygon(
  points: Point[],
  polygon: Point[],
): boolean {
  for (let index = 1; index < points.length; index += 1) {
    if (segmentIntersectsPolygon(points[index - 1], points[index], polygon)) {
      return true;
    }
  }
  return false;
}

function polylinesIntersect(left: Point[], right: Point[]): boolean {
  for (let leftIndex = 1; leftIndex < left.length; leftIndex += 1) {
    for (let rightIndex = 1; rightIndex < right.length; rightIndex += 1) {
      if (
        segmentsIntersect(
          left[leftIndex - 1],
          left[leftIndex],
          right[rightIndex - 1],
          right[rightIndex],
        )
      ) return true;
    }
  }
  return false;
}

interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

interface SpatialItem<T> {
  value: T;
  bounds: Bounds;
}

interface SpatialTree<T> {
  bounds: Bounds;
  left?: SpatialTree<T>;
  right?: SpatialTree<T>;
  items?: SpatialItem<T>[];
}

function pointBounds(point: Point, radius = 0): Bounds {
  return {
    minX: point.x - radius,
    minY: point.y - radius,
    maxX: point.x + radius,
    maxY: point.y + radius,
  };
}

function polygonBounds(points: Point[], padding = 0): Bounds {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    minX: Math.min(...xs) - padding,
    minY: Math.min(...ys) - padding,
    maxX: Math.max(...xs) + padding,
    maxY: Math.max(...ys) + padding,
  };
}

function mergeBounds(items: Array<{ bounds: Bounds }>): Bounds {
  return {
    minX: Math.min(...items.map((item) => item.bounds.minX)),
    minY: Math.min(...items.map((item) => item.bounds.minY)),
    maxX: Math.max(...items.map((item) => item.bounds.maxX)),
    maxY: Math.max(...items.map((item) => item.bounds.maxY)),
  };
}

function expandedBounds(bounds: Bounds, padding: number): Bounds {
  return {
    minX: bounds.minX - padding,
    minY: bounds.minY - padding,
    maxX: bounds.maxX + padding,
    maxY: bounds.maxY + padding,
  };
}

function boundsOverlap(left: Bounds, right: Bounds): boolean {
  return !(
    left.maxX < right.minX ||
    left.minX > right.maxX ||
    left.maxY < right.minY ||
    left.minY > right.maxY
  );
}

function buildSpatialTree<T>(
  items: SpatialItem<T>[],
): SpatialTree<T> | null {
  if (!items.length) return null;
  const bounds = mergeBounds(items);
  if (items.length <= 12) return { bounds, items };
  const splitX =
    bounds.maxX - bounds.minX >= bounds.maxY - bounds.minY;
  const ordered = items.slice().sort((left, right) => {
    const leftCenter = splitX
      ? left.bounds.minX + left.bounds.maxX
      : left.bounds.minY + left.bounds.maxY;
    const rightCenter = splitX
      ? right.bounds.minX + right.bounds.maxX
      : right.bounds.minY + right.bounds.maxY;
    return leftCenter - rightCenter;
  });
  const middle = Math.floor(ordered.length / 2);
  return {
    bounds,
    left: buildSpatialTree(ordered.slice(0, middle)) || undefined,
    right: buildSpatialTree(ordered.slice(middle)) || undefined,
  };
}

function querySpatialTree<T>(
  tree: SpatialTree<T> | null,
  bounds: Bounds,
  matches: SpatialItem<T>[],
): void {
  if (!tree || !boundsOverlap(tree.bounds, bounds)) return;
  if (tree.items) {
    for (const item of tree.items) {
      if (boundsOverlap(item.bounds, bounds)) matches.push(item);
    }
    return;
  }
  querySpatialTree(tree.left || null, bounds, matches);
  querySpatialTree(tree.right || null, bounds, matches);
}

function sceneCollisionCounts(
  nodes: Map<string, { point: Point; radius: number; polygon: Point[] }>,
  segments: Segment[],
  placements: Map<string, EdgeLabelPlacement>,
  visualScale: number,
): SceneCollisions {
  const gap = 3 * visualScale;
  const nodeEntries = Array.from(nodes.entries()).map(
    ([id, geometry], index) => ({
      index,
      id,
      geometry,
    }),
  );
  const nodeItems = nodeEntries.map((value) => ({
    value,
    bounds: pointBounds(value.geometry.point, value.geometry.radius),
  }));
  const nodeTree = buildSpatialTree(nodeItems);
  let nodeOverlaps = 0;
  const nodeConflictPairs = new Set<string>();
  for (const node of nodeItems) {
    const candidates: typeof nodeItems = [];
    querySpatialTree(nodeTree, expandedBounds(node.bounds, gap), candidates);
    for (const candidate of candidates) {
      if (candidate.value.index <= node.value.index) continue;
      const first = node.value.geometry;
      const second = candidate.value.geometry;
      if (
        Math.hypot(
          first.point.x - second.point.x,
          first.point.y - second.point.y,
        ) <
        first.radius + second.radius + gap
      ) {
        nodeOverlaps += 1;
        nodeConflictPairs.add(
          `${node.value.id}\u0000${candidate.value.id}`,
        );
      }
    }
  }

  const segmentItems = segments.map((value, index) => ({
    value: { index, segment: value },
    bounds: polygonBounds(value.points),
  }));
  const segmentTree = buildSpatialTree(segmentItems);
  let edgeNodeOverlaps = 0;
  const edgeNodeConflictEdges = new Set<string>();
  for (const node of nodeItems) {
    const candidates: typeof segmentItems = [];
    querySpatialTree(segmentTree, expandedBounds(node.bounds, gap), candidates);
    for (const candidate of candidates) {
      const segment = candidate.value.segment;
      if (
        node.value.id === segment.source ||
        node.value.id === segment.target
      ) continue;
      if (
        distanceToPolyline(node.value.geometry.point, segment.points) <
        node.value.geometry.radius + 8 * visualScale
      ) {
        edgeNodeOverlaps += 1;
        edgeNodeConflictEdges.add(segment.edge);
      }
    }
  }

  const edgeCrossings = 0;

  const labelGeometries = segments
    .map((segment) => {
      const placement = placements.get(segment.edge);
      return placement
        ? {
            segment,
            polygon: edgeGeometry(placement, segment).polygon,
            edgeExclusionPolygon: edgeGeometry(
              placement,
              segment,
              3 * visualScale,
            ).polygon,
          }
        : null;
    })
    .filter(
      (
        value,
      ): value is {
        segment: Segment;
        polygon: Point[];
        edgeExclusionPolygon: Point[];
      } => value !== null,
    );
  const labelItems = labelGeometries.map((value, index) => ({
    value: { index, label: value },
    bounds: polygonBounds(value.edgeExclusionPolygon),
  }));
  const labelTree = buildSpatialTree(labelItems);
  let labelNodeOverlaps = 0;
  let labelEndpointOverlaps = 0;
  let labelEdgeOverlaps = 0;
  let labelOverlaps = 0;
  const labelNodeConflictEdges = new Set<string>();
  const labelEndpointConflictEdges = new Set<string>();
  const labelEdgeConflictEdges = new Set<string>();
  const labelEdgeConflictPairs = new Set<string>();
  const labelConflictEdges = new Set<string>();
  for (const labelItem of labelItems) {
    const label = labelItem.value.label;
    const nodeCandidates: typeof nodeItems = [];
    querySpatialTree(
      nodeTree,
      expandedBounds(labelItem.bounds, gap),
      nodeCandidates,
    );
    for (const node of nodeCandidates) {
      if (
        polygonOverlapsCircle(
          label.polygon,
          node.value.geometry.point,
          node.value.geometry.radius,
          gap,
        )
      ) {
        labelNodeOverlaps += 1;
        labelNodeConflictEdges.add(label.segment.edge);
      }
    }

    for (const endpointId of [label.segment.source, label.segment.target]) {
      const endpoint = nodes.get(endpointId);
      if (
        endpoint &&
        polygonOverlapsCircle(
          label.polygon,
          endpoint.point,
          endpoint.radius + 14 * visualScale,
          gap,
        )
      ) {
        labelEndpointOverlaps += 1;
        labelEndpointConflictEdges.add(label.segment.edge);
      }
    }

    const edgeCandidates: typeof segmentItems = [];
    querySpatialTree(
      segmentTree,
      expandedBounds(labelItem.bounds, gap),
      edgeCandidates,
    );
    for (const edge of edgeCandidates) {
      const segment = edge.value.segment;
      if (segment.edge === label.segment.edge) continue;
      if (
        polylineIntersectsPolygon(
          segment.points,
          label.edgeExclusionPolygon,
        )
      ) {
        labelEdgeOverlaps += 1;
        labelEdgeConflictEdges.add(label.segment.edge);
        labelEdgeConflictPairs.add(
          `${label.segment.edge}\u0000${segment.edge}`,
        );
      }
    }

    const labelCandidates: typeof labelItems = [];
    querySpatialTree(
      labelTree,
      expandedBounds(labelItem.bounds, gap),
      labelCandidates,
    );
    for (const candidate of labelCandidates) {
      if (candidate.value.index <= labelItem.value.index) continue;
      if (
        polygonsOverlap(
          label.polygon,
          candidate.value.label.polygon,
          gap,
        )
      ) {
        labelOverlaps += 1;
        labelConflictEdges.add(label.segment.edge);
        labelConflictEdges.add(candidate.value.label.segment.edge);
      }
    }
  }
  return {
    nodeOverlaps,
    edgeNodeOverlaps,
    edgeCrossings,
    labelNodeOverlaps,
    labelEndpointOverlaps,
    labelEdgeOverlaps,
    labelOverlaps,
    nodeConflictPairs: Array.from(nodeConflictPairs),
    edgeNodeConflictEdges: Array.from(edgeNodeConflictEdges),
    labelNodeConflictEdges: Array.from(labelNodeConflictEdges),
    labelEndpointConflictEdges: Array.from(labelEndpointConflictEdges),
    labelEdgeConflictEdges: Array.from(labelEdgeConflictEdges),
    labelEdgeConflictPairs: Array.from(labelEdgeConflictPairs),
    labelConflictEdges: Array.from(labelConflictEdges),
  };
}

export function drawCollisionLabels({
  renderer,
  graph,
  canvas,
  fullyLabeledNodes,
  edgeLabeledNodes,
  selectedNode,
  hoveredNode,
  selectedEdge,
  selectedEdges,
  hoveredEdge,
}: CollisionLabelOptions): CollisionLabelMetrics {
  const { width, height } = renderer.getDimensions();
  const pixelRatio = window.devicePixelRatio || 1;
  const targetWidth = Math.round(width * pixelRatio);
  const targetHeight = Math.round(height * pixelRatio);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const context = canvas.getContext("2d");
  const emptyMetrics: CollisionLabelMetrics = {
    edgeLabels: 0,
    requiredEdgeLabels: 0,
    nodeOverlaps: 0,
    edgeNodeOverlaps: 0,
    edgeCrossings: 0,
    labelNodeOverlaps: 0,
    labelEndpointOverlaps: 0,
    labelEdgeOverlaps: 0,
    labelOverlaps: 0,
    nodeConflictPairs: [],
    edgeNodeConflictEdges: [],
    labelNodeConflictEdges: [],
    labelEndpointConflictEdges: [],
    labelEdgeConflictEdges: [],
    labelEdgeConflictPairs: [],
    labelConflictEdges: [],
    forbiddenCollisions: 0,
    totalCollisions: 0,
  };
  if (!context) return emptyMetrics;
  edgeLabelHitRegions.set(canvas, []);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  let cache = caches.get(canvas);
  if (!cache || cache.graph !== graph) {
    cache = {
      graph,
      width,
      height,
      revision: nextLayoutRevision,
      edgeLabels: new Map(),
      parallelOffsets: parallelEdgeOffsets(graph),
      collisionKey: null,
      collisions: null,
    };
    nextLayoutRevision += 1;
    caches.set(canvas, cache);
  }
  cache.width = width;
  cache.height = height;
  ensurePlacements(
    graph,
    context,
    cache,
    edgeLabeledNodes,
    selectedEdge,
    selectedEdges,
    hoveredEdge,
  );

  const camera = renderer.getCamera().getState();
  const customRendered = Boolean(graph.getAttribute("customRenderedEdges"));
  const collisionAudit = Boolean(graph.getAttribute("collisionAudit"));
  const uniformScaling = Boolean(graph.getAttribute("uniformScaling"));
  const categoryMode = Boolean(graph.getAttribute("categoryMode"));
  const visualScale = uniformScaling
    ? 1 / camera.ratio
    : 1;
  const { nodes, segments } = graphGeometry(
    renderer,
    graph,
    cache.parallelOffsets,
    selectedNode,
    selectedEdge,
    selectedEdges,
    visualScale,
    false,
  );
  canvas.dataset.renderedEdges = String(segments.length);
  const segmentById = new Map(segments.map((segment) => [segment.edge, segment]));
  if (customRendered) {
    for (const segment of segments) {
      const source = nodes.get(segment.source);
      const target = nodes.get(segment.target);
      if (!source || !target) continue;
      drawCustomEdge(
        context,
        segment,
        source.radius,
        target.radius,
        segment.edge === hoveredEdge ||
          segment.edge === selectedEdge ||
          Boolean(selectedEdges?.has(segment.edge)),
        visualScale,
        Number(graph.getEdgeAttribute(segment.edge, "size") || 1.35),
        categoryMode,
      );
    }
  }

  const hitRegions: Array<{ edge: string; polygon: Point[] }> = [];
  const scaledPlacements = new Map<string, EdgeLabelPlacement>();
  const progressiveEdgeLabels = Boolean(
    graph.getAttribute("progressiveEdgeLabels"),
  );
  const occupiedLabelBounds: Bounds[] = [];
  let edgeLabels = 0;
  let requiredEdgeLabels = 0;
  for (const basePlacement of cache.edgeLabels.values()) {
    const placement = scaledPlacement(basePlacement, visualScale);
    const segment = segmentById.get(placement.edge);
    if (!segment) continue;
    const active =
      placement.edge === hoveredEdge ||
      placement.edge === selectedEdge ||
      Boolean(selectedEdges?.has(placement.edge));
    if (
      progressiveEdgeLabels &&
      !active &&
      placement.fontSize < 4.75
    ) continue;
    const geometry = edgeGeometry(placement, segment);
    if (!polygonTouchesViewport(geometry.polygon, width, height)) continue;
    requiredEdgeLabels += 1;
    if (progressiveEdgeLabels && !active) {
      const bounds = polygonBounds(geometry.polygon, 2 * visualScale);
      if (
        occupiedLabelBounds.some((occupied) =>
          boundsOverlap(bounds, occupied),
        ) ||
        Array.from(nodes.values()).some((node) =>
          polygonOverlapsCircle(
            geometry.polygon,
            node.point,
            node.radius,
            3 * visualScale,
          ),
        )
      ) continue;
      occupiedLabelBounds.push(bounds);
    }
    scaledPlacements.set(placement.edge, placement);
    drawEdgeLabel(
      context,
      placement,
      segment,
      active,
    );
    hitRegions.push({
      edge: placement.edge,
      polygon: rectPolygon(
        geometry.center,
        placement.width + 8 * visualScale,
        placement.height + 8 * visualScale,
        geometry.angle,
      ),
    });
    edgeLabels += 1;
  }
  const edgeLabelRatios: number[] = [];
  // Bare line left visible between one end of the tag and the node rim,
  // reported in unscaled pixels so it can be compared against the arrowhead
  // length directly. The ratio above cannot stand in for this: a long tag on
  // a long edge has a poor ratio while still leaving plenty of room, and a
  // short tag on a short edge does the reverse.
  const edgeLabelClearances: number[] = [];
  let offCenterEdgeLabels = 0;
  for (const placement of scaledPlacements.values()) {
    if (Math.abs(placement.fraction - 0.5) > 0.0001) {
      offCenterEdgeLabels += 1;
    }
    const segment = segmentById.get(placement.edge);
    if (!segment) continue;
    const source = nodes.get(segment.source);
    const target = nodes.get(segment.target);
    if (!source || !target || placement.width <= 0) continue;
    const points = trimmedPoints(
      segment,
      source.radius,
      target.radius,
      visualScale,
    );
    let length = 0;
    for (let index = 1; index < points.length; index += 1) {
      length += Math.hypot(
        points[index].x - points[index - 1].x,
        points[index].y - points[index - 1].y,
      );
    }
    if (length > 0) {
      edgeLabelRatios.push(length / placement.width);
      edgeLabelClearances.push(
        (length - placement.width) / 2 / (visualScale || 1),
      );
    }
  }
  edgeLabelRatios.sort((left, right) => left - right);
  edgeLabelClearances.sort((left, right) => left - right);
  canvas.dataset.minimumEdgeLabelClearance = String(
    edgeLabelClearances.length ? edgeLabelClearances[0] : 0,
  );
  canvas.dataset.medianEdgeLabelClearance = String(
    edgeLabelClearances.length
      ? edgeLabelClearances[Math.floor(edgeLabelClearances.length / 2)]
      : 0,
  );
  // Pixels drawn per graph unit, with the camera ratio divided back out.
  // The layout sizes its clearances in pixels, so this has to read 1: any
  // other value means graph distances and node sizes disagree about what a
  // pixel is, and clearances computed during layout shrink on the way to the
  // screen. It is a function of the custom bbox aspect ratio, which
  // prepareReadableLayout() matches to the container for exactly this reason.
  const originProbe = renderer.graphToViewport({ x: 0, y: 0 });
  const unitProbe = renderer.graphToViewport({ x: 100, y: 0 });
  canvas.dataset.graphPixelsPerUnit = String(
    (Math.hypot(
      unitProbe.x - originProbe.x,
      unitProbe.y - originProbe.y,
    ) /
      100) *
      camera.ratio,
  );
  canvas.dataset.offCenterEdgeLabels = String(offCenterEdgeLabels);
  canvas.dataset.minimumEdgeLabelRatio = String(
    edgeLabelRatios[0] || 0,
  );
  canvas.dataset.medianEdgeLabelRatio = String(
    edgeLabelRatios.length
      ? edgeLabelRatios[Math.floor(edgeLabelRatios.length / 2)]
      : 0,
  );
  canvas.dataset.maximumEdgeLabelRatio = String(
    edgeLabelRatios[edgeLabelRatios.length - 1] || 0,
  );
  edgeLabelHitRegions.set(canvas, hitRegions);

  if (hoveredNode) {
    const geometry = nodes.get(hoveredNode);
    if (geometry) {
      drawNodeHover(
        context,
        geometry.point,
        geometry.radius,
        visualScale,
      );
    }
  }

  let entityLabels = 0;
  let visibleNodeCount = 0;
  const showEveryEntity = Boolean(graph.getAttribute("showAllLabels"));
  for (const [node, geometry] of nodes) {
    if (
      geometry.point.x + geometry.radius < 0 ||
      geometry.point.x - geometry.radius > width ||
      geometry.point.y + geometry.radius < 0 ||
      geometry.point.y - geometry.radius > height
    ) continue;
    visibleNodeCount += 1;
    if (
      !showEveryEntity &&
      !fullyLabeledNodes.has(node) &&
      node !== selectedNode
    ) continue;
    drawEntityLabel(
      context,
      String(graph.getNodeAttribute(node, "label")),
      geometry.point,
      geometry.radius,
      visualScale,
    );
    entityLabels += 1;
  }

  const collisionKey = uniformScaling
    ? [
        "uniform",
        selectedNode || "",
        selectedEdge || "",
        selectedEdges ? Array.from(selectedEdges).sort().join(",") : "",
        scaledPlacements.size,
        Number(graph.getAttribute("geometryRevision") || 0),
        progressiveEdgeLabels ? camera.ratio : "",
      ].join(":")
    : [
        width,
        height,
        camera.ratio,
        camera.angle,
        selectedNode || "",
        selectedEdge || "",
        selectedEdges ? Array.from(selectedEdges).sort().join(",") : "",
        cache.edgeLabels.size,
        Number(graph.getAttribute("geometryRevision") || 0),
        visualScale,
      ].join(":");
  let collisions: SceneCollisions;
  if (
    customRendered &&
    cache.collisionKey === collisionKey &&
    cache.collisions
  ) {
    collisions = cache.collisions;
  } else if (
    customRendered &&
    collisionAudit
  ) {
    const auditGeometry = graphGeometry(
      renderer,
      graph,
      cache.parallelOffsets,
      selectedNode,
      selectedEdge,
      selectedEdges,
      visualScale,
      true,
    );
    collisions = sceneCollisionCounts(
      auditGeometry.nodes,
      auditGeometry.segments,
      scaledPlacements,
      visualScale,
    );
    cache.collisionKey = collisionKey;
    cache.collisions = collisions;
  } else {
    collisions = {
        nodeOverlaps: 0,
        edgeNodeOverlaps: 0,
        edgeCrossings: 0,
        labelNodeOverlaps: 0,
        labelEndpointOverlaps: 0,
        labelEdgeOverlaps: 0,
        labelOverlaps: 0,
        nodeConflictPairs: [],
        edgeNodeConflictEdges: [],
        labelNodeConflictEdges: [],
        labelEndpointConflictEdges: [],
        labelEdgeConflictEdges: [],
        labelEdgeConflictPairs: [],
        labelConflictEdges: [],
    };
  }
  const forbiddenCollisions =
    collisions.nodeOverlaps +
    collisions.edgeNodeOverlaps +
    collisions.labelNodeOverlaps +
    collisions.labelEndpointOverlaps +
    (Boolean(graph.getAttribute("allowEdgeUnderLabels"))
      ? 0
      : collisions.labelEdgeOverlaps) +
    collisions.labelOverlaps;
  const totalCollisions =
    forbiddenCollisions + collisions.edgeCrossings;
  canvas.dataset.nodeLabels = String(entityLabels);
  canvas.dataset.requiredNodeLabels = String(
    showEveryEntity ? visibleNodeCount : fullyLabeledNodes.size,
  );
  canvas.dataset.edgeLabels = String(edgeLabels);
  canvas.dataset.requiredEdgeLabels = String(requiredEdgeLabels);
  canvas.dataset.nodeOverlaps = String(collisions.nodeOverlaps);
  canvas.dataset.edgeNodeOverlaps = String(collisions.edgeNodeOverlaps);
  canvas.dataset.edgeCrossings = String(collisions.edgeCrossings);
  canvas.dataset.labelNodeOverlaps = String(collisions.labelNodeOverlaps);
  canvas.dataset.labelEndpointOverlaps = String(
    collisions.labelEndpointOverlaps,
  );
  canvas.dataset.labelEdgeOverlaps = String(collisions.labelEdgeOverlaps);
  canvas.dataset.labelOverlaps = String(collisions.labelOverlaps);
  canvas.dataset.nodeConflictPairs =
    collisions.nodeConflictPairs.join(",");
  canvas.dataset.edgeNodeConflictEdges =
    collisions.edgeNodeConflictEdges.join(",");
  canvas.dataset.labelEdgeConflictEdges =
    collisions.labelEdgeConflictEdges.join(",");
  canvas.dataset.labelEdgeConflictPairs =
    collisions.labelEdgeConflictPairs.join(",");
  canvas.dataset.forbiddenCollisions = String(forbiddenCollisions);
  canvas.dataset.totalCollisions = String(totalCollisions);
  canvas.dataset.layoutRevision = String(cache.revision);
  canvas.dataset.layoutSolveScale = String(
    graph.getAttribute("layoutSolveScale") || 1,
  );
  canvas.dataset.layoutSolveSteps = String(
    graph.getAttribute("layoutSolveSteps") || 0,
  );
  canvas.dataset.layoutVerified = String(
    graph.getAttribute("layoutVerified") !== false,
  );
  return {
    edgeLabels,
    requiredEdgeLabels,
    ...collisions,
    forbiddenCollisions,
    totalCollisions,
  };
}
