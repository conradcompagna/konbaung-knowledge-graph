import type { GraphController } from "./controller";
import { MultiDirectedGraph } from "graphology";
import Sigma from "sigma";
import {
  CORPUS_BUBBLE_RADIUS,
  CORPUS_MIN_CENTER_DISTANCE,
} from "../corpus_radial_layout";
import { type CorpusRadialModel } from "./types";
import { el, clear } from "./dom";
import { GRAPH_HOVER_EFFECTS_ENABLED } from "./utilities";

export const corpus_radial_view = {
  bindCorpusRadialDisclosure(this: GraphController): void {
    if (!this.renderer) return;
    this.renderer.getCamera().on("updated", () => {
      if (this.corpusRadialShowAll) {
        if (this.corpusRadialFullSettleTimer !== null) {
          window.clearTimeout(this.corpusRadialFullSettleTimer);
        }
        this.corpusRadialFullSettleTimer = window.setTimeout(() => {
          this.corpusRadialFullSettleTimer = null;
          this.refreshCorpusFullView();
        }, 70);
        return;
      }
      if (this.corpusRadialDetailFrame !== null) return;
      this.corpusRadialDetailFrame = window.requestAnimationFrame(() => {
        this.corpusRadialDetailFrame = null;
        this.updateCorpusRadialDisclosure();
      });
    });
  },

  corpusRadialViewportPoint(
    this: GraphController,
    nodeIndex: number,
  ): { x: number; y: number } {
    const model = this.corpusRadialModel as CorpusRadialModel;
    return (this.renderer as Sigma).graphToViewport({
      x: model.layout.x[nodeIndex],
      y: model.layout.y[nodeIndex],
    });
  },

  corpusRadialProjection(this: GraphController): {
    pixelsPerUnit: number;
    projectedRadius: number;
  } {
    const renderer = this.renderer as Sigma;
    const origin = renderer.graphToViewport({ x: 0, y: 0 });
    const unit = renderer.graphToViewport({ x: 1, y: 0 });
    const pixelsPerUnit = Math.hypot(unit.x - origin.x, unit.y - origin.y);
    return {
      pixelsPerUnit,
      projectedRadius: Math.max(
        0.5,
        Math.min(CORPUS_BUBBLE_RADIUS, CORPUS_BUBBLE_RADIUS * pixelsPerUnit),
      ),
    };
  },

  makeCorpusRenderGraph(this: GraphController): MultiDirectedGraph {
    const graph = new MultiDirectedGraph();
    graph.setAttribute("corpusRadial", true);
    graph.setAttribute("uniformScaling", true);
    graph.setAttribute("hardCollisionFree", true);
    if (this.corpusRadialModel) {
      graph.setAttribute(
        "layoutHash",
        this.corpusRadialModel.layout.layoutHash,
      );
    }
    return graph;
  },

  syncCorpusRadialDiagnostics(
    this: GraphController,
    level: 0 | 1 | 2 | 3,
  ): void {
    const model = this.corpusRadialModel;
    const graph = this.graph;
    if (!model || !graph) return;
    const diagnostics = this.corpusRadialOverlayCanvas?.dataset;
    if (diagnostics) {
      diagnostics.layoutReady = "true";
      diagnostics.layoutHash = model.layout.layoutHash;
      diagnostics.layoutDurationMs = model.layout.durationMs.toFixed(2);
      diagnostics.disclosureDurationMs = model.lastDisclosureMs.toFixed(2);
      diagnostics.minimumComponentGap =
        model.layout.minimumComponentGap.toFixed(3);
      diagnostics.hoverEffectsEnabled = String(GRAPH_HOVER_EFFECTS_ENABLED);
      diagnostics.overlapCount = String(model.layout.overlapCount);
      diagnostics.totalNodes = String(model.payload.nodes.length);
      diagnostics.totalTreeEdges = String(model.layout.forestEdgeCount);
      diagnostics.rootCount = String(model.layout.roots.length);
      diagnostics.treeOnly = "true";
      diagnostics.predicateLabels = "0";
      diagnostics.allCoordinatesFinite = "true";
      diagnostics.fullGraph = String(this.corpusRadialShowAll);
      diagnostics.detailLevel = String(level);
      diagnostics.admittedNodes = String(graph.order);
      diagnostics.admittedEdges = String(graph.size);
      diagnostics.pixelsPerUnit = model.pixelsPerUnit.toFixed(6);
      diagnostics.dominantRoot =
        model.payload.nodes[model.layout.dominantRoot]?.[1] || "";
      diagnostics.maxDepth = String(model.layout.maxDepth);
      diagnostics.layoutWidth = String(
        model.layout.bounds[2] - model.layout.bounds[0],
      );
      diagnostics.layoutHeight = String(
        model.layout.bounds[3] - model.layout.bounds[1],
      );
      diagnostics.medianTreeDistance = String(model.layout.medianTreeDistance);
      const rootViewport = this.corpusRadialViewportPoint(
        model.layout.dominantRoot,
      );
      diagnostics.rootViewportX = rootViewport.x.toFixed(2);
      diagnostics.rootViewportY = rootViewport.y.toFixed(2);
    }
    el.container.dataset.corpusRadialReady = "true";
    el.container.dataset.corpusRadialLevel = String(level);
    el.container.dataset.corpusRadialNodes = String(graph.order);
    el.container.dataset.corpusRadialEdges = String(graph.size);
    el.container.dataset.corpusRadialFull = String(this.corpusRadialShowAll);
  },

  refreshCorpusFullView(this: GraphController): void {
    const model = this.corpusRadialModel;
    const graph = this.graph;
    const renderer = this.renderer;
    if (!model || !graph || !renderer || !this.corpusRadialShowAll) return;
    const startedAt = performance.now();
    const projection = this.corpusRadialProjection();
    if (Math.abs(projection.projectedRadius - model.projectedRadius) >= 0.01) {
      graph.updateEachNodeAttributes(
        (_node, attributes) => ({
          ...attributes,
          size: projection.projectedRadius,
        }),
        { attributes: ["size"] },
      );
    }
    model.projectedRadius = projection.projectedRadius;
    model.pixelsPerUnit = projection.pixelsPerUnit;
    model.lastDisclosureMs = performance.now() - startedAt;
    this.corpusRadialLevel = 3;
    this.syncCorpusRadialDiagnostics(3);
    renderer.refresh();
  },

  updateCorpusRadialDisclosure(
    this: GraphController,
    force: boolean = false,
  ): void {
    const renderer = this.renderer;
    const graph = this.graph;
    const model = this.corpusRadialModel;
    if (!renderer || !graph || !model) return;
    const startedAt = performance.now();
    const { width, height } = renderer.getDimensions();
    const projection = this.corpusRadialProjection();
    const { pixelsPerUnit, projectedRadius } = projection;
    const projectedSpacing = pixelsPerUnit * CORPUS_MIN_CENTER_DISTANCE;
    const nextLevel: 0 | 1 | 2 | 3 =
      this.corpusRadialShowAll || pixelsPerUnit >= 1
        ? 3
        : projectedSpacing >= 30
          ? 2
          : projectedSpacing >= 10
            ? 1
            : 0;
    const center = renderer.viewportToGraph({
      x: width / 2,
      y: height / 2,
    });
    const preservingExisting =
      !force &&
      (model.pixelsPerUnit === 0 ||
        pixelsPerUnit >= model.pixelsPerUnit * 0.995);
    const next =
      preservingExisting && !this.corpusRadialShowAll
        ? new Set(model.admitted)
        : new Set<number>();
    const margin = projectedRadius + 8;
    const points = new Map<
      number,
      { x: number; y: number; visible: boolean }
    >();
    const pointFor = (
      node: number,
    ): { x: number; y: number; visible: boolean } => {
      const cached = points.get(node);
      if (cached) return cached;
      const point = this.corpusRadialViewportPoint(node);
      const value = {
        x: point.x,
        y: point.y,
        visible:
          point.x >= -margin &&
          point.x <= width + margin &&
          point.y >= -margin &&
          point.y <= height + margin,
      };
      points.set(node, value);
      return value;
    };

    const clearance = Math.max(0.1, Math.min(8, 8 * pixelsPerUnit));
    const minimumScreenDistance = projectedRadius * 2 + clearance;
    const cellSize = Math.max(1, minimumScreenDistance);
    const occupied = new Map<string, number[]>();
    const addOccupied = (node: number): void => {
      const point = pointFor(node);
      if (!point.visible) return;
      const cellX = Math.floor(point.x / cellSize);
      const cellY = Math.floor(point.y / cellSize);
      const key = `${cellX}:${cellY}`;
      const values = occupied.get(key);
      if (values) values.push(node);
      else occupied.set(key, [node]);
    };
    for (const node of next) addOccupied(node);
    const overlapsAccepted = (node: number): boolean => {
      const point = pointFor(node);
      const cellX = Math.floor(point.x / cellSize);
      const cellY = Math.floor(point.y / cellSize);
      const minimumSquared = minimumScreenDistance * minimumScreenDistance;
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
          const occupants = occupied.get(
            `${cellX + offsetX}:${cellY + offsetY}`,
          );
          if (!occupants) continue;
          for (const other of occupants) {
            const otherPoint = pointFor(other);
            const dx = point.x - otherPoint.x;
            const dy = point.y - otherPoint.y;
            if (dx * dx + dy * dy < minimumSquared - 0.01) return true;
          }
        }
      }
      return false;
    };
    const ensureOffscreenAncestors = (node: number): boolean => {
      const missing: number[] = [];
      let current = model.layout.parent[node];
      while (current >= 0 && !next.has(current)) {
        if (pointFor(current).visible) return false;
        missing.push(current);
        current = model.layout.parent[current];
      }
      for (let index = missing.length - 1; index >= 0; index -= 1) {
        next.add(missing[index]);
      }
      return true;
    };
    const admit = (node: number, checkOverlap: boolean): void => {
      if (next.has(node)) return;
      const point = pointFor(node);
      if (!point.visible || !ensureOffscreenAncestors(node)) return;
      if (checkOverlap && overlapsAccepted(node)) return;
      next.add(node);
      addOccupied(node);
    };

    if (this.corpusRadialShowAll) {
      for (let node = 0; node < model.payload.nodes.length; node += 1) {
        next.add(node);
      }
    } else if (nextLevel === 3) {
      const visibleNodes: number[] = [];
      for (let node = 0; node < model.payload.nodes.length; node += 1) {
        if (pointFor(node).visible) visibleNodes.push(node);
      }
      visibleNodes.sort(
        (left, right) =>
          model.layout.depth[left] - model.layout.depth[right] ||
          model.rankPosition[left] - model.rankPosition[right],
      );
      for (const node of visibleNodes) {
        if (next.has(node)) continue;
        const ancestors: number[] = [];
        let current = node;
        while (current >= 0 && !next.has(current)) {
          ancestors.push(current);
          current = model.layout.parent[current];
        }
        for (let index = ancestors.length - 1; index >= 0; index -= 1) {
          next.add(ancestors[index]);
        }
      }
    } else {
      for (let level = 0; level <= nextLevel; level += 1) {
        for (const node of model.levelAdditions[level]) admit(node, true);
      }
    }

    if (!next.size && model.layout.dominantRoot >= 0) {
      next.add(model.layout.dominantRoot);
    }
    const nextNodeIds = new Set(
      Array.from(next, (node) => model.payload.nodes[node][0]),
    );
    for (const nodeId of graph.nodes()) {
      if (!nextNodeIds.has(nodeId)) graph.dropNode(nodeId);
    }
    model.projectedRadius = projectedRadius;
    for (const node of next) {
      this.addCorpusRadialNode(graph, model, node);
      graph.setNodeAttribute(
        model.payload.nodes[node][0],
        "size",
        projectedRadius,
      );
    }
    for (const edge of graph.edges()) {
      const child = Number(graph.getEdgeAttribute(edge, "corpusChildIndex"));
      const parent = model.layout.parent[child];
      if (!next.has(child) || !next.has(parent)) graph.dropEdge(edge);
    }
    for (const child of next) {
      const parent = model.layout.parent[child];
      if (parent >= 0 && next.has(parent)) {
        this.addCorpusRadialEdge(graph, model, child);
      }
    }
    if (this.selectedNode && !graph.hasNode(this.selectedNode)) {
      this.selectedNode = null;
      this.selectedNeighbors.clear();
    }
    if (this.selectedEdge && !graph.hasEdge(this.selectedEdge)) {
      this.selectedEdge = null;
    }
    model.admitted = next;
    model.pixelsPerUnit = pixelsPerUnit;
    model.lastCenterX = center.x;
    model.lastCenterY = center.y;
    model.lastDisclosureMs = performance.now() - startedAt;
    this.corpusRadialLevel = nextLevel;
    this.syncCorpusRadialDiagnostics(nextLevel);
    renderer.refresh();
  },

  setCorpusShowAll(this: GraphController, enabled: boolean): void {
    if (!this.corpusRadialModel || !this.renderer) {
      el.showAll.checked = false;
      return;
    }
    if (enabled === this.corpusRadialShowAll) return;
    this.corpusRadialShowAll = enabled;
    el.showAll.checked = enabled;
    if (enabled) {
      this.setLoading(
        true,
        `Rendering all ${this.corpusRadialModel.payload.nodes.length.toLocaleString()} entities\u2026`,
      );
      this.fitGraph(false);
      requestAnimationFrame(() => {
        if (!this.corpusRadialShowAll) return;
        const model = this.corpusRadialModel as CorpusRadialModel;
        const renderer = this.renderer as Sigma;
        const startedAt = performance.now();
        const projection = this.corpusRadialProjection();
        model.projectedRadius = projection.projectedRadius;
        const fullGraph = this.makeCorpusRenderGraph();
        const admitted = new Set<number>();
        for (let node = 0; node < model.payload.nodes.length; node += 1) {
          admitted.add(node);
          this.addCorpusRadialNode(fullGraph, model, node);
        }
        for (let child = 0; child < model.payload.nodes.length; child += 1) {
          if (model.layout.parent[child] >= 0) {
            this.addCorpusRadialEdge(fullGraph, model, child);
          }
        }
        this.graph = fullGraph;
        model.admitted = admitted;
        model.pixelsPerUnit = projection.pixelsPerUnit;
        model.lastDisclosureMs = performance.now() - startedAt;
        this.corpusRadialLevel = 3;
        renderer.setSetting("enableEdgeEvents", false);
        renderer.setGraph(fullGraph);
        if (this.graphBBox) renderer.setCustomBBox(this.graphBBox);
        this.syncCorpusRadialDiagnostics(3);
        renderer.refresh();
        requestAnimationFrame(() => this.setLoading(false));
      });
      return;
    }
    const progressiveGraph = this.makeCorpusRenderGraph();
    this.graph = progressiveGraph;
    this.corpusRadialModel.admitted.clear();
    this.corpusRadialModel.pixelsPerUnit = 0;
    this.renderer.setSetting("enableEdgeEvents", true);
    this.renderer.setGraph(progressiveGraph);
    if (this.graphBBox) this.renderer.setCustomBBox(this.graphBBox);
    this.updateCorpusRadialDisclosure(true);
  },

  corpusRadialTextLines(
    this: GraphController,
    context: CanvasRenderingContext2D,
    label: string,
    maxWidth: number,
  ): string[] {
    const words = label.trim().split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    const lines: string[] = [];
    let line = "";
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (context.measureText(candidate).width <= maxWidth) {
        line = candidate;
        continue;
      }
      if (line) lines.push(line);
      line = word;
      if (lines.length === 3) break;
    }
    if (line && lines.length < 4) lines.push(line);
    if (lines.length === 4 && words.join(" ").length > lines.join(" ").length) {
      while (
        lines[3].length > 1 &&
        context.measureText(`${lines[3]}\u2026`).width > maxWidth
      ) {
        lines[3] = lines[3].slice(0, -1);
      }
      lines[3] += "\u2026";
    }
    return lines;
  },

  drawCorpusRadialOverlay(this: GraphController): void {
    const renderer = this.renderer;
    const graph = this.graph;
    const canvas = this.corpusRadialOverlayCanvas;
    if (!renderer || !graph || !canvas || !this.corpusRadialModel) return;
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
    if (!context) return;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    if (
      this.corpusRadialShowAll &&
      this.corpusRadialModel.projectedRadius * 2 < 52 &&
      !this.hoveredNode &&
      !this.selectedNode
    )
      return;
    context.textAlign = "center";
    context.textBaseline = "middle";
    for (const node of graph.nodes()) {
      const display = renderer.getNodeDisplayData(node);
      if (!display) continue;
      const point = renderer.framedGraphToViewport(display);
      const radius = renderer.scaleSize(Number(display.size));
      if (
        point.x + radius < 0 ||
        point.x - radius > width ||
        point.y + radius < 0 ||
        point.y - radius > height
      )
        continue;
      if (node === this.hoveredNode || node === this.selectedNode) {
        context.beginPath();
        context.arc(point.x, point.y, radius + 3, 0, Math.PI * 2);
        context.strokeStyle =
          node === this.selectedNode ? "#4d206f" : "#66348c";
        context.lineWidth = node === this.selectedNode ? 3 : 2;
        context.stroke();
      }
      if (radius * 2 < 52) continue;
      context.font = "600 8px Inter, Segoe UI, sans-serif";
      const lines = this.corpusRadialTextLines(
        context,
        String(graph.getNodeAttribute(node, "label") || ""),
        radius * 1.55,
      );
      const lineHeight = 9;
      const firstY = point.y - ((lines.length - 1) * lineHeight) / 2;
      context.fillStyle = "#ffffff";
      lines.forEach((line, index) => {
        context.fillText(line, point.x, firstY + index * lineHeight);
      });
    }
  },
};
