import type { GraphController } from "./controller";
import FA2Layout from "graphology-layout-forceatlas2/worker";
import { sameSet } from "./utilities";

export const camera = {
  cameraCenterForGraphPoint(
    this: GraphController,
    graphPoint: { x: number; y: number },
    ratio: number = 1,
  ): { x: number; y: number } {
    if (!this.renderer) return { x: 0.5, y: 0.5 };
    const identity = { x: 0.5, y: 0.5, ratio, angle: 0 };
    const viewport = this.renderer.graphToViewport(graphPoint, {
      cameraState: identity,
    });
    const viewportX = this.renderer.graphToViewport(graphPoint, {
      cameraState: { ...identity, x: 1.5 },
    });
    const viewportY = this.renderer.graphToViewport(graphPoint, {
      cameraState: { ...identity, y: 1.5 },
    });
    const target = {
      x: this.renderer.getDimensions().width / 2,
      y: this.renderer.getDimensions().height / 2,
    };
    const a = viewportX.x - viewport.x;
    const b = viewportY.x - viewport.x;
    const c = viewportX.y - viewport.y;
    const d = viewportY.y - viewport.y;
    const targetX = target.x - viewport.x;
    const targetY = target.y - viewport.y;
    const determinant = a * d - b * c;
    if (Math.abs(determinant) < 0.000001) {
      return { x: 0.5, y: 0.5 };
    }
    return {
      x: 0.5 + (targetX * d - b * targetY) / determinant,
      y: 0.5 + (a * targetY - targetX * c) / determinant,
    };
  },

  readableCameraCenter(this: GraphController): { x: number; y: number } {
    if (!this.renderer || !this.graph) return { x: 0.5, y: 0.5 };
    if (Boolean(this.graph.getAttribute("centerWholeGraph"))) {
      return this.cameraCenterForGraphPoint(this.readableLayoutCenter);
    }
    const startNode = String(
      this.graph.getAttribute("readableStartNode") || "",
    );
    if (!startNode || !this.graph.hasNode(startNode)) {
      return { x: 0.5, y: 0.5 };
    }
    const attributes = this.graph.getNodeAttributes(startNode);
    return this.cameraCenterForGraphPoint({
      x: Number(attributes.x),
      y: Number(attributes.y),
    });
  },

  recenterReadableView(this: GraphController): void {
    if (!this.renderer || !this.graph) return;
    let graphPoint = this.readableLayoutCenter;
    if (!Boolean(this.graph.getAttribute("centerWholeGraph"))) {
      const startNode = String(
        this.graph.getAttribute("readableStartNode") || "",
      );
      if (!startNode || !this.graph.hasNode(startNode)) return;
      const attributes = this.graph.getNodeAttributes(startNode);
      graphPoint = {
        x: Number(attributes.x),
        y: Number(attributes.y),
      };
    }
    const camera = this.renderer.getCamera();
    const state = camera.getState();
    const viewport = this.renderer.graphToViewport(graphPoint);
    const viewportX = this.renderer.graphToViewport(graphPoint, {
      cameraState: { ...state, x: state.x + 1 },
    });
    const viewportY = this.renderer.graphToViewport(graphPoint, {
      cameraState: { ...state, y: state.y + 1 },
    });
    const target = {
      x: this.renderer.getDimensions().width / 2,
      y: this.renderer.getDimensions().height / 2,
    };
    const a = viewportX.x - viewport.x;
    const b = viewportY.x - viewport.x;
    const c = viewportX.y - viewport.y;
    const d = viewportY.y - viewport.y;
    const targetX = target.x - viewport.x;
    const targetY = target.y - viewport.y;
    const determinant = a * d - b * c;
    if (Math.abs(determinant) < 0.000001) return;
    camera.setState({
      x: state.x + (targetX * d - b * targetY) / determinant,
      y: state.y + (a * targetY - targetX * c) / determinant,
      ratio: 1,
      angle: 0,
    });
    this.renderer.refresh();
  },

  bindAdaptiveDetail(this: GraphController): void {
    if (!this.renderer) return;
    this.renderer.getCamera().on("updated", () => {
      this.scheduleProgressiveDetail();
    });
    this.scheduleProgressiveDetail();
  },

  scheduleProgressiveDetail(this: GraphController): void {
    if (this.progressiveDetailFrame !== null) return;
    this.progressiveDetailFrame = window.requestAnimationFrame(() => {
      this.progressiveDetailFrame = null;
      this.updateProgressiveDetail();
    });
  },

  updateProgressiveDetail(this: GraphController): void {
    if (!this.renderer || !this.graph) return;
    const { width, height } = this.renderer.getDimensions();
    const labelCapacity = Math.max(24, Math.floor((width * height) / 5000));
    const edgeLabelCapacity = labelCapacity * 2;
    const visibleNodes = new Set<string>();
    const selectedExtremities =
      this.selectedEdge && this.graph.hasEdge(this.selectedEdge)
        ? new Set(this.graph.extremities(this.selectedEdge))
        : null;

    for (const node of this.graph.nodes()) {
      if (
        this.selectedNode &&
        node !== this.selectedNode &&
        !this.selectedNeighbors.has(node)
      )
        continue;
      if (selectedExtremities && !selectedExtremities.has(node)) {
        continue;
      }
      const attributes = this.graph.getNodeAttributes(node);
      const point = this.renderer.graphToViewport({
        x: Number(attributes.x),
        y: Number(attributes.y),
      });
      if (point.x < 0 || point.x > width || point.y < 0 || point.y > height) {
        continue;
      }
      visibleNodes.add(node);
      if (visibleNodes.size > edgeLabelCapacity) break;
    }

    const nextFullyLabeled =
      visibleNodes.size <= labelCapacity ? visibleNodes : new Set<string>();
    const nextEdgeLabeled =
      visibleNodes.size <= edgeLabelCapacity ? visibleNodes : new Set<string>();
    if (
      sameSet(this.fullyLabeledNodes, nextFullyLabeled) &&
      sameSet(this.edgeLabeledNodes, nextEdgeLabeled)
    )
      return;
    this.fullyLabeledNodes = nextFullyLabeled;
    this.edgeLabeledNodes = nextEdgeLabeled;
    this.renderer.refresh();
  },

  runForceLayout(this: GraphController): void {
    if (!this.graph || !this.renderer) return;
    this.stopLayout();
    this.layout = new FA2Layout(this.graph, {
      settings: {
        barnesHutOptimize: this.graph.order > 150,
        barnesHutTheta: 0.7,
        edgeWeightInfluence: 0.65,
        gravity: 1,
        scalingRatio: 8,
        slowDown: 4,
      },
      getEdgeWeight: "occurrenceCount",
    });
    this.layout.start();
    const duration = this.graph.order > 250 ? 1250 : 850;
    this.layoutTimer = window.setTimeout(() => {
      this.layout?.kill();
      this.layout = null;
      this.layoutTimer = null;
      this.renderer?.refresh();
      this.fitGraph(true);
    }, duration);
  },

  centerNode(this: GraphController, id: string): void {
    const display = this.renderer?.getNodeDisplayData(id);
    if (!display || !this.renderer) return;
    const camera = this.renderer.getCamera();
    const state = camera.getState();
    let targetRatio = Math.min(state.ratio, 0.2);
    let targetCenter = { x: display.x, y: display.y };
    if (
      this.payload?.layout.kind === "atlas" &&
      this.payload.layout.medianSpacing &&
      this.graph?.hasNode(id)
    ) {
      const attributes = this.graph.getNodeAttributes(id);
      targetCenter = this.cameraCenterForGraphPoint({
        x: Number(attributes.x),
        y: Number(attributes.y),
      });
      const origin = this.renderer.graphToViewport({ x: 0, y: 0 });
      const spaced = this.renderer.graphToViewport({
        x: this.payload.layout.medianSpacing,
        y: 0,
      });
      const projectedSpacing = Math.max(
        0.001,
        Math.hypot(spaced.x - origin.x, spaced.y - origin.y),
      );
      targetRatio = Math.max(
        0.0005,
        Math.min(state.ratio, (state.ratio * projectedSpacing) / 64),
      );
    }
    camera.animate(
      {
        x: targetCenter.x,
        y: targetCenter.y,
        ratio: targetRatio,
      },
      { duration: 450 },
    );
  },

  fitGraph(this: GraphController, animated: boolean = true): void {
    if (!this.renderer) return;
    if (this.corpusRadialModel) {
      const rootCenter = this.cameraCenterForGraphPoint({ x: 0, y: 0 }, 1.08);
      const state = { ...rootCenter, ratio: 1.08, angle: 0 };
      if (animated) {
        this.renderer.getCamera().animate(state, { duration: 450 });
      } else {
        this.renderer.getCamera().setState(state);
        requestAnimationFrame(() => this.renderer?.refresh());
      }
      return;
    }
    const graph = this.graphBBox;
    let center = { x: 0.5, y: 0.5 };
    let ratio = 1;
    if (graph) {
      const graphCenter = {
        x: (graph.x[0] + graph.x[1]) / 2,
        y: (graph.y[0] + graph.y[1]) / 2,
      };
      center = this.cameraCenterForGraphPoint(graphCenter);
      const centeredState = { ...center, ratio: 1, angle: 0 };
      const corners = [
        { x: graph.x[0], y: graph.y[0] },
        { x: graph.x[1], y: graph.y[0] },
        { x: graph.x[1], y: graph.y[1] },
        { x: graph.x[0], y: graph.y[1] },
      ].map(
        (point) =>
          this.renderer?.graphToViewport(point, {
            cameraState: centeredState,
          }) as { x: number; y: number },
      );
      const xs = corners.map((point) => point.x);
      const ys = corners.map((point) => point.y);
      const dimensions = this.renderer.getDimensions();
      ratio =
        Math.max(
          1,
          (Math.max(...xs) - Math.min(...xs)) /
            Math.max(1, dimensions.width - 56),
          (Math.max(...ys) - Math.min(...ys)) /
            Math.max(1, dimensions.height - 56),
        ) * (this.thematicPayload ? 0.98 : 1.35);
    }
    if (animated) {
      this.renderer
        .getCamera()
        .animate(
          { x: center.x, y: center.y, ratio, angle: 0 },
          { duration: 450 },
        );
    } else {
      this.renderer.getCamera().setState({
        x: center.x,
        y: center.y,
        ratio,
        angle: 0,
      });
      requestAnimationFrame(() => this.renderer?.refresh());
    }
  },

  zoomIn(this: GraphController): void {
    this.zoomAtViewportCenter(true);
  },

  zoomOut(this: GraphController): void {
    this.zoomAtViewportCenter(false);
  },

  zoomAtViewportCenter(this: GraphController, zoomingIn: boolean): void {
    if (!this.renderer) return;
    const dimensions = this.renderer.getDimensions();
    const viewportPoint = {
      x: dimensions.width / 2,
      y: dimensions.height / 2,
    };
    const camera = this.renderer.getCamera();
    const zoomingRatio = Number(this.renderer.getSetting("zoomingRatio"));
    const ratio = camera.getBoundedRatio(
      camera.getState().ratio * (zoomingIn ? 1 / zoomingRatio : zoomingRatio),
    );
    if (this.corpusRadialModel) {
      const rootViewport = this.corpusRadialViewportPoint(
        this.corpusRadialModel.layout.dominantRoot,
      );
      const rootIsCentral =
        Math.hypot(
          rootViewport.x - viewportPoint.x,
          rootViewport.y - viewportPoint.y,
        ) <
        Math.min(dimensions.width, dimensions.height) * 0.2;
      camera.animate(
        this.renderer.getViewportZoomedState(
          rootIsCentral ? rootViewport : viewportPoint,
          ratio,
        ),
        { duration: 140 },
      );
      return;
    }
    camera.animate(this.renderer.getViewportZoomedState(viewportPoint, ratio), {
      duration: 220,
    });
  },
};
