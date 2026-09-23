import type { ThematicRelationLabelHit } from "./types";
import type { GraphController } from "./controller";
import Sigma from "sigma";
import { EdgeLineProgram, NodePointProgram } from "sigma/rendering";
import { animateNodes } from "sigma/utils";
import {
  drawCollisionLabels,
  edgeLabelAtViewportPoint,
} from "../collision_labels";
import {
  type TopologyPayload,
  type CategoryDescriptor,
  type ThematicEntityCategory,
  type ThematicTopologyPayload,
} from "./types";
import { emptyNavigationState } from "./navigation_state";
import { el, clear } from "./dom";
import {
  GRAPH_HOVER_EFFECTS_ENABLED,
  LargeArrowProgram,
  bubbleNodeSize,
} from "./utilities";

export const renderer_lifecycle = {
  edgeLabelAtEvent(this: GraphController, event: MouseEvent): string | null {
    if (!this.labelCanvas) return null;
    const bounds = el.container.getBoundingClientRect();
    return edgeLabelAtViewportPoint(
      this.labelCanvas,
      event.clientX - bounds.left,
      event.clientY - bounds.top,
    );
  },

  syncHoveredEdge(this: GraphController): void {
    const edge = this.labelHoveredEdge || this.lineHoveredEdge;
    if (edge === this.hoveredEdge) return;
    this.hoveredEdge = edge;
    if (this.payload?.layout.kind === "atlas") {
      this.renderer?.scheduleRefresh();
    } else {
      this.renderer?.refresh();
    }
  },

  stopLayout(this: GraphController): void {
    if (this.layoutTimer !== null) {
      window.clearTimeout(this.layoutTimer);
      this.layoutTimer = null;
    }
    if (this.layout) {
      this.layout.kill();
      this.layout = null;
    }
  },

  destroyRenderer(this: GraphController): void {
    this.stopLayout();
    this.thematicLayoutRequest += 1;
    this.thematicLayoutWorker?.terminate();
    this.thematicLayoutWorker = null;
    this.corpusRadialRequest += 1;
    this.corpusRadialWorker?.terminate();
    this.corpusRadialWorker = null;
    if (this.corpusRadialDetailFrame !== null) {
      window.cancelAnimationFrame(this.corpusRadialDetailFrame);
      this.corpusRadialDetailFrame = null;
    }
    if (this.corpusRadialFullSettleTimer !== null) {
      window.clearTimeout(this.corpusRadialFullSettleTimer);
      this.corpusRadialFullSettleTimer = null;
    }
    this.atlasScopeAnimation?.();
    this.atlasScopeAnimation = null;
    if (this.progressiveDetailFrame !== null) {
      window.cancelAnimationFrame(this.progressiveDetailFrame);
      this.progressiveDetailFrame = null;
    }
    if (this.atlasDetailFrame !== null) {
      window.cancelAnimationFrame(this.atlasDetailFrame);
      this.atlasDetailFrame = null;
    }
    if (this.atlasSettleTimer !== null) {
      window.clearTimeout(this.atlasSettleTimer);
      this.atlasSettleTimer = null;
    }
    this.readableLayoutPending = false;
    this.readableLayoutCenter = { x: 0, y: 0 };
    this.fixedBBox = null;
    this.graphBBox = null;
    this.fullyLabeledNodes.clear();
    this.edgeLabeledNodes.clear();
    this.labelCanvas = null;
    this.atlasOverlayCanvas = null;
    this.atlasLevel = 0;
    this.atlasOverviewLabels.clear();
    this.atlasVisibleNodes.clear();
    this.atlasContinuityEdges.clear();
    this.atlasPredicateEdges.clear();
    this.atlasViewportLabels.clear();
    this.atlasViewportPredicateEdges.clear();
    this.atlasPredicateLabelPositions.clear();
    this.atlasViewportRegionalEdges.clear();
    this.atlasProjectedSpacing = 0;
    this.corpusRadialModel = null;
    this.corpusRadialLevel = 0;
    this.corpusRadialOverlayCanvas = null;
    this.corpusRadialShowAll = false;
    this.thematicPayload = null;
    this.thematicLayoutResult = null;
    this.thematicOverlayCanvas = null;
    this.thematicPatternsByNode.clear();
    this.thematicRelationLabelHits = [];
    this.thematicHoveredRelationLabel = null;
    this.thematicHoveredRelationPoint = null;
    el.container.classList.remove("edge-label-hover");
    // The whole navigation record goes at once; the category lists survive
    // because they describe the filter, not the selection.
    const survivingCategories = this.nav.categories;
    this.nav = emptyNavigationState();
    this.nav.categories = survivingCategories;
    this.thematicEvidencePattern = null;
    if (this.thematicInspectorFrame !== null) {
      window.cancelAnimationFrame(this.thematicInspectorFrame);
      this.thematicInspectorFrame = null;
    }
    el.showAll.checked = false;
    el.showAllControl.hidden = true;
    if (this.renderer) {
      this.renderer.kill();
      this.renderer = null;
    }
    clear(el.container);
    el.container.style.visibility = "";
    this.graph = null;
  },

  setLoading(
    this: GraphController,
    loading: boolean,
    message: string = "Loading graph…",
  ): void {
    el.loading.textContent = message;
    el.loading.hidden = !loading;
  },

  thematicNodeAtEvent(this: GraphController, event: MouseEvent): string | null {
    if (!this.thematicPayload || !this.renderer || !this.graph) return null;
    const bounds = el.container.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    let entityHit: { node: string; distance: number } | null = null;
    for (const category of this.thematicPayload.entityCategories) {
      const node = this.thematicEntityNodeId(category.id);
      if (!this.graph.hasNode(node)) continue;
      const attributes = this.graph.getNodeAttributes(node);
      const point = this.renderer.graphToViewport({
        x: Number(attributes.x),
        y: Number(attributes.y),
      });
      const distance = Math.hypot(point.x - x, point.y - y);
      if (
        distance <= this.renderer.scaleSize(Number(attributes.size)) + 4 &&
        (!entityHit || distance < entityHit.distance)
      ) {
        entityHit = { node, distance };
      }
    }
    // Relation pills are painted over the entity circles, so they take the
    // click where the two overlap. Testing the entity first made a pill that
    // happened to land on a circle unclickable: the click fell through to the
    // node underneath it. Nothing here moves a label; only the hit order.
    const relationHit = this.thematicRelationLabelAtPoint(x, y);
    if (relationHit) return relationHit.node;
    if (
      this.thematicPrimaryEntity &&
      entityHit &&
      (event.shiftKey || entityHit.node !== this.thematicPrimaryEntity)
    ) {
      return entityHit.node;
    }
    return entityHit?.node || null;
  },

  thematicRelationLabelAtPoint(
    this: GraphController,
    x: number,
    y: number,
  ): ThematicRelationLabelHit | null {
    return (
      this.thematicRelationLabelHits.find(
        (hit) =>
          Math.abs(hit.x - x) <= hit.halfWidth + 3 &&
          Math.abs(hit.y - y) <= hit.halfHeight + 3,
      ) || null
    );
  },

  thematicRelationLabelAtEvent(
    this: GraphController,
    event: MouseEvent,
  ): ThematicRelationLabelHit | null {
    if (!this.thematicPayload) return null;
    const bounds = el.container.getBoundingClientRect();
    return this.thematicRelationLabelAtPoint(
      event.clientX - bounds.left,
      event.clientY - bounds.top,
    );
  },

  updateThematicRelationHover(this: GraphController, event: MouseEvent): void {
    if (!this.thematicPayload || !this.thematicOverlayCanvas) return;
    const bounds = el.container.getBoundingClientRect();
    const hit = this.thematicRelationLabelAtPoint(
      event.clientX - bounds.left,
      event.clientY - bounds.top,
    );
    const next = hit?.node || null;
    if (
      next === this.thematicHoveredRelationLabel &&
      (!hit ||
        (this.thematicHoveredRelationPoint?.x === hit.x &&
          this.thematicHoveredRelationPoint?.y === hit.y))
    )
      return;
    this.thematicHoveredRelationLabel = next;
    this.thematicHoveredPattern = hit ? hit.patternIndex : null;
    this.thematicHoveredRelationPoint = hit ? { x: hit.x, y: hit.y } : null;
    el.container.classList.toggle("edge-label-hover", Boolean(next));
    if (next) {
      const relationIndex = this.thematicRelationIndex(next);
      const relation =
        relationIndex === null
          ? null
          : this.thematicPayload.relationCategories[relationIndex];
      this.thematicOverlayCanvas.dataset.hoveredRelation = relation?.id || "";
      // The exact pattern, so the two directions of one relation category are
      // distinguishable rather than both lighting up together.
      this.thematicOverlayCanvas.dataset.hoveredPattern = String(
        this.thematicHoveredPattern ?? -1,
      );
      this.thematicOverlayCanvas.dataset.hoveredRelationTag =
        relation?.tagId || "";
      this.thematicOverlayCanvas.dataset.hoveredRelationLabel =
        relation?.label || "";
    } else {
      delete this.thematicOverlayCanvas.dataset.hoveredRelation;
      delete this.thematicOverlayCanvas.dataset.hoveredRelationTag;
      delete this.thematicOverlayCanvas.dataset.hoveredRelationLabel;
      delete this.thematicOverlayCanvas.dataset.hoveredPattern;
    }
    this.renderer?.scheduleRefresh();
  },

  clearThematicRelationHover(this: GraphController): void {
    if (!this.thematicHoveredRelationLabel) return;
    this.thematicHoveredRelationLabel = null;
    this.thematicHoveredPattern = null;
    this.thematicHoveredRelationPoint = null;
    el.container.classList.remove("edge-label-hover");
    if (this.thematicOverlayCanvas) {
      delete this.thematicOverlayCanvas.dataset.hoveredRelation;
      delete this.thematicOverlayCanvas.dataset.hoveredRelationTag;
      delete this.thematicOverlayCanvas.dataset.hoveredRelationLabel;
      delete this.thematicOverlayCanvas.dataset.hoveredPattern;
    }
    this.renderer?.scheduleRefresh();
  },

  async installPayload(
    this: GraphController,
    payload: TopologyPayload,
  ): Promise<void> {
    if (payload.layout.kind === "thematic") {
      if (payload.schemaVersion !== 5) {
        throw new Error(
          `Unsupported thematic topology schema ${payload.schemaVersion}`,
        );
      }
      await this.installThematicPayload(payload as ThematicTopologyPayload);
      return;
    }
    const isAtlas = payload.layout.kind === "atlas";
    if (
      (isAtlas && payload.schemaVersion !== 3) ||
      (!isAtlas && payload.schemaVersion !== 2)
    ) {
      throw new Error(
        `Unsupported graph topology schema ${payload.schemaVersion}`,
      );
    }
    if (isAtlas && payload.focus.kind === "corpus") {
      await this.installCorpusRadialPayload(payload);
      return;
    }
    const previousAtlas =
      isAtlas &&
      this.payload?.layout.kind === "atlas" &&
      this.graph &&
      this.payload.layout.bounds
        ? {
            bounds: this.payload.layout.bounds,
            positions: new Map(
              this.payload.nodes.map((node) => [
                node[0],
                {
                  x: Number(this.graph?.getNodeAttribute(node[0], "x") || 0),
                  y: Number(this.graph?.getNodeAttribute(node[0], "y") || 0),
                },
              ]),
            ),
          }
        : null;
    this.destroyRenderer();
    this.payload = payload;
    this.selectedNode = null;
    this.selectedEdge = null;
    this.rawPrimaryEntity = null;
    this.rawPairEntity = null;
    this.rawContextRelation = null;
    this.rawSelectedEdges.clear();
    this.navigationHistory = [];
    this.hoveredNode = null;
    this.hoveredEdge = null;
    this.lineHoveredEdge = null;
    this.labelHoveredEdge = null;
    el.container.classList.remove("edge-label-hover");
    this.selectedNeighbors.clear();
    this.evidenceItems = [];

    const graph = this.makeGraph(payload);
    this.readableLayoutPending =
      !isAtlas && Boolean(graph.getAttribute("showAllLabels"));
    el.container.style.visibility = this.readableLayoutPending ? "hidden" : "";
    if (this.readableLayoutPending) {
      this.setLoading(true, "Drawing graph…");
    }
    let atlasAnimationTargets: Record<string, { x: number; y: number }> | null =
      null;
    if (previousAtlas && isAtlas && payload.layout.bounds) {
      const oldBounds = previousAtlas.bounds;
      const newBounds = payload.layout.bounds;
      const oldWidth = Math.max(1, oldBounds[2] - oldBounds[0]);
      const oldHeight = Math.max(1, oldBounds[3] - oldBounds[1]);
      const newWidth = newBounds[2] - newBounds[0];
      const newHeight = newBounds[3] - newBounds[1];
      atlasAnimationTargets = {};
      for (const node of payload.nodes) {
        const previous = previousAtlas.positions.get(node[0]);
        if (!previous) continue;
        const target = graph.getNodeAttributes(node[0]);
        atlasAnimationTargets[node[0]] = {
          x: Number(target.x),
          y: Number(target.y),
        };
        graph.mergeNodeAttributes(node[0], {
          x: newBounds[0] + ((previous.x - oldBounds[0]) / oldWidth) * newWidth,
          y:
            newBounds[1] +
            ((previous.y - oldBounds[1]) / oldHeight) * newHeight,
        });
      }
    }
    this.graph = graph;
    if (Boolean(graph.getAttribute("showAllLabels"))) {
      this.prepareReadableLayout(graph);
    } else if (isAtlas && payload.layout.bounds) {
      this.graphBBox = {
        x: [payload.layout.bounds[0], payload.layout.bounds[2]],
        y: [payload.layout.bounds[1], payload.layout.bounds[3]],
      };
    }
    this.renderer = new Sigma(graph, el.container, {
      edgeProgramClasses: {
        arrow: LargeArrowProgram,
        line: EdgeLineProgram,
      },
      nodeProgramClasses: isAtlas ? { point: NodePointProgram } : undefined,
      defaultEdgeType: "arrow",
      enableEdgeEvents: true,
      hideEdgesOnMove: isAtlas,
      hideLabelsOnMove: isAtlas,
      renderLabels: isAtlas,
      renderEdgeLabels: isAtlas,
      defaultDrawNodeHover: () => undefined,
      labelFont: "Inter, Segoe UI, sans-serif",
      labelSize: 12,
      labelWeight: "600",
      labelColor: { color: "#17212b" },
      edgeLabelFont: "Inter, Segoe UI, sans-serif",
      edgeLabelSize: 10,
      edgeLabelWeight: "600",
      edgeLabelColor: { color: "#5c2b5d" },
      labelDensity: isAtlas ? 0.75 : 1,
      labelGridCellSize: isAtlas ? 120 : 100,
      labelRenderedSizeThreshold: isAtlas ? 4 : 0,
      minEdgeThickness: 0.45,
      stagePadding: 28,
      zIndex: true,
      minCameraRatio: isAtlas ? 0.0005 : 0.02,
      maxCameraRatio: 512,
      zoomToSizeRatioFunction:
        Boolean(graph.getAttribute("uniformScaling")) && !isAtlas
          ? (ratio: number) => ratio
          : () => 1,
      cameraPanBoundaries: null,
      nodeReducer: (node, data) => this.reduceNode(node, data),
      edgeReducer: (edge, data) => this.reduceEdge(edge, data),
    });
    if (this.fixedBBox) {
      this.renderer.setCustomBBox(this.fixedBBox);
      this.renderer.refresh();
    } else if (isAtlas && this.graphBBox) {
      this.renderer.setCustomBBox(this.graphBBox);
      this.renderer.refresh();
    }
    if (isAtlas) {
      this.atlasOverlayCanvas = this.renderer.createCanvas("atlasRegions", {
        beforeLayer: "nodes",
        style: { pointerEvents: "none" },
      });
      this.renderer.on("afterRender", () => this.drawAtlasRegions());
    } else {
      this.labelCanvas = this.renderer.createCanvas("collisionLabels", {
        beforeLayer: "mouse",
        style: { pointerEvents: "none" },
      });
      this.renderer.on("afterRender", () => {
        if (!this.renderer || !this.graph || !this.labelCanvas) return;
        drawCollisionLabels({
          renderer: this.renderer,
          graph: this.graph,
          canvas: this.labelCanvas,
          fullyLabeledNodes: this.fullyLabeledNodes,
          edgeLabeledNodes: this.edgeLabeledNodes,
          selectedNode: this.selectedNode,
          hoveredNode: this.hoveredNode,
          selectedEdge: this.selectedEdge,
          selectedEdges:
            this.rawPairEntity || this.rawContextRelation
              ? this.rawSelectedEdges
              : null,
          hoveredEdge: this.hoveredEdge,
        });
        this.revealReadableLayout();
      });
    }
    this.bindRendererEvents();
    if (isAtlas) this.bindAtlasDetail();
    else this.bindAdaptiveDetail();
    this.updateHeader();
    this.renderFocusDetail();

    const scopeKinds = new Set(["corpus", "vol1", "vol2", "vol3"]);
    if (payload.focus.kind === "page") el.scope.value = "page";
    else if (payload.focus.kind === "range") {
      const rangeOption = el.scope.querySelector<HTMLOptionElement>(
        'option[value="range"]',
      );
      if (rangeOption) rangeOption.textContent = payload.focus.label;
      el.scope.value = "range";
    } else if (scopeKinds.has(payload.focus.kind))
      el.scope.value = payload.focus.kind;
    else el.scope.value = "focus";

    requestAnimationFrame(() => {
      this.renderer?.resize(true);
      if (Boolean(graph.getAttribute("showAllLabels"))) {
        const center = this.readableCameraCenter();
        this.renderer?.getCamera().setState({
          ...center,
          ratio: 1,
          angle: 0,
        });
        this.renderer?.refresh();
      } else {
        this.fitGraph(false);
        if (
          atlasAnimationTargets &&
          Object.keys(atlasAnimationTargets).length
        ) {
          if (this.atlasOverlayCanvas) {
            this.atlasOverlayCanvas.dataset.scopeTransition = String(
              Object.keys(atlasAnimationTargets).length,
            );
          }
          this.atlasScopeAnimation = animateNodes(
            graph,
            atlasAnimationTargets,
            { duration: 420, easing: "quadraticInOut" },
            () => {
              if (this.graph !== graph) return;
              this.atlasScopeAnimation = null;
              if (this.atlasOverlayCanvas) {
                delete this.atlasOverlayCanvas.dataset.scopeTransition;
              }
              this.renderer?.refresh();
            },
          );
        }
      }
    });

    if (payload.layout.kind === "force" && graph.order > 2) {
      this.runForceLayout();
    } else if (payload.focus.kind === "claim" && graph.size) {
      const firstEdge = graph.edges()[0];
      if (firstEdge) this.selectEdge(firstEdge);
    }
  },

  reduceNode(
    this: GraphController,
    node: string,
    data: Record<string, unknown>,
  ): Record<string, unknown> {
    if (Boolean(data.thematicNode)) {
      const active =
        Number(
          (data.categoryDetail as ThematicEntityCategory).activeMentionCount,
        ) > 0;
      const selected =
        node === this.selectedNode ||
        node === this.thematicPrimaryEntity ||
        node === this.thematicPairEntity;
      const foreground = this.selectedNeighbors.has(node);
      const hovering = node === this.hoveredNode;
      if (this.isSemanticTagSlice() && !active) {
        return {
          ...data,
          hidden: true,
          label: "",
          highlighted: false,
          zIndex: 0,
        };
      }
      if (this.thematicSelectionCandidates.length) {
        if (selected) {
          return {
            ...data,
            label: "",
            color: "#6d3a91",
            highlighted: true,
            size: Number(data.size) + 2,
            zIndex: 7,
          };
        }
        if (foreground) {
          return {
            ...data,
            label: "",
            color: "#14907a",
            highlighted: hovering,
            zIndex: 6,
          };
        }
        return {
          ...data,
          label: "",
          color: "rgba(151, 167, 162, 0.44)",
          highlighted: false,
          zIndex: 2,
        };
      }
      return {
        ...data,
        label: "",
        color: active ? "#17866f" : "rgba(113, 142, 135, 0.46)",
        highlighted: hovering,
        zIndex: hovering ? 7 : 3,
      };
    }
    if (Boolean(data.atlasAnchor)) {
      return {
        ...data,
        label: "",
        size: 0.01,
        forceLabel: false,
        highlighted: false,
        zIndex: 0,
      };
    }
    const activeNode = this.selectedNode;
    const activeEdge = this.selectedEdge;
    if (activeNode) {
      if (node === activeNode) {
        return {
          ...data,
          color: "#66348c",
          highlighted: true,
          forceLabel: true,
          zIndex: 4,
          size: Number(data.size) + 2,
        };
      }
      if (this.selectedNeighbors.has(node)) {
        const atlasLabel =
          Boolean(data.atlasNode) && this.atlasViewportLabels.has(node);
        return {
          ...data,
          color: "#15968a",
          label: Boolean(data.atlasNode)
            ? atlasLabel
              ? data.label
              : ""
            : data.label,
          forceLabel: Boolean(data.atlasNode)
            ? atlasLabel
            : this.fullyLabeledNodes.has(node),
          zIndex: 2,
        };
      }
      return { ...data, color: "#d3dade", label: "", zIndex: 0 };
    }
    if (activeEdge && this.graph) {
      const [source, target] = this.graph.extremities(activeEdge);
      if (node === source || node === target) {
        return {
          ...data,
          color: "#75419a",
          highlighted: true,
          forceLabel: true,
          zIndex: 3,
        };
      }
      return { ...data, color: "#d9dee2", label: "", zIndex: 0 };
    }
    if (Boolean(data.corpusRadial)) {
      const hovered = node === this.hoveredNode;
      return {
        ...data,
        label: "",
        forceLabel: false,
        highlighted: hovered,
        zIndex: hovered ? 4 : 1,
      };
    }
    if (Boolean(data.atlasNode)) {
      const frequency = Number(data.frequency || 1);
      const hovered = node === this.hoveredNode;
      if (this.atlasLevel === 0) {
        const labeled = this.atlasOverviewLabels.has(node) || hovered;
        return {
          ...data,
          label: labeled ? data.label : "",
          size: Math.min(3.1, 0.9 + Math.log2(frequency + 1) * 0.24),
          forceLabel: hovered,
          highlighted: hovered,
          zIndex: hovered ? 4 : 1,
        };
      }
      if (this.atlasLevel === 1) {
        const labeled = this.atlasViewportLabels.has(node) || hovered;
        return {
          ...data,
          label: labeled ? data.label : "",
          size: Math.min(6, 1.6 + Math.log2(frequency + 1) * 0.42),
          forceLabel: labeled,
          highlighted: hovered,
          zIndex: hovered ? 4 : 1,
        };
      }
      const bubbleRadius = bubbleNodeSize(String(data.label || ""));
      const spacingScale =
        this.atlasLevel === 2
          ? Math.max(4, this.atlasProjectedSpacing * 0.15)
          : Math.max(7, this.atlasProjectedSpacing * 0.22);
      return {
        ...data,
        label: this.atlasViewportLabels.has(node) || hovered ? data.label : "",
        size: Math.min(bubbleRadius, spacingScale),
        forceLabel: this.atlasViewportLabels.has(node) || hovered,
        highlighted: hovered,
        zIndex: hovered ? 4 : 1,
      };
    }
    if (this.fullyLabeledNodes.has(node)) {
      return { ...data, forceLabel: true };
    }
    return { ...data, forceLabel: false };
  },

  reduceEdge(
    this: GraphController,
    edge: string,
    data: Record<string, unknown>,
  ): Record<string, unknown> {
    if (Boolean(data.thematicExact)) {
      const patternIndex = Number(data.patternIndex);
      const active = Number(data.activeCount) > 0;
      if (this.isSemanticTagSlice() && !active) {
        return { ...data, hidden: true, label: null };
      }
      if (this.thematicSelectionCandidates.length) {
        if (!this.thematicSelectedPatterns.has(patternIndex)) {
          return {
            ...data,
            hidden: false,
            label: null,
            type: "line",
            color: "rgba(116, 121, 127, 0.012)",
            size: 0.2,
            zIndex: 0,
          };
        }
        return {
          ...data,
          hidden: false,
          label: null,
          type: "line",
          color: "rgba(116, 121, 127, 0.018)",
          size: 0.2,
          zIndex: 0,
        };
      }
      return {
        ...data,
        hidden: false,
        label: null,
        type: "line",
        color: active
          ? "rgba(83, 64, 100, 0.055)"
          : "rgba(116, 121, 127, 0.018)",
        size: 0.42,
        zIndex: 0,
      };
    }
    if (this.rawPairEntity || this.rawContextRelation) {
      if (!this.rawSelectedEdges.has(edge)) {
        return { ...data, hidden: true, label: null };
      }
      return {
        ...data,
        hidden: false,
        color: "#66348c",
        label: String(data.relationLabel),
        forceLabel: true,
        size: Number(data.size) + 1.2,
        zIndex: 4,
      };
    }
    if (Boolean(data.customRendered)) {
      return { ...data, hidden: true };
    }
    if (Boolean(data.corpusRadialTree)) {
      const source = String(data.sourceId);
      const target = String(data.targetId);
      const activeNode = this.selectedNode || this.hoveredNode;
      if (activeNode) {
        const incident = source === activeNode || target === activeNode;
        if (!incident) return { ...data, hidden: true, label: null };
        return {
          ...data,
          hidden: false,
          label: null,
          forceLabel: false,
          color: "#66348c",
          size: Number(data.size) + 0.8,
          zIndex: 2,
        };
      }
      if (this.selectedEdge) {
        if (edge !== this.selectedEdge) {
          return { ...data, hidden: true, label: null };
        }
        return {
          ...data,
          hidden: false,
          label: null,
          forceLabel: false,
          color: "#66348c",
          size: Number(data.size) + 1.2,
          zIndex: 3,
        };
      }
      if (edge === this.hoveredEdge) {
        return {
          ...data,
          hidden: false,
          label: null,
          forceLabel: false,
          color: "#66348c",
          size: Number(data.size) + 0.8,
          zIndex: 2,
        };
      }
      return {
        ...data,
        hidden: false,
        label: null,
        forceLabel: false,
      };
    }
    const activeAtlasNode =
      this.payload?.layout.kind === "atlas" ? this.hoveredNode : null;
    const activeNode = this.selectedNode || activeAtlasNode;
    if (activeNode && this.graph) {
      const source = String(data.sourceId);
      const target = String(data.targetId);
      const other =
        source === activeNode ? target : target === activeNode ? source : null;
      if (
        other === null ||
        (this.selectedNode && !this.selectedNeighbors.has(other))
      ) {
        return { ...data, hidden: true };
      }
      return {
        ...data,
        color: "#75419a",
        label: Boolean(data.atlasRawEdge)
          ? null
          : this.edgeLabeledNodes.has(source) &&
              this.edgeLabeledNodes.has(target)
            ? String(data.relationLabel)
            : null,
        forceLabel:
          !Boolean(data.atlasRawEdge) &&
          this.fullyLabeledNodes.has(source) &&
          this.fullyLabeledNodes.has(target),
        size: Number(data.size) + 0.8,
        zIndex: 2,
      };
    }
    if (this.selectedEdge) {
      if (edge !== this.selectedEdge) return { ...data, hidden: true };
      return {
        ...data,
        color: "#66348c",
        label: String(data.relationLabel),
        size: Number(data.size) + 1.5,
        zIndex: 4,
      };
    }
    if (edge === this.hoveredEdge) {
      return {
        ...data,
        color: "#66348c",
        label: String(data.relationLabel),
        size: Number(data.size) + 1,
        zIndex: 3,
      };
    }
    if (Boolean(data.atlasBundle)) {
      if (this.atlasLevel >= 1 || !Boolean(data.atlasOverviewBundle)) {
        return { ...data, hidden: true };
      }
      return {
        ...data,
        hidden: false,
        label: null,
        forceLabel: false,
        color: "#a891bc",
        size: Math.min(2.2, Number(data.size)),
        zIndex: 0,
      };
    }
    if (Boolean(data.atlasRawEdge)) {
      if (this.atlasLevel === 0 && !Boolean(data.atlasSmallComponent)) {
        return { ...data, hidden: true };
      }
      if (
        this.atlasLevel === 1 &&
        !Boolean(data.atlasInternal) &&
        !Boolean(data.atlasSmallComponent)
      ) {
        return { ...data, hidden: true };
      }
      if (this.atlasLevel === 1 && !this.atlasViewportRegionalEdges.has(edge)) {
        return { ...data, hidden: true };
      }
      if (this.atlasLevel >= 1) {
        const source = String(data.sourceId);
        const target = String(data.targetId);
        if (
          !(
            this.atlasVisibleNodes.has(source) &&
            this.atlasVisibleNodes.has(target)
          ) &&
          !this.atlasContinuityEdges.has(edge)
        ) {
          return { ...data, hidden: true };
        }
      }
      return {
        ...data,
        hidden: false,
        label: null,
        forceLabel: false,
        color: this.atlasLevel >= 2 ? "#8761a8" : "#b6a1c6",
        size:
          this.atlasLevel >= 2
            ? Number(data.size)
            : Math.max(0.35, Number(data.size) * 0.55),
        zIndex: 1,
      };
    }
    const source = String(data.sourceId);
    const target = String(data.targetId);
    if (
      this.fullyLabeledNodes.has(source) &&
      this.fullyLabeledNodes.has(target)
    ) {
      return {
        ...data,
        label: String(data.relationLabel),
        forceLabel: true,
      };
    }
    if (
      this.edgeLabeledNodes.has(source) &&
      this.edgeLabeledNodes.has(target)
    ) {
      return {
        ...data,
        label: String(data.relationLabel),
        forceLabel: false,
      };
    }
    return { ...data, label: null, forceLabel: false };
  },

  bindRendererEvents(this: GraphController): void {
    if (!this.renderer) return;
    const thematic = this.payload?.layout.kind === "thematic";
    if (GRAPH_HOVER_EFFECTS_ENABLED || thematic) {
      this.renderer.on("enterNode", ({ node }) => {
        if (Boolean(this.graph?.getNodeAttribute(node, "atlasAnchor"))) return;
        this.hoveredNode = node;
        el.container.classList.add("node-hover");
        if (this.payload?.layout.kind === "atlas") {
          this.renderer?.scheduleRefresh();
        } else {
          this.renderer?.refresh();
        }
      });
      this.renderer.on("leaveNode", () => {
        this.hoveredNode = null;
        el.container.classList.remove("node-hover");
        if (this.payload?.layout.kind === "atlas") {
          this.renderer?.scheduleRefresh();
        } else {
          this.renderer?.refresh();
        }
      });
    }
    this.renderer.on("clickNode", ({ node, event }) => {
      if (Boolean(this.graph?.getNodeAttribute(node, "atlasAnchor"))) return;
      if (thematic) {
        const original = event.original as MouseEvent;
        this.selectThematicNode(node, Boolean(original?.shiftKey));
        return;
      }
      this.selectNode(node);
    });
    this.renderer.on("doubleClickNode", ({ node, preventSigmaDefault }) => {
      if (Boolean(this.graph?.getNodeAttribute(node, "atlasAnchor"))) return;
      preventSigmaDefault();
      if (thematic) {
        const data = this.graph?.getNodeAttributes(node);
        const detail = data?.categoryDetail as CategoryDescriptor | undefined;
        if (detail && data?.thematicKind === "entity") {
          void this.filterByEntityCategory(detail.id);
        } else if (detail) {
          void this.filterByRelationCategory(detail.id);
        }
      } else {
        void this.loadEntity(node, 1);
      }
    });
    this.renderer.on("clickEdge", ({ edge }) => {
      if (Boolean(this.graph?.getEdgeAttribute(edge, "atlasBundle"))) return;
      this.selectEdge(edge);
    });
    if (GRAPH_HOVER_EFFECTS_ENABLED && !thematic) {
      this.renderer.on("enterEdge", ({ edge }) => {
        if (Boolean(this.graph?.getEdgeAttribute(edge, "atlasBundle"))) return;
        this.lineHoveredEdge = edge;
        this.syncHoveredEdge();
      });
      this.renderer.on("leaveEdge", () => {
        this.lineHoveredEdge = null;
        this.syncHoveredEdge();
      });
    }
    this.renderer.on("clickStage", () => {
      if (!this.consumeGraphClickSuppression()) this.navigateBackOneLayer();
    });
  },
};
