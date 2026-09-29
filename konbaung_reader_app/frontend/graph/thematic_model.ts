import type { GraphController } from "./controller";
import { MultiDirectedGraph } from "graphology";
import Sigma from "sigma";
import { EdgeLineProgram } from "sigma/rendering";
import {
  type ThematicLayoutInput,
  type ThematicLayoutResult,
} from "../thematic_layout";
import ThematicLayoutWorker from "../thematic_layout_worker?worker";
import {
  type ThematicEntityCategory,
  type ThematicTopologyPayload,
  type ThematicLayoutWorkerResponse,
} from "./types";
import { emptyNavigationState } from "./navigation_state";
import { el, clear } from "./dom";
import { LargeArrowProgram } from "./utilities";

export const thematic_model = {
  thematicEntityNodeId(this: GraphController, categoryId: string): string {
    return `entity-category:${categoryId}`;
  },

  thematicRelationNodeId(this: GraphController, categoryId: string): string {
    return `relation-category:${categoryId}`;
  },

  thematicRelationIndex(this: GraphController, node: string): number | null {
    if (!this.thematicPayload || !node.startsWith("relation-category:")) {
      return null;
    }
    const categoryId = node.slice("relation-category:".length);
    const index = this.thematicPayload.relationCategories.findIndex(
      (category) => category.id === categoryId,
    );
    return index >= 0 ? index : null;
  },

  thematicPatternsForCombination(
    this: GraphController,
    entityIndices: number[],
    relationIndices: number[],
    exactPattern: number | null = null,
  ): number[] {
    if (!this.thematicPayload) return [];
    if (exactPattern !== null) {
      const focus = this.thematicPayload.patterns[exactPattern];
      if (!focus) return [];
      return this.thematicPayload.patterns.flatMap((pattern, index) =>
        pattern[0] === focus[0] &&
        pattern[1] === focus[1] &&
        pattern[2] === focus[2]
          ? [index]
          : [],
      );
    }
    const entities = new Set(entityIndices);
    const relations = new Set(relationIndices);
    return this.thematicPayload.patterns.flatMap((pattern, index) => {
      if (relations.size && !relations.has(pattern[1])) return [];
      if (entities.size === 1) {
        const entity = entityIndices[0];
        if (pattern[0] !== entity && pattern[2] !== entity) return [];
      } else if (
        entities.size > 1 &&
        (!entities.has(pattern[0]) ||
          !entities.has(pattern[2]) ||
          pattern[0] === pattern[2])
      ) {
        return [];
      }
      return [index];
    });
  },

  /**
   * Record which categories a selection corresponds to.
   *
   * The controls are deliberately not written here. commitThematicSelection()
   * renders them from the state record, so a navigation change cannot leave
   * one surface updated and another stale.
   */
  syncThematicCategoryControls(
    this: GraphController,
    subjectIndices: number[],
    relationIndices: number[],
    objectIndices: number[] = [],
  ): void {
    const payload = this.thematicPayload;
    if (!payload) return;
    const entityId = (index: number): string[] => {
      const id = payload.entityCategories[index]?.id;
      return id ? [id] : [];
    };
    const relationId = (index: number): string[] => {
      const id = payload.relationCategories[index]?.id;
      return id ? [id] : [];
    };
    this.nav.categories = {
      subjects: subjectIndices.flatMap(entityId),
      relations: relationIndices.flatMap(relationId),
      objects: objectIndices.flatMap(entityId),
    };
  },

  thematicEntitySize(
    this: GraphController,
    category: ThematicEntityCategory,
    minimum: number,
    maximum: number,
  ): number {
    const low = Math.log1p(Math.max(0, minimum));
    const high = Math.log1p(Math.max(0, maximum));
    const value = Math.log1p(Math.max(0, category.mentionCount));
    const ratio = high > low ? (value - low) / (high - low) : 1;
    const minimumRadius = 7;
    const maximumRadius = minimumRadius * 10;
    const sizeRatio = Math.pow(Math.max(0, Math.min(1, ratio)), 6);
    return minimumRadius + sizeRatio * (maximumRadius - minimumRadius);
  },

  runThematicLayoutWorker(
    this: GraphController,
    payload: ThematicTopologyPayload,
  ): Promise<ThematicLayoutResult> {
    const requestId = ++this.thematicLayoutRequest;
    const entityCount = payload.entityCategories.length;
    const frequencies = new Float64Array(entityCount);
    const connectivity = new Float64Array(entityCount);
    const radii = new Float64Array(entityCount);
    const entityMentions = payload.entityCategories.map(
      (category) => category.mentionCount,
    );
    const minimumMention = Math.min(...entityMentions);
    const maximumMention = Math.max(...entityMentions);
    const maximumPatternCount = Math.max(
      1,
      ...payload.entityCategories.map((category) => category.patternCount),
    );
    const maximumRelationCount = Math.max(
      1,
      ...payload.entityCategories.map((category) => category.relationCount),
    );
    const maximumCounterpartCount = Math.max(
      1,
      ...payload.entityCategories.map((category) => category.counterpartCount),
    );
    const ids = payload.entityCategories.map((category, index) => {
      frequencies[index] = category.mentionCount;
      connectivity[index] =
        0.5 * (category.patternCount / maximumPatternCount) +
        0.25 * (category.relationCount / maximumRelationCount) +
        0.25 * (category.counterpartCount / maximumCounterpartCount);
      const size = this.thematicEntitySize(
        category,
        minimumMention,
        maximumMention,
      );
      radii[index] = size * 1.15 + 22;
      return this.thematicEntityNodeId(category.id);
    });
    const input: ThematicLayoutInput = {
      ids,
      entityCount,
      frequencies,
      connectivity,
      radii,
    };
    const worker = new ThematicLayoutWorker({
      name: "thematic-entity-layout",
    });
    this.thematicLayoutWorker = worker;
    return new Promise((resolve, reject) => {
      const finish = (): void => {
        worker.terminate();
        if (this.thematicLayoutWorker === worker) {
          this.thematicLayoutWorker = null;
        }
      };
      worker.addEventListener(
        "message",
        (event: MessageEvent<ThematicLayoutWorkerResponse>) => {
          const response = event.data;
          if (response.kind !== "layout" || response.requestId !== requestId)
            return;
          finish();
          if (requestId !== this.thematicLayoutRequest) {
            reject(
              new DOMException("Superseded thematic layout", "AbortError"),
            );
            return;
          }
          if (response.error || !response.result) {
            reject(new Error(response.error || "Thematic layout failed."));
            return;
          }
          resolve(response.result);
        },
      );
      worker.addEventListener("error", (event) => {
        finish();
        reject(new Error(event.message || "Thematic layout worker failed."));
      });
      worker.postMessage({ kind: "layout", requestId, input }, [
        frequencies.buffer,
        connectivity.buffer,
        radii.buffer,
      ]);
    });
  },

  makeThematicGraph(
    this: GraphController,
    payload: ThematicTopologyPayload,
    layout: ThematicLayoutResult,
  ): MultiDirectedGraph {
    const graph = new MultiDirectedGraph();
    const entityMentions = payload.entityCategories.map(
      (category) => category.mentionCount,
    );
    const minimumMention = Math.min(...entityMentions);
    const maximumMention = Math.max(...entityMentions);
    graph.setAttribute("thematic", true);
    graph.setAttribute("categoryMode", true);
    graph.setAttribute("layoutHash", layout.layoutHash);
    graph.setAttribute("hardCollisionFree", layout.overlapCount === 0);
    graph.setAttribute("uniformScaling", false);
    graph.setAttribute("patternCount", payload.patterns.length);

    payload.entityCategories.forEach((category, index) => {
      const node = this.thematicEntityNodeId(category.id);
      graph.addNode(node, {
        label: category.label,
        x: layout.x[index],
        y: layout.y[index],
        size: this.thematicEntitySize(category, minimumMention, maximumMention),
        color:
          category.activeMentionCount > 0
            ? "#17866f"
            : "rgba(113, 142, 135, 0.46)",
        zIndex: 3,
        type: "circle",
        thematicNode: true,
        thematicKind: "entity",
        categoryIndex: index,
        categoryDetail: category,
      });
      this.thematicPatternsByNode.set(node, []);
    });
    payload.relationCategories.forEach((category) => {
      this.thematicPatternsByNode.set(
        this.thematicRelationNodeId(category.id),
        [],
      );
    });

    payload.patterns.forEach((pattern, patternIndex) => {
      const sourceCategory = payload.entityCategories[pattern[0]];
      const relationCategory = payload.relationCategories[pattern[1]];
      const targetCategory = payload.entityCategories[pattern[2]];
      if (!sourceCategory || !relationCategory || !targetCategory) return;
      const source = this.thematicEntityNodeId(sourceCategory.id);
      const relationKey = this.thematicRelationNodeId(relationCategory.id);
      const target = this.thematicEntityNodeId(targetCategory.id);
      this.thematicPatternsByNode.get(source)?.push(patternIndex);
      this.thematicPatternsByNode.get(relationKey)?.push(patternIndex);
      if (target !== source) {
        this.thematicPatternsByNode.get(target)?.push(patternIndex);
      }
      const common = {
        label: null,
        relationId: relationCategory.id,
        relationLabel: relationCategory.label,
        occurrenceCount: pattern[3],
        activeCount: pattern[4],
        patternIndex,
        sourceEntityNode: source,
        targetEntityNode: target,
        size: 0.42,
        color:
          pattern[4] > 0
            ? "rgba(83, 64, 100, 0.07)"
            : "rgba(116, 121, 127, 0.018)",
        type: "line",
        forceLabel: false,
        zIndex: 0,
        thematicExact: true,
      };
      graph.addDirectedEdgeWithKey(`pattern:${patternIndex}`, source, target, {
        ...common,
        sourceId: source,
        targetId: target,
      });
    });
    return graph;
  },

  isSemanticTagSlice(this: GraphController): boolean {
    return new Set(["tag-filter", "filtered-tags"]).has(
      this.thematicPayload?.focus.kind || "",
    );
  },

  semanticTagSlicePatterns(this: GraphController): number[] {
    if (!this.isSemanticTagSlice() || !this.thematicPayload) return [];
    return this.thematicPayload.patterns.flatMap((pattern, index) =>
      pattern[4] > 0 ? [index] : [],
    );
  },

  activateSemanticTagUnion(this: GraphController): boolean {
    const patterns = this.semanticTagSlicePatterns();
    if (!patterns.length) return false;
    el.themeGranularity.value = "1";
    el.themeGranularityValue.value = "1+";
    this.thematicSelectionCandidates = patterns;
    this.setThematicSelection(patterns);
    if (this.thematicOverlayCanvas) {
      this.thematicOverlayCanvas.dataset.thematicFocusMode =
        this.thematicPayload?.focus.kind === "filtered-tags"
          ? "filtered-tags"
          : "semantic-union";
      this.thematicOverlayCanvas.dataset.scopeTagCount = String(
        this.thematicPayload?.focus.tagIds?.length || 0,
      );
      this.thematicOverlayCanvas.dataset.activePatterns = String(
        patterns.length,
      );
      this.thematicOverlayCanvas.dataset.activeEntityNodes = String(
        this.thematicPayload?.entityCategories.filter(
          (category) => category.activeMentionCount > 0,
        ).length || 0,
      );
      this.thematicOverlayCanvas.dataset.activeRelationCategories = String(
        this.thematicPayload?.relationCategories.filter(
          (category) => category.activeTripleCount > 0,
        ).length || 0,
      );
      this.thematicOverlayCanvas.dataset.minimumPatternInstances = "1";
    }
    return true;
  },

  async installThematicPayload(
    this: GraphController,
    payload: ThematicTopologyPayload,
  ): Promise<void> {
    this.destroyRenderer();
    this.payload = payload;
    this.thematicPayload = payload;
    this.syncCategoryOptionCounts(payload);
    // A newly loaded slice starts from a clean selection. The category lists
    // are kept: they describe the filter that produced this slice.
    const requestedCategories = this.nav.categories;
    this.nav = emptyNavigationState();
    this.nav.categories = requestedCategories;
    this.rawPrimaryEntity = null;
    this.rawPairEntity = null;
    this.rawContextRelation = null;
    this.rawSelectedEdges.clear();
    this.navigationHistory = [];
    this.hoveredNode = null;
    this.selectedNeighbors.clear();
    this.evidenceItems = [];
    el.container.style.visibility = "hidden";
    this.setLoading(true, "Arranging the direct thematic entity network…");

    const layout = await this.runThematicLayoutWorker(payload);
    if (layout.overlapCount !== 0) {
      throw new Error(
        `The thematic layout retained ${layout.overlapCount} category overlaps.`,
      );
    }
    this.thematicLayoutResult = layout;
    const graph = this.makeThematicGraph(payload, layout);
    this.graph = graph;
    this.graphBBox = {
      x: [layout.bounds[0], layout.bounds[2]],
      y: [layout.bounds[1], layout.bounds[3]],
    };
    if (new Set(["tag-filter", "filtered-tags"]).has(payload.focus.kind)) {
      const activePoints = payload.entityCategories.flatMap(
        (category, index) =>
          category.activeMentionCount > 0
            ? [{ x: layout.x[index], y: layout.y[index] }]
            : [],
      );
      if (activePoints.length) {
        const xs = activePoints.map((point) => point.x);
        const ys = activePoints.map((point) => point.y);
        const width = Math.max(1, Math.max(...xs) - Math.min(...xs));
        const height = Math.max(1, Math.max(...ys) - Math.min(...ys));
        const padding = Math.max(1, Math.max(width, height) * 0.04);
        this.graphBBox = {
          x: [Math.min(...xs) - padding, Math.max(...xs) + padding],
          y: [Math.min(...ys) - padding, Math.max(...ys) + padding],
        };
      }
    }
    this.renderer = new Sigma(graph, el.container, {
      edgeProgramClasses: {
        pattern: LargeArrowProgram,
        line: EdgeLineProgram,
      },
      defaultEdgeType: "pattern",
      enableEdgeEvents: false,
      hideEdgesOnMove: false,
      hideLabelsOnMove: true,
      renderLabels: false,
      renderEdgeLabels: false,
      defaultDrawNodeHover: () => undefined,
      minEdgeThickness: 0.18,
      stagePadding: 36,
      zIndex: true,
      minCameraRatio: 0.03,
      maxCameraRatio: 64,
      zoomToSizeRatioFunction: Math.sqrt,
      cameraPanBoundaries: null,
      nodeReducer: (node, data) => this.reduceNode(node, data),
      edgeReducer: (edge, data) => this.reduceEdge(edge, data),
    });
    this.renderer.setCustomBBox(this.graphBBox);
    this.thematicOverlayCanvas = this.renderer.createCanvas("thematicLabels", {
      beforeLayer: "mouse",
      style: { pointerEvents: "none" },
    });
    this.thematicOverlayCanvas.dataset.schemaVersion = String(
      payload.schemaVersion,
    );
    this.thematicOverlayCanvas.dataset.entityNodes = String(
      payload.entityCategories.length,
    );
    this.thematicOverlayCanvas.dataset.relationNodes = String(
      payload.relationCategories.length,
    );
    this.thematicOverlayCanvas.dataset.renderedGraphNodes = String(graph.order);
    this.thematicOverlayCanvas.dataset.patterns = String(
      payload.patterns.length,
    );
    this.thematicOverlayCanvas.dataset.patternEdges = String(
      payload.patterns.length,
    );
    this.thematicOverlayCanvas.dataset.renderedRelationNodes = "0";
    this.thematicOverlayCanvas.dataset.layoutHash = layout.layoutHash;
    this.thematicOverlayCanvas.dataset.overlaps = String(layout.overlapCount);
    this.thematicOverlayCanvas.dataset.coreMedianHigh =
      layout.coreMedianHigh.toFixed(3);
    this.thematicOverlayCanvas.dataset.coreMedianLow =
      layout.coreMedianLow.toFixed(3);
    const entitySizes = payload.entityCategories.map((category) =>
      Number(
        graph.getNodeAttribute(this.thematicEntityNodeId(category.id), "size"),
      ),
    );
    const minimumEntitySize = Math.min(...entitySizes);
    const maximumEntitySize = Math.max(...entitySizes);
    this.thematicOverlayCanvas.dataset.minimumEntitySize =
      minimumEntitySize.toFixed(3);
    this.thematicOverlayCanvas.dataset.maximumEntitySize =
      maximumEntitySize.toFixed(3);
    this.thematicOverlayCanvas.dataset.entitySizeRatio = (
      maximumEntitySize / minimumEntitySize
    ).toFixed(3);
    this.thematicOverlayCanvas.dataset.minimumPatternInstances =
      el.themeGranularity.value;
    this.thematicOverlayCanvas.dataset.direction = el.themeDirection.value;
    this.activateSemanticTagUnion();
    this.renderer.on("afterRender", () => this.drawThematicOverlay());
    this.bindRendererEvents();
    this.updateHeader();
    this.renderFocusDetail();
    const focusKind = payload.focus.kind;
    if (focusKind === "page") el.scope.value = "page";
    else if (focusKind === "range") el.scope.value = "range";
    else if (new Set(["tag-filter", "filtered-tags"]).has(focusKind)) {
      el.scope.value = "focus";
    } else if (new Set(["corpus", "vol1", "vol2", "vol3"]).has(focusKind)) {
      el.scope.value = focusKind;
    }
    await this.ensureCategoryCatalog();

    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => {
        if (!this.renderer || this.thematicPayload !== payload) {
          resolve();
          return;
        }
        this.renderer.resize(true);
        this.fitGraph(false);
        requestAnimationFrame(() => {
          if (this.renderer && this.graph && this.thematicPayload === payload) {
            if (
              this.selectedCategoryValues(el.entityCategories).length ||
              this.selectedCategoryValues(el.relationCategories).length ||
              this.selectedCategoryValues(el.objectCategories).length
            ) {
              this.applyThematicCategorySelection();
            }
            this.renderer.refresh();
            el.container.style.visibility = "";
            if (this.thematicOverlayCanvas) {
              this.thematicOverlayCanvas.dataset.ready = "true";
            }
            this.setLoading(false);
          }
          resolve();
        });
      });
    });
  },
};
