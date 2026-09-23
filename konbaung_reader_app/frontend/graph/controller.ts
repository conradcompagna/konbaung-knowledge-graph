/** Internal composition host: feature modules share one graph and navigation state. */
import { MultiDirectedGraph } from "graphology";
import FA2Layout from "graphology-layout-forceatlas2/worker";
import Sigma from "sigma";
import { type ThematicLayoutResult } from "../thematic_layout";
import {
  type TopologyPayload,
  type ThematicTopologyPayload,
  type EvidenceItem,
  type FilteredTagRole,
  type FilteredTagResult,
  type SimilarPatternResult,
  type ChronicleVolume,
  type ThematicNavigationState,
  type GraphNavigationSnapshot,
  type CorpusRadialModel,
} from "./types";
import { emptyNavigationState } from "./navigation_state";
import { el } from "./dom";
import { GRAPH_HOVER_EFFECTS_ENABLED } from "./utilities";
import { renderer_lifecycle } from "./renderer_lifecycle";
import { page_navigation } from "./page_navigation";
import { graph_layout } from "./graph_layout";
import { corpus_radial_model } from "./corpus_radial_model";
import { thematic_model } from "./thematic_model";
import { thematic_rendering } from "./thematic_rendering";
import { thematic_navigation } from "./thematic_navigation";
import { thematic_evidence } from "./thematic_evidence";
import { camera } from "./camera";
import { atlas_view } from "./atlas_view";
import { category_filters } from "./category_filters";
import { filtered_tags } from "./filtered_tags";
import { corpus_radial_view } from "./corpus_radial_view";
import { selection_inspector } from "./selection_inspector";
import { similar_patterns } from "./similar_patterns";
import { workspace } from "./workspace";

export class GraphController {
  graph: MultiDirectedGraph | null = null;

  renderer: Sigma | null = null;

  layout: FA2Layout | null = null;

  layoutTimer: number | null = null;

  graphAbort: AbortController | null = null;

  evidenceAbort: AbortController | null = null;

  payload: TopologyPayload | null = null;

  page = { volumeId: "vol1", pageNumber: 47 };

  pageVolumes = new Map<string, ChronicleVolume>();

  pageIndexPromise: Promise<void> | null = null;

  categoryCatalogLoaded = false;

  categoryCatalogPromise: Promise<void> | null = null;

  get selectedNode(): string | null {
    return this.nav.selectedNode;
  }

  set selectedNode(value: string | null) {
    this.nav.selectedNode = value;
  }

  get selectedEdge(): string | null {
    return this.nav.selectedEdge;
  }

  set selectedEdge(value: string | null) {
    this.nav.selectedEdge = value;
  }

  rawPrimaryEntity: string | null = null;

  rawPairEntity: string | null = null;

  rawContextRelation: string | null = null;

  rawSelectedEdges = new Set<string>();

  navigationHistory: GraphNavigationSnapshot[] = [];

  graphGesture: {
    active: boolean;
    pointerId: number;
    startX: number;
    startY: number;
    dragged: boolean;
  } = {
    active: false,
    pointerId: -1,
    startX: 0,
    startY: 0,
    dragged: false,
  };

  suppressGraphClickUntil = 0;

  hoveredNode: string | null = null;

  hoveredEdge: string | null = null;

  lineHoveredEdge: string | null = null;

  labelHoveredEdge: string | null = null;

  selectedNeighbors = new Set<string>();

  filteredTagsAbort: AbortController | null = null;

  filteredTagsItems: FilteredTagResult[] = [];

  filteredTagsOffset = 0;

  filteredTagsTotal = 0;

  filteredTagsHasMore = false;

  filteredTagsDebounce: number | null = null;

  filteredTagsSyncTimer: number | null = null;

  filteredTagsBucketKey = "";

  filteredTagsSelected = new Map<FilteredTagRole, Set<string>>([
    ["subject", new Set<string>()],
    ["relation", new Set<string>()],
    ["object", new Set<string>()],
  ]);

  // Anchor tag IDs whose averaged vector currently orders each bucket. Empty
  // means the bucket is back to plain frequency ordering.
  filteredTagsSimilarAnchors = new Map<FilteredTagRole, string[]>([
    ["subject", []],
    ["relation", []],
    ["object", []],
  ]);

  // The low-level tag filter currently applied to the graph. Distinct from
  // filteredTagsSelected, which is what is merely checked but not yet applied.
  appliedTagFilters = new Map<FilteredTagRole, string[]>([
    ["subject", []],
    ["relation", []],
    ["object", []],
  ]);

  // True while the selection came from the entity-group controls, whose
  // candidates already encode the pairing direction.
  get thematicGroupSelection(): boolean {
    return this.nav.groupSelection;
  }

  set thematicGroupSelection(value: boolean) {
    this.nav.groupSelection = value;
  }

  // Entity category indices that orient the edge colouring: a pattern with one
  // of these on its subject side is drawn as outgoing, on its object side as
  // incoming. Group A when it has anything in it, otherwise group B, otherwise
  // the single clicked entity.
  get thematicOrientationAnchors(): Set<number> {
    return this.nav.orientationAnchors;
  }

  set thematicOrientationAnchors(value: Set<number>) {
    this.nav.orientationAnchors = value;
  }

  similarPatternsAbort: AbortController | null = null;

  similarPatterns: SimilarPatternResult[] = [];

  similarPatternsSelected = new Set<string>();

  evidenceItems: EvidenceItem[] = [];

  evidenceHasMore = false;

  evidenceNextOffset: number | null = null;

  evidenceLoading = false;

  fullyLabeledNodes = new Set<string>();

  edgeLabeledNodes = new Set<string>();

  progressiveDetailFrame: number | null = null;

  readableLayoutPending = false;

  readableLayoutCenter = { x: 0, y: 0 };

  labelCanvas: HTMLCanvasElement | null = null;

  fixedBBox: {
    x: [number, number];
    y: [number, number];
  } | null = null;

  graphBBox: {
    x: [number, number];
    y: [number, number];
  } | null = null;

  atlasLevel: 0 | 1 | 2 | 3 = 0;

  atlasDetailFrame: number | null = null;

  atlasSettleTimer: number | null = null;

  atlasOverlayCanvas: HTMLCanvasElement | null = null;

  atlasOverviewLabels = new Set<string>();

  atlasVisibleNodes = new Set<string>();

  atlasContinuityEdges = new Set<string>();

  atlasPredicateEdges = new Set<string>();

  atlasViewportLabels = new Set<string>();

  atlasViewportPredicateEdges = new Set<string>();

  atlasPredicateLabelPositions = new Map<
    string,
    { x: number; y: number; width: number; label: string }
  >();

  atlasViewportRegionalEdges = new Set<string>();

  atlasScopeAnimation: (() => void) | null = null;

  atlasProjectedSpacing = 0;

  corpusRadialModel: CorpusRadialModel | null = null;

  corpusRadialWorker: Worker | null = null;

  corpusRadialRequest = 0;

  corpusRadialLevel: 0 | 1 | 2 | 3 = 0;

  corpusRadialDetailFrame: number | null = null;

  corpusRadialFullSettleTimer: number | null = null;

  corpusRadialOverlayCanvas: HTMLCanvasElement | null = null;

  corpusRadialShowAll = false;

  thematicPayload: ThematicTopologyPayload | null = null;

  thematicLayoutWorker: Worker | null = null;

  thematicLayoutRequest = 0;

  thematicLayoutResult: ThematicLayoutResult | null = null;

  thematicOverlayCanvas: HTMLCanvasElement | null = null;

  thematicPatternsByNode = new Map<string, number[]>();

  thematicRelationLabelHits: Array<{
    node: string;
    patternIndex: number;
    x: number;
    y: number;
    halfWidth: number;
    halfHeight: number;
  }> = [];

  thematicHoveredRelationLabel: string | null = null;

  // The exact pattern under the cursor. One relation category can appear once
  // per direction between the same pair, so hovering a pill must light only
  // the line that pill belongs to.
  thematicHoveredPattern: number | null = null;

  thematicHoveredRelationPoint: { x: number; y: number } | null = null;

  /**
   * The one place the thematic graph's navigation state lives.
   *
   * Filter themes, Filtered tags, the canvas, and the inspector are all views
   * of this record: every interaction writes here and then calls
   * commitNavigation(), which re-renders all four from it. The named accessors
   * below keep the older field names working as aliases onto this object, so
   * there is exactly one copy of the state rather than one per surface.
   */
  nav: ThematicNavigationState = emptyNavigationState();

  get thematicSelectionCandidates(): number[] {
    return this.nav.candidates;
  }

  set thematicSelectionCandidates(value: number[]) {
    this.nav.candidates = value;
  }

  get thematicSelectedPatterns(): Set<number> {
    return this.nav.selectedPatterns;
  }

  set thematicSelectedPatterns(value: Set<number>) {
    this.nav.selectedPatterns = value;
  }

  get thematicPrimaryEntity(): string | null {
    return this.nav.primaryEntity;
  }

  set thematicPrimaryEntity(value: string | null) {
    this.nav.primaryEntity = value;
  }

  get thematicPairEntity(): string | null {
    return this.nav.pairEntity;
  }

  set thematicPairEntity(value: string | null) {
    this.nav.pairEntity = value;
  }

  get thematicContextRelation(): number | null {
    return this.nav.contextRelation;
  }

  set thematicContextRelation(value: number | null) {
    this.nav.contextRelation = value;
  }

  get thematicExactPatternFocus(): number | null {
    return this.nav.exactPatternFocus;
  }

  set thematicExactPatternFocus(value: number | null) {
    this.nav.exactPatternFocus = value;
  }

  thematicEvidencePattern: number | null = null;

  thematicInspectorFrame: number | null = null;

  constructor() {
    // Sigma emits a stage click when a pan gesture ends. Track pointer travel
    // independently so releasing a drag can never be interpreted as the
    // deliberate empty-space click used to move up one navigation layer.
    el.container.addEventListener(
      "pointerdown",
      (event) => {
        if (event.button !== 0) return;
        // A new press is a new intentional gesture; any unconsumed guard from
        // the preceding release must not suppress this interaction.
        this.suppressGraphClickUntil = 0;
        this.graphGesture = {
          active: true,
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          dragged: false,
        };
      },
      true,
    );
    el.container.addEventListener(
      "pointermove",
      (event) => {
        if (
          !this.graphGesture.active ||
          event.pointerId !== this.graphGesture.pointerId
        )
          return;
        if (
          Math.hypot(
            event.clientX - this.graphGesture.startX,
            event.clientY - this.graphGesture.startY,
          ) >= 6
        ) {
          this.graphGesture.dragged = true;
        }
      },
      true,
    );
    const finishGraphGesture = (event: PointerEvent): void => {
      if (
        !this.graphGesture.active ||
        event.pointerId !== this.graphGesture.pointerId
      )
        return;
      if (this.graphGesture.dragged) {
        // Cover both the DOM click and Sigma's synthetic clickStage callback,
        // which arrive just after pointerup.
        this.suppressGraphClickUntil = performance.now() + 300;
      }
      this.graphGesture.active = false;
      this.graphGesture.pointerId = -1;
    };
    el.container.addEventListener("pointerup", finishGraphGesture, true);
    el.container.addEventListener("pointercancel", finishGraphGesture, true);
    el.container.addEventListener(
      "click",
      (event) => {
        if (this.consumeGraphClickSuppression()) {
          event.preventDefault();
          event.stopImmediatePropagation();
          return;
        }
        const relationHit = this.thematicRelationLabelAtEvent(event);
        const thematicNode = this.thematicNodeAtEvent(event);
        if (thematicNode) {
          event.preventDefault();
          event.stopImmediatePropagation();
          if (relationHit && thematicNode === relationHit.node) {
            this.selectThematicRelationLabel(relationHit);
          } else {
            this.selectThematicNode(thematicNode, event.shiftKey);
          }
          return;
        }
        const edge = this.edgeLabelAtEvent(event);
        if (!edge) {
          if (this.thematicPayload) {
            event.preventDefault();
            event.stopImmediatePropagation();
            this.navigateBackOneLayer();
          }
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        this.selectEdge(edge);
      },
      true,
    );
    if (GRAPH_HOVER_EFFECTS_ENABLED) {
      el.container.addEventListener("mousemove", (event) => {
        const edge = this.edgeLabelAtEvent(event);
        if (edge === this.labelHoveredEdge) return;
        this.labelHoveredEdge = edge;
        el.container.classList.toggle("edge-label-hover", Boolean(edge));
        this.syncHoveredEdge();
      });
      el.container.addEventListener("mouseleave", () => {
        if (!this.labelHoveredEdge) return;
        this.labelHoveredEdge = null;
        el.container.classList.remove("edge-label-hover");
        this.syncHoveredEdge();
      });
    }
    el.container.addEventListener("mousemove", (event) => {
      this.updateThematicRelationHover(event);
    });
    el.container.addEventListener("mouseleave", () => {
      this.clearThematicRelationHover();
    });
  }
}

type renderer_lifecycleMethods = typeof renderer_lifecycle;
type page_navigationMethods = typeof page_navigation;
type graph_layoutMethods = typeof graph_layout;
type corpus_radial_modelMethods = typeof corpus_radial_model;
type thematic_modelMethods = typeof thematic_model;
type thematic_renderingMethods = typeof thematic_rendering;
type thematic_navigationMethods = typeof thematic_navigation;
type thematic_evidenceMethods = typeof thematic_evidence;
type cameraMethods = typeof camera;
type atlas_viewMethods = typeof atlas_view;
type category_filtersMethods = typeof category_filters;
type filtered_tagsMethods = typeof filtered_tags;
type corpus_radial_viewMethods = typeof corpus_radial_view;
type selection_inspectorMethods = typeof selection_inspector;
type similar_patternsMethods = typeof similar_patterns;
type workspaceMethods = typeof workspace;

export interface GraphController
  extends renderer_lifecycleMethods,
    page_navigationMethods,
    graph_layoutMethods,
    corpus_radial_modelMethods,
    thematic_modelMethods,
    thematic_renderingMethods,
    thematic_navigationMethods,
    thematic_evidenceMethods,
    cameraMethods,
    atlas_viewMethods,
    category_filtersMethods,
    filtered_tagsMethods,
    corpus_radial_viewMethods,
    selection_inspectorMethods,
    similar_patternsMethods,
    workspaceMethods {}

// Keep methods on the prototype with the same descriptors as class methods.
for (const feature of [
  renderer_lifecycle,
  page_navigation,
  graph_layout,
  corpus_radial_model,
  thematic_model,
  thematic_rendering,
  thematic_navigation,
  thematic_evidence,
  camera,
  atlas_view,
  category_filters,
  filtered_tags,
  corpus_radial_view,
  selection_inspector,
  similar_patterns,
  workspace,
]) {
  for (const [name, descriptor] of Object.entries(
    Object.getOwnPropertyDescriptors(feature),
  )) {
    if (Object.prototype.hasOwnProperty.call(GraphController.prototype, name)) {
      throw new Error(`Duplicate graph feature method: ${name}`);
    }
    Object.defineProperty(GraphController.prototype, name, {
      ...descriptor,
      enumerable: false,
    });
  }
}
