import { MultiDirectedGraph } from "graphology";
import FA2Layout from "graphology-layout-forceatlas2/worker";
import Sigma from "sigma";
import {
  EdgeLineProgram,
  NodePointProgram,
  createEdgeArrowProgram,
} from "sigma/rendering";
import { animateNodes } from "sigma/utils";
import {
  drawCollisionLabels,
  edgeLabelAtViewportPoint,
} from "./collision_labels";
import {
  CORPUS_BUBBLE_RADIUS,
  CORPUS_MIN_CENTER_DISTANCE,
  type CorpusRadialLayoutResult,
} from "./corpus_radial_layout";
import CorpusRadialWorker from "./corpus_radial_worker?worker";
import type {
  ThematicLayoutInput,
  ThematicLayoutResult,
} from "./thematic_layout";
import ThematicLayoutWorker from "./thematic_layout_worker?worker";

const GRAPH_HOVER_EFFECTS_ENABLED = false;

const LargeArrowProgram = createEdgeArrowProgram({
  lengthToThicknessRatio: 5.2,
  widenessToThicknessRatio: 4,
});

type NodeTuple = [
  id: string,
  label: string,
  frequency: number,
  x: number | null,
  y: number | null,
  componentIndex?: number,
  communityIndex?: number,
  priority?: number,
];
type EdgeTuple = [
  sourceIndex: number,
  targetIndex: number,
  relationId: string,
  label: string,
  occurrenceCount: number,
  bundleIndex?: number,
];
type ComponentTuple = [
  id: string,
  label: string,
  nodeCount: number,
  edgeCount: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  communityStart: number,
  communityCount: number,
];
type CommunityTuple = [
  componentIndex: number,
  label: string,
  nodeCount: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  centerX: number,
  centerY: number,
];
type BundleTuple = [
  sourceCommunity: number,
  targetCommunity: number,
  edgeCount: number,
  claimCount: number,
  dominantRelation: string,
  overview: boolean,
  sourceX: number,
  sourceY: number,
  targetX: number,
  targetY: number,
];

interface GraphFocus {
  kind: string;
  id: string;
  label: string;
  frequency?: number;
  depth?: number;
  sentenceId?: string;
  ordinal?: number;
  volumeId?: string;
  pageNumber?: number;
  startPage?: number;
  endPage?: number;
  tagKind?: "entity" | "relation";
  tagIds?: string[];
  tagLabels?: string[];
  subjectTagIds?: string[];
  relationTagIds?: string[];
  objectTagIds?: string[];
}

interface TopologyPayload {
  ok: boolean;
  schemaVersion: number;
  focus: GraphFocus;
  layout: {
    kind:
      | "triples"
      | "force"
      | "radial"
      | "bipartite"
      | "claim"
      | "thematic"
      | "atlas";
    scope?: string;
    bounds?: [number, number, number, number];
    zoomStops?: [number, number, number];
    medianSpacing?: number;
  };
  nodes: NodeTuple[];
  edges: EdgeTuple[];
  components?: ComponentTuple[];
  communities?: CommunityTuple[];
  bundles?: BundleTuple[];
  claimCount: number;
  entityMentionCount?: number;
  availableCount: number | null;
  truncated: boolean;
  limit: number | null;
  categoryMode?: boolean;
  filters?: {
    entities: string[];
    relations: string[];
    rawTagKind?: "entity" | "relation" | null;
    rawTagIds?: string[];
    subjects?: string[];
    objects?: string[];
    rawSubjectTagIds?: string[];
    rawRelationTagIds?: string[];
    rawObjectTagIds?: string[];
  };
}

interface CategoryDescriptor {
  id: string;
  tagId: string;
  label: string;
  definition: string;
  count: number;
  provisional: boolean;
  justification?: string;
  sourcePage?: string;
}

interface ThematicEntityCategory extends CategoryDescriptor {
  mentionCount: number;
  activeMentionCount: number;
  counterpartCount: number;
  activeCounterpartCount: number;
  relationCount: number;
  activeRelationCount: number;
  patternCount: number;
  activePatternCount: number;
}

interface ThematicRelationCategory extends CategoryDescriptor {
  tripleCount: number;
  activeTripleCount: number;
  entityCount: number;
  activeEntityCount: number;
  patternCount: number;
  activePatternCount: number;
}

type ThematicPatternTuple = [
  sourceEntityIndex: number,
  relationIndex: number,
  targetEntityIndex: number,
  supportingTripleCount: number,
  activeTripleCount: number,
];

type ThematicBundleTuple = [
  entityIndex: number,
  relationIndex: number,
  role: 0 | 1,
  supportingTripleCount: number,
  activeTripleCount: number,
  childPatternIndices: number[],
];

interface ThematicTopologyPayload extends TopologyPayload {
  entityCategories: ThematicEntityCategory[];
  relationCategories: ThematicRelationCategory[];
  patterns: ThematicPatternTuple[];
  incidenceBundles: ThematicBundleTuple[];
}

interface ThematicLayoutWorkerResponse {
  kind: "layout";
  requestId: number;
  result?: ThematicLayoutResult;
  error?: string;
}

interface CategoryCatalogPayload {
  ok: boolean;
  entities: CategoryDescriptor[];
  relations: CategoryDescriptor[];
}

interface EvidenceItem {
  claimId: string;
  sentenceId: string;
  sentenceMy: string;
  sentenceEn: string;
  volumeId: string;
  ownerPage: number;
  ordinal: number;
  subject?: string;
  predicate?: string;
  object?: string;
  relationCategory?: CategoryDescriptor;
}

interface EvidencePayload {
  ok: boolean;
  items: EvidenceItem[];
  hasMore: boolean;
  nextOffset: number | null;
}

type FilteredTagRole = "subject" | "relation" | "object";

interface FilteredTagResult {
  id: string | null;
  kind: "entity" | "relation";
  role: FilteredTagRole;
  label: string;
  frequency: number;
  similarity: number | null;
}

interface SimilarPatternResult {
  subjectId: string;
  relationId: string;
  objectId: string;
  subjectLabel: string;
  relationLabel: string;
  objectLabel: string;
  tripleCount: number;
  similarity: number;
}

interface SimilarPatternsPayload {
  ok: boolean;
  minimumSimilarity: number;
  comparedRoles: string[];
  narrowedBy?: Partial<Record<FilteredTagRole, string[]>>;
  anchor: {
    claimCount: number;
    patternCount: number;
    subjects: string[];
    relations: string[];
    objects: string[];
  };
  totalCandidates: number;
  items: SimilarPatternResult[];
}

interface FilteredTagsPayload {
  ok: boolean;
  role: FilteredTagRole;
  query: string;
  items: FilteredTagResult[];
  categoryScoped: boolean;
  narrowedBy: Partial<Record<FilteredTagRole, string[]>>;
  similarTo: string[];
  minimumSimilarity: number | null;
  matchingClaimCount: number;
  pagination: {
    offset: number;
    limit: number;
    returned: number;
    total: number;
    hasMore: boolean;
    nextOffset: number | null;
  };
  filters: {
    subjects: string[];
    relations: string[];
    objects: string[];
  };
}

interface OpenOptions {
  volumeId?: string;
  pageNumber?: number;
  categories?: { entities?: string[]; relations?: string[] } | null;
}

interface ChronicleVolume {
  id: string;
  label: string;
  availablePages: number[];
}

interface ChronicleIndex {
  volumes: ChronicleVolume[];
}

type ThematicDirection = "either" | "ab" | "ba";

/** The complete navigation state of the thematic graph. */
interface ThematicNavigationState {
  selectedNode: string | null;
  selectedEdge: string | null;
  primaryEntity: string | null;
  pairEntity: string | null;
  contextRelation: number | null;
  exactPatternFocus: number | null;
  candidates: number[];
  selectedPatterns: Set<number>;
  groupSelection: boolean;
  orientationAnchors: Set<number>;
  /** Category IDs shown in the three theme lists for this selection. */
  categories: { subjects: string[]; relations: string[]; objects: string[] };
}

function emptyNavigationState(): ThematicNavigationState {
  return {
    selectedNode: null,
    selectedEdge: null,
    primaryEntity: null,
    pairEntity: null,
    contextRelation: null,
    exactPatternFocus: null,
    candidates: [],
    selectedPatterns: new Set<number>(),
    groupSelection: false,
    orientationAnchors: new Set<number>(),
    categories: { subjects: [], relations: [], objects: [] },
  };
}

function cloneNavigationState(
  state: ThematicNavigationState,
): ThematicNavigationState {
  return {
    ...state,
    candidates: [...state.candidates],
    selectedPatterns: new Set(state.selectedPatterns),
    orientationAnchors: new Set(state.orientationAnchors),
    categories: {
      subjects: [...state.categories.subjects],
      relations: [...state.categories.relations],
      objects: [...state.categories.objects],
    },
  };
}

interface GraphNavigationSnapshot {
  nav: ThematicNavigationState;
  selectedNeighbors: string[];
  rawPrimaryEntity: string | null;
  rawPairEntity: string | null;
  rawContextRelation: string | null;
  rawSelectedEdges: string[];
}

interface CorpusRadialModel {
  payload: TopologyPayload;
  layout: CorpusRadialLayoutResult;
  rankPosition: Int32Array;
  levelFlags: [Uint8Array, Uint8Array, Uint8Array];
  levelAdditions: [number[], number[], number[]];
  admitted: Set<number>;
  pixelsPerUnit: number;
  projectedRadius: number;
  lastDisclosureMs: number;
  lastCenterX: number;
  lastCenterY: number;
}

interface CorpusRadialWorkerResponse {
  kind: "layout";
  requestId: number;
  result?: CorpusRadialLayoutResult;
  error?: string;
}

declare global {
  interface Window {
    ChronicleGraph: {
      open(options: OpenOptions): Promise<void>;
      close(): void;
      setPage(volumeId: string, pageNumber: number): void;
      isOpen(): boolean;
    };
  }
}

const byId = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing graph UI element: ${id}`);
  return element as T;
};

function ensureCorpusShowAllControl(): {
  control: HTMLLabelElement;
  input: HTMLInputElement;
} {
  const existingControl = document.getElementById(
    "graph-show-all-control",
  ) as HTMLLabelElement | null;
  const existingInput = document.getElementById(
    "graph-show-all",
  ) as HTMLInputElement | null;
  if (existingControl && existingInput) {
    return { control: existingControl, input: existingInput };
  }
  const control = document.createElement("label");
  control.id = "graph-show-all-control";
  control.className = "graph-show-all-control";
  control.hidden = true;
  const input = document.createElement("input");
  input.id = "graph-show-all";
  input.type = "checkbox";
  const label = document.createElement("span");
  label.textContent = "Show entire graph";
  control.append(input, label);
  byId<HTMLButtonElement>("graph-fit").insertAdjacentElement(
    "afterend",
    control,
  );
  return { control, input };
}

const corpusShowAllControl = ensureCorpusShowAllControl();

const el = {
  workspace: byId<HTMLElement>("graph-workspace"),
  reader: byId<HTMLElement>("workspace"),
  close: byId<HTMLButtonElement>("graph-close"),
  scope: byId<HTMLSelectElement>("graph-scope"),
  zoomOut: byId<HTMLButtonElement>("graph-zoom-out"),
  zoomIn: byId<HTMLButtonElement>("graph-zoom-in"),
  fit: byId<HTMLButtonElement>("graph-fit"),
  showAllControl: corpusShowAllControl.control,
  showAll: corpusShowAllControl.input,
  pagesPanel: byId<HTMLDetailsElement>("graph-pages-panel"),
  pageVolume: byId<HTMLSelectElement>("graph-page-volume"),
  pageNumber: byId<HTMLSelectElement>("graph-page-number"),
  pagePrev: byId<HTMLButtonElement>("graph-page-prev"),
  pageNext: byId<HTMLButtonElement>("graph-page-next"),
  pageLoad: byId<HTMLButtonElement>("graph-page-load"),
  rangeStart: byId<HTMLSelectElement>("graph-range-start"),
  rangeEnd: byId<HTMLSelectElement>("graph-range-end"),
  rangeLoad: byId<HTMLButtonElement>("graph-range-load"),
  pageStatus: byId<HTMLElement>("graph-page-status"),
  heading: byId<HTMLElement>("graph-heading"),
  counts: byId<HTMLElement>("graph-counts"),
  browserPanel: byId<HTMLDetailsElement>("graph-browser-panel"),
  browserSummary: byId<HTMLElement>("graph-browser-summary"),
  filteredTagsPanel: byId<HTMLDetailsElement>("graph-filtered-tags-panel"),
  themeControls: byId<HTMLElement>("graph-theme-controls"),
  entityCategories: byId<HTMLElement>("graph-entity-categories"),
  objectCategories: byId<HTMLElement>("graph-object-categories"),
  relationCategories: byId<HTMLElement>("graph-relation-categories"),
  themeDirection: byId<HTMLSelectElement>("graph-theme-direction"),
  themeGranularity: byId<HTMLInputElement>("graph-theme-granularity"),
  themeGranularityValue: byId<HTMLOutputElement>(
    "graph-theme-granularity-value",
  ),
  themeApply: byId<HTMLButtonElement>("graph-theme-apply"),
  themeClear: byId<HTMLButtonElement>("graph-theme-clear"),
  themeSimilarity: byId<HTMLInputElement>("graph-theme-similarity"),
  themeSimilarityValue: byId<HTMLOutputElement>("graph-theme-similarity-value"),
  themeSimilar: byId<HTMLButtonElement>("graph-theme-similar"),
  themeSimilarClear: byId<HTMLButtonElement>("graph-theme-similar-clear"),
  themeSimilarAnchor: byId<HTMLElement>("graph-theme-similar-anchor"),
  themeSimilarNote: byId<HTMLElement>("graph-theme-similar-note"),
  themeSimilarResults: byId<HTMLElement>("graph-theme-similar-results"),
  themeSimilarApply: byId<HTMLButtonElement>("graph-theme-similar-apply"),
  filteredTagsRole: byId<HTMLSelectElement>("graph-filtered-tags-role"),
  filteredTagsSearch: byId<HTMLInputElement>("graph-filtered-tags-search"),
  filteredTagsSearchButton: byId<HTMLButtonElement>(
    "graph-filtered-tags-search-button",
  ),
  filteredTagsNote: byId<HTMLElement>("graph-filtered-tags-note"),
  filteredTagsResults: byId<HTMLElement>("graph-filtered-tags-results"),
  filteredTagsSelectPage: byId<HTMLButtonElement>(
    "graph-filtered-tags-select-page",
  ),
  filteredTagsClear: byId<HTMLButtonElement>("graph-filtered-tags-clear"),
  filteredTagsPrev: byId<HTMLButtonElement>("graph-filtered-tags-prev"),
  filteredTagsNext: byId<HTMLButtonElement>("graph-filtered-tags-next"),
  filteredTagsPage: byId<HTMLElement>("graph-filtered-tags-page"),
  filteredTagsApply: byId<HTMLButtonElement>("graph-filtered-tags-apply"),
  filteredTagsRelease: byId<HTMLButtonElement>("graph-filtered-tags-release"),
  filteredTagsSelectionNote: byId<HTMLElement>(
    "graph-filtered-tags-selection-note",
  ),
  filteredTagsSimilarity: byId<HTMLInputElement>(
    "graph-filtered-tags-similarity",
  ),
  filteredTagsSimilarityValue: byId<HTMLOutputElement>(
    "graph-filtered-tags-similarity-value",
  ),
  filteredTagsSimilar: byId<HTMLButtonElement>("graph-filtered-tags-similar"),
  filteredTagsSimilarClear: byId<HTMLButtonElement>(
    "graph-filtered-tags-similar-clear",
  ),
  filteredTagsSimilarNote: byId<HTMLElement>(
    "graph-filtered-tags-similar-note",
  ),
  nodeLegend: byId<HTMLElement>("graph-node-legend"),
  edgeLegend: byId<HTMLElement>("graph-edge-legend"),
  container: byId<HTMLDivElement>("graph-canvas"),
  loading: byId<HTMLElement>("graph-loading"),
  detail: byId<HTMLElement>("graph-detail-content"),
  detailPanel: byId<HTMLDetailsElement>("graph-detail-panel"),
};

function clear(element: Element): void {
  element.replaceChildren();
}

function textElement(
  tag: keyof HTMLElementTagNameMap,
  className: string,
  text: string,
): HTMLElement {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function actionButton(
  label: string,
  action: string,
  value: string,
  className = "",
): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.dataset.graphAction = action;
  button.dataset.value = value;
  button.className = className;
  return button;
}

async function fetchJson<T>(
  url: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(url, {
    signal,
    headers: { Accept: "application/json" },
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.error || `Graph request failed (${response.status})`);
  }
  return payload as T;
}

function nodeSize(frequency: number): number {
  return Math.min(11, 2.2 + Math.log2(frequency + 1) * 0.62);
}

function bubbleNodeSize(label: string): number {
  const charactersPerLine = 10;
  const lines = Math.ceil(label.length / charactersPerLine);
  const textWidth = Math.min(charactersPerLine, label.length) * 4.3;
  const textHeight = lines * 8.5;
  return Math.min(
    36,
    Math.max(23, Math.ceil(Math.hypot(textWidth / 2, textHeight / 2) + 6)),
  );
}

const atlasPalette = [
  "#226f8a",
  "#0f8978",
  "#6f4a96",
  "#9a6534",
  "#376c51",
  "#8a4967",
  "#4f6fa5",
  "#75712b",
];

function atlasNodeColor(communityIndex: number): string {
  const mixed = Math.imul(communityIndex + 1, 2654435761) >>> 0;
  return atlasPalette[mixed % atlasPalette.length];
}

function sameSet(left: Set<string>, right: Set<string>): boolean {
  if (left.size !== right.size) return false;
  for (const value of left) {
    if (!right.has(value)) return false;
  }
  return true;
}

class GraphController {
  private graph: MultiDirectedGraph | null = null;
  private renderer: Sigma | null = null;
  private layout: FA2Layout | null = null;
  private layoutTimer: number | null = null;
  private graphAbort: AbortController | null = null;
  private evidenceAbort: AbortController | null = null;
  private payload: TopologyPayload | null = null;
  private page = { volumeId: "vol1", pageNumber: 47 };
  private pageVolumes = new Map<string, ChronicleVolume>();
  private pageIndexPromise: Promise<void> | null = null;
  private categoryCatalogLoaded = false;
  private categoryCatalogPromise: Promise<void> | null = null;
  private get selectedNode(): string | null {
    return this.nav.selectedNode;
  }
  private set selectedNode(value: string | null) {
    this.nav.selectedNode = value;
  }
  private get selectedEdge(): string | null {
    return this.nav.selectedEdge;
  }
  private set selectedEdge(value: string | null) {
    this.nav.selectedEdge = value;
  }
  private rawPrimaryEntity: string | null = null;
  private rawPairEntity: string | null = null;
  private rawContextRelation: string | null = null;
  private rawSelectedEdges = new Set<string>();
  private navigationHistory: GraphNavigationSnapshot[] = [];
  private graphGesture: {
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
  private suppressGraphClickUntil = 0;
  private hoveredNode: string | null = null;
  private hoveredEdge: string | null = null;
  private lineHoveredEdge: string | null = null;
  private labelHoveredEdge: string | null = null;
  private selectedNeighbors = new Set<string>();
  private filteredTagsAbort: AbortController | null = null;
  private filteredTagsItems: FilteredTagResult[] = [];
  private filteredTagsOffset = 0;
  private filteredTagsTotal = 0;
  private filteredTagsHasMore = false;
  private filteredTagsDebounce: number | null = null;
  private filteredTagsSyncTimer: number | null = null;
  private filteredTagsBucketKey = "";
  private filteredTagsSelected = new Map<FilteredTagRole, Set<string>>([
    ["subject", new Set<string>()],
    ["relation", new Set<string>()],
    ["object", new Set<string>()],
  ]);
  // Anchor tag IDs whose averaged vector currently orders each bucket. Empty
  // means the bucket is back to plain frequency ordering.
  private filteredTagsSimilarAnchors = new Map<FilteredTagRole, string[]>([
    ["subject", []],
    ["relation", []],
    ["object", []],
  ]);
  // The low-level tag filter currently applied to the graph. Distinct from
  // filteredTagsSelected, which is what is merely checked but not yet applied.
  private appliedTagFilters = new Map<FilteredTagRole, string[]>([
    ["subject", []],
    ["relation", []],
    ["object", []],
  ]);
  // True while the selection came from the entity-group controls, whose
  // candidates already encode the pairing direction.
  private get thematicGroupSelection(): boolean {
    return this.nav.groupSelection;
  }
  private set thematicGroupSelection(value: boolean) {
    this.nav.groupSelection = value;
  }
  // Entity category indices that orient the edge colouring: a pattern with one
  // of these on its subject side is drawn as outgoing, on its object side as
  // incoming. Group A when it has anything in it, otherwise group B, otherwise
  // the single clicked entity.
  private get thematicOrientationAnchors(): Set<number> {
    return this.nav.orientationAnchors;
  }
  private set thematicOrientationAnchors(value: Set<number>) {
    this.nav.orientationAnchors = value;
  }
  private similarPatternsAbort: AbortController | null = null;
  private similarPatterns: SimilarPatternResult[] = [];
  private similarPatternsSelected = new Set<string>();
  private evidenceItems: EvidenceItem[] = [];
  private evidenceHasMore = false;
  private evidenceNextOffset: number | null = null;
  private evidenceLoading = false;
  private fullyLabeledNodes = new Set<string>();
  private edgeLabeledNodes = new Set<string>();
  private progressiveDetailFrame: number | null = null;
  private readableLayoutPending = false;
  private readableLayoutCenter = { x: 0, y: 0 };
  private labelCanvas: HTMLCanvasElement | null = null;
  private fixedBBox: {
    x: [number, number];
    y: [number, number];
  } | null = null;
  private graphBBox: {
    x: [number, number];
    y: [number, number];
  } | null = null;
  private atlasLevel: 0 | 1 | 2 | 3 = 0;
  private atlasDetailFrame: number | null = null;
  private atlasSettleTimer: number | null = null;
  private atlasOverlayCanvas: HTMLCanvasElement | null = null;
  private atlasOverviewLabels = new Set<string>();
  private atlasVisibleNodes = new Set<string>();
  private atlasContinuityEdges = new Set<string>();
  private atlasPredicateEdges = new Set<string>();
  private atlasViewportLabels = new Set<string>();
  private atlasViewportPredicateEdges = new Set<string>();
  private atlasPredicateLabelPositions = new Map<
    string,
    { x: number; y: number; width: number; label: string }
  >();
  private atlasViewportRegionalEdges = new Set<string>();
  private atlasScopeAnimation: (() => void) | null = null;
  private atlasProjectedSpacing = 0;
  private corpusRadialModel: CorpusRadialModel | null = null;
  private corpusRadialWorker: Worker | null = null;
  private corpusRadialRequest = 0;
  private corpusRadialLevel: 0 | 1 | 2 | 3 = 0;
  private corpusRadialDetailFrame: number | null = null;
  private corpusRadialFullSettleTimer: number | null = null;
  private corpusRadialOverlayCanvas: HTMLCanvasElement | null = null;
  private corpusRadialShowAll = false;
  private thematicPayload: ThematicTopologyPayload | null = null;
  private thematicLayoutWorker: Worker | null = null;
  private thematicLayoutRequest = 0;
  private thematicLayoutResult: ThematicLayoutResult | null = null;
  private thematicOverlayCanvas: HTMLCanvasElement | null = null;
  private thematicPatternsByNode = new Map<string, number[]>();
  private thematicRelationLabelHits: Array<{
    node: string;
    patternIndex: number;
    x: number;
    y: number;
    halfWidth: number;
    halfHeight: number;
  }> = [];
  private thematicHoveredRelationLabel: string | null = null;
  // The exact pattern under the cursor. One relation category can appear once
  // per direction between the same pair, so hovering a pill must light only
  // the line that pill belongs to.
  private thematicHoveredPattern: number | null = null;
  private thematicHoveredRelationPoint: { x: number; y: number } | null = null;
  /**
   * The one place the thematic graph's navigation state lives.
   *
   * Filter themes, Filtered tags, the canvas, and the inspector are all views
   * of this record: every interaction writes here and then calls
   * commitNavigation(), which re-renders all four from it. The named accessors
   * below keep the older field names working as aliases onto this object, so
   * there is exactly one copy of the state rather than one per surface.
   */
  private nav: ThematicNavigationState = emptyNavigationState();

  private get thematicSelectionCandidates(): number[] {
    return this.nav.candidates;
  }
  private set thematicSelectionCandidates(value: number[]) {
    this.nav.candidates = value;
  }
  private get thematicSelectedPatterns(): Set<number> {
    return this.nav.selectedPatterns;
  }
  private set thematicSelectedPatterns(value: Set<number>) {
    this.nav.selectedPatterns = value;
  }
  private get thematicPrimaryEntity(): string | null {
    return this.nav.primaryEntity;
  }
  private set thematicPrimaryEntity(value: string | null) {
    this.nav.primaryEntity = value;
  }
  private get thematicPairEntity(): string | null {
    return this.nav.pairEntity;
  }
  private set thematicPairEntity(value: string | null) {
    this.nav.pairEntity = value;
  }
  private get thematicContextRelation(): number | null {
    return this.nav.contextRelation;
  }
  private set thematicContextRelation(value: number | null) {
    this.nav.contextRelation = value;
  }
  private get thematicExactPatternFocus(): number | null {
    return this.nav.exactPatternFocus;
  }
  private set thematicExactPatternFocus(value: number | null) {
    this.nav.exactPatternFocus = value;
  }
  private thematicEvidencePattern: number | null = null;
  private thematicInspectorFrame: number | null = null;

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
        ) return;
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
      ) return;
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

  private edgeLabelAtEvent(event: MouseEvent): string | null {
    if (!this.labelCanvas) return null;
    const bounds = el.container.getBoundingClientRect();
    return edgeLabelAtViewportPoint(
      this.labelCanvas,
      event.clientX - bounds.left,
      event.clientY - bounds.top,
    );
  }

  private syncHoveredEdge(): void {
    const edge = this.labelHoveredEdge || this.lineHoveredEdge;
    if (edge === this.hoveredEdge) return;
    this.hoveredEdge = edge;
    if (this.payload?.layout.kind === "atlas") {
      this.renderer?.scheduleRefresh();
    } else {
      this.renderer?.refresh();
    }
  }

  setPage(volumeId: string, pageNumber: number): void {
    this.page = { volumeId, pageNumber: Number(pageNumber) };
    this.updatePageScopeLabel();
    if (this.pageVolumes.size) {
      this.populatePageSelectors(volumeId, Number(pageNumber));
    }
  }

  private updatePageScopeLabel(): void {
    const { volumeId, pageNumber } = this.page;
    const pageOption = el.scope.querySelector<HTMLOptionElement>(
      'option[value="page"]',
    );
    if (pageOption) {
      pageOption.textContent =
        `Current page · ${volumeId.toUpperCase()} ${pageNumber}`;
    }
  }

  private async ensurePageIndex(): Promise<void> {
    if (this.pageVolumes.size) return;
    if (this.pageIndexPromise) return this.pageIndexPromise;
    this.pageIndexPromise = (async () => {
      const response = await fetch("/api/chronicles/index", {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`Page index request failed (${response.status})`);
      }
      const index = (await response.json()) as ChronicleIndex;
      this.pageVolumes = new Map(
        index.volumes.map((volume) => [
          volume.id,
          {
            ...volume,
            availablePages: volume.availablePages.map(Number),
          },
        ]),
      );
      this.populatePageSelectors(this.page.volumeId, this.page.pageNumber);
    })();
    try {
      await this.pageIndexPromise;
    } finally {
      this.pageIndexPromise = null;
    }
  }

  private populatePageSelect(
    select: HTMLSelectElement,
    pages: number[],
    selectedPage: number,
  ): void {
    select.replaceChildren(
      ...pages.map((page) => {
        const option = document.createElement("option");
        option.value = String(page);
        option.textContent = String(page);
        return option;
      }),
    );
    select.value = String(selectedPage);
  }

  private populatePageSelectors(
    volumeId: string,
    preferredPage?: number,
  ): void {
    const volume = this.pageVolumes.get(volumeId);
    if (!volume || !volume.availablePages.length) return;
    const pages = volume.availablePages;
    const target = Number(preferredPage || pages[0]);
    const selectedPage = pages.includes(target)
      ? target
      : pages.reduce((closest, page) =>
          Math.abs(page - target) < Math.abs(closest - target)
            ? page
            : closest,
        );
    el.pageVolume.value = volumeId;
    this.populatePageSelect(el.pageNumber, pages, selectedPage);
    this.populatePageSelect(el.rangeStart, pages, selectedPage);
    this.populatePageSelect(el.rangeEnd, pages, selectedPage);
    el.pageStatus.textContent =
      `${volume.label} has ${pages.length.toLocaleString()} available pages.`;
    this.updatePageNavigation();
  }

  private updatePageNavigation(): void {
    const pages =
      this.pageVolumes.get(el.pageVolume.value)?.availablePages || [];
    const index = pages.indexOf(Number(el.pageNumber.value));
    el.pagePrev.disabled = index <= 0;
    el.pageNext.disabled = index < 0 || index >= pages.length - 1;
  }

  private normalizeRange(changed: "start" | "end"): void {
    const start = Number(el.rangeStart.value);
    const end = Number(el.rangeEnd.value);
    if (start <= end) return;
    if (changed === "start") el.rangeEnd.value = String(start);
    else el.rangeStart.value = String(end);
  }

  handlePageVolumeChange(): void {
    this.populatePageSelectors(el.pageVolume.value);
  }

  handlePageNumberChange(): void {
    this.updatePageNavigation();
  }

  handleRangeChange(changed: "start" | "end"): void {
    this.normalizeRange(changed);
  }

  async loadSelectedPage(): Promise<void> {
    const volumeId = el.pageVolume.value;
    const pageNumber = Number(el.pageNumber.value);
    if (!pageNumber) return;
    this.setPage(volumeId, pageNumber);
    el.scope.value = "page";
    el.pagesPanel.open = false;
    await this.loadScope();
  }

  async moveSelectedPage(direction: -1 | 1): Promise<void> {
    const pages =
      this.pageVolumes.get(el.pageVolume.value)?.availablePages || [];
    const index = pages.indexOf(Number(el.pageNumber.value));
    const nextPage = pages[index + direction];
    if (nextPage === undefined) return;
    el.pageNumber.value = String(nextPage);
    this.updatePageNavigation();
    await this.loadSelectedPage();
  }

  async loadSelectedRange(): Promise<void> {
    const volumeId = el.pageVolume.value;
    const startPage = Number(el.rangeStart.value);
    const endPage = Number(el.rangeEnd.value);
    if (!startPage || !endPage || startPage > endPage) return;
    const selectedPages =
      this.pageVolumes
        .get(volumeId)
        ?.availablePages.filter(
          (page) => page >= startPage && page <= endPage,
        ).length || 0;
    this.page = { volumeId, pageNumber: startPage };
    this.updatePageScopeLabel();
    const rangeOption = el.scope.querySelector<HTMLOptionElement>(
      'option[value="range"]',
    );
    if (rangeOption) {
      rangeOption.textContent =
        `${volumeId.toUpperCase()} ${startPage}–${endPage}`;
    }
    el.pageStatus.textContent =
      `Loading all triples from ${selectedPages.toLocaleString()} available pages…`;
    el.scope.value = "range";
    el.pagesPanel.open = false;
    const path =
      `/api/graph/categories/topology/range/${volumeId}/${startPage}/${endPage}`;
    await this.loadGraph(path);
    if (this.activeFilteredTagRoles().length) {
      await this.refreshFilteredTags(true, 0);
    }
    el.pageStatus.textContent =
      `${selectedPages.toLocaleString()} available pages selected.`;
  }

  private stopLayout(): void {
    if (this.layoutTimer !== null) {
      window.clearTimeout(this.layoutTimer);
      this.layoutTimer = null;
    }
    if (this.layout) {
      this.layout.kill();
      this.layout = null;
    }
  }

  private destroyRenderer(): void {
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
  }

  private setLoading(loading: boolean, message = "Loading graph…"): void {
    el.loading.textContent = message;
    el.loading.hidden = !loading;
  }

  private thematicNodeAtEvent(event: MouseEvent): string | null {
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
  }

  private thematicRelationLabelAtPoint(
    x: number,
    y: number,
  ): (typeof this.thematicRelationLabelHits)[number] | null {
    return (
      this.thematicRelationLabelHits.find(
        (hit) =>
          Math.abs(hit.x - x) <= hit.halfWidth + 3 &&
          Math.abs(hit.y - y) <= hit.halfHeight + 3,
      ) || null
    );
  }

  private thematicRelationLabelAtEvent(
    event: MouseEvent,
  ): (typeof this.thematicRelationLabelHits)[number] | null {
    if (!this.thematicPayload) return null;
    const bounds = el.container.getBoundingClientRect();
    return this.thematicRelationLabelAtPoint(
      event.clientX - bounds.left,
      event.clientY - bounds.top,
    );
  }

  private updateThematicRelationHover(event: MouseEvent): void {
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
    ) return;
    this.thematicHoveredRelationLabel = next;
    this.thematicHoveredPattern = hit ? hit.patternIndex : null;
    this.thematicHoveredRelationPoint = hit
      ? { x: hit.x, y: hit.y }
      : null;
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
  }

  private clearThematicRelationHover(): void {
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
  }

  private graphUrlForScope(): string {
    if (el.scope.value === "page") {
      return `/api/graph/categories/topology/page/${this.page.volumeId}/${this.page.pageNumber}`;
    }
    return `/api/graph/categories/topology/overview/${el.scope.value}`;
  }

  private thematicBaseUrl(): string {
    const scope = this.thematicPayload?.layout.scope || "corpus";
    const focus = this.thematicPayload?.focus;
    if (scope === "page") {
      const volumeId = focus?.volumeId || this.page.volumeId;
      const pageNumber = focus?.pageNumber || this.page.pageNumber;
      return `/api/graph/categories/topology/page/${volumeId}/${pageNumber}`;
    }
    if (scope === "range") {
      const volumeId = focus?.volumeId || el.pageVolume.value;
      const startPage = focus?.startPage || Number(el.rangeStart.value);
      const endPage = focus?.endPage || Number(el.rangeEnd.value);
      return `/api/graph/categories/topology/range/${volumeId}/${startPage}/${endPage}`;
    }
    return `/api/graph/categories/topology/overview/${scope}`;
  }

  async loadScope(): Promise<void> {
    if (el.scope.value === "focus" || el.scope.value === "range") return;
    await this.loadGraph(this.graphUrlForScope());
    if (this.activeFilteredTagRoles().length) {
      await this.refreshFilteredTags(true, 0);
    }
  }

  /** Resolves true only when this request's payload was actually installed. */
  async loadGraph(url: string): Promise<boolean> {
    this.graphAbort?.abort();
    this.evidenceAbort?.abort();
    this.graphAbort = new AbortController();
    const request = this.graphAbort;
    this.stopLayout();
    this.setLoading(true);
    try {
      const payload = await fetchJson<TopologyPayload>(url, request.signal);
      if (request !== this.graphAbort) return false;
      await this.installPayload(payload);
      this.scheduleFilteredTagsSync();
      return true;
    } catch (error) {
      if (request.signal.aborted) return false;
      clear(el.detail);
      el.detail.appendChild(
        textElement(
          "div",
          "graph-empty",
          error instanceof Error ? error.message : String(error),
        ),
      );
      return false;
    } finally {
      if (request === this.graphAbort && !this.readableLayoutPending) {
        this.setLoading(false);
      }
    }
  }

  private assignLayout(
    graph: MultiDirectedGraph,
    payload: TopologyPayload,
  ): void {
    const kind = payload.layout.kind;

    if (kind === "atlas") return;

    if (kind === "claim") {
      const edge = payload.edges[0];
      if (!edge) return;
      graph.mergeNodeAttributes(payload.nodes[edge[0]][0], {
        x: -1,
        y: 0,
        layoutParent: "",
        layoutDepth: 0,
      });
      graph.mergeNodeAttributes(payload.nodes[edge[1]][0], {
        x: 1,
        y: 0,
        layoutParent: payload.nodes[edge[0]][0],
        layoutDepth: 1,
      });
      return;
    }

    if (kind === "triples") {
      // Constructive radial layout. Nothing here repairs overlap after the
      // fact: the geometry is sized so that the overlaps we care about
      // cannot occur, and a single exact verification confirms it.
      //
      //  1. Every subtree owns a disjoint angular sector, centred on its
      //     parent's outward direction and never wider than OUTWARD_SPAN.
      //     A sector below 180 degrees is convex, so a straight edge drawn
      //     between two points inside one never leaves it.
      //  2. Every node sits exactly on a ring, and every ring gap is wider
      //     than the diameters it separates, so no node ever lies in the
      //     open space between two rings.
      //  3. An edge therefore spans exactly one ring gap inside one sector:
      //     the only nodes it can come near are its own endpoints and its
      //     siblings.
      //  4. A relation tag is budgeted into both its sector width and its
      //     ring gap, and every edge is long enough to hold a centred tag
      //     with bare line and an arrowhead beyond each end of it.
      //
      // Angular widths use asin(radius / ringRadius) rather than the
      // arc-length approximation, which makes the spacing exact instead of
      // merely close: for half-angles a and b,
      //     sin a + sin b = 2 sin((a+b)/2) cos((a-b)/2) <= 2 sin((a+b)/2)
      // so the chord between two centres placed a + b apart is never
      // shorter than the sum of the two radii.
      //
      // The construction above leaves a short tail of cases it does not
      // prove on its own -- chiefly the handful of non-tree edges in a
      // cyclic component, which no sector argument can contain. Those are
      // caught by verifyLayout() and answered by scaling the component up.
      // Scaling is monotone (distances grow, radii and tags do not), so the
      // solve terminates, and in practice it lands on the first or second
      // attempt rather than needing the scale at all.

      const NODE_CLEARANCE = 7; // bare space between two node discs
      const TAG_CLEARANCE = 7; // bare space around a relation tag
      const TAG_ENDPOINT_CLEARANCE = 17;
      // drawCollisionLabels flags an edge as touching a node at radius + 8,
      // so the layout has to leave more than that or the audit disagrees
      // with the geometry that produced it.
      const EDGE_NODE_CLEARANCE = 11;
      const TAG_STANDOFF = 24; // tag to endpoint rim: arrowhead plus bare line
      const OUTWARD_SPAN = (Math.PI * 5) / 9; // 100deg: children fan outward
      const COMPONENT_GAP = 28;
      const MAX_SOLVE_STEPS = 24;
      const TAG_HEIGHT = 15; // font size 8 plus the 7px padding drawEdgeLabel adds

      const nodeIds = payload.nodes.map((node) => node[0]);
      const nodeById = new Map(payload.nodes.map((node) => [node[0], node]));
      const measurementCanvas = document.createElement("canvas");
      const measurementContext = measurementCanvas.getContext("2d");
      if (measurementContext) {
        measurementContext.font = "600 8px Inter, Segoe UI, sans-serif";
      }
      // Must stay in step with ensurePlacements() in collision_labels.ts.
      const tagWidth = (label: string) =>
        (measurementContext
          ? measurementContext.measureText(label).width
          : label.length * 4.6) + 8;

      const adjacency = new Map(
        nodeIds.map((id) => [id, new Set<string>()]),
      );
      const sources = new Set<string>();
      const targets = new Set<string>();
      // A self-loop has no length to hang its tag on, so it is kept out of
      // the tree entirely and given an anchor of its own once the component
      // is placed. Leaving it in adjacency would make a node its own
      // neighbour and skew the ranking that picks each component's root.
      const selfLoops = new Map<string, string[]>();
      for (const edge of graph.edges()) {
        const [source, target] = graph.extremities(edge);
        sources.add(source);
        targets.add(target);
        if (source === target) {
          const loops = selfLoops.get(source) || [];
          loops.push(edge);
          selfLoops.set(source, loops);
          continue;
        }
        adjacency.get(source)?.add(target);
        adjacency.get(target)?.add(source);
      }
      const roleColor = (id: string) =>
        sources.has(id) && targets.has(id)
          ? "#75419a"
          : sources.has(id)
            ? "#2c718f"
            : "#0f8a74";
      const nodeRadius = (id: string) =>
        Number(graph.getNodeAttribute(id, "size"));

      // Mirrors parallelEdgeOffsets() in collision_labels.ts: the nth edge of
      // a parallel group bows out by slot * 48, which displaces its tag by
      // half that. Budgeting it here keeps parallel relations inside the
      // sector their pair was given.
      const pairKey = (left: string, right: string) =>
        left < right
          ? `${left}\u0000${right}`
          : `${right}\u0000${left}`;
      const parallelGroups = new Map<string, string[]>();
      for (const edge of graph.edges()) {
        const [source, target] = graph.extremities(edge);
        if (source === target) continue;
        const key = pairKey(source, target);
        const group = parallelGroups.get(key) || [];
        group.push(edge);
        parallelGroups.set(key, group);
      }
      const tagBow = new Map<string, number>();
      for (const group of parallelGroups.values()) {
        group
          .slice()
          .sort()
          .forEach((edge, index) => {
            tagBow.set(
              edge,
              (Math.abs(index - (group.length - 1) / 2) * 48) / 2,
            );
          });
      }

      // A tag is a thin rectangle lying along its edge, so its two extents
      // do completely different jobs and must not be conflated. The half
      // WIDTH runs along the edge and is what the edge has to be long enough
      // to hold. The half HEIGHT sticks out sideways and is all the sector
      // has to be wide enough to hold. Treating the tag as a disc of its
      // half-width -- roughly 100px for the longest relation names here --
      // demands over ten times the sideways room it actually occupies, which
      // is enough to make every layout look impossible to satisfy.
      const tagHalfLengthByEdge = new Map<string, number>();
      for (const edge of graph.edges()) {
        const width = tagWidth(
          String(graph.getEdgeAttribute(edge, "relationLabel")),
        );
        tagHalfLengthByEdge.set(edge, width / 2);
      }
      // Along the edge.
      const tagHalfLength = (edge: string) =>
        tagHalfLengthByEdge.get(edge) || 0;
      // Across the edge, including any bow a parallel relation is drawn with.
      const tagHalfDepth = (edge: string) =>
        TAG_HEIGHT / 2 + (tagBow.get(edge) || 0);

      // Length that lets the tag sit at the midpoint with an arrowhead and a
      // run of bare line still visible past each side of it.
      const edgeSpan = (edge: string) => {
        const [source, target] = graph.extremities(edge);
        return (
          2 *
          (tagHalfLength(edge) +
            Math.max(nodeRadius(source), nodeRadius(target)) +
            TAG_STANDOFF)
        );
      };

      const nodeRank = (left: string, right: string) =>
        (adjacency.get(right)?.size || 0) -
          (adjacency.get(left)?.size || 0) ||
        Number(nodeById.get(right)?.[2] || 0) -
          Number(nodeById.get(left)?.[2] || 0) ||
        left.localeCompare(right);

      const unseen = new Set(nodeIds);
      const components: string[][] = [];
      while (unseen.size) {
        const first = unseen.values().next().value as string;
        const stack = [first];
        const component: string[] = [];
        unseen.delete(first);
        while (stack.length) {
          const id = stack.pop() as string;
          component.push(id);
          for (const neighbor of adjacency.get(id) || []) {
            if (!unseen.has(neighbor)) continue;
            unseen.delete(neighbor);
            stack.push(neighbor);
          }
        }
        components.push(component);
      }
      const rootOf = new Map<string[], string>();
      for (const component of components) {
        rootOf.set(component, component.slice().sort(nodeRank)[0]);
      }
      components.sort(
        (left, right) =>
          right.length - left.length ||
          nodeRank(
            rootOf.get(left) as string,
            rootOf.get(right) as string,
          ),
      );

      type LayoutPoint = { x: number; y: number };

      const distanceToSegment = (
        point: LayoutPoint,
        start: LayoutPoint,
        end: LayoutPoint,
      ): number => {
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const lengthSquared = dx * dx + dy * dy;
        if (!lengthSquared) {
          return Math.hypot(point.x - start.x, point.y - start.y);
        }
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
      };

      // Exact sweep over the finished component, using the same shapes the
      // renderer will draw: a tag is a rotated rectangle, not a disc. This
      // reports what it finds; it never resizes anything. Inflating the
      // layout to escape a failure here is what produced mile-long edges,
      // and it papered over the real bug rather than fixing it.
      const rectClearsCircle = (
        centre: LayoutPoint,
        halfLength: number,
        halfDepth: number,
        along: LayoutPoint,
        point: LayoutPoint,
        keepOff: number,
      ): boolean => {
        const dx = point.x - centre.x;
        const dy = point.y - centre.y;
        // Into the tag's own frame: x runs along the edge, y across it.
        const localX = dx * along.x + dy * along.y;
        const localY = -dx * along.y + dy * along.x;
        const nearestX = Math.max(-halfLength, Math.min(halfLength, localX));
        const nearestY = Math.max(-halfDepth, Math.min(halfDepth, localY));
        return Math.hypot(localX - nearestX, localY - nearestY) >= keepOff;
      };

      const rectClearsSegment = (
        centre: LayoutPoint,
        halfLength: number,
        halfDepth: number,
        along: LayoutPoint,
        start: LayoutPoint,
        end: LayoutPoint,
        keepOff: number,
      ): boolean => {
        const toLocal = (point: LayoutPoint) => {
          const dx = point.x - centre.x;
          const dy = point.y - centre.y;
          return {
            x: dx * along.x + dy * along.y,
            y: -dx * along.y + dy * along.x,
          };
        };
        const a = toLocal(start);
        const b = toLocal(end);
        const boxX = halfLength + keepOff;
        const boxY = halfDepth + keepOff;
        if (a.x < -boxX && b.x < -boxX) return true;
        if (a.x > boxX && b.x > boxX) return true;
        if (a.y < -boxY && b.y < -boxY) return true;
        if (a.y > boxY && b.y > boxY) return true;
        for (let step = 0; step <= 16; step += 1) {
          const t = step / 16;
          const x = a.x + (b.x - a.x) * t;
          const y = a.y + (b.y - a.y) * t;
          const nearestX = Math.max(-halfLength, Math.min(halfLength, x));
          const nearestY = Math.max(-halfDepth, Math.min(halfDepth, y));
          if (Math.hypot(x - nearestX, y - nearestY) < keepOff) return false;
        }
        return true;
      };

      interface TagBox {
        edge: string;
        centre: LayoutPoint;
        along: LayoutPoint;
        halfLength: number;
        halfDepth: number;
      }

      const tagBoxesFor = (
        componentEdges: string[],
        positions: Map<string, LayoutPoint>,
      ): TagBox[] => {
        const boxes: TagBox[] = [];
        for (const edge of componentEdges) {
          const [source, target] = graph.extremities(edge);
          if (source === target) continue;
          const start = positions.get(source);
          const end = positions.get(target);
          if (!start || !end) continue;
          const length = Math.hypot(end.x - start.x, end.y - start.y) || 1;
          const along = {
            x: (end.x - start.x) / length,
            y: (end.y - start.y) / length,
          };
          const bow = tagBow.get(edge) || 0;
          const bowSign = bow === 0 ? 0 : source < target ? 1 : -1;
          boxes.push({
            edge,
            centre: {
              x: (start.x + end.x) / 2 - along.y * bow * bowSign,
              y: (start.y + end.y) / 2 + along.x * bow * bowSign,
            },
            along,
            halfLength: tagHalfLength(edge),
            halfDepth: TAG_HEIGHT / 2,
          });
        }
        return boxes;
      };

      const rectCorners = (box: TagBox): LayoutPoint[] =>
        [-1, 1].flatMap((alongSign) =>
          [-1, 1].map((depthSign) => ({
            x:
              box.centre.x +
              box.along.x * box.halfLength * alongSign -
              box.along.y * box.halfDepth * depthSign,
            y:
              box.centre.y +
              box.along.y * box.halfLength * alongSign +
              box.along.x * box.halfDepth * depthSign,
          })),
        );

      const verifyLayout = (
        component: string[],
        componentEdges: string[],
        positions: Map<string, LayoutPoint>,
      ): boolean => {
        const tags = tagBoxesFor(componentEdges, positions);
        let widestNode = 0;
        for (const id of component) {
          widestNode = Math.max(widestNode, nodeRadius(id));
        }
        let longestTag = 0;
        for (const tag of tags) {
          longestTag = Math.max(longestTag, tag.halfLength);
        }

        const cell = 200;
        const key = (x: number, y: number) =>
          `${Math.floor(x / cell)}|${Math.floor(y / cell)}`;
        const nodeGrid = new Map<string, string[]>();
        for (const id of component) {
          const point = positions.get(id);
          if (!point) continue;
          const bucket = key(point.x, point.y);
          const list = nodeGrid.get(bucket) || [];
          list.push(id);
          nodeGrid.set(bucket, list);
        }
        const nodesNear = (point: LayoutPoint, reach: number): string[] => {
          const span = Math.ceil(reach / cell);
          const baseX = Math.floor(point.x / cell);
          const baseY = Math.floor(point.y / cell);
          const found: string[] = [];
          for (let ix = baseX - span; ix <= baseX + span; ix += 1) {
            for (let iy = baseY - span; iy <= baseY + span; iy += 1) {
              const list = nodeGrid.get(`${ix}|${iy}`);
              if (list) found.push(...list);
            }
          }
          return found;
        };

        // Node against node.
        for (const id of component) {
          const point = positions.get(id);
          if (!point) continue;
          for (const other of nodesNear(
            point,
            nodeRadius(id) + widestNode + NODE_CLEARANCE,
          )) {
            if (other <= id) continue;
            const otherPoint = positions.get(other);
            if (!otherPoint) continue;
            if (
              Math.hypot(point.x - otherPoint.x, point.y - otherPoint.y) <
              nodeRadius(id) + nodeRadius(other) + NODE_CLEARANCE
            ) return false;
          }
        }

        // Edge line against node.
        for (const edge of componentEdges) {
          const [source, target] = graph.extremities(edge);
          if (source === target) continue;
          const start = positions.get(source);
          const end = positions.get(target);
          if (!start || !end) continue;
          const midpoint = {
            x: (start.x + end.x) / 2,
            y: (start.y + end.y) / 2,
          };
          const half = Math.hypot(end.x - start.x, end.y - start.y) / 2;
          for (const id of nodesNear(
            midpoint,
            half + widestNode + EDGE_NODE_CLEARANCE,
          )) {
            if (id === source || id === target) continue;
            const point = positions.get(id);
            if (!point) continue;
            if (
              distanceToSegment(point, start, end) <
              nodeRadius(id) + EDGE_NODE_CLEARANCE
            ) return false;
          }
        }

        // Tag against node.
        for (const tag of tags) {
          for (const id of nodesNear(
            tag.centre,
            tag.halfLength + widestNode + TAG_CLEARANCE,
          )) {
            const point = positions.get(id);
            if (!point) continue;
            if (
              !rectClearsCircle(
                tag.centre,
                tag.halfLength,
                tag.halfDepth,
                tag.along,
                point,
                nodeRadius(id) + TAG_CLEARANCE,
              )
            ) return false;
          }
        }

        // Tag against tag, and tag against every other edge line. Neither
        // follows from the sector argument on its own: a tag sits at its
        // edge's midpoint, which lies inside its parent's sector but not
        // inside its own child's sub-sector, so a sibling's line can still
        // reach it.
        for (let index = 0; index < tags.length; index += 1) {
          const tag = tags[index];
          for (let other = index + 1; other < tags.length; other += 1) {
            const rival = tags[other];
            if (
              Math.hypot(
                tag.centre.x - rival.centre.x,
                tag.centre.y - rival.centre.y,
              ) >
              tag.halfLength + rival.halfLength + longestTag + TAG_CLEARANCE
            ) continue;
            const corners = rectCorners(rival);
            const sides: Array<[LayoutPoint, LayoutPoint]> = [
              [corners[0], corners[1]],
              [corners[1], corners[3]],
              [corners[3], corners[2]],
              [corners[2], corners[0]],
            ];
            for (const [from, to] of sides) {
              if (
                !rectClearsSegment(
                  tag.centre,
                  tag.halfLength,
                  tag.halfDepth,
                  tag.along,
                  from,
                  to,
                  TAG_CLEARANCE,
                )
              ) return false;
            }
          }

          for (const edge of componentEdges) {
            if (edge === tag.edge) continue;
            const [source, target] = graph.extremities(edge);
            if (source === target) continue;
            const start = positions.get(source);
            const end = positions.get(target);
            if (!start || !end) continue;
            if (
              !rectClearsSegment(
                tag.centre,
                tag.halfLength,
                tag.halfDepth,
                tag.along,
                start,
                end,
                TAG_CLEARANCE,
              )
            ) return false;
          }
        }
        return true;
      };

      let layoutVerified = true;
      let solveScale = 1;
      let solveSteps = 0;

      interface ComponentLayout {
        root: string;
        positions: Map<string, LayoutPoint>;
        loopAnchors: Map<string, LayoutPoint>;
        minX: number;
        minY: number;
        maxX: number;
        maxY: number;
        width: number;
        height: number;
      }

      const layoutComponent = (component: string[]): ComponentLayout => {
        const root = rootOf.get(component) as string;
        const componentSet = new Set(component);
        const componentEdges = graph
          .edges()
          .filter((edge) =>
            componentSet.has(graph.extremities(edge)[0]),
          );

        // Spanning tree. Children are ordered by rank so the densest branch
        // keeps the same visual position it had before.
        const parentOf = new Map<string, string | null>([[root, null]]);
        const depthOf = new Map<string, number>([[root, 0]]);
        const childrenOf = new Map<string, string[]>(
          component.map((id) => [id, [] as string[]]),
        );
        const order = [root];
        for (let cursor = 0; cursor < order.length; cursor += 1) {
          const parent = order[cursor];
          const neighbors = Array.from(adjacency.get(parent) || []).sort(
            nodeRank,
          );
          for (const neighbor of neighbors) {
            if (parentOf.has(neighbor)) continue;
            parentOf.set(neighbor, parent);
            depthOf.set(neighbor, Number(depthOf.get(parent) || 0) + 1);
            childrenOf.get(parent)?.push(neighbor);
            order.push(neighbor);
          }
        }
        const maxDepth = order.reduce(
          (deepest, id) => Math.max(deepest, depthOf.get(id) || 0),
          0,
        );
        const isInSubtree = (top: string, id: string): boolean => {
          let current: string | null | undefined = id;
          while (current) {
            if (current === top) return true;
            current = parentOf.get(current);
          }
          return false;
        };
        const descendantsOf = (top: string): string[] => {
          const descendants: string[] = [];
          const stack = [top];
          while (stack.length) {
            const current = stack.pop() as string;
            descendants.push(current);
            for (const child of childrenOf.get(current) || []) {
              stack.push(child);
            }
          }
          return descendants;
        };
        const tagMover = (edge: string): string => {
          const [source, target] = graph.extremities(edge);
          if (parentOf.get(target) === source) return target;
          if (parentOf.get(source) === target) return source;
          const sourceDepth = Number(depthOf.get(source) || 0);
          const targetDepth = Number(depthOf.get(target) || 0);
          return targetDepth > sourceDepth ? target : source;
        };

        // Every edge joining a parent to one of its children shares that
        // pair's ring gap, so the gap has to satisfy the longest of them.
        const linkEdges = new Map<string, string[]>();
        const baseGap = new Array(maxDepth + 1).fill(0);
        for (const edge of componentEdges) {
          const [source, target] = graph.extremities(edge);
          const linksParentToChild =
            parentOf.get(target) === source ||
            parentOf.get(source) === target;
          if (!linksParentToChild) continue;
          const child = parentOf.get(target) === source ? target : source;
          const list = linkEdges.get(child) || [];
          list.push(edge);
          linkEdges.set(child, list);
          const depth = Math.min(
            Number(depthOf.get(source) || 0),
            Number(depthOf.get(target) || 0),
          );
          baseGap[depth] = Math.max(baseGap[depth], edgeSpan(edge));
        }
        for (let depth = 0; depth <= maxDepth; depth += 1) {
          if (baseGap[depth] <= 0) baseGap[depth] = 96;
        }

        let positions = new Map<string, LayoutPoint>();
        let scale = 1;
        let usedSteps = 0;
        // Sideways room a tag needs in its sector. A tag lies along its edge,
        // and an edge is only roughly radial, so the tag's long side leans
        // partly across the sector. How far it leans depends on where the
        // child was placed, which depends on this number -- so start from the
        // tag's depth alone and let the loop below raise it once it can
        // measure the real lean. It only ever rises, so this settles.
        const tagSpread = new Map<string, number>();
        for (let step = 0; step < MAX_SOLVE_STEPS; step += 1) {
          usedSteps = step + 1;
          const ring = [0];
          for (let depth = 0; depth <= maxDepth; depth += 1) {
            ring.push(ring[depth] + baseGap[depth] * scale);
          }

          // How much angle each subtree needs, bottom up. Angles measured
          // from the origin are additive across depths, so a child's demand
          // can be summed into its parent's directly.
          const halfAngle = (extent: number, atRadius: number) =>
            atRadius <= 0
              ? Math.PI
              : Math.asin(Math.min(1, extent / atRadius));
          const demand = new Map<string, number>();
          for (let index = order.length - 1; index >= 0; index -= 1) {
            const id = order[index];
            const depth = Number(depthOf.get(id) || 0);
            let own =
              2 * halfAngle(nodeRadius(id) + NODE_CLEARANCE / 2, ring[depth]);
            // The tag on the link down from the parent lives in this same
            // sector, so the sector has to be wide enough to hold it too.
            if (depth > 0) {
              const tagAt = (ring[depth - 1] + ring[depth]) / 2;
              // A tag rides the middle of its edge, and the middle of a chord
              // sits nearer the parent than the child does. So sibling tags
              // are squeezed together by exactly
              //     ringChild / (ringParent + ringChild)
              // relative to the angle their children were given -- a factor
              // of about a half once past the root. Budgeting the tag as if
              // it sat centred in its child's sector under-allocates by that
              // same factor, which is what let tags pile up around a hub.
              // Divide it back out.
              const crowding =
                (ring[depth - 1] + ring[depth]) / (ring[depth] || 1);
              for (const edge of linkEdges.get(id) || []) {
                const spread = Math.max(
                  tagSpread.get(edge) || 0,
                  tagHalfDepth(edge),
                );
                own = Math.max(
                  own,
                  2 * halfAngle(spread + TAG_CLEARANCE / 2, tagAt) * crowding,
                );
              }
            }
            let childSum = 0;
            for (const child of childrenOf.get(id) || []) {
              childSum += demand.get(child) || 0;
            }
            demand.set(id, Math.max(own, childSum));
          }

          // Does every fan fit the span it is allowed?
          let overflow = 1;
          for (const id of order) {
            const children = childrenOf.get(id) || [];
            if (!children.length) continue;
            let childSum = 0;
            for (const child of children) childSum += demand.get(child) || 0;
            const cap = id === root ? Math.PI * 2 : OUTWARD_SPAN;
            if (childSum > cap) {
              overflow = Math.max(overflow, childSum / cap);
            }
          }
          if (overflow > 1) {
            scale *= overflow * 1.02;
            continue;
          }

          // Place, top down. Each node receives a sector; its children
          // divide that sector in proportion to their demand, centred on
          // the direction that points away from the graph's centre. That
          // last part is what makes a branch fan outward instead of
          // curling back towards its grandparent.
          positions = new Map<string, LayoutPoint>([[root, { x: 0, y: 0 }]]);
          const sectorOf = new Map<string, { center: number; width: number }>([
            [root, { center: -Math.PI / 2, width: Math.PI * 2 }],
          ]);
          for (const id of order) {
            const children = childrenOf.get(id) || [];
            if (!children.length) continue;
            const sector = sectorOf.get(id) as {
              center: number;
              width: number;
            };
            const depth = Number(depthOf.get(id) || 0);
            let childSum = 0;
            for (const child of children) childSum += demand.get(child) || 0;
            const available =
              id === root
                ? Math.PI * 2
                : Math.min(OUTWARD_SPAN, sector.width);
            // Spreading to fill the available span only ever adds
            // clearance, never removes it.
            const stretch = childSum > 0 ? available / childSum : 1;
            let cursor = sector.center - available / 2;
            for (const child of children) {
              const width = (demand.get(child) || 0) * stretch;
              const center = cursor + width / 2;
              cursor += width;
              sectorOf.set(child, { center, width });
              positions.set(child, {
                x: Math.cos(center) * ring[depth + 1],
                y: Math.sin(center) * ring[depth + 1],
              });
            }
          }

          // Now that the children are placed, measure how far each tag
          // really leans across its sector and feed that back. Widening is
          // monotone, so this reaches a fixed point rather than oscillating.
          let widened = false;
          for (const edge of componentEdges) {
            const [source, target] = graph.extremities(edge);
            if (source === target) continue;
            const start = positions.get(source);
            const end = positions.get(target);
            if (!start || !end) continue;
            const child = parentOf.get(target) === source ? target : source;
            const length =
              Math.hypot(end.x - start.x, end.y - start.y) || 1;
            const along = {
              x: (end.x - start.x) / length,
              y: (end.y - start.y) / length,
            };
            const centre = {
              x: (start.x + end.x) / 2,
              y: (start.y + end.y) / 2,
            };
            const reach = Math.hypot(centre.x, centre.y) || 1;
            const radial = { x: centre.x / reach, y: centre.y / reach };
            const lean = Math.abs(radial.x * along.y - radial.y * along.x);
            const face = Math.abs(radial.x * along.x + radial.y * along.y);
            const needed =
              tagHalfLength(edge) * lean + tagHalfDepth(edge) * face;
            const key = childrenOf.has(child) ? edge : edge;
            if (needed > (tagSpread.get(key) || 0) + 0.5) {
              tagSpread.set(key, needed);
              widened = true;
            }
          }
          if (widened && step < MAX_SOLVE_STEPS - 1) continue;

          // Nodes are laid out first. Only after the edge tags have their
          // real rectangles do we relieve a tag that covers a node. Move the
          // offending node and its whole subtree by the same small
          // translation, then rebuild the tag rectangles before deciding
          // whether anything else needs to move. Internal subtree edges keep
          // their exact lengths; only the link back to the parent grows.
          for (let correction = 0; correction < 256; correction += 1) {
            const tags = tagBoxesFor(componentEdges, positions);
            let mover: string | null = null;
            for (const tag of tags) {
              const [tagSource, tagTarget] = graph.extremities(tag.edge);
              for (const id of component) {
                const point = positions.get(id);
                if (!point) continue;
                if (
                  Math.hypot(
                    point.x - tag.centre.x,
                    point.y - tag.centre.y,
                  ) >
                  tag.halfLength +
                    nodeRadius(id) +
                    TAG_ENDPOINT_CLEARANCE
                ) continue;
                const clearance =
                  id === tagSource || id === tagTarget
                    ? TAG_ENDPOINT_CLEARANCE
                    : TAG_CLEARANCE;
                if (
                  rectClearsCircle(
                    tag.centre,
                    tag.halfLength,
                    tag.halfDepth,
                    tag.along,
                    point,
                    nodeRadius(id) + clearance,
                  )
                ) continue;

                // Moving an ancestor of both tag endpoints would carry the
                // tag and node together and change nothing. In that case
                // lengthen the tag's own tree edge instead.
                const carriesWholeTag =
                  isInSubtree(id, tagSource) &&
                  isInSubtree(id, tagTarget);
                mover =
                  id !== root && !carriesWholeTag
                    ? id
                    : tagMover(tag.edge);
                if (mover === root) mover = null;
                break;
              }
              if (mover) break;
            }
            if (!mover) break;
            const parent = parentOf.get(mover);
            const point = positions.get(mover);
            const parentPoint = parent ? positions.get(parent) : null;
            if (!point || !parentPoint) {
              layoutVerified = false;
              break;
            }
            const dx = point.x - parentPoint.x;
            const dy = point.y - parentPoint.y;
            const length = Math.hypot(dx, dy) || 1;
            const shiftX = (dx / length) * 4;
            const shiftY = (dy / length) * 4;
            for (const member of descendantsOf(mover)) {
              const memberPoint = positions.get(member);
              if (!memberPoint) continue;
              positions.set(member, {
                x: memberPoint.x + shiftX,
                y: memberPoint.y + shiftY,
              });
            }
            if (correction === 255) layoutVerified = false;
          }

          // The construction above is the whole of the sizing. If the sweep
          // still finds something, that is a bug to fix here, not something
          // to escape by making the component bigger.
          layoutVerified =
            verifyLayout(component, componentEdges, positions) &&
            layoutVerified;
          break;
        }
        solveScale = Math.max(solveScale, scale);
        solveSteps = Math.max(solveSteps, usedSteps);

        // Anchor each self-loop. The loop is drawn as a circle tangent to its
        // node, reaching out to the anchor, with the tag riding the far side
        // of it -- so the anchor is where the tag will sit. Sweep directions
        // at a growing radius and keep the first that clears everything
        // already placed. This always terminates: past the component's own
        // extent there is nothing left to hit.
        const loopAnchors = new Map<string, LayoutPoint>();
        for (const [id, loops] of selfLoops) {
          const centre = positions.get(id);
          if (!centre) continue;
          loops.forEach((edge, loopIndex) => {
            const radius = tagHalfLength(edge);
            // Far enough out that the loop can clear its own node.
            const minimumReach =
              nodeRadius(id) + 2 * ((radius + TAG_STANDOFF) / 2);
            let placed: LayoutPoint | null = null;
            for (let ringStep = 0; ringStep < 24 && !placed; ringStep += 1) {
              const reach = minimumReach * (1 + ringStep * 0.18);
              const loopRadius = (reach - nodeRadius(id)) / 2;
              for (let spoke = 0; spoke < 36 && !placed; spoke += 1) {
                // Alternate either side of straight up so the loop lands as
                // close to a natural reading position as the space allows.
                const turn =
                  ((spoke % 2 ? 1 : -1) * Math.ceil(spoke / 2) * Math.PI) / 18;
                const angle = -Math.PI / 2 + turn + loopIndex * 0.6;
                const anchor = {
                  x: centre.x + Math.cos(angle) * reach,
                  y: centre.y + Math.sin(angle) * reach,
                };
                const loopCentre = {
                  x: (centre.x + anchor.x) / 2,
                  y: (centre.y + anchor.y) / 2,
                };
                let clear = true;
                for (const [otherId, point] of positions) {
                  if (otherId === id) continue;
                  const keepOff = nodeRadius(otherId) + NODE_CLEARANCE;
                  if (
                    Math.hypot(point.x - anchor.x, point.y - anchor.y) <
                      radius + keepOff ||
                    Math.hypot(point.x - loopCentre.x, point.y - loopCentre.y) <
                      loopRadius + keepOff
                  ) {
                    clear = false;
                    break;
                  }
                }
                if (clear) {
                  for (const other of componentEdges) {
                    const [otherSource, otherTarget] =
                      graph.extremities(other);
                    if (otherSource === otherTarget) continue;
                    const start = positions.get(otherSource);
                    const end = positions.get(otherTarget);
                    if (!start || !end) continue;
                    if (
                      distanceToSegment(anchor, start, end) <
                        radius + TAG_CLEARANCE ||
                      distanceToSegment(loopCentre, start, end) <
                        loopRadius + TAG_CLEARANCE
                    ) {
                      clear = false;
                      break;
                    }
                    const otherTag = {
                      x: (start.x + end.x) / 2,
                      y: (start.y + end.y) / 2,
                    };
                    if (
                      Math.hypot(otherTag.x - anchor.x, otherTag.y - anchor.y) <
                      radius + tagHalfLength(other) + TAG_CLEARANCE
                    ) {
                      clear = false;
                      break;
                    }
                  }
                }
                if (clear) placed = anchor;
              }
            }
            if (placed) loopAnchors.set(edge, placed);
          });
        }

        let minX = Number.POSITIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;
        const stretchBounds = (point: LayoutPoint, radius: number) => {
          minX = Math.min(minX, point.x - radius);
          minY = Math.min(minY, point.y - radius);
          maxX = Math.max(maxX, point.x + radius);
          maxY = Math.max(maxY, point.y + radius);
        };
        for (const [id, point] of positions) {
          stretchBounds(point, nodeRadius(id) + NODE_CLEARANCE);
        }
        for (const edge of componentEdges) {
          const [source, target] = graph.extremities(edge);
          const start = positions.get(source);
          const end = positions.get(target);
          if (!start || !end) continue;
          const anchor = loopAnchors.get(edge);
          stretchBounds(
            anchor || { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 },
            tagHalfLength(edge) + TAG_CLEARANCE,
          );
        }
        return {
          root,
          positions,
          loopAnchors,
          minX,
          minY,
          maxX,
          maxY,
          width: Math.max(1, maxX - minX),
          height: Math.max(1, maxY - minY),
        };
      };

      const componentLayouts = components.map(layoutComponent);
      // Diagnostics: how far the sizing solve had to push beyond its first,
      // fully constructed attempt. A scale of 1 means the construction alone
      // was enough, which is what should normally happen.
      graph.setAttribute("layoutSolveScale", solveScale);
      graph.setAttribute("layoutVerified", layoutVerified);
      graph.setAttribute("layoutSolveSteps", solveSteps);

      // Edges are straight, and the tag sits dead centre on every one.
      for (const edge of graph.edges()) {
        graph.setEdgeAttribute(edge, "layoutCurve", 0);
        graph.setEdgeAttribute(edge, "labelFraction", 0.5);
        graph.setEdgeAttribute(
          edge,
          "layoutLabelRadius",
          tagHalfLength(edge),
        );
      }

      // Shelf-pack the components. Boxes are disjoint by construction, so
      // nothing one component contains can reach another.
      const targetWidth = Math.max(
        ...componentLayouts.map((layout) => layout.width),
        Math.sqrt(
          componentLayouts.reduce(
            (total, layout) =>
              total +
              (layout.width + COMPONENT_GAP) *
                (layout.height + COMPONENT_GAP),
            0,
          ) * 1.8,
        ),
      );
      let cursorX = 0;
      let cursorY = 0;
      let rowHeight = 0;
      let packedWidth = 0;
      for (const layout of componentLayouts) {
        if (cursorX > 0 && cursorX + layout.width > targetWidth) {
          cursorX = 0;
          cursorY += rowHeight + COMPONENT_GAP;
          rowHeight = 0;
        }
        const offsetX = cursorX - layout.minX;
        const offsetY = cursorY - layout.minY;
        for (const [id, point] of layout.positions) {
          graph.mergeNodeAttributes(id, {
            x: point.x + offsetX,
            y: point.y + offsetY,
            color: roleColor(id),
          });
        }
        for (const [edge, anchor] of layout.loopAnchors) {
          graph.setEdgeAttribute(edge, "loopAnchorX", anchor.x + offsetX);
          graph.setEdgeAttribute(edge, "loopAnchorY", anchor.y + offsetY);
        }
        cursorX += layout.width + COMPONENT_GAP;
        rowHeight = Math.max(rowHeight, layout.height);
        packedWidth = Math.max(packedWidth, cursorX - COMPONENT_GAP);
      }
      const packedHeight = cursorY + rowHeight;
      graph.updateEachNodeAttributes((_node, attributes) => ({
        ...attributes,
        x: Number(attributes.x) - packedWidth / 2,
        y: Number(attributes.y) - packedHeight / 2,
      }));
      // Loop anchors are absolute positions, so they take the same shift.
      for (const edge of graph.edges()) {
        if (!graph.hasEdgeAttribute(edge, "loopAnchorX")) continue;
        graph.setEdgeAttribute(
          edge,
          "loopAnchorX",
          Number(graph.getEdgeAttribute(edge, "loopAnchorX")) -
            packedWidth / 2,
        );
        graph.setEdgeAttribute(
          edge,
          "loopAnchorY",
          Number(graph.getEdgeAttribute(edge, "loopAnchorY")) -
            packedHeight / 2,
        );
      }

      const primary = componentLayouts[0];
      if (primary) {
        graph.setAttribute("readableStartNode", primary.root);
        const attributes = graph.getNodeAttributes(primary.root);
        graph.setAttribute("readableStartX", Number(attributes.x));
        graph.setAttribute("readableStartY", Number(attributes.y));
      }
      return;
    }

    if (kind === "radial") {
      const focusId = payload.focus.id;
      const neighbors = payload.nodes
        .map((node) => node[0])
        .filter((id) => id !== focusId);
      graph.mergeNodeAttributes(focusId, { x: 0, y: 0, fixed: true });
      let cursor = 0;
      let ring = 0;
      while (cursor < neighbors.length) {
        const capacity = 14 + ring * 10;
        const ringNodes = neighbors.slice(cursor, cursor + capacity);
        const radius = 15 + ring * 13;
        ringNodes.forEach((id, index) => {
          const angle = (Math.PI * 2 * index) / ringNodes.length - Math.PI / 2;
          graph.mergeNodeAttributes(id, {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
          });
        });
        cursor += ringNodes.length;
        ring += 1;
      }
      return;
    }

    if (kind === "bipartite") {
      const sources = new Set(payload.edges.map((edge) => payload.nodes[edge[0]][0]));
      const targets = new Set(payload.edges.map((edge) => payload.nodes[edge[1]][0]));
      const columns: string[][] = [[], [], []];
      for (const node of payload.nodes) {
        const isSource = sources.has(node[0]);
        const isTarget = targets.has(node[0]);
        columns[isSource && isTarget ? 1 : isSource ? 0 : 2].push(node[0]);
      }
      const placeSide = (
        ids: string[],
        direction: -1 | 1,
        color: string,
      ) => {
        const rows = Math.min(34, Math.max(1, Math.ceil(Math.sqrt(ids.length * 4))));
        ids.forEach((id, index) => {
          const column = Math.floor(index / rows);
          const row = index % rows;
          const rowsInColumn = Math.min(rows, ids.length - column * rows);
          graph.mergeNodeAttributes(id, {
            x: direction * (24 + column * 11),
            y: (row - (rowsInColumn - 1) / 2) * 4.2,
            color,
          });
        });
      };
      placeSide(columns[0], -1, "#2c718f");
      placeSide(columns[2], 1, "#0f8a74");
      const centerRows = Math.min(24, Math.max(1, columns[1].length));
      columns[1].forEach((id, index) => {
        const column = Math.floor(index / centerRows);
        const row = index % centerRows;
        graph.mergeNodeAttributes(id, {
          x: (column - Math.floor(columns[1].length / centerRows) / 2) * 8,
          y: (row - (centerRows - 1) / 2) * 4.2,
          color: "#75419a",
        });
      });
      return;
    }

    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    payload.nodes.forEach((node, index) => {
      const radius = 1.35 * Math.sqrt(index + 1);
      graph.mergeNodeAttributes(node[0], {
        x: Math.cos(index * goldenAngle) * radius,
        y: Math.sin(index * goldenAngle) * radius,
        fixed: node[0] === payload.focus.id,
      });
    });
  }

  private makeGraph(payload: TopologyPayload): MultiDirectedGraph {
    const graph = new MultiDirectedGraph();
    const isAtlas = payload.layout.kind === "atlas";
    const usesEntityBubbles =
      payload.layout.kind === "triples" ||
      payload.layout.kind === "claim";
    graph.setAttribute(
      "showAllLabels",
      payload.layout.kind === "triples" ||
        payload.layout.kind === "claim",
    );
    graph.setAttribute("customRenderedEdges", usesEntityBubbles);
    graph.setAttribute("uniformScaling", true);
    graph.setAttribute("atlas", isAtlas);
    graph.setAttribute("parallelEdgeGap", 48);
    graph.setAttribute("progressiveEdgeLabels", false);
    graph.setAttribute("allowEdgeUnderLabels", false);
    graph.setAttribute("collisionAudit", usesEntityBubbles);
    graph.setAttribute("centerWholeGraph", false);
    if (isAtlas) {
      const ranked = payload.nodes
        .slice()
        .sort(
          (left, right) =>
            Number(right[7] || 0) - Number(left[7] || 0) ||
            left[0].localeCompare(right[0]),
        );
      this.atlasOverviewLabels = new Set(
        ranked.slice(0, 15).map((node) => node[0]),
      );
      this.atlasPredicateEdges = new Set(
        payload.edges
          .map((edge, index) => ({ edge, index }))
          .sort(
            (left, right) =>
              Number(right.edge[4]) - Number(left.edge[4]) ||
              left.edge[3].localeCompare(right.edge[3]) ||
              left.index - right.index,
          )
          .slice(0, 500)
          .map(({ index }) => `edge-${index}`),
      );
    }

    payload.nodes.forEach((node) => {
      const isFocus = node[0] === payload.focus.id;
      const communityIndex = Number(node[6] || 0);
      const componentIndex = Number(node[5] || 0);
      const componentSize = Number(
        payload.components?.[componentIndex]?.[2] || 0,
      );
      graph.addNode(node[0], {
        label: node[1],
        frequency: node[2],
        x: node[3] ?? 0,
        y: node[4] ?? 0,
        size:
          (usesEntityBubbles
            ? bubbleNodeSize(node[1])
            : nodeSize(node[2])) + (isFocus ? 1.5 : 0),
        color:
          isFocus
            ? "#75419a"
            : isAtlas
              ? atlasNodeColor(communityIndex)
              : "#087f75",
        forceLabel: isFocus,
        highlighted: isFocus,
        zIndex: isFocus ? 2 : 1,
        atlasNode: isAtlas,
        componentIndex,
        communityIndex,
        atlasPriority: Number(node[7] || 0),
        atlasSmallComponent: componentSize <= 16,
        type: isAtlas ? "point" : "circle",
      });
    });

    payload.edges.forEach((edge, index) => {
      const sourceId = payload.nodes[edge[0]][0];
      const targetId = payload.nodes[edge[1]][0];
      const sourceCommunity = Number(payload.nodes[edge[0]][6] || 0);
      const targetCommunity = Number(payload.nodes[edge[1]][6] || 0);
      const componentIndex = Number(payload.nodes[edge[0]][5] || 0);
      const componentSize = Number(
        payload.components?.[componentIndex]?.[2] || 0,
      );
      graph.addDirectedEdgeWithKey(`edge-${index}`, sourceId, targetId, {
        label: null,
        relationLabel: edge[3],
        relationId: edge[2],
        occurrenceCount: edge[4],
        sourceId,
        targetId,
        size: usesEntityBubbles
            ? Math.min(4, 1.3 + Math.log2(edge[4] + 1) * 0.35)
            : Math.min(4, 0.5 + Math.log2(edge[4] + 1) * 0.55),
        color: "#8761a8",
        type: "arrow",
        forceLabel: false,
        zIndex: 1,
        customRendered: usesEntityBubbles,
        atlasRawEdge: isAtlas,
        atlasInternal: sourceCommunity === targetCommunity,
        atlasSmallComponent: componentSize <= 16,
        bundleIndex: Number(edge[5] ?? -1),
      });
    });

    if (isAtlas) {
      (payload.bundles || []).forEach((bundle, index) => {
        if (!bundle[5]) return;
        const sourceAnchor = `atlas-bundle-source-${index}`;
        const targetAnchor = `atlas-bundle-target-${index}`;
        graph.addNode(sourceAnchor, {
          label: "",
          x: bundle[6],
          y: bundle[7],
          size: 0.01,
          color: "#ffffff",
          zIndex: 0,
          atlasAnchor: true,
          type: "point",
        });
        graph.addNode(targetAnchor, {
          label: "",
          x: bundle[8],
          y: bundle[9],
          size: 0.01,
          color: "#ffffff",
          zIndex: 0,
          atlasAnchor: true,
          type: "point",
        });
        graph.addDirectedEdgeWithKey(
          `atlas-bundle-${index}`,
          sourceAnchor,
          targetAnchor,
          {
            label: null,
            relationLabel: bundle[4],
            relationId: "",
            occurrenceCount: bundle[3],
            sourceId: sourceAnchor,
            targetId: targetAnchor,
            size: Math.min(3, 0.45 + Math.log2(bundle[3] + 1) * 0.32),
            color: "#a891bc",
            type: "line",
            forceLabel: false,
            zIndex: 0,
            atlasBundle: true,
            atlasOverviewBundle: bundle[5],
          },
        );
      });
    }

    this.assignLayout(graph, payload);
    return graph;
  }

  private prepareReadableLayout(graph: MultiDirectedGraph): void {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const node of graph.nodes()) {
      const x = Number(graph.getNodeAttribute(node, "x"));
      const y = Number(graph.getNodeAttribute(node, "y"));
      xs.push(x);
      ys.push(y);
    }
    if (!xs.length) return;
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    this.readableLayoutCenter = {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
    };
    this.graphBBox = {
      x: [minX, maxX],
      y: [minY, maxY],
    };
    const fullWidth = Math.max(1, maxX - minX);
    const fullHeight = Math.max(1, maxY - minY);
    const wholeGraphFits =
      graph.order <= 60 || (fullWidth <= 120 && fullHeight <= 80);
    graph.setAttribute("centerWholeGraph", wholeGraphFits);
    // The custom bbox has to keep the same aspect ratio as the container.
    // Sigma normalises the bbox by its longest side and then rescales by the
    // viewport's shortest side, so matching the two aspect ratios is what
    // makes one graph unit render as exactly one pixel at camera ratio 1.
    // With uniformScaling that equality then holds at every zoom, which is
    // what lets the layout below reason in pixels: node sizes and tag
    // metrics are pixel quantities, and the clearances the layout computes
    // between them have to survive the trip to the screen unchanged. Any
    // other aspect ratio shrinks graph distances by up to 15% relative to
    // node sizes and silently reintroduces overlap.
    // Sigma rescales by the viewport's shortest side less twice the 28px
    // stage padding, so the bbox side matching that shortest side is what
    // has to absorb the padding; the other side follows the aspect ratio.
    const stageWidth = Math.max(320, el.container.clientWidth);
    const stageHeight = Math.max(240, el.container.clientHeight);
    const stageAspect = stageWidth / stageHeight;
    const viewHeight =
      stageAspect >= 1 ? stageHeight - 56 : (stageWidth - 56) / stageAspect;
    const viewWidth = viewHeight * stageAspect;
    const viewCenter = wholeGraphFits
      ? this.readableLayoutCenter
      : {
          x: Number(
            graph.getAttribute("readableStartX") ??
              this.readableLayoutCenter.x,
          ),
          y: Number(
            graph.getAttribute("readableStartY") ??
              this.readableLayoutCenter.y,
          ),
        };
    this.fixedBBox = {
      x: [
        viewCenter.x - viewWidth / 2,
        viewCenter.x + viewWidth / 2,
      ],
      y: [
        viewCenter.y - viewHeight / 2,
        viewCenter.y + viewHeight / 2,
      ],
    };
    graph.setAttribute("geometryRevision", 0);
  }

  private refreshReadableBounds(): void {
    if (!this.graph) return;
    const xs: number[] = [];
    const ys: number[] = [];
    for (const node of this.graph.nodes()) {
      xs.push(Number(this.graph.getNodeAttribute(node, "x")));
      ys.push(Number(this.graph.getNodeAttribute(node, "y")));
    }
    if (!xs.length) return;
    this.graphBBox = {
      x: [Math.min(...xs), Math.max(...xs)],
      y: [Math.min(...ys), Math.max(...ys)],
    };
  }

  // The layout is overlap-free by construction, so there is nothing left to
  // resolve once the first frame has been drawn: reveal the graph and record
  // the audit that the collision test reads back.
  private revealReadableLayout(): void {
    if (!this.readableLayoutPending || !this.graph || !this.renderer) return;
    this.readableLayoutPending = false;
    this.refreshReadableBounds();
    this.graph.setAttribute("hardCollisionFree", true);
    if (this.labelCanvas) {
      this.labelCanvas.dataset.hardCollisionFree = "true";
      this.labelCanvas.dataset.hardCollisionPasses = "0";
    }
    el.container.style.visibility = "";
    this.setLoading(false);
    window.requestAnimationFrame(() => {
      if (!this.renderer || !this.graph) return;
      if (Boolean(this.graph.getAttribute("centerWholeGraph"))) {
        this.fitGraph(false);
      } else {
        this.recenterReadableView();
      }
    });
  }

  private runCorpusRadialWorker(
    payload: TopologyPayload,
  ): Promise<CorpusRadialLayoutResult> {
    const requestId = ++this.corpusRadialRequest;
    const frequencies = new Float64Array(payload.nodes.length);
    const sources = new Int32Array(payload.edges.length);
    const targets = new Int32Array(payload.edges.length);
    const ids = payload.nodes.map((node, index) => {
      frequencies[index] = Number(node[2] || 0);
      return node[0];
    });
    payload.edges.forEach((edge, index) => {
      sources[index] = edge[0];
      targets[index] = edge[1];
    });
    const worker = new CorpusRadialWorker({
      name: "corpus-radial-layout",
    });
    this.corpusRadialWorker = worker;
    return new Promise((resolve, reject) => {
      const finish = (): void => {
        worker.terminate();
        if (this.corpusRadialWorker === worker) {
          this.corpusRadialWorker = null;
        }
      };
      worker.addEventListener("message", (event: MessageEvent<CorpusRadialWorkerResponse>) => {
        const response = event.data;
        if (response.kind !== "layout" || response.requestId !== requestId) return;
        finish();
        if (requestId !== this.corpusRadialRequest) {
          reject(new DOMException("Superseded corpus layout", "AbortError"));
          return;
        }
        if (response.error || !response.result) {
          reject(new Error(response.error || "Corpus radial layout failed."));
          return;
        }
        resolve(response.result);
      });
      worker.addEventListener("error", (event) => {
        finish();
        reject(new Error(event.message || "Corpus radial worker failed."));
      });
      worker.postMessage(
        {
          kind: "layout",
          requestId,
          input: { ids, frequencies, sources, targets },
        },
        [frequencies.buffer, sources.buffer, targets.buffer],
      );
    });
  }

  private graphClickIsSuppressed(): boolean {
    return performance.now() < this.suppressGraphClickUntil;
  }

  private consumeGraphClickSuppression(): boolean {
    if (!this.graphClickIsSuppressed()) return false;
    this.suppressGraphClickUntil = 0;
    return true;
  }

  private makeCorpusRadialModel(
    payload: TopologyPayload,
    layout: CorpusRadialLayoutResult,
  ): CorpusRadialModel {
    const nodeCount = payload.nodes.length;
    const rankPosition = new Int32Array(nodeCount);
    layout.ranked.forEach((node, rank) => {
      rankPosition[node] = rank;
    });
    const markAncestors = (flag: Uint8Array, node: number): void => {
      let current = node;
      while (current >= 0 && !flag[current]) {
        flag[current] = 1;
        current = layout.parent[current];
      }
    };
    const buildLevel = (
      count: number,
      previous?: Uint8Array,
    ): Uint8Array => {
      const flag = previous ? previous.slice() : new Uint8Array(nodeCount);
      for (
        let rank = 0;
        rank < Math.min(count, layout.ranked.length);
        rank += 1
      ) {
        markAncestors(flag, layout.ranked[rank]);
      }
      return flag;
    };
    const level0 = buildLevel(Math.min(20, nodeCount));
    const level1 = buildLevel(Math.ceil(nodeCount * 0.01), level0);
    const level2 = buildLevel(Math.ceil(nodeCount * 0.1), level1);
    const order = (nodes: number[]): number[] =>
      nodes.sort(
        (left, right) =>
          layout.depth[left] - layout.depth[right] ||
          rankPosition[left] - rankPosition[right] ||
          payload.nodes[left][0].localeCompare(payload.nodes[right][0]),
      );
    const additions: [number[], number[], number[]] = [
      order(Array.from(layout.ranked).filter((node) => level0[node] === 1)),
      order(
        Array.from(layout.ranked).filter(
          (node) => level1[node] === 1 && level0[node] === 0,
        ),
      ),
      order(
        Array.from(layout.ranked).filter(
          (node) => level2[node] === 1 && level1[node] === 0,
        ),
      ),
    ];
    return {
      payload,
      layout,
      rankPosition,
      levelFlags: [level0, level1, level2],
      levelAdditions: additions,
      admitted: new Set<number>(),
      pixelsPerUnit: 0,
      projectedRadius: 4,
      lastDisclosureMs: 0,
      lastCenterX: Number.NaN,
      lastCenterY: Number.NaN,
    };
  }

  private addCorpusRadialNode(
    graph: MultiDirectedGraph,
    model: CorpusRadialModel,
    nodeIndex: number,
  ): void {
    const node = model.payload.nodes[nodeIndex];
    const nodeId = node[0];
    if (graph.hasNode(nodeId)) return;
    graph.addNode(nodeId, {
      label: node[1],
      frequency: node[2],
      x: model.layout.x[nodeIndex],
      y: model.layout.y[nodeIndex],
      size: model.projectedRadius,
      color: atlasNodeColor(Number(node[6] || 0)),
      forceLabel: false,
      highlighted: false,
      zIndex: 1,
      type: "circle",
      corpusRadial: true,
      corpusIndex: nodeIndex,
      corpusDegree: model.layout.degree[nodeIndex],
    });
  }

  private addCorpusRadialEdge(
    graph: MultiDirectedGraph,
    model: CorpusRadialModel,
    childIndex: number,
  ): void {
    const sourceEdgeIndex = model.layout.parentEdge[childIndex];
    if (sourceEdgeIndex < 0) return;
    const edge = model.payload.edges[sourceEdgeIndex];
    if (!edge) return;
    const sourceId = model.payload.nodes[edge[0]][0];
    const targetId = model.payload.nodes[edge[1]][0];
    if (!graph.hasNode(sourceId) || !graph.hasNode(targetId)) return;
    const edgeId = `corpus-tree-${childIndex}`;
    if (graph.hasEdge(edgeId)) return;
    graph.addDirectedEdgeWithKey(edgeId, sourceId, targetId, {
      label: null,
      relationLabel: edge[3],
      relationId: edge[2],
      occurrenceCount: edge[4],
      sourceId,
      targetId,
      size: Math.min(2.8, 0.65 + Math.log2(edge[4] + 1) * 0.32),
      color: "#9a7caf",
      type: "arrow",
      forceLabel: false,
      zIndex: 0,
      corpusRadialTree: true,
      corpusChildIndex: childIndex,
    });
  }

  private async installCorpusRadialPayload(
    payload: TopologyPayload,
  ): Promise<void> {
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
    this.selectedNeighbors.clear();
    this.evidenceItems = [];
    el.container.style.visibility = "hidden";
    this.setLoading(
      true,
      `Laying out ${payload.nodes.length.toLocaleString()} corpus entities\u2026`,
    );

    const layout = await this.runCorpusRadialWorker(payload);
    for (let node = 0; node < payload.nodes.length; node += 1) {
      if (!Number.isFinite(layout.x[node]) || !Number.isFinite(layout.y[node])) {
        throw new Error(`Corpus radial layout produced a non-finite coordinate at ${node}.`);
      }
    }
    const dominantRoot = payload.nodes[layout.dominantRoot];
    if (!dominantRoot || dominantRoot[1] !== "King") {
      throw new Error(
        `Expected King as the dominant corpus root; got ${dominantRoot?.[1] || "none"}.`,
      );
    }
    const model = this.makeCorpusRadialModel(payload, layout);
    this.corpusRadialModel = model;
    el.showAllControl.hidden = false;
    el.showAll.checked = false;
    const graph = new MultiDirectedGraph();
    graph.setAttribute("corpusRadial", true);
    graph.setAttribute("uniformScaling", true);
    graph.setAttribute("hardCollisionFree", layout.overlapCount === 0);
    graph.setAttribute("layoutHash", layout.layoutHash);
    this.graph = graph;
    this.graphBBox = {
      x: [layout.bounds[0], layout.bounds[2]],
      y: [layout.bounds[1], layout.bounds[3]],
    };
    this.addCorpusRadialNode(graph, model, layout.dominantRoot);
    model.admitted.add(layout.dominantRoot);

    this.renderer = new Sigma(graph, el.container, {
      edgeProgramClasses: {
        arrow: LargeArrowProgram,
        line: EdgeLineProgram,
      },
      defaultEdgeType: "arrow",
      enableEdgeEvents: true,
      hideEdgesOnMove: false,
      hideLabelsOnMove: true,
      renderLabels: false,
      renderEdgeLabels: false,
      defaultDrawNodeHover: () => undefined,
      minEdgeThickness: 0.45,
      stagePadding: 28,
      zIndex: true,
      minCameraRatio: 0.0000001,
      maxCameraRatio: 10000000,
      zoomToSizeRatioFunction: () => 1,
      cameraPanBoundaries: null,
      nodeReducer: (node, data) => this.reduceNode(node, data),
      edgeReducer: (edge, data) => this.reduceEdge(edge, data),
    });
    this.renderer.setCustomBBox(this.graphBBox);
    this.corpusRadialOverlayCanvas = this.renderer.createCanvas(
      "corpusRadialLabels",
      {
        beforeLayer: "mouse",
        style: { pointerEvents: "none" },
      },
    );
    this.renderer.on("afterRender", () => this.drawCorpusRadialOverlay());
    this.bindRendererEvents();
    this.bindCorpusRadialDisclosure();
    this.updateHeader();
    this.renderFocusDetail();
    el.scope.value = "corpus";

    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => {
        if (!this.renderer || this.corpusRadialModel !== model) {
          resolve();
          return;
        }
        this.renderer.resize(true);
        this.fitGraph(false);
        requestAnimationFrame(() => {
          if (this.corpusRadialModel === model) {
            this.updateCorpusRadialDisclosure(true);
            el.container.style.visibility = "";
            this.setLoading(false);
          }
          resolve();
        });
      });
    });
  }

  private thematicEntityNodeId(categoryId: string): string {
    return `entity-category:${categoryId}`;
  }

  private thematicRelationNodeId(categoryId: string): string {
    return `relation-category:${categoryId}`;
  }

  private thematicRelationIndex(node: string): number | null {
    if (!this.thematicPayload || !node.startsWith("relation-category:")) {
      return null;
    }
    const categoryId = node.slice("relation-category:".length);
    const index = this.thematicPayload.relationCategories.findIndex(
      (category) => category.id === categoryId,
    );
    return index >= 0 ? index : null;
  }

  private thematicPatternsForCombination(
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
  }

  /**
   * Record which categories a selection corresponds to.
   *
   * The controls are deliberately not written here. commitThematicSelection()
   * renders them from the state record, so a navigation change cannot leave
   * one surface updated and another stale.
   */
  private syncThematicCategoryControls(
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
  }

  private thematicEntitySize(
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
    return (
      minimumRadius +
      sizeRatio * (maximumRadius - minimumRadius)
    );
  }

  private runThematicLayoutWorker(
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
          if (
            response.kind !== "layout" ||
            response.requestId !== requestId
          ) return;
          finish();
          if (requestId !== this.thematicLayoutRequest) {
            reject(new DOMException("Superseded thematic layout", "AbortError"));
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
      worker.postMessage(
        { kind: "layout", requestId, input },
        [
          frequencies.buffer,
          connectivity.buffer,
          radii.buffer,
        ],
      );
    });
  }

  private makeThematicGraph(
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
        size: this.thematicEntitySize(
          category,
          minimumMention,
          maximumMention,
        ),
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
      graph.addDirectedEdgeWithKey(
        `pattern:${patternIndex}`,
        source,
        target,
        {
          ...common,
          sourceId: source,
          targetId: target,
        },
      );
    });
    return graph;
  }

  private isSemanticTagSlice(): boolean {
    return new Set(["tag-filter", "filtered-tags"]).has(
      this.thematicPayload?.focus.kind || "",
    );
  }

  private semanticTagSlicePatterns(): number[] {
    if (!this.isSemanticTagSlice() || !this.thematicPayload) return [];
    return this.thematicPayload.patterns.flatMap((pattern, index) =>
      pattern[4] > 0 ? [index] : [],
    );
  }

  private activateSemanticTagUnion(): boolean {
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
  }

  private async installThematicPayload(
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
      const activePoints = payload.entityCategories.flatMap((category, index) =>
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
    this.thematicOverlayCanvas = this.renderer.createCanvas(
      "thematicLabels",
      {
        beforeLayer: "mouse",
        style: { pointerEvents: "none" },
      },
    );
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
      Number(graph.getNodeAttribute(this.thematicEntityNodeId(category.id), "size")),
    );
    const minimumEntitySize = Math.min(...entitySizes);
    const maximumEntitySize = Math.max(...entitySizes);
    this.thematicOverlayCanvas.dataset.minimumEntitySize =
      minimumEntitySize.toFixed(3);
    this.thematicOverlayCanvas.dataset.maximumEntitySize =
      maximumEntitySize.toFixed(3);
    this.thematicOverlayCanvas.dataset.entitySizeRatio =
      (maximumEntitySize / minimumEntitySize).toFixed(3);
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
    }
    else if (new Set(["corpus", "vol1", "vol2", "vol3"]).has(focusKind)) {
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
          if (
            this.renderer &&
            this.graph &&
            this.thematicPayload === payload
          ) {
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
  }

  private wrapThematicLabel(
    context: CanvasRenderingContext2D,
    label: string,
    maximumWidth: number,
    maximumLines: number,
  ): string[] | null {
    if (maximumLines < 1 || maximumWidth <= 0) return null;
    const words = label
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replace(/[_-]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .split(" ");
    const lines: string[] = [];
    let line = "";
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (context.measureText(candidate).width <= maximumWidth) {
        line = candidate;
        continue;
      }
      if (line) {
        lines.push(line);
        line = "";
      }
      if (context.measureText(word).width <= maximumWidth) {
        line = word;
        continue;
      }
      let fragment = "";
      for (const character of word) {
        const next = fragment + character;
        if (
          fragment &&
          context.measureText(next).width > maximumWidth
        ) {
          lines.push(fragment);
          fragment = character;
        } else {
          fragment = next;
        }
      }
      line = fragment;
      if (lines.length > maximumLines) return null;
    }
    if (line) lines.push(line);
    return lines.length <= maximumLines ? lines : null;
  }

  private drawThematicDirectEdges(
    context: CanvasRenderingContext2D,
    width: number,
    height: number,
  ): Array<{
    kind: "rectangle";
    x: number;
    y: number;
    halfWidth: number;
    halfHeight: number;
  }> {
    if (!this.renderer || !this.graph || !this.thematicPayload) return [];
    this.thematicRelationLabelHits = [];
    const canvas = this.thematicOverlayCanvas;
    const hoveredRelationIndex = this.thematicHoveredRelationLabel
      ? this.thematicRelationIndex(this.thematicHoveredRelationLabel)
      : null;
    const hoveredPattern = this.thematicHoveredPattern;
    if (!this.thematicSelectionCandidates.length) {
      if (canvas) {
        canvas.dataset.visibleDirectEdgeLabels = "0";
        canvas.dataset.hiddenDirectEdgeLabels = "0";
        canvas.dataset.outgoingPaths = "0";
        canvas.dataset.incomingPaths = "0";
        canvas.dataset.coincidentPaths = "0";
        canvas.dataset.foregroundRelationTags = "0";
        canvas.dataset.ghostRelationTags = String(
          this.thematicPayload.relationCategories.length,
        );
      }
      return [];
    }

    type DirectPath = {
      patternIndex: number;
      relationIndex: number;
      pairKey: string;
      count: number;
      startX: number;
      startY: number;
      controlX: number;
      controlY: number;
      endX: number;
      endY: number;
      labelX: number;
      labelY: number;
      color: string;
    };
    const groups = new Map<string, number[]>();
    for (const patternIndex of this.thematicSelectedPatterns) {
      const pattern = this.thematicPayload.patterns[patternIndex];
      if (!pattern) continue;
      // Keyed on the unordered pair, matching parallelEdgeOffsets() in
      // collision_labels.ts. Keying on the written subject/object order split
      // one pairing into two groups that then laid themselves out
      // independently on a shared midpoint.
      const key = pattern[0] <= pattern[2]
        ? `${pattern[0]}:${pattern[2]}`
        : `${pattern[2]}:${pattern[0]}`;
      const group = groups.get(key) || [];
      group.push(patternIndex);
      groups.set(key, group);
    }
    const paths: DirectPath[] = [];
    const quadraticPoint = (
      start: number,
      control: number,
      end: number,
      t: number,
    ): number => {
      const inverse = 1 - t;
      return (
        inverse * inverse * start +
        2 * inverse * t * control +
        t * t * end
      );
    };
    for (const [pairKey, group] of groups) {
      group.sort((left, right) => {
        const leftPattern = this.thematicPayload?.patterns[left];
        const rightPattern = this.thematicPayload?.patterns[right];
        if (!leftPattern || !rightPattern) return left - right;
        return (
          leftPattern[1] - rightPattern[1] ||
          rightPattern[4] - leftPattern[4] ||
          left - right
        );
      });
      group.forEach((patternIndex, index) => {
        const pattern = this.thematicPayload?.patterns[patternIndex];
        if (!pattern) return;
        const sourceNode = this.thematicEntityNodeId(
          this.thematicPayload?.entityCategories[pattern[0]].id || "",
        );
        const targetNode = this.thematicEntityNodeId(
          this.thematicPayload?.entityCategories[pattern[2]].id || "",
        );
        if (!this.graph?.hasNode(sourceNode) || !this.graph.hasNode(targetNode)) {
          return;
        }
        const sourceAttributes = this.graph.getNodeAttributes(sourceNode);
        const targetAttributes = this.graph.getNodeAttributes(targetNode);
        const source = this.renderer?.graphToViewport({
          x: Number(sourceAttributes.x),
          y: Number(sourceAttributes.y),
        });
        const target = this.renderer?.graphToViewport({
          x: Number(targetAttributes.x),
          y: Number(targetAttributes.y),
        });
        if (!source || !target) return;
        const sourceRadius = this.renderer?.scaleSize(
          Number(sourceAttributes.size),
        ) || 0;
        const targetRadius = this.renderer?.scaleSize(
          Number(targetAttributes.size),
        ) || 0;
        const centeredIndex = index - (group.length - 1) / 2;
        let startX = source.x;
        let startY = source.y;
        let endX = target.x;
        let endY = target.y;
        let controlX = (source.x + target.x) / 2;
        let controlY = (source.y + target.y) / 2;
        if (sourceNode === targetNode) {
          const angle =
            pattern[1] * 2.399963 + centeredIndex * 0.19;
          const spread = sourceRadius + 24 + Math.abs(centeredIndex) * 2.8;
          startX = source.x + Math.cos(angle - 0.46) * (sourceRadius + 1);
          startY = source.y + Math.sin(angle - 0.46) * (sourceRadius + 1);
          endX = source.x + Math.cos(angle + 0.46) * (sourceRadius + 1);
          endY = source.y + Math.sin(angle + 0.46) * (sourceRadius + 1);
          controlX = source.x + Math.cos(angle) * spread * 2;
          controlY = source.y + Math.sin(angle) * spread * 2;
        } else {
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const distance = Math.max(1, Math.hypot(dx, dy));
          const unitX = dx / distance;
          const unitY = dy / distance;
          startX += unitX * (sourceRadius + 2);
          startY += unitY * (sourceRadius + 2);
          endX -= unitX * (targetRadius + 7);
          endY -= unitY * (targetRadius + 7);
          const curveStep = Math.max(
            4.5,
            Math.min(9, 150 / Math.max(1, group.length)),
          );
          // Same canonicalDirection trick parallelEdgeOffsets() uses. The
          // perpendicular comes from this claim's own subject-to-object
          // vector, which reverses for the opposite orientation, so without
          // the sign a forward and a reverse path with mirrored offsets
          // resolve to one control point and trace the same curve twice.
          const canonicalSide = pattern[0] <= pattern[2] ? 1 : -1;
          const offset = centeredIndex * curveStep * canonicalSide;
          controlX = (startX + endX) / 2 - unitY * offset;
          controlY = (startY + endY) / 2 + unitX * offset;
        }
        let labelX: number;
        let labelY: number;
        if (
          this.thematicPairEntity &&
          sourceNode !== targetNode &&
          group.length > 1
        ) {
          const columns = Math.min(
            8,
            Math.ceil(Math.sqrt(group.length * 1.4)),
          );
          const rows = Math.ceil(group.length / columns);
          const column = index % columns;
          const row = Math.floor(index / columns);
          // Centre the grid on the two node centres rather than on this
          // path's own trimmed endpoints. The trim differs per orientation
          // (and with each node's radius), so deriving the centre per path
          // gave the pairing's two orientations two offset grids.
          const gridCenterX = (source.x + target.x) / 2;
          const gridCenterY = (source.y + target.y) / 2;
          labelX = gridCenterX + (column - (columns - 1) / 2) * 34;
          labelY = gridCenterY + (row - (rows - 1) / 2) * 19;
          controlX = labelX * 2 - (startX + endX) / 2;
          controlY = labelY * 2 - (startY + endY) / 2;
        } else {
          const labelT =
            group.length === 1
              ? 0.5
              : 0.25 + 0.5 * ((index + 0.5) / group.length);
          labelX = quadraticPoint(startX, controlX, endX, labelT);
          labelY = quadraticPoint(startY, controlY, endY, labelT);
        }
        // Orientation wins over the pair highlight. Flat orange for every
        // paired edge threw away the one thing the reader is looking for once
        // both slots are filled: which side is acting on which.
        const anchors = this.thematicOrientationAnchors;
        const color = anchors.has(pattern[0])
          ? "#8c45b5"
          : anchors.has(pattern[2])
            ? "#168f91"
            : this.thematicPairEntity
              ? "#d9792b"
              : this.thematicPrimaryEntity === sourceNode
                ? "#8c45b5"
                : this.thematicPrimaryEntity === targetNode
                  ? "#168f91"
                  : "#70419a";
        paths.push({
          patternIndex,
          relationIndex: pattern[1],
          pairKey,
          count: pattern[4],
          startX,
          startY,
          controlX,
          controlY,
          endX,
          endY,
          labelX,
          labelY,
          color,
        });
      });
    }

    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";
    for (const path of paths) {
      const relationHovered = hoveredPattern === path.patternIndex;
      const anotherRelationHovered =
        hoveredPattern !== null && !relationHovered;
      context.beginPath();
      context.moveTo(path.startX, path.startY);
      context.quadraticCurveTo(
        path.controlX,
        path.controlY,
        path.endX,
        path.endY,
      );
      context.globalAlpha = relationHovered
        ? 0.98
        : anotherRelationHovered
          ? 0.14
          : 0.58;
      context.strokeStyle = path.color;
      context.lineWidth =
        Math.min(3.2, 0.75 + Math.log1p(path.count) * 0.32) +
        (relationHovered ? 1.6 : 0);
      context.stroke();
      const angle = Math.atan2(
        path.endY - path.controlY,
        path.endX - path.controlX,
      );
      context.beginPath();
      context.moveTo(path.endX, path.endY);
      context.lineTo(
        path.endX - Math.cos(angle - 0.48) * 7,
        path.endY - Math.sin(angle - 0.48) * 7,
      );
      context.lineTo(
        path.endX - Math.cos(angle + 0.48) * 7,
        path.endY - Math.sin(angle + 0.48) * 7,
      );
      context.closePath();
      context.fillStyle = path.color;
      context.fill();
    }
    context.globalAlpha = 1;

    const labelHalfWidth = 15;
    const labelHalfHeight = 7.5;
    const occupied: Array<[number, number, number, number]> = [];
    for (const category of this.thematicPayload.entityCategories) {
      if (this.isSemanticTagSlice() && category.activeMentionCount <= 0) {
        continue;
      }
      const node = this.thematicEntityNodeId(category.id);
      if (!this.graph.hasNode(node)) continue;
      const attributes = this.graph.getNodeAttributes(node);
      const point = this.renderer.graphToViewport({
        x: Number(attributes.x),
        y: Number(attributes.y),
      });
      const radius = this.renderer.scaleSize(Number(attributes.size)) + 2;
      occupied.push([
        point.x - radius,
        point.y - radius,
        point.x + radius,
        point.y + radius,
      ]);
    }
    const selectedRelationFocus = this.selectedNode
      ? this.thematicRelationIndex(this.selectedNode)
      : null;
    const forceAllDirectLabels =
      Boolean(this.thematicPairEntity) ||
      this.thematicContextRelation !== null ||
      selectedRelationFocus !== null;
    const labels = paths
      .slice()
      .sort(
        (left, right) =>
          right.count - left.count ||
          left.relationIndex - right.relationIndex ||
          left.patternIndex - right.patternIndex,
      );
    const footprints: Array<{
      kind: "rectangle";
      x: number;
      y: number;
      halfWidth: number;
      halfHeight: number;
    }> = [];
    const visibleRelations = new Set<number>();
    let hoveredAnchor: DirectPath | null = null;
    for (const path of labels) {
      let box: [number, number, number, number] = [
        path.labelX - labelHalfWidth,
        path.labelY - labelHalfHeight,
        path.labelX + labelHalfWidth,
        path.labelY + labelHalfHeight,
      ];
      if (
        box[2] < 0 ||
        box[0] > width ||
        box[3] < 0 ||
        box[1] > height
      ) continue;
      let collision = occupied.some(
        (other) =>
          box[0] < other[2] + 2 &&
          box[2] > other[0] - 2 &&
          box[1] < other[3] + 2 &&
          box[3] > other[1] - 2,
      );
      if (
        collision &&
        forceAllDirectLabels &&
        !this.thematicPairEntity
      ) {
        for (const alternativeT of [0.35, 0.65, 0.2, 0.8, 0.1, 0.9]) {
          const alternativeX = quadraticPoint(
            path.startX,
            path.controlX,
            path.endX,
            alternativeT,
          );
          const alternativeY = quadraticPoint(
            path.startY,
            path.controlY,
            path.endY,
            alternativeT,
          );
          const alternativeBox: [number, number, number, number] = [
            alternativeX - labelHalfWidth,
            alternativeY - labelHalfHeight,
            alternativeX + labelHalfWidth,
            alternativeY + labelHalfHeight,
          ];
          const alternativeCollision = occupied.some(
            (other) =>
              alternativeBox[0] < other[2] + 2 &&
              alternativeBox[2] > other[0] - 2 &&
              alternativeBox[1] < other[3] + 2 &&
              alternativeBox[3] > other[1] - 2,
          );
          if (alternativeCollision) continue;
          path.labelX = alternativeX;
          path.labelY = alternativeY;
          box = alternativeBox;
          collision = false;
          break;
        }
      }
      if (collision && !forceAllDirectLabels) continue;
      occupied.push(box);
      const relation = this.thematicPayload.relationCategories[path.relationIndex];
      if (!relation) continue;
      const relationHovered = hoveredPattern === path.patternIndex;
      if (
        relationHovered &&
        (!hoveredAnchor ||
          !this.thematicHoveredRelationPoint ||
          Math.hypot(
            path.labelX - this.thematicHoveredRelationPoint.x,
            path.labelY - this.thematicHoveredRelationPoint.y,
          ) <
            Math.hypot(
              hoveredAnchor.labelX - this.thematicHoveredRelationPoint.x,
              hoveredAnchor.labelY - this.thematicHoveredRelationPoint.y,
            ))
      ) {
        hoveredAnchor = path;
      }
      context.beginPath();
      context.roundRect(
        box[0],
        box[1],
        labelHalfWidth * 2,
        labelHalfHeight * 2,
        7,
      );
      context.fillStyle = relationHovered ? "#9a4fc4" : "#6a3290";
      context.fill();
      context.lineWidth = relationHovered ? 2.2 : 1.1;
      context.strokeStyle = relationHovered
        ? "rgba(255, 222, 112, 1)"
        : "rgba(244, 201, 93, 0.9)";
      context.stroke();
      context.font = "800 8px Inter, Segoe UI, sans-serif";
      context.fillStyle = "#ffffff";
      context.fillText(relation.tagId, path.labelX, path.labelY + 0.5);
      const relationNode = this.thematicRelationNodeId(relation.id);
      this.thematicRelationLabelHits.push({
        node: relationNode,
        patternIndex: path.patternIndex,
        x: path.labelX,
        y: path.labelY,
        halfWidth: labelHalfWidth,
        halfHeight: labelHalfHeight,
      });
      footprints.push({
        kind: "rectangle",
        x: path.labelX,
        y: path.labelY,
        halfWidth: labelHalfWidth,
        halfHeight: labelHalfHeight,
      });
      visibleRelations.add(path.relationIndex);
      if (canvas && !canvas.dataset.firstRelationLabelX) {
        canvas.dataset.firstRelationLabelId = relation.id;
        canvas.dataset.firstRelationLabelX = path.labelX.toFixed(2);
        canvas.dataset.firstRelationLabelY = path.labelY.toFixed(2);
      }
      if (relation.id === "R01" && canvas && !canvas.dataset.r01X) {
        canvas.dataset.r01X = path.labelX.toFixed(2);
        canvas.dataset.r01Y = path.labelY.toFixed(2);
      }
    }
    if (hoveredAnchor && hoveredRelationIndex !== null) {
      const relation =
        this.thematicPayload.relationCategories[hoveredRelationIndex];
      if (relation) {
        const tooltip = `${relation.tagId} · ${relation.label}`;
        context.font = "650 11px Inter, Segoe UI, sans-serif";
        const tooltipWidth = Math.min(
          380,
          Math.max(120, context.measureText(tooltip).width + 20),
        );
        const tooltipHeight = 25;
        const tooltipLeft = Math.max(
          6,
          Math.min(width - tooltipWidth - 6, hoveredAnchor.labelX - tooltipWidth / 2),
        );
        const preferredTop =
          hoveredAnchor.labelY - labelHalfHeight - tooltipHeight - 7;
        const tooltipTop =
          preferredTop >= 6
            ? preferredTop
            : hoveredAnchor.labelY + labelHalfHeight + 7;
        context.beginPath();
        context.roundRect(
          tooltipLeft,
          tooltipTop,
          tooltipWidth,
          tooltipHeight,
          6,
        );
        context.fillStyle = "rgba(29, 22, 34, 0.96)";
        context.fill();
        context.lineWidth = 1.2;
        context.strokeStyle = "rgba(154, 79, 196, 0.95)";
        context.stroke();
        context.fillStyle = "#ffffff";
        context.textAlign = "left";
        context.fillText(
          tooltip,
          tooltipLeft + 10,
          tooltipTop + tooltipHeight / 2 + 0.5,
          tooltipWidth - 20,
        );
        context.textAlign = "center";
      }
    }
    context.restore();
    if (canvas) {
      canvas.dataset.visibleDirectEdgeLabels = String(footprints.length);
      canvas.dataset.hiddenDirectEdgeLabels = String(
        Math.max(0, paths.length - footprints.length),
      );
      // Curves of one pairing that still resolve to the same control point,
      // i.e. that would be drawn one on top of the other.
      const byPair = new Map<string, DirectPath[]>();
      for (const path of paths) {
        const list = byPair.get(path.pairKey);
        if (list) list.push(path);
        else byPair.set(path.pairKey, [path]);
      }
      let coincidentPaths = 0;
      for (const list of byPair.values()) {
        for (let left = 0; left < list.length; left += 1) {
          for (let right = left + 1; right < list.length; right += 1) {
            if (
              Math.hypot(
                list[left].controlX - list[right].controlX,
                list[left].controlY - list[right].controlY,
              ) < 1.5
            ) {
              coincidentPaths += 1;
            }
          }
        }
      }
      canvas.dataset.coincidentPaths = String(coincidentPaths);
      // Label anchor points, so a test can hover a specific direction of a
      // relation rather than whichever pill happens to be first.
      // Entity circle geometry, so a test can aim at a pill that really does
      // sit on a node and prove the click resolves to the pill.
      canvas.dataset.entityCircles = this.thematicPayload.entityCategories
        .flatMap((category) => {
          const node = this.thematicEntityNodeId(category.id);
          if (!this.graph?.hasNode(node)) return [];
          const attributes = this.graph.getNodeAttributes(node);
          const point = this.renderer?.graphToViewport({
            x: Number(attributes.x),
            y: Number(attributes.y),
          });
          if (!point) return [];
          const radius = this.renderer?.scaleSize(Number(attributes.size)) || 0;
          return [
            `${point.x.toFixed(1)},${point.y.toFixed(1)},${radius.toFixed(1)}`,
          ];
        })
        .join(";");
      canvas.dataset.relationLabelPoints = this.thematicRelationLabelHits
        .slice(0, 80)
        .map((hit) => `${hit.patternIndex},${hit.x.toFixed(1)},${hit.y.toFixed(1)}`)
        .join(";");
      // Direction of each drawn path, by the colour it was given.
      canvas.dataset.outgoingPaths = String(
        paths.filter((path) => path.color === "#8c45b5").length,
      );
      canvas.dataset.incomingPaths = String(
        paths.filter((path) => path.color === "#168f91").length,
      );
      canvas.dataset.foregroundRelationTags = String(visibleRelations.size);
      canvas.dataset.ghostRelationTags = String(
        this.thematicPayload.relationCategories.length -
          visibleRelations.size,
      );
    }
    return footprints;
  }

  private drawThematicOverlay(): void {
    if (
      !this.renderer ||
      !this.graph ||
      !this.thematicPayload ||
      !this.thematicOverlayCanvas
    ) return;
    const canvas = this.thematicOverlayCanvas;
    const context = canvas.getContext("2d");
    if (!context) return;
    const { width, height } = this.renderer.getDimensions();
    const dpr = window.devicePixelRatio || 1;
    const targetWidth = Math.max(1, Math.round(width * dpr));
    const targetHeight = Math.max(1, Math.round(height * dpr));
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    context.textAlign = "center";
    context.textBaseline = "middle";
    delete canvas.dataset.r01X;
    delete canvas.dataset.r01Y;
    delete canvas.dataset.firstRelationLabelId;
    delete canvas.dataset.firstRelationLabelX;
    delete canvas.dataset.firstRelationLabelY;
    const footprints: Array<
      | { kind: "circle"; x: number; y: number; radius: number }
      | { kind: "rectangle"; x: number; y: number; halfWidth: number; halfHeight: number }
    > = this.drawThematicDirectEdges(context, width, height);
    let visibleEntityTags = 0;
    let fullEntityLabels = 0;

    for (const category of this.thematicPayload.entityCategories) {
      if (this.isSemanticTagSlice() && category.activeMentionCount <= 0) {
        continue;
      }
      const node = this.thematicEntityNodeId(category.id);
      if (!this.graph.hasNode(node)) continue;
      const renderedRadius = this.renderer.scaleSize(
        Number(this.graph.getNodeAttribute(node, "size")),
      );
      const point = this.renderer.graphToViewport({
        x: Number(this.graph.getNodeAttribute(node, "x")),
        y: Number(this.graph.getNodeAttribute(node, "y")),
      });
      if (category.id === "E01" || category.id === "E24") {
        const key = category.id.toLowerCase();
        canvas.dataset[`${key}X`] = point.x.toFixed(2);
        canvas.dataset[`${key}Y`] = point.y.toFixed(2);
      }
      if (
        point.x + renderedRadius < 0 ||
        point.y + renderedRadius < 0 ||
        point.x - renderedRadius > width ||
        point.y - renderedRadius > height
      ) continue;
      const selected =
        node === this.selectedNode ||
        node === this.thematicPrimaryEntity ||
        node === this.thematicPairEntity;
      context.font = "800 9px Inter, Segoe UI, sans-serif";
      const tagFits =
        context.measureText(category.tagId).width + 5 <= renderedRadius * 2;
      if (tagFits) {
        context.font = "600 7px Inter, Segoe UI, sans-serif";
        const labelLines = this.wrapThematicLabel(
          context,
          category.label,
          renderedRadius * 1.55,
          Math.max(
            0,
            Math.floor((renderedRadius * 1.45 - 12) / 8),
          ),
        );
        if (labelLines) {
          const totalHeight = 10 + labelLines.length * 8;
          let lineY = point.y - totalHeight / 2 + 5;
          context.font = "800 9px Inter, Segoe UI, sans-serif";
          context.fillStyle = selected ? "#ffffff" : "#f6fffc";
          context.fillText(category.tagId, point.x, lineY);
          lineY += 10;
          context.font = "600 7px Inter, Segoe UI, sans-serif";
          context.fillStyle = selected ? "#f3eaff" : "#dff8ef";
          for (const line of labelLines) {
            context.fillText(line, point.x, lineY);
            lineY += 8;
          }
          fullEntityLabels += 1;
        } else {
          context.font = "800 9px Inter, Segoe UI, sans-serif";
          context.fillStyle = selected ? "#ffffff" : "#f6fffc";
          context.fillText(category.tagId, point.x, point.y);
        }
        visibleEntityTags += 1;
      }
      footprints.push({
        kind: "circle",
        x: point.x,
        y: point.y,
        radius: renderedRadius,
      });
    }

    let visualOverlaps = 0;
    let entityEntityOverlaps = 0;
    let entityRelationOverlaps = 0;
    let relationRelationOverlaps = 0;
    const clearance = 0.25;
    for (let left = 0; left < footprints.length; left += 1) {
      for (let right = left + 1; right < footprints.length; right += 1) {
        const first = footprints[left];
        const second = footprints[right];
        if (first.kind === "circle" && second.kind === "circle") {
          if (
            Math.hypot(first.x - second.x, first.y - second.y) <
            first.radius + second.radius + clearance
          ) {
            visualOverlaps += 1;
            entityEntityOverlaps += 1;
          }
          continue;
        }
        if (first.kind === "rectangle" && second.kind === "rectangle") {
          if (
            Math.abs(first.x - second.x) <
              first.halfWidth + second.halfWidth + clearance &&
            Math.abs(first.y - second.y) <
              first.halfHeight + second.halfHeight + clearance
          ) {
            visualOverlaps += 1;
            relationRelationOverlaps += 1;
          }
          continue;
        }
        const circle = first.kind === "circle" ? first : second;
        const rectangle = first.kind === "rectangle" ? first : second;
        const dx = Math.max(
          Math.abs(circle.x - rectangle.x) - rectangle.halfWidth,
          0,
        );
        const dy = Math.max(
          Math.abs(circle.y - rectangle.y) - rectangle.halfHeight,
          0,
        );
        if (
          Math.hypot(dx, dy) <
          circle.radius + clearance
        ) {
          visualOverlaps += 1;
          entityRelationOverlaps += 1;
        }
      }
    }
    canvas.dataset.visibleTags = String(footprints.length);
    canvas.dataset.visibleEntityTags = String(visibleEntityTags);
    canvas.dataset.fullEntityLabels = String(fullEntityLabels);
    canvas.dataset.visualOverlaps = String(visualOverlaps);
    canvas.dataset.entityEntityOverlaps = String(entityEntityOverlaps);
    canvas.dataset.entityRelationOverlaps = String(entityRelationOverlaps);
    canvas.dataset.relationRelationOverlaps = String(
      relationRelationOverlaps,
    );

    const detailNode = this.hoveredNode || this.selectedNode;
    if (detailNode && this.graph.hasNode(detailNode)) {
      const data = this.graph.getNodeAttributes(detailNode);
      if (data.thematicKind === "relation") return;
      const category = data.categoryDetail as CategoryDescriptor | null;
      if (category) {
        const point = this.renderer.graphToViewport({
          x: Number(data.x),
          y: Number(data.y),
        });
        context.font = "600 11px Inter, Segoe UI, sans-serif";
        const label = `${category.tagId} · ${category.label}`;
        const labelWidth = Math.min(320, context.measureText(label).width + 18);
        const left = Math.min(
          width - labelWidth - 6,
          Math.max(6, point.x - labelWidth / 2),
        );
        const top = Math.min(height - 29, Math.max(6, point.y + 21));
        context.beginPath();
        context.roundRect(left, top, labelWidth, 23, 6);
        context.fillStyle = "rgba(26, 29, 36, 0.92)";
        context.fill();
        context.textAlign = "left";
        context.fillStyle = "#ffffff";
        context.fillText(label, left + 9, top + 12);
        context.textAlign = "center";
      }
    }
  }

  private thematicDirection(): ThematicDirection {
    const direction = el.themeDirection.value;
    return direction === "ab" || direction === "ba" ? direction : "either";
  }

  private filteredThematicPatterns(
    patternIndices: Iterable<number>,
  ): number[] {
    if (!this.thematicPayload) return [];
    const minimumInstances = Math.max(
      1,
      Number(el.themeGranularity.value) || 1,
    );
    const direction = this.thematicDirection();
    const primaryIndex =
      this.thematicPrimaryEntity && this.graph
        ? Number(
            this.graph.getNodeAttribute(
              this.thematicPrimaryEntity,
              "categoryIndex",
            ),
          )
        : null;
    return Array.from(patternIndices).filter((patternIndex) => {
      const pattern = this.thematicPayload?.patterns[patternIndex];
      if (!pattern || pattern[4] < minimumInstances) return false;
      if (this.thematicExactPatternFocus !== null) return true;
      // A group selection already encoded the pairing direction when it built
      // its candidates; re-applying it here relative to one primary entity
      // would drop the reverse orientation again.
      if (this.thematicGroupSelection) return true;
      if (primaryIndex === null || direction === "either") return true;
      return direction === "ab"
        ? pattern[0] === primaryIndex
        : pattern[2] === primaryIndex;
    });
  }

  refreshThematicExplorationFilters(): void {
    el.themeGranularityValue.value = `${el.themeGranularity.value}+`;
    if (this.thematicOverlayCanvas) {
      this.thematicOverlayCanvas.dataset.minimumPatternInstances =
        el.themeGranularity.value;
      this.thematicOverlayCanvas.dataset.direction = this.thematicDirection();
    }
    if (
      !this.thematicPayload ||
      !this.selectedNode ||
      !this.thematicSelectionCandidates.length
    ) return;
    this.setThematicSelection(
      this.filteredThematicPatterns(this.thematicSelectionCandidates),
    );
    // Changing the threshold or the pairing direction changes what the graph
    // shows, so it travels the same publish path as any other change.
    this.commitThematicSelection(performance.now());
  }

  private setThematicSelection(patternIndices: Iterable<number>): void {
    const payload = this.thematicPayload;
    if (!payload) return;
    this.thematicSelectedPatterns = new Set(patternIndices);
    this.selectedNeighbors.clear();
    const relationNodes = new Set<number>();
    const counterpartNodes = new Set<number>();
    const primaryIndex =
      this.thematicPrimaryEntity && this.graph
        ? Number(
            this.graph.getNodeAttribute(
              this.thematicPrimaryEntity,
              "categoryIndex",
            ),
          )
        : null;
    for (const patternIndex of this.thematicSelectedPatterns) {
      const pattern = payload.patterns[patternIndex];
      if (!pattern) continue;
      this.selectedNeighbors.add(
        this.thematicEntityNodeId(
          payload.entityCategories[pattern[0]].id,
        ),
      );
      this.selectedNeighbors.add(
        this.thematicEntityNodeId(
          payload.entityCategories[pattern[2]].id,
        ),
      );
      relationNodes.add(pattern[1]);
      if (primaryIndex !== null) {
        if (pattern[0] === primaryIndex) counterpartNodes.add(pattern[2]);
        if (pattern[2] === primaryIndex) counterpartNodes.add(pattern[0]);
      }
    }
    if (this.thematicOverlayCanvas) {
      this.thematicOverlayCanvas.dataset.selectedPatterns = String(
        this.thematicSelectedPatterns.size,
      );
      this.thematicOverlayCanvas.dataset.selectedRelations = String(
        relationNodes.size,
      );
      this.thematicOverlayCanvas.dataset.selectedCounterparts = String(
        counterpartNodes.size,
      );
      this.thematicOverlayCanvas.dataset.availableSelectionPatterns = String(
        this.thematicSelectionCandidates.length,
      );
    }
  }

  private appendThematicPatternList(patternIndices: Iterable<number>): void {
    if (!this.thematicPayload) return;
    const indices = Array.from(patternIndices).sort((left, right) => {
      const leftPattern = this.thematicPayload?.patterns[left];
      const rightPattern = this.thematicPayload?.patterns[right];
      if (!leftPattern || !rightPattern) return left - right;
      return (
        rightPattern[3] - leftPattern[3] ||
        leftPattern[0] - rightPattern[0] ||
        leftPattern[1] - rightPattern[1] ||
        leftPattern[2] - rightPattern[2] ||
        left - right
      );
    });
    if (!indices.length) {
      el.detail.appendChild(
        textElement(
          "div",
          "graph-empty",
          `No exact pattern meets the ${el.themeGranularity.value}+ instance and ${this.thematicDirection()} direction filters.`,
        ),
      );
      return;
    }
    const select = document.createElement("select");
    select.id = "graph-thematic-patterns";
    select.className = "graph-thematic-patterns";
    select.size = Math.min(12, Math.max(4, indices.length));
    select.setAttribute("aria-label", "Foregrounded exact thematic patterns");
    for (const patternIndex of indices) {
      const pattern = this.thematicPayload.patterns[patternIndex];
      if (!pattern) continue;
      const source = this.thematicPayload.entityCategories[pattern[0]];
      const relation = this.thematicPayload.relationCategories[pattern[1]];
      const target = this.thematicPayload.entityCategories[pattern[2]];
      const option = document.createElement("option");
      option.value = String(patternIndex);
      option.dataset.patternCount = String(pattern[3]);
      // Always state the active share. Omitting it when everything was
      // active read as "not active at all" on the fully in-view rows.
      option.textContent =
        `${source.tagId} → ${relation.tagId} → ${target.tagId} · ` +
        `${pattern[4].toLocaleString()} of ${pattern[3].toLocaleString()} ` +
        `triples in view`;
      option.title =
        `${source.label} → ${relation.label} → ${target.label}\n` +
        relation.definition;
      select.appendChild(option);
    }
    select.selectedIndex = 0;
    const label = document.createElement("label");
    label.className = "graph-field graph-thematic-pattern-list";
    label.append(
      textElement(
        "span",
        "",
        `${indices.length.toLocaleString()} of ${this.thematicSelectionCandidates.length.toLocaleString()} patterns foregrounded · ${el.themeGranularity.value}+ instances`,
      ),
      select,
    );
    el.detail.appendChild(label);
    el.detail.appendChild(
      actionButton(
        "Inspect selected pattern and evidence",
        "thematic-evidence-selected",
        "",
        "graph-evidence-more",
      ),
    );
  }

  private selectThematicNode(node: string, _pairFocus = false): void {
    const startedAt = performance.now();
    if (!this.graph || !this.thematicPayload) {
      return;
    }
    const relationIndex = this.thematicRelationIndex(node);
    const isEntity = this.graph.hasNode(node);
    if (!isEntity && relationIndex === null) return;
    const data = isEntity ? this.graph.getNodeAttributes(node) : null;
    const kind = isEntity ? String(data?.thematicKind) : "relation";
    const contextRelation = this.thematicContextRelation;
    this.pushNavigationState();
    this.thematicExactPatternFocus = null;
    if (
      kind === "entity" &&
      this.thematicPrimaryEntity &&
      this.thematicPrimaryEntity !== node
    ) {
      const sourceIndex = Number(
        this.graph.getNodeAttribute(
          this.thematicPrimaryEntity,
          "categoryIndex",
        ),
      );
      const targetIndex = Number(data?.categoryIndex);
      const direction = this.thematicDirection();
      const selected: number[] = [];
      this.thematicPayload.patterns.forEach((pattern, index) => {
        if (contextRelation !== null && pattern[1] !== contextRelation) return;
        // Pairing two entities on the canvas honours the same direction rule
        // as the group controls, so a pair whose claims are only ever written
        // the other way round still resolves instead of coming back empty.
        const { forward, reverse } = this.patternOrientations(
          pattern,
          new Set([sourceIndex]),
          new Set([targetIndex]),
          direction,
        );
        if (forward || reverse) selected.push(index);
      });
      this.thematicGroupSelection = true;
      this.thematicOrientationAnchors = new Set([sourceIndex]);
      this.thematicContextRelation = contextRelation;
      this.thematicPairEntity = node;
      this.selectedNode = this.thematicPrimaryEntity;
      this.thematicSelectionCandidates = selected;
      this.setThematicSelection(this.filteredThematicPatterns(selected));
      this.syncThematicCategoryControls(
        [sourceIndex],
        contextRelation === null ? [] : [contextRelation],
        [targetIndex],
      );
    } else {
      this.thematicContextRelation = null;
      this.thematicPairEntity = null;
      this.selectedNode = node;
      // A single node is a plain incidence selection, so the direction control
      // once again means outgoing/incoming relative to it.
      this.thematicGroupSelection = false;
      this.thematicOrientationAnchors = kind === "entity"
        ? new Set([Number(data?.categoryIndex)])
        : new Set<number>();
      if (kind === "entity") this.thematicPrimaryEntity = node;
      else this.thematicPrimaryEntity = null;
      this.thematicSelectionCandidates =
        this.thematicPatternsByNode.get(node) || [];
      this.setThematicSelection(
        this.filteredThematicPatterns(this.thematicSelectionCandidates),
      );
      this.syncThematicCategoryControls(
        kind === "entity" ? [Number(data?.categoryIndex)] : [],
        kind === "relation" && relationIndex !== null ? [relationIndex] : [],
      );
    }
    this.commitThematicSelection(startedAt);
  }

  /**
   * Leave an evidence list and return to the foregrounded pattern set.
   *
   * This steps back out of the layer that focusing a pattern pushed, so the
   * graph and both panels widen with the inspector instead of the inspector
   * alone changing.
   */
  private returnToForegroundedPatterns(): void {
    this.evidenceAbort?.abort();
    this.thematicEvidencePattern = null;
    this.evidenceItems = [];
    this.evidenceHasMore = false;
    this.evidenceNextOffset = null;
    const previous = this.navigationHistory.pop();
    if (previous) {
      this.restoreNavigationState(previous);
      return;
    }
    this.commitThematicSelection(performance.now());
  }

  /**
   * Narrow the whole workspace to one exact pattern, then open its evidence.
   *
   * Inspecting a pattern is a navigation move like any other, so it goes
   * through the state record and commitThematicSelection() rather than only
   * swapping the inspector's contents.
   */
  private focusThematicPattern(patternIndex: number): void {
    const payload = this.thematicPayload;
    const pattern = payload?.patterns[patternIndex];
    if (!payload || !pattern) return;
    const startedAt = performance.now();
    this.pushNavigationState();
    this.thematicExactPatternFocus = patternIndex;
    this.thematicContextRelation = pattern[1];
    this.thematicGroupSelection = true;
    this.thematicOrientationAnchors = new Set([pattern[0]]);
    this.thematicPrimaryEntity = this.thematicEntityNodeId(
      payload.entityCategories[pattern[0]].id,
    );
    this.thematicPairEntity = this.thematicEntityNodeId(
      payload.entityCategories[pattern[2]].id,
    );
    this.selectedNode = this.thematicPrimaryEntity;
    this.thematicSelectionCandidates = [patternIndex];
    this.syncThematicCategoryControls(
      [pattern[0]],
      [pattern[1]],
      [pattern[2]],
    );
    this.setThematicSelection([patternIndex]);
    this.commitThematicSelection(startedAt);
    void this.loadThematicPatternEvidence(patternIndex);
  }

  private selectThematicRelationLabel(
    hit: (typeof this.thematicRelationLabelHits)[number],
  ): void {
    const startedAt = performance.now();
    if (!this.graph || !this.thematicPayload) return;
    const pattern = this.thematicPayload.patterns[hit.patternIndex];
    if (!pattern) return;
    const relationKey = this.thematicRelationNodeId(
      this.thematicPayload.relationCategories[pattern[1]].id,
    );
    const selectedRelation = this.selectedNode
      ? this.thematicRelationIndex(this.selectedNode)
      : null;

    if (
      this.thematicPrimaryEntity &&
      this.graph.hasNode(this.thematicPrimaryEntity)
    ) {
      this.pushNavigationState();
      const entityIndex = Number(
        this.graph.getNodeAttribute(
          this.thematicPrimaryEntity,
          "categoryIndex",
        ),
      );
      const pairIndex =
        this.thematicPairEntity && this.graph.hasNode(this.thematicPairEntity)
          ? Number(
              this.graph.getNodeAttribute(
                this.thematicPairEntity,
                "categoryIndex",
              ),
            )
          : null;
      const entityIndices =
        pairIndex === null ? [entityIndex] : [entityIndex, pairIndex];
      this.thematicContextRelation = pattern[1];
      this.thematicGroupSelection = pairIndex !== null;
      this.thematicExactPatternFocus =
        pairIndex === null ? null : hit.patternIndex;
      this.selectedNode = this.thematicPrimaryEntity;
      this.thematicSelectionCandidates =
        this.thematicPatternsForCombination(
          entityIndices,
          [pattern[1]],
          this.thematicExactPatternFocus,
        );
      this.syncThematicCategoryControls(
        [entityIndex],
        [pattern[1]],
        pairIndex === null ? [] : [pairIndex],
      );
    } else if (selectedRelation !== null) {
      this.pushNavigationState();
      this.thematicPrimaryEntity = null;
      this.thematicPairEntity = null;
      this.thematicContextRelation = null;
      this.thematicExactPatternFocus = hit.patternIndex;
      this.selectedNode = relationKey;
      this.thematicSelectionCandidates =
        this.thematicPatternsForCombination(
          [pattern[0], pattern[2]],
          [pattern[1]],
          hit.patternIndex,
        );
      this.syncThematicCategoryControls(
        [pattern[0]],
        [pattern[1]],
        [pattern[2]],
      );
    } else {
      this.selectThematicNode(relationKey);
      return;
    }

    this.setThematicSelection(
      this.filteredThematicPatterns(this.thematicSelectionCandidates),
    );
    this.commitThematicSelection(startedAt);
  }

  /**
   * Publish `this.nav` to every surface.
   *
   * This is the only place the four views are refreshed from a selection
   * change. Callers mutate the state record and then call this; a caller that
   * updates a panel itself is how the surfaces drifted apart before.
   */
  /**
   * Push the record out to the two side panels.
   *
   * Shared by commitThematicSelection() and by clearSelection(), so clearing
   * is not a second, slightly different publish path.
   */
  private publishNavigationToPanels(): void {
    this.setCategoryValues(el.entityCategories, this.nav.categories.subjects);
    this.setCategoryValues(el.relationCategories, this.nav.categories.relations);
    this.setCategoryValues(el.objectCategories, this.nav.categories.objects);
    this.refreshCategoryFacets();
    this.scheduleFilteredTagsSync();
  }

  private commitThematicSelection(startedAt: number): void {
    if (!this.renderer) return;
    this.publishNavigationToPanels();
    this.selectedEdge = null;
    el.detailPanel.open = true;
    el.container.classList.add("node-hover");
    this.renderer.scheduleRefresh();
    if (this.thematicOverlayCanvas) {
      const selectedRelation = this.selectedNode
        ? this.thematicRelationIndex(this.selectedNode)
        : null;
      this.thematicOverlayCanvas.dataset.thematicFocusMode =
        this.thematicExactPatternFocus !== null
          ? "exact-pattern"
          : this.thematicContextRelation !== null && this.thematicPairEntity
            ? "entity-pair-relation"
          : this.thematicContextRelation !== null
            ? "entity-relation"
            : this.thematicPairEntity
              ? "entity-pair"
              : selectedRelation !== null
                ? "relation"
                : "entity";
      this.thematicOverlayCanvas.dataset.selectionDurationMs = (
        performance.now() - startedAt
      ).toFixed(3);
    }
    if (this.thematicInspectorFrame !== null) {
      window.cancelAnimationFrame(this.thematicInspectorFrame);
    }
    this.thematicInspectorFrame = window.requestAnimationFrame(() => {
      this.thematicInspectorFrame = window.requestAnimationFrame(() => {
        this.thematicInspectorFrame = null;
        if (this.thematicPayload && this.selectedNode) {
          this.renderThematicSelection();
        }
      });
    });
  }

  private renderThematicSelection(): void {
    if (
      !this.graph ||
      !this.thematicPayload ||
      !this.selectedNode
    ) return;
    const payload = this.thematicPayload;
    const relationIndex = this.thematicRelationIndex(this.selectedNode);
    const data = this.graph.hasNode(this.selectedNode)
      ? this.graph.getNodeAttributes(this.selectedNode)
      : relationIndex !== null
        ? {
            thematicKind: "relation",
            categoryIndex: relationIndex,
            categoryDetail: payload.relationCategories[relationIndex],
          }
        : null;
    if (!data) return;
    const category = data.categoryDetail as
      | ThematicEntityCategory
      | ThematicRelationCategory;
    clear(el.detail);

    if (this.thematicExactPatternFocus !== null) {
      const pattern = payload.patterns[this.thematicExactPatternFocus];
      if (!pattern) return;
      const source = payload.entityCategories[pattern[0]];
      const relation = payload.relationCategories[pattern[1]];
      const target = payload.entityCategories[pattern[2]];
      el.detail.appendChild(
        textElement(
          "h3",
          "graph-detail-title",
          `${source.tagId} → ${relation.tagId} → ${target.tagId}`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-type",
          "specific source–relation–target focus",
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-summary",
          `${pattern[4].toLocaleString()} in-scope triples · ` +
            `${pattern[3].toLocaleString()} corpus triples`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          `${source.label} → ${relation.label} → ${target.label}`,
        ),
      );
      const actions = document.createElement("div");
      actions.className = "graph-detail-actions";
      actions.append(
        actionButton(
          `View all ${source.tagId} relations`,
          "thematic-entity-group",
          source.id,
        ),
        actionButton(
          `View complete ${relation.tagId} group`,
          "thematic-relation-group",
          relation.id,
        ),
        actionButton("Return to complete graph", "thematic-reset", ""),
      );
      el.detail.appendChild(actions);
      this.appendThematicPatternList(this.thematicSelectedPatterns);
    } else if (
      this.thematicContextRelation !== null &&
      this.thematicPairEntity
    ) {
      const source = data.categoryDetail as ThematicEntityCategory;
      const target = this.graph.getNodeAttribute(
        this.thematicPairEntity,
        "categoryDetail",
      ) as ThematicEntityCategory;
      const relation =
        payload.relationCategories[this.thematicContextRelation];
      const inScopeClaims = Array.from(this.thematicSelectedPatterns).reduce(
        (total, patternIndex) => total + payload.patterns[patternIndex][4],
        0,
      );
      el.detail.appendChild(
        textElement(
          "h3",
          "graph-detail-title",
          `${source.tagId} → ${relation.tagId} → ${target.tagId}`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-type",
          "entity-pair + relation focus",
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-summary",
          `${this.thematicSelectedPatterns.size.toLocaleString()} exact patterns · ` +
            `${inScopeClaims.toLocaleString()} in-scope triples`,
        ),
      );
      const actions = document.createElement("div");
      actions.className = "graph-detail-actions";
      actions.append(
        actionButton(
          `View all ${source.tagId} relations`,
          "thematic-entity-group",
          source.id,
        ),
        actionButton(
          `View complete ${relation.tagId} group`,
          "thematic-relation-group",
          relation.id,
        ),
        actionButton("Return to complete graph", "thematic-reset", ""),
      );
      el.detail.appendChild(actions);
      this.appendThematicPatternList(this.thematicSelectedPatterns);
    } else if (this.thematicContextRelation !== null) {
      const entity = data.categoryDetail as ThematicEntityCategory;
      const relation =
        payload.relationCategories[this.thematicContextRelation];
      const counterparts = new Set<number>();
      let inScopeClaims = 0;
      const entityIndex = Number(data.categoryIndex);
      for (const patternIndex of this.thematicSelectedPatterns) {
        const pattern = payload.patterns[patternIndex];
        inScopeClaims += pattern[4];
        if (pattern[0] === entityIndex) counterparts.add(pattern[2]);
        if (pattern[2] === entityIndex) counterparts.add(pattern[0]);
      }
      el.detail.appendChild(
        textElement(
          "h3",
          "graph-detail-title",
          `${entity.tagId} + ${relation.tagId}`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-type",
          "entity + relation focus",
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-summary",
          `${this.thematicSelectedPatterns.size.toLocaleString()} exact patterns · ` +
            `${counterparts.size.toLocaleString()} counterparts · ` +
            `${inScopeClaims.toLocaleString()} in-scope triples`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          `${entity.label} combined with ${relation.label}`,
        ),
      );
      const actions = document.createElement("div");
      actions.className = "graph-detail-actions";
      actions.append(
        actionButton(
          `View all ${entity.tagId} relations`,
          "thematic-entity-group",
          entity.id,
        ),
        actionButton(
          `View complete ${relation.tagId} group`,
          "thematic-relation-group",
          relation.id,
        ),
        actionButton("Return to complete graph", "thematic-reset", ""),
      );
      el.detail.appendChild(actions);
      this.appendThematicPatternList(this.thematicSelectedPatterns);
    } else if (this.thematicPairEntity) {
      const source = data.categoryDetail as ThematicEntityCategory;
      const target = this.graph.getNodeAttribute(
        this.thematicPairEntity,
        "categoryDetail",
      ) as ThematicEntityCategory;
      el.detail.appendChild(
        textElement(
          "h3",
          "graph-detail-title",
          `${source.tagId} → ${target.tagId}`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-type",
          "ordered entity-pair focus",
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-summary",
          `${this.thematicSelectedPatterns.size.toLocaleString()} exact patterns`,
        ),
      );
      this.appendThematicPatternList(this.thematicSelectedPatterns);
    } else if (String(data.thematicKind) === "entity") {
      const entity = category as ThematicEntityCategory;
      const relations = new Set<number>();
      const counterparts = new Set<number>();
      const entityIndex = Number(data.categoryIndex);
      let incoming = 0;
      let outgoing = 0;
      for (const patternIndex of this.thematicSelectedPatterns) {
        const pattern = payload.patterns[patternIndex];
        relations.add(pattern[1]);
        if (pattern[0] === entityIndex) {
          outgoing += 1;
          counterparts.add(pattern[2]);
        }
        if (pattern[2] === entityIndex) {
          incoming += 1;
          counterparts.add(pattern[0]);
        }
      }
      el.detail.appendChild(
        textElement("h3", "graph-detail-title", entity.label),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-type",
          `${entity.tagId} · entity category${entity.provisional ? " · provisional" : ""}`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-summary",
          `${this.thematicSelectedPatterns.size.toLocaleString()} exact patterns · ` +
            `${relations.size.toLocaleString()} relations · ` +
            `${counterparts.size.toLocaleString()} counterparts`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          `${outgoing.toLocaleString()} outgoing · ${incoming.toLocaleString()} incoming · ` +
            `${entity.mentionCount.toLocaleString()} corpus mentions`,
        ),
      );
      el.detail.appendChild(
        textElement("div", "graph-detail-status", entity.definition),
      );
      const actions = document.createElement("div");
      actions.className = "graph-detail-actions";
      actions.append(actionButton("Return to complete graph", "thematic-reset", ""));
      el.detail.appendChild(actions);
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          "Click a second entity to show only that ordered subject/object pair and every relation category connecting it.",
        ),
      );
      this.appendThematicPatternList(this.thematicSelectedPatterns);
    } else {
      const relation = category as ThematicRelationCategory;
      const entities = new Set<number>();
      const direction = this.thematicDirection();
      for (const patternIndex of this.thematicSelectedPatterns) {
        const pattern = payload.patterns[patternIndex];
        if (direction !== "object") entities.add(pattern[0]);
        if (direction !== "subject") entities.add(pattern[2]);
      }
      el.detail.appendChild(
        textElement("h3", "graph-detail-title", relation.label),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-type",
          `${relation.tagId} · relation category${relation.provisional ? " · provisional" : ""}`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-summary",
          `${this.thematicSelectedPatterns.size.toLocaleString()} exact patterns · ` +
            `${entities.size.toLocaleString()} entity categories · ` +
            `${relation.tripleCount.toLocaleString()} triples`,
        ),
      );
      el.detail.appendChild(
        textElement("div", "graph-detail-status", relation.definition),
      );
      const actions = document.createElement("div");
      actions.className = "graph-detail-actions";
      actions.append(
        actionButton("Return to complete graph", "thematic-reset", ""),
        actionButton(
          "Filter by this relation",
          "category-relation",
          relation.id,
        ),
      );
      el.detail.appendChild(actions);
      this.appendThematicPatternList(this.thematicSelectedPatterns);
    }
  }

  private renderThematicEvidence(errorMessage = ""): void {
    if (
      !this.thematicPayload ||
      this.thematicEvidencePattern === null
    ) return;
    const pattern =
      this.thematicPayload.patterns[this.thematicEvidencePattern];
    if (!pattern) return;
    const source = this.thematicPayload.entityCategories[pattern[0]];
    const relation = this.thematicPayload.relationCategories[pattern[1]];
    const target = this.thematicPayload.entityCategories[pattern[2]];
    clear(el.detail);
    el.detail.appendChild(
      textElement(
        "h3",
        "graph-detail-title",
        `${source.tagId} → ${relation.tagId} → ${target.tagId}`,
      ),
    );
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-type",
        "exact thematic pattern",
      ),
    );
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-summary",
        `${source.label} → ${relation.label} → ${target.label} · ` +
          `${pattern[3].toLocaleString()} supporting triples`,
      ),
    );
    el.detail.appendChild(
      textElement("div", "graph-detail-status", relation.definition),
    );
    const actions = document.createElement("div");
    actions.className = "graph-detail-actions";
    actions.append(
      actionButton("Back to foregrounded patterns", "thematic-back", ""),
      actionButton("Return to complete graph", "thematic-reset", ""),
    );
    el.detail.appendChild(actions);
    for (const item of this.evidenceItems) {
      const claim = document.createElement("article");
      claim.className = "graph-claim";
      claim.appendChild(
        textElement(
          "div",
          "graph-claim-spo",
          `${item.subject || source.label} — ${item.predicate || relation.label} → ${item.object || target.label}`,
        ),
      );
      claim.appendChild(
        textElement("div", "graph-claim-my", item.sentenceMy),
      );
      claim.appendChild(
        textElement("div", "graph-claim-en", item.sentenceEn),
      );
      const link = document.createElement("a");
      link.className = "graph-page-link";
      link.href = `/chronicles/${item.volumeId}/${item.ownerPage}`;
      link.textContent =
        `${item.volumeId.toUpperCase()}, page ${item.ownerPage} · ` +
        `${item.sentenceId} #${item.ordinal}`;
      claim.appendChild(link);
      el.detail.appendChild(claim);
    }
    if (this.evidenceLoading) {
      el.detail.appendChild(
        textElement("div", "graph-detail-status", "Loading source evidence…"),
      );
    } else if (errorMessage) {
      el.detail.appendChild(
        textElement("div", "graph-empty", errorMessage),
      );
    } else if (!this.evidenceItems.length) {
      el.detail.appendChild(
        textElement("div", "graph-empty", "No source evidence found."),
      );
    }
    if (this.evidenceHasMore && !this.evidenceLoading) {
      el.detail.appendChild(
        actionButton(
          "Load more evidence",
          "thematic-evidence-more",
          "",
          "graph-evidence-more",
        ),
      );
    }
  }

  private async loadThematicPatternEvidence(
    patternIndex: number,
    offset = 0,
  ): Promise<void> {
    if (!this.thematicPayload) return;
    const pattern = this.thematicPayload.patterns[patternIndex];
    if (!pattern) return;
    if (offset === 0 || this.thematicEvidencePattern !== patternIndex) {
      this.evidenceItems = [];
      this.evidenceHasMore = false;
      this.evidenceNextOffset = null;
    }
    this.thematicEvidencePattern = patternIndex;
    this.evidenceAbort?.abort();
    this.evidenceAbort = new AbortController();
    const request = this.evidenceAbort;
    const source = this.thematicPayload.entityCategories[pattern[0]];
    const relation = this.thematicPayload.relationCategories[pattern[1]];
    const target = this.thematicPayload.entityCategories[pattern[2]];
    const parameters = new URLSearchParams({
      source: source.id,
      target: target.id,
      relation: relation.id,
      scope: this.thematicPayload.layout.scope || "corpus",
      offset: String(offset),
      limit: "20",
    });
    const focus = this.thematicPayload.focus;
    if (focus.kind === "tag-filter" && focus.tagKind) {
      parameters.set("tagKind", focus.tagKind);
      for (const tagId of focus.tagIds || []) {
        parameters.append("tagId", tagId);
      }
    } else if (focus.kind === "filtered-tags") {
      for (const tagId of focus.subjectTagIds || []) {
        parameters.append("subjectTag", tagId);
      }
      for (const tagId of focus.relationTagIds || []) {
        parameters.append("relationTag", tagId);
      }
      for (const tagId of focus.objectTagIds || []) {
        parameters.append("objectTag", tagId);
      }
    }
    if (focus.volumeId) parameters.set("volumeId", focus.volumeId);
    if (focus.kind === "page") {
      parameters.set(
        "pageNumber",
        String(focus.pageNumber || this.page.pageNumber),
      );
    } else if (focus.kind === "range") {
      parameters.set("startPage", String(focus.startPage));
      parameters.set("endPage", String(focus.endPage));
    }
    this.evidenceLoading = true;
    this.renderThematicEvidence();
    try {
      const evidence = await fetchJson<EvidencePayload>(
        `/api/graph/categories/evidence?${parameters}`,
        request.signal,
      );
      if (
        request !== this.evidenceAbort ||
        patternIndex !== this.thematicEvidencePattern
      ) return;
      this.evidenceItems.push(...evidence.items);
      this.evidenceHasMore = evidence.hasMore;
      this.evidenceNextOffset = evidence.nextOffset;
      this.evidenceLoading = false;
      this.renderThematicEvidence();
    } catch (error) {
      if (request.signal.aborted) return;
      this.evidenceLoading = false;
      this.renderThematicEvidence(
        error instanceof Error ? error.message : String(error),
      );
    }
  }


  private async installPayload(payload: TopologyPayload): Promise<void> {
    if (payload.layout.kind === "thematic") {
      if (payload.schemaVersion !== 5) {
        throw new Error(
          `Unsupported thematic topology schema ${payload.schemaVersion}`,
        );
      }
      await this.installThematicPayload(
        payload as ThematicTopologyPayload,
      );
      return;
    }
    const isAtlas = payload.layout.kind === "atlas";
    if (
      (isAtlas && payload.schemaVersion !== 3) ||
      (!isAtlas && payload.schemaVersion !== 2)
    ) {
      throw new Error(`Unsupported graph topology schema ${payload.schemaVersion}`);
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
    el.container.style.visibility =
      this.readableLayoutPending ? "hidden" : "";
    if (this.readableLayoutPending) {
      this.setLoading(true, "Drawing graph…");
    }
    let atlasAnimationTargets:
      | Record<string, { x: number; y: number }>
      | null = null;
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
          x:
            newBounds[0] +
            ((previous.x - oldBounds[0]) / oldWidth) * newWidth,
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
      nodeProgramClasses: isAtlas
        ? { point: NodePointProgram }
        : undefined,
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
      zoomToSizeRatioFunction: Boolean(
        graph.getAttribute("uniformScaling"),
      ) && !isAtlas
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
    }
    else if (scopeKinds.has(payload.focus.kind)) el.scope.value = payload.focus.kind;
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
  }

  private cameraCenterForGraphPoint(
    graphPoint: { x: number; y: number },
    ratio = 1,
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
  }

  private readableCameraCenter(): { x: number; y: number } {
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
  }

  private recenterReadableView(): void {
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
  }

  private drawAtlasRegions(): void {
    if (
      !this.renderer ||
      !this.atlasOverlayCanvas ||
      this.payload?.layout.kind !== "atlas"
    ) return;
    const canvas = this.atlasOverlayCanvas;
    const { width, height } = this.renderer.getDimensions();
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
    const mouse = this.renderer.getMouseCaptor();
    if (
      this.renderer.getCamera().isAnimated() ||
      mouse.isMoving ||
      mouse.draggedEvents > 0 ||
      mouse.currentWheelDirection !== 0
    ) return;
    if (this.atlasLevel >= 2) {
      if (this.atlasLevel >= 3) {
        context.font = "600 10px Inter, Segoe UI, sans-serif";
        context.textAlign = "center";
        context.textBaseline = "middle";
        for (const [edge, placement] of this.atlasPredicateLabelPositions) {
          if (edge === this.selectedEdge || edge === this.hoveredEdge) continue;
          context.beginPath();
          context.roundRect(
            placement.x - placement.width / 2,
            placement.y - 7,
            placement.width,
            14,
            3,
          );
          context.fillStyle = "rgba(250, 247, 252, 0.98)";
          context.fill();
          context.strokeStyle = "#d4c1df";
          context.lineWidth = 0.7;
          context.stroke();
          context.fillStyle = "#5c2b5d";
          context.fillText(placement.label, placement.x, placement.y + 0.4);
        }
      }
      this.drawAtlasOffscreenMarkers(context, width, height);
      return;
    }

    const origin = this.renderer.graphToViewport({ x: 0, y: 0 });
    const unitX = this.renderer.graphToViewport({ x: 1, y: 0 });
    const unitY = this.renderer.graphToViewport({ x: 0, y: 1 });
    const xAxis = {
      x: unitX.x - origin.x,
      y: unitX.y - origin.y,
    };
    const yAxis = {
      x: unitY.x - origin.x,
      y: unitY.y - origin.y,
    };
    const toViewport = (x: number, y: number) => ({
      x: origin.x + x * xAxis.x + y * yAxis.x,
      y: origin.y + x * xAxis.y + y * yAxis.y,
    });
    const labelBoxes: Array<[number, number, number, number]> = [];
    let labelsDrawn = 0;
    const drawRegion = (
      label: string,
      nodeCount: number,
      minimumX: number,
      minimumY: number,
      maximumX: number,
      maximumY: number,
      community: boolean,
    ) => {
      const topLeft = toViewport(minimumX, minimumY);
      const bottomRight = toViewport(maximumX, maximumY);
      const left = Math.min(topLeft.x, bottomRight.x);
      const right = Math.max(topLeft.x, bottomRight.x);
      const top = Math.min(topLeft.y, bottomRight.y);
      const bottom = Math.max(topLeft.y, bottomRight.y);
      if (right < 0 || left > width || bottom < 0 || top > height) return;
      const boxWidth = right - left;
      const boxHeight = bottom - top;
      if (boxWidth < 2 || boxHeight < 2) return;
      context.strokeStyle = community
        ? "rgba(111,74,150,0.12)"
        : "rgba(65,82,96,0.18)";
      context.lineWidth = community ? 0.7 : 1;
      context.setLineDash(community ? [4, 4] : []);
      context.strokeRect(left, top, boxWidth, boxHeight);
      if (
        labelsDrawn >= 36 ||
        nodeCount < 2 ||
        boxWidth < 88 ||
        boxHeight < 28
      ) return;
      label = label.replaceAll("_", " ");
      context.font = "600 10px Inter, Segoe UI, sans-serif";
      const labelWidth = Math.min(240, context.measureText(label).width + 12);
      const labelBox: [number, number, number, number] = [
        left + 5,
        top + 5,
        left + 5 + labelWidth,
        top + 23,
      ];
      if (
        labelBoxes.some(
          (box) =>
            labelBox[0] < box[2] &&
            labelBox[2] > box[0] &&
            labelBox[1] < box[3] &&
            labelBox[3] > box[1],
        )
      ) return;
      labelBoxes.push(labelBox);
      context.setLineDash([]);
      context.fillStyle = "rgba(248,249,250,0.88)";
      context.fillRect(
        labelBox[0],
        labelBox[1],
        labelBox[2] - labelBox[0],
        labelBox[3] - labelBox[1],
      );
      context.fillStyle = "#52616c";
      context.textBaseline = "middle";
      context.fillText(
        label,
        labelBox[0] + 6,
        (labelBox[1] + labelBox[3]) / 2,
        labelWidth - 12,
      );
      labelsDrawn += 1;
    };
    for (const component of this.payload.components || []) {
      drawRegion(
        component[1],
        component[2],
        component[4],
        component[5],
        component[6],
        component[7],
        false,
      );
    }
    if (this.atlasLevel === 1) {
      for (const community of this.payload.communities || []) {
        if (community[2] < 3) continue;
        drawRegion(
          community[1],
          community[2],
          community[3],
          community[4],
          community[5],
          community[6],
          true,
        );
      }
    }
    context.setLineDash([]);
    this.drawAtlasOffscreenMarkers(context, width, height);
  }

  private drawAtlasOffscreenMarkers(
    context: CanvasRenderingContext2D,
    width: number,
    height: number,
  ): void {
    if (!this.renderer || !this.graph) return;
    const activeNode = this.selectedNode || this.hoveredNode;
    if (!activeNode || !this.graph.hasNode(activeNode)) return;
    const activeAttributes = this.graph.getNodeAttributes(activeNode);
    const activePoint = this.renderer.graphToViewport({
      x: Number(activeAttributes.x),
      y: Number(activeAttributes.y),
    });
    if (
      activePoint.x < 0 ||
      activePoint.x > width ||
      activePoint.y < 0 ||
      activePoint.y > height
    ) return;

    const inset = 11;
    const buckets = new Map<
      string,
      {
        x: number;
        y: number;
        angle: number;
        outgoing: boolean;
        count: number;
      }
    >();
    for (const edge of this.graph.edges(activeNode)) {
      if (!Boolean(this.graph.getEdgeAttribute(edge, "atlasRawEdge"))) {
        continue;
      }
      const [source, target] = this.graph.extremities(edge);
      if (source === target) continue;
      const other = source === activeNode ? target : source;
      const attributes = this.graph.getNodeAttributes(other);
      const otherPoint = this.renderer.graphToViewport({
        x: Number(attributes.x),
        y: Number(attributes.y),
      });
      if (
        otherPoint.x >= inset &&
        otherPoint.x <= width - inset &&
        otherPoint.y >= inset &&
        otherPoint.y <= height - inset
      ) continue;
      const deltaX = otherPoint.x - activePoint.x;
      const deltaY = otherPoint.y - activePoint.y;
      const candidates: number[] = [];
      if (deltaX > 0) {
        candidates.push((width - inset - activePoint.x) / deltaX);
      } else if (deltaX < 0) {
        candidates.push((inset - activePoint.x) / deltaX);
      }
      if (deltaY > 0) {
        candidates.push((height - inset - activePoint.y) / deltaY);
      } else if (deltaY < 0) {
        candidates.push((inset - activePoint.y) / deltaY);
      }
      const scale = Math.min(
        ...candidates.filter((value) => value >= 0 && value <= 1),
      );
      if (!Number.isFinite(scale)) continue;
      const x = activePoint.x + deltaX * scale;
      const y = activePoint.y + deltaY * scale;
      const outgoing = source === activeNode;
      const side =
        x <= inset + 0.5
          ? "left"
          : x >= width - inset - 0.5
            ? "right"
            : y <= inset + 0.5
              ? "top"
              : "bottom";
      const along = side === "left" || side === "right" ? y : x;
      const key = `${side}:${Math.round(along / 28)}:${outgoing ? 1 : 0}`;
      const existing = buckets.get(key);
      if (existing) {
        existing.count += 1;
      } else {
        buckets.set(key, {
          x,
          y,
          angle: Math.atan2(deltaY, deltaX),
          outgoing,
          count: 1,
        });
      }
    }

    context.setLineDash([]);
    for (const marker of buckets.values()) {
      context.save();
      context.translate(marker.x, marker.y);
      context.rotate(marker.angle);
      context.beginPath();
      context.moveTo(7, 0);
      context.lineTo(-4, -4.5);
      context.lineTo(-4, 4.5);
      context.closePath();
      context.fillStyle = marker.outgoing ? "#75419a" : "#15968a";
      if (marker.outgoing) context.fill();
      else {
        context.lineWidth = 2;
        context.strokeStyle = "#15968a";
        context.stroke();
      }
      context.restore();
      if (marker.count > 1) {
        context.font = "600 9px Inter, Segoe UI, sans-serif";
        context.textBaseline = "middle";
        context.fillStyle = marker.outgoing ? "#66348c" : "#087f75";
        context.fillText(
          marker.count.toLocaleString(),
          Math.min(width - 34, Math.max(3, marker.x + 7)),
          Math.min(height - 8, Math.max(8, marker.y)),
        );
      }
    }
  }

  private async ensureCategoryCatalog(): Promise<void> {
    if (this.categoryCatalogLoaded) return;
    if (this.categoryCatalogPromise) return this.categoryCatalogPromise;
    this.categoryCatalogPromise = (async () => {
      const catalog = await fetchJson<CategoryCatalogPayload>(
        "/api/graph/categories/catalog",
      );
      const populate = (
        container: HTMLElement,
        categories: CategoryDescriptor[],
      ): void => {
        container.replaceChildren(
          ...categories.map((category) => {
            const row = document.createElement("label");
            row.className = "graph-category-option";
            row.title = category.definition;
            row.dataset.search =
              `${category.tagId} ${category.label}`.toLowerCase();
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.value = category.id;
            const copy = document.createElement("span");
            copy.className = "graph-category-option-copy";
            copy.appendChild(
              textElement(
                "span",
                "graph-category-option-label",
                `${category.tagId} · ${category.label}`,
              ),
            );
            copy.appendChild(
              textElement(
                "span",
                "graph-category-option-count",
                `${Number(category.count).toLocaleString()} claims`,
              ),
            );
            row.append(checkbox, copy);
            return row;
          }),
        );
      };
      populate(el.entityCategories, catalog.entities);
      populate(el.objectCategories, catalog.entities);
      populate(el.relationCategories, catalog.relations);
      this.categoryCatalogLoaded = true;
    })();
    try {
      await this.categoryCatalogPromise;
    } finally {
      this.categoryCatalogPromise = null;
    }
  }

  /**
   * Re-derive every theme list from the loaded slice and the current selection.
   *
   * Each list is a facet: its options are counted under the *other* two lists'
   * constraints but not its own, so checking two entity categories immediately
   * narrows the relation list to the relations that actually hold between
   * them, while the entity list itself stays open for adding a third. Counts
   * come from the loaded slice's active triple totals, so they also track the
   * scope and any focused slice.
   */
  private syncCategoryOptionCounts(payload: ThematicTopologyPayload): void {
    const indexOf = (
      categories: { id: string }[],
      container: HTMLElement,
    ): Set<number> => {
      const chosen = new Set(this.selectedCategoryValues(container));
      return new Set(
        categories.flatMap((category, index) =>
          chosen.has(category.id) ? [index] : [],
        ),
      );
    };
    const groupA = indexOf(payload.entityCategories, el.entityCategories);
    const groupB = indexOf(payload.entityCategories, el.objectCategories);
    const relations = indexOf(payload.relationCategories, el.relationCategories);
    const direction = this.thematicDirection();
    const fits = (index: number, group: Set<number>): boolean =>
      group.size === 0 || group.has(index);

    const facetA = new Map<number, number>();
    const facetB = new Map<number, number>();
    const facetRelation = new Map<number, number>();
    const add = (
      facet: Map<number, number>,
      index: number,
      triples: number,
    ): void => facet.set(index, (facet.get(index) || 0) + triples) as never;

    for (const pattern of payload.patterns) {
      const [source, relation, target] = pattern;
      const triples = pattern[4];
      const forwardAllowed = direction !== "ba";
      const reverseAllowed = direction !== "ab";
      const relationOk = fits(relation, relations);

      // Group A's own constraint is excluded from its own facet, and so on.
      if (relationOk && forwardAllowed && fits(target, groupB)) {
        add(facetA, source, triples);
      }
      if (relationOk && reverseAllowed && fits(source, groupB)) {
        add(facetA, target, triples);
      }
      if (relationOk && forwardAllowed && fits(source, groupA)) {
        add(facetB, target, triples);
      }
      if (relationOk && reverseAllowed && fits(target, groupA)) {
        add(facetB, source, triples);
      }
      const pairs =
        (forwardAllowed && fits(source, groupA) && fits(target, groupB)) ||
        (reverseAllowed && fits(target, groupA) && fits(source, groupB));
      if (pairs) add(facetRelation, relation, triples);
    }

    const apply = (
      container: HTMLElement,
      categories: { id: string }[],
      facet: Map<number, number>,
    ): void => {
      const byId = new Map(
        categories.map((category, index) => [category.id, index] as const),
      );
      for (const input of this.categoryInputs(container)) {
        const row = input.closest<HTMLElement>(".graph-category-option");
        if (!row) continue;
        const index = byId.get(input.value);
        const available = index === undefined ? 0 : facet.get(index) || 0;
        const count = row.querySelector(".graph-category-option-count");
        if (count) {
          count.textContent =
            `${available.toLocaleString()} claim` +
            `${available === 1 ? "" : "s"} available`;
        }
        // A checked category is never disabled: unchecking it must stay
        // possible even once the slice it produced contains nothing else.
        input.disabled = available === 0 && !input.checked;
        row.classList.toggle("is-empty", input.disabled);
      }
      this.syncCategoryPickerFooter(container);
    };
    apply(el.entityCategories, payload.entityCategories, facetA);
    apply(el.objectCategories, payload.entityCategories, facetB);
    apply(el.relationCategories, payload.relationCategories, facetRelation);
  }

  /** Recompute the facets against whatever is checked right now. */
  refreshCategoryFacets(): void {
    if (this.thematicPayload) {
      this.syncCategoryOptionCounts(this.thematicPayload);
    }
  }

  /**
   * Copy the three category lists into the state record.
   *
   * The lists are the input device for the category part of the state, so an
   * edit there is written straight through. Without this, commitNavigation
   * republishing the record would revert checks a reader had made but not yet
   * applied, the moment anything else committed.
   */
  recordCategoryControls(): void {
    this.nav.categories = {
      subjects: this.selectedCategoryValues(el.entityCategories),
      relations: this.selectedCategoryValues(el.relationCategories),
      objects: this.selectedCategoryValues(el.objectCategories),
    };
  }

  private categoryInputs(container: HTMLElement): HTMLInputElement[] {
    return Array.from(
      container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'),
    );
  }

  private selectedCategoryValues(container: HTMLElement): string[] {
    return this.categoryInputs(container)
      .filter((input) => input.checked)
      .map((input) => input.value);
  }

  private setCategoryValues(
    container: HTMLElement,
    values: string[] = [],
  ): void {
    const selected = new Set(values);
    let first: HTMLElement | null = null;
    for (const input of this.categoryInputs(container)) {
      input.checked = selected.has(input.value);
      if (input.checked && !first) {
        first = input.closest<HTMLElement>(".graph-category-option");
      }
    }
    // A selection driven by a graph click can land outside the scrolled
    // viewport, where it reads as nothing having happened.
    if (first) {
      const offset = first.offsetTop - container.offsetTop;
      if (
        offset < container.scrollTop ||
        offset + first.offsetHeight >
          container.scrollTop + container.clientHeight
      ) {
        container.scrollTop = Math.max(
          0,
          offset - (container.clientHeight - first.offsetHeight) / 2,
        );
      }
    }
    this.syncCategoryPickerFooter(container);
  }

  /** Keep each picker's "N checked" line and its Clear button honest. */
  syncCategoryPickerFooter(container: HTMLElement): void {
    for (const input of this.categoryInputs(container)) {
      input
        .closest<HTMLElement>(".graph-category-option")
        ?.classList.toggle("is-checked", input.checked);
    }
    const count = this.selectedCategoryValues(container).length;
    const picker = container.parentElement;
    const label = picker?.querySelector<HTMLElement>(".graph-category-count");
    const clear = picker?.querySelector<HTMLButtonElement>(
      ".graph-category-foot button",
    );
    if (label) {
      label.textContent = count
        ? `${count} checked · combined with OR`
        : "None checked";
    }
    if (clear) clear.disabled = count === 0;
  }

  clearCategoryPicker(container: HTMLElement): void {
    for (const input of this.categoryInputs(container)) input.checked = false;
    this.syncCategoryPickerFooter(container);
    this.recordCategoryControls();
    this.refreshCategoryFacets();
    this.scheduleFilteredTagsSync();
  }

  /** Narrow a long category list to the typed term. */
  filterCategoryPicker(container: HTMLElement, query: string): void {
    const needle = query.trim().toLowerCase();
    for (const row of container.querySelectorAll<HTMLElement>(
      ".graph-category-option",
    )) {
      row.hidden = Boolean(needle) &&
        !(row.dataset.search || "").includes(needle);
    }
  }

  /**
   * Whether a pattern satisfies the two entity groups in each orientation.
   *
   * This mirrors the server's matching exactly: "forward" puts group A on the
   * pattern's subject side, "reverse" puts it on the object side. An empty
   * group matches anything. The graph highlight and the panels both go through
   * here, so they cannot drift apart the way they did when the highlight was
   * hard-coded to the subject/object order.
   */
  private patternOrientations(
    pattern: ThematicPatternTuple,
    groupA: Set<number>,
    groupB: Set<number>,
    direction: ThematicDirection,
  ): { forward: boolean; reverse: boolean } {
    const fits = (index: number, group: Set<number>): boolean =>
      group.size === 0 || group.has(index);
    return {
      forward: direction !== "ba" &&
        fits(pattern[0], groupA) && fits(pattern[2], groupB),
      reverse: direction !== "ab" &&
        fits(pattern[2], groupA) && fits(pattern[0], groupB),
    };
  }

  private applyThematicCategorySelection(): void {
    if (!this.thematicPayload || !this.graph || !this.renderer) return;
    const selectedSubjectIds = new Set(
      this.selectedCategoryValues(el.entityCategories),
    );
    const selectedObjectIds = new Set(
      this.selectedCategoryValues(el.objectCategories),
    );
    const selectedRelationIds = new Set(
      this.selectedCategoryValues(el.relationCategories),
    );
    const subjectIndices = this.thematicPayload.entityCategories.flatMap(
      (category, index) => selectedSubjectIds.has(category.id) ? [index] : [],
    );
    const objectIndices = this.thematicPayload.entityCategories.flatMap(
      (category, index) => selectedObjectIds.has(category.id) ? [index] : [],
    );
    const relationIndices = this.thematicPayload.relationCategories.flatMap(
      (category, index) => selectedRelationIds.has(category.id) ? [index] : [],
    );
    if (
      !subjectIndices.length &&
      !objectIndices.length &&
      !relationIndices.length
    ) {
      this.clearSelection();
      return;
    }

    this.pushNavigationState();
    const startedAt = performance.now();
    const primaryIndex = subjectIndices[0] ?? objectIndices[0];
    this.thematicPrimaryEntity = primaryIndex !== undefined
      ? this.thematicEntityNodeId(
          this.thematicPayload.entityCategories[primaryIndex].id,
        )
      : null;
    this.thematicPairEntity =
      subjectIndices.length && objectIndices.length
      ? this.thematicEntityNodeId(
          this.thematicPayload.entityCategories[objectIndices[0]].id,
        )
      : null;
    this.thematicContextRelation =
      (subjectIndices.length || objectIndices.length) &&
      relationIndices.length === 1
        ? relationIndices[0]
        : null;
    this.thematicExactPatternFocus = null;
    this.selectedNode =
      this.thematicPrimaryEntity ||
      (relationIndices.length
        ? this.thematicRelationNodeId(
            this.thematicPayload.relationCategories[relationIndices[0]].id,
          )
        : null);
    if (!this.selectedNode) return;
    const subjects = new Set(subjectIndices);
    const objects = new Set(objectIndices);
    const relations = new Set(relationIndices);
    const direction = this.thematicDirection();
    this.thematicGroupSelection = true;
    this.thematicOrientationAnchors = new Set(
      subjectIndices.length ? subjectIndices : objectIndices,
    );
    // Driven from the controls, so the state records what they already say.
    this.syncThematicCategoryControls(
      subjectIndices,
      relationIndices,
      objectIndices,
    );
    this.thematicSelectionCandidates = this.thematicPayload.patterns.flatMap(
      (pattern, index) => {
        if (relations.size && !relations.has(pattern[1])) return [];
        const { forward, reverse } = this.patternOrientations(
          pattern,
          subjects,
          objects,
          direction,
        );
        return forward || reverse ? [index] : [];
      },
    );
    this.setThematicSelection(
      this.filteredThematicPatterns(this.thematicSelectionCandidates),
    );
    this.commitThematicSelection(startedAt);
  }

  private syncViewModeUi(): void {
    el.themeControls.hidden = false;
    el.browserPanel.hidden = false;
    el.browserSummary.textContent = "Filter themes";
    el.nodeLegend.textContent = "entity category · size = mentions";
    el.edgeLegend.textContent =
      "purple = anchor is the subject · teal = anchor is the object";
    if (el.edgeLegend.parentElement) {
      el.edgeLegend.parentElement.hidden = false;
      el.edgeLegend.parentElement.style.display = "";
    }
  }

  async applyThemeFilters(): Promise<void> {
    if (this.thematicPayload?.focus.kind === "filtered-tags") {
      await this.loadGraph(this.thematicBaseUrl());
    } else {
      this.applyThematicCategorySelection();
    }
    this.clearSimilarPatterns();
    this.syncSimilarPatternAnchor();
    await this.refreshFilteredTags(true, 0);
  }

  async clearThemeFilters(): Promise<void> {
    const filteredTagScope = this.thematicPayload?.focus.kind === "filtered-tags"
      ? this.thematicBaseUrl()
      : null;
    this.setCategoryValues(el.entityCategories);
    this.setCategoryValues(el.objectCategories);
    this.setCategoryValues(el.relationCategories);
    this.recordCategoryControls();
    el.themeDirection.value = "either";
    el.themeGranularity.value = "5";
    el.themeGranularityValue.value = "5+";
    this.resetFilteredTags(true);
    this.clearSimilarPatterns();
    this.syncSimilarPatternAnchor();
    if (filteredTagScope) await this.loadGraph(filteredTagScope);
    else this.clearSelection();
  }

  private activeFilteredTagRoles(): FilteredTagRole[] {
    const roles: FilteredTagRole[] = [];
    if (this.selectedCategoryValues(el.entityCategories).length) {
      roles.push("subject");
    }
    if (this.selectedCategoryValues(el.relationCategories).length) {
      roles.push("relation");
    }
    if (this.selectedCategoryValues(el.objectCategories).length) {
      roles.push("object");
    }
    return roles;
  }

  private filteredTagRoleLabel(role: FilteredTagRole): string {
    const select = role === "subject"
      ? el.entityCategories
      : role === "relation"
      ? el.relationCategories
      : el.objectCategories;
    const categories = this.selectedCategoryValues(select);
    const title = role === "subject"
      ? "Entity group A"
      : role === "object"
      ? "Entity group B"
      : "Relation";
    const narrowed = (["subject", "relation", "object"] as FilteredTagRole[])
      .some((other) => other !== role && (this.appliedTagFilters.get(other) || []).length);
    const base = categories.length
      ? `${title} tags / ${categories.join(", ")}`
      : `${title} tags / every tag in scope`;
    return narrowed ? `${base} / narrowed by applied tags` : base;
  }

  /** Every role is always listed; unfiltered roles fall back to the scope. */
  private populateFilteredTagRoles(): FilteredTagRole[] {
    const roles: FilteredTagRole[] = ["subject", "relation", "object"];
    const active = this.activeFilteredTagRoles();
    const previous = el.filteredTagsRole.value as FilteredTagRole;
    el.filteredTagsRole.replaceChildren(
      ...roles.map((role) => {
        const option = document.createElement("option");
        option.value = role;
        option.textContent = this.filteredTagRoleLabel(role);
        return option;
      }),
    );
    if (roles.includes(previous)) el.filteredTagsRole.value = previous;
    else if (active.length) el.filteredTagsRole.value = active[0];
    return roles;
  }

  private selectedFilteredTagCount(): number {
    return Array.from(this.filteredTagsSelected.values()).reduce(
      (total, selected) => total + selected.size,
      0,
    );
  }

  private resetFilteredTags(clearSearch = false): void {
    this.filteredTagsAbort?.abort();
    if (this.filteredTagsDebounce !== null) {
      window.clearTimeout(this.filteredTagsDebounce);
      this.filteredTagsDebounce = null;
    }
    for (const selected of this.filteredTagsSelected.values()) selected.clear();
    for (const role of this.filteredTagsSimilarAnchors.keys()) {
      this.filteredTagsSimilarAnchors.set(role, []);
    }
    this.clearAppliedTagFilters();
    this.filteredTagsItems = [];
    this.filteredTagsOffset = 0;
    this.filteredTagsTotal = 0;
    this.filteredTagsHasMore = false;
    this.filteredTagsBucketKey = "";
    if (clearSearch) el.filteredTagsSearch.value = "";
    this.populateFilteredTagRoles();
    clear(el.filteredTagsResults);
    el.filteredTagsPrev.disabled = true;
    el.filteredTagsNext.disabled = true;
    el.filteredTagsSelectPage.disabled = true;
    el.filteredTagsClear.disabled = true;
    el.filteredTagsApply.disabled = true;
    el.filteredTagsSimilar.disabled = true;
    el.filteredTagsSimilarClear.disabled = true;
    el.filteredTagsPage.textContent = "0 tags";
    el.filteredTagsNote.textContent = this.activeFilteredTagRoles().length
      ? "Open a theme bucket to load its low-level tags."
      : "Listing every low-level tag in the current scope. Apply categories in Filter themes to narrow the bucket.";
    el.filteredTagsSelectionNote.textContent =
      "Up to 100 checked tags can be combined. Tags within one role are unioned; different roles intersect.";
    el.filteredTagsSimilarNote.textContent =
      "Check one or more tags, then rank this bucket by embedding similarity to their averaged vector.";
  }

  /**
   * Identity of the tag list currently on screen. Reloading is only worth it
   * when this changes; without the check a stray sync would drop the reader
   * back onto page one of the bucket they were already paging through.
   */
  private filteredTagsSignature(): string {
    const role = el.filteredTagsRole.value as FilteredTagRole;
    return [
      role,
      el.filteredTagsSearch.value,
      this.selectedCategoryValues(el.entityCategories).join(","),
      this.selectedCategoryValues(el.relationCategories).join(","),
      this.selectedCategoryValues(el.objectCategories).join(","),
      (this.filteredTagsSimilarAnchors.get(role) || []).join(","),
      (this.appliedTagFilters.get("subject") || []).join(","),
      (this.appliedTagFilters.get("relation") || []).join(","),
      (this.appliedTagFilters.get("object") || []).join(","),
      el.filteredTagsSimilarity.value,
      el.themeDirection.value,
      this.thematicPayload?.layout.scope || el.scope.value,
      this.payload?.focus.kind || "",
      this.payload?.focus.id || "",
      this.page.volumeId,
      String(this.page.pageNumber),
    ].join("|");
  }

  /** Reload only when the bucket differs from what is already rendered. */
  refreshFilteredTagsIfStale(): void {
    if (
      this.filteredTagsItems.length &&
      this.filteredTagsSignature() === this.filteredTagsBucketKey
    ) return;
    void this.refreshFilteredTags(false, 0);
  }

  /**
   * Reload the open tag bucket after the graph selection changed. Debounced
   * because one click can move several category selects at once.
   */
  scheduleFilteredTagsSync(): void {
    this.syncSimilarPatternAnchor();
    if (this.filteredTagsSyncTimer !== null) {
      window.clearTimeout(this.filteredTagsSyncTimer);
    }
    this.filteredTagsSyncTimer = window.setTimeout(() => {
      this.filteredTagsSyncTimer = null;
      this.populateFilteredTagRoles();
      if (!el.filteredTagsPanel.open) return;
      if (this.filteredTagsSignature() === this.filteredTagsBucketKey) return;
      void this.refreshFilteredTags(false, 0);
    }, 180);
  }

  private appendFilteredTagScope(parameters: URLSearchParams): void {
    const scope = this.thematicPayload?.layout.scope ||
      (el.scope.value === "focus" ? "corpus" : el.scope.value);
    parameters.set("scope", scope);
    if (scope === "page") {
      parameters.set("volumeId", this.page.volumeId);
      parameters.set("pageNumber", String(this.page.pageNumber));
    } else if (scope === "range") {
      parameters.set("volumeId", el.pageVolume.value);
      parameters.set("startPage", el.rangeStart.value);
      parameters.set("endPage", el.rangeEnd.value);
    }
  }

  private appendPairingDirection(parameters: URLSearchParams): void {
    parameters.set("direction", this.thematicDirection());
  }

  private appendFilteredTagCategories(
    parameters: URLSearchParams,
    topology = false,
  ): void {
    const suffix = topology ? "Category" : "";
    for (const id of this.selectedCategoryValues(el.entityCategories)) {
      parameters.append(`subject${suffix}`, id);
    }
    for (const id of this.selectedCategoryValues(el.relationCategories)) {
      parameters.append(`relation${suffix}`, id);
    }
    for (const id of this.selectedCategoryValues(el.objectCategories)) {
      parameters.append(`object${suffix}`, id);
    }
  }

  private appendAppliedTagFilters(parameters: URLSearchParams): void {
    for (const role of ["subject", "relation", "object"] as FilteredTagRole[]) {
      for (const id of this.appliedTagFilters.get(role) || []) {
        parameters.append(`${role}Tag`, id);
      }
    }
  }

  private appliedTagFilterCount(): number {
    return [...this.appliedTagFilters.values()].reduce(
      (total, ids) => total + ids.length,
      0,
    );
  }

  clearAppliedTagFilters(): void {
    for (const role of this.appliedTagFilters.keys()) {
      this.appliedTagFilters.set(role, []);
    }
    el.filteredTagsRelease.hidden = true;
  }

  /** Drop the live tag filter, restore the base graph, and widen the facets. */
  async releaseAppliedTagFilters(): Promise<void> {
    if (!this.appliedTagFilterCount()) return;
    this.clearAppliedTagFilters();
    el.filteredTagsRelease.disabled = true;
    await this.loadGraph(this.thematicBaseUrl());
    el.filteredTagsRelease.disabled = false;
    await this.refreshFilteredTags(false, 0);
  }

  async refreshFilteredTags(
    resetSelection = false,
    offset = 0,
  ): Promise<void> {
    this.populateFilteredTagRoles();
    if (resetSelection) {
      for (const selected of this.filteredTagsSelected.values()) selected.clear();
      for (const role of this.filteredTagsSimilarAnchors.keys()) {
        this.filteredTagsSimilarAnchors.set(role, []);
      }
      for (const input of el.filteredTagsResults.querySelectorAll<HTMLInputElement>(
        'input[type="checkbox"]',
      )) input.checked = false;
    }
    const role = el.filteredTagsRole.value as FilteredTagRole;
    this.filteredTagsAbort?.abort();
    this.filteredTagsAbort = new AbortController();
    const request = this.filteredTagsAbort;
    const anchors = this.filteredTagsSimilarAnchors.get(role) || [];
    const parameters = new URLSearchParams({
      role,
      q: el.filteredTagsSearch.value,
      offset: String(Math.max(0, offset)),
      limit: "100",
    });
    if (anchors.length) {
      parameters.set(
        "minSimilarity",
        (Number(el.filteredTagsSimilarity.value) / 100).toFixed(4),
      );
      for (const id of anchors) parameters.append("similarTo", id);
    }
    this.appendFilteredTagCategories(parameters);
    this.appendAppliedTagFilters(parameters);
    this.appendFilteredTagScope(parameters);
    this.appendPairingDirection(parameters);
    el.filteredTagsNote.textContent = anchors.length
      ? "Averaging the checked tag vectors…"
      : "Loading low-level tags by frequency...";
    el.filteredTagsSearchButton.disabled = true;
    try {
      const payload = await fetchJson<FilteredTagsPayload>(
        `/api/graph/categories/tags?${parameters}`,
        request.signal,
      );
      if (request !== this.filteredTagsAbort) return;
      this.filteredTagsBucketKey = this.filteredTagsSignature();
      this.filteredTagsItems = payload.items;
      this.filteredTagsOffset = payload.pagination.offset;
      this.filteredTagsTotal = payload.pagination.total;
      this.filteredTagsHasMore = payload.pagination.hasMore;
      this.renderFilteredTags(payload);
    } catch (error) {
      if (request.signal.aborted) return;
      this.filteredTagsItems = [];
      clear(el.filteredTagsResults);
      el.filteredTagsNote.textContent =
        error instanceof Error ? error.message : String(error);
    } finally {
      if (request === this.filteredTagsAbort) {
        el.filteredTagsSearchButton.disabled = false;
      }
    }
  }

  private renderFilteredTags(payload: FilteredTagsPayload): void {
    clear(el.filteredTagsResults);
    const selected = this.filteredTagsSelected.get(payload.role) as Set<string>;
    if (!payload.items.length) {
      el.filteredTagsResults.appendChild(
        textElement("div", "graph-empty", "No matching low-level tags."),
      );
    }
    for (const item of payload.items) {
      const label = document.createElement("label");
      label.className = "graph-filtered-tag";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = item.id || "";
      checkbox.dataset.role = item.role;
      // Tags with no embedding record have no stable ID to filter on.
      checkbox.disabled = !item.id;
      checkbox.checked = Boolean(item.id) && selected.has(item.id as string);
      const copy = document.createElement("span");
      copy.className = "graph-filtered-tag-copy";
      copy.appendChild(textElement("span", "graph-result-label", item.label));
      const similarity = item.similarity === null || item.similarity === undefined
        ? ""
        : ` / ${(item.similarity * 100).toFixed(1)}% similar`;
      copy.appendChild(
        textElement(
          "span",
          "graph-result-meta",
          `${item.role} / ${item.frequency.toLocaleString()} matching claims` +
            similarity,
        ),
      );
      label.append(checkbox, copy);
      el.filteredTagsResults.appendChild(label);
    }
    const start = payload.pagination.total ? payload.pagination.offset + 1 : 0;
    const end = payload.pagination.offset + payload.pagination.returned;
    el.filteredTagsPage.textContent =
      `${start.toLocaleString()}-${end.toLocaleString()} of ` +
      `${payload.pagination.total.toLocaleString()} tags`;
    el.filteredTagsPrev.disabled = payload.pagination.offset === 0;
    el.filteredTagsNext.disabled = !payload.pagination.hasMore;
    el.filteredTagsSelectPage.disabled = payload.items.length === 0;
    // Fixed subject/relation/object order; the JSON object's key order is not
    // guaranteed to be meaningful.
    const narrowing = (["subject", "relation", "object"] as FilteredTagRole[])
      .flatMap((role) => {
        const count = (payload.narrowedBy?.[role] || []).length;
        const name = role === "subject"
          ? "group A"
          : role === "object"
          ? "group B"
          : "relation";
        return count
          ? [`${count} ${name} tag${count === 1 ? "" : "s"}`]
          : [];
      });
    const slot = payload.role === "subject"
      ? "group A"
      : payload.role === "object"
      ? "group B"
      : "relation";
    const bucket = payload.categoryScoped
      ? `${slot} tags in the applied categories`
      : `${slot} tags in this scope`;
    const narrowed = narrowing.length
      ? ` Narrowed by the applied ${narrowing.join(" and ")}; ` +
        `the ${slot} filter is not applied to its own list so you can ` +
        "still add or swap tags here."
      : "";
    el.filteredTagsNote.textContent = payload.similarTo.length
      ? `${payload.pagination.total.toLocaleString()} ${bucket} are at least ` +
        `${el.filteredTagsSimilarity.value}% similar to the averaged vector of ` +
        `${payload.similarTo.length.toLocaleString()} checked tag` +
        `${payload.similarTo.length === 1 ? "" : "s"}, ranked by similarity. ` +
        `At most 100 are shown per page.${narrowed}`
      : `${payload.pagination.total.toLocaleString()} ${bucket}, drawn from ` +
        `${payload.matchingClaimCount.toLocaleString()} claims and ranked by ` +
        `frequency. At most 100 are shown per page.${narrowed}`;
    el.filteredTagsSimilarClear.disabled = payload.similarTo.length === 0;
    this.syncFilteredTagSelection();
  }

  /**
   * Refresh the controls that depend on the checked set. This deliberately
   * does not re-derive that set from the DOM: the list re-renders whenever a
   * search, a page turn, or a graph click reloads the bucket, and rebuilding
   * the set from a freshly painted list would silently drop tags the reader
   * had checked on another page.
   */
  syncFilteredTagSelection(): void {
    const role = el.filteredTagsRole.value as FilteredTagRole;
    const selected = this.filteredTagsSelected.get(role);
    const count = this.selectedFilteredTagCount();
    el.filteredTagsApply.disabled = count === 0 || count > 100;
    el.filteredTagsClear.disabled = count === 0;
    el.filteredTagsSimilar.disabled = (selected?.size || 0) === 0;
    el.filteredTagsSelectionNote.textContent = count
      ? `${count.toLocaleString()} of 100 tags checked. Tags within a role use OR; checked roles combine with AND.`
      : "Up to 100 checked tags can be combined. Tags within one role are unioned; different roles intersect.";
  }

  /**
   * Switch buckets. The search box is cleared because a term that matched
   * subject tags almost never matches relation tags, and carrying it across
   * silently empties the new bucket.
   */
  async changeFilteredTagRole(): Promise<void> {
    el.filteredTagsSearch.value = "";
    await this.refreshFilteredTags(false, 0);
  }

  /** Rank the open bucket by similarity to the averaged checked vectors. */
  async findSimilarTags(): Promise<void> {
    this.syncFilteredTagSelection();
    const role = el.filteredTagsRole.value as FilteredTagRole;
    const checked = [...(this.filteredTagsSelected.get(role) || [])];
    if (!checked.length) return;
    this.filteredTagsSimilarAnchors.set(role, checked);
    el.filteredTagsSimilarNote.textContent =
      `Ranking this bucket against the mean vector of ${checked.length} checked ` +
      `tag${checked.length === 1 ? "" : "s"}. Checked tags stay at the top.`;
    await this.refreshFilteredTags(false, 0);
  }

  async clearSimilarTags(): Promise<void> {
    const role = el.filteredTagsRole.value as FilteredTagRole;
    if (!(this.filteredTagsSimilarAnchors.get(role) || []).length) return;
    this.filteredTagsSimilarAnchors.set(role, []);
    el.filteredTagsSimilarNote.textContent =
      "Check one or more tags, then rank this bucket by embedding similarity to their averaged vector.";
    await this.refreshFilteredTags(false, 0);
  }

  /** Re-rank at the new threshold when a similarity ordering is active. */
  scheduleSimilarTagThreshold(): void {
    const role = el.filteredTagsRole.value as FilteredTagRole;
    if (!(this.filteredTagsSimilarAnchors.get(role) || []).length) return;
    if (this.filteredTagsDebounce !== null) {
      window.clearTimeout(this.filteredTagsDebounce);
    }
    this.filteredTagsDebounce = window.setTimeout(() => {
      this.filteredTagsDebounce = null;
      void this.refreshFilteredTags(false, 0);
    }, 250);
  }

  selectFilteredTagPage(): void {
    const role = el.filteredTagsRole.value as FilteredTagRole;
    const selected = this.filteredTagsSelected.get(role);
    if (!selected) return;
    let remaining = 100 - this.selectedFilteredTagCount();
    for (const input of el.filteredTagsResults.querySelectorAll<HTMLInputElement>(
      'input[type="checkbox"]',
    )) {
      if (remaining <= 0) break;
      if (input.checked || input.disabled) continue;
      input.checked = true;
      selected.add(input.value);
      remaining -= 1;
    }
    this.syncFilteredTagSelection();
  }

  clearFilteredTagSelection(): void {
    for (const selected of this.filteredTagsSelected.values()) selected.clear();
    for (const input of el.filteredTagsResults.querySelectorAll<HTMLInputElement>(
      'input[type="checkbox"]',
    )) input.checked = false;
    this.syncFilteredTagSelection();
  }

  handleFilteredTagChange(input: HTMLInputElement): void {
    const role = (input.dataset.role || el.filteredTagsRole.value) as FilteredTagRole;
    const selected = this.filteredTagsSelected.get(role);
    if (!selected || !input.value) return;
    if (!input.checked) {
      selected.delete(input.value);
    } else if (this.selectedFilteredTagCount() >= 100) {
      input.checked = false;
      this.syncFilteredTagSelection();
      el.filteredTagsSelectionNote.textContent =
        "The filtered graph accepts at most 100 checked tags.";
      return;
    } else {
      selected.add(input.value);
    }
    this.syncFilteredTagSelection();
  }

  previousFilteredTagsPage(): void {
    void this.refreshFilteredTags(false, Math.max(0, this.filteredTagsOffset - 100));
  }

  nextFilteredTagsPage(): void {
    if (!this.filteredTagsHasMore) return;
    void this.refreshFilteredTags(false, this.filteredTagsOffset + 100);
  }

  scheduleFilteredTagSearch(): void {
    if (this.filteredTagsDebounce !== null) {
      window.clearTimeout(this.filteredTagsDebounce);
    }
    this.filteredTagsDebounce = window.setTimeout(() => {
      this.filteredTagsDebounce = null;
      void this.refreshFilteredTags(false, 0);
    }, 250);
  }

  async applyFilteredTags(): Promise<void> {
    this.syncFilteredTagSelection();
    const count = this.selectedFilteredTagCount();
    if (!count || count > 100) return;
    const parameters = new URLSearchParams();
    this.appendFilteredTagScope(parameters);
    this.appendFilteredTagCategories(parameters, true);
    this.appendPairingDirection(parameters);
    for (const role of ["subject", "relation", "object"] as FilteredTagRole[]) {
      const ids = [...(this.filteredTagsSelected.get(role) || [])];
      this.appliedTagFilters.set(role, ids);
      for (const id of ids) parameters.append(`${role}Tag`, id);
    }
    el.themeGranularity.value = "1";
    el.themeGranularityValue.value = "1+";
    el.scope.value = "focus";
    el.filteredTagsApply.disabled = true;
    el.filteredTagsSelectionNote.textContent =
      "Building the thematic graph from the checked low-level tags...";
    const installed = await this.loadGraph(
      `/api/graph/categories/topology/filtered-tags?${parameters}`,
    );
    if (installed && this.payload?.focus.kind === "filtered-tags") {
      el.filteredTagsRelease.hidden = false;
    } else {
      // The slice failed to build or was superseded, so nothing is applied.
      this.clearAppliedTagFilters();
    }
    this.syncFilteredTagSelection();
    // The other role buckets now describe only the claims the graph shows.
    await this.refreshFilteredTags(false, 0);
  }

  private bindCorpusRadialDisclosure(): void {
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
  }

  private corpusRadialViewportPoint(
    nodeIndex: number,
  ): { x: number; y: number } {
    const model = this.corpusRadialModel as CorpusRadialModel;
    return (this.renderer as Sigma).graphToViewport({
      x: model.layout.x[nodeIndex],
      y: model.layout.y[nodeIndex],
    });
  }

  private corpusRadialProjection(): {
    pixelsPerUnit: number;
    projectedRadius: number;
  } {
    const renderer = this.renderer as Sigma;
    const origin = renderer.graphToViewport({ x: 0, y: 0 });
    const unit = renderer.graphToViewport({ x: 1, y: 0 });
    const pixelsPerUnit = Math.hypot(
      unit.x - origin.x,
      unit.y - origin.y,
    );
    return {
      pixelsPerUnit,
      projectedRadius: Math.max(
        0.5,
        Math.min(
          CORPUS_BUBBLE_RADIUS,
          CORPUS_BUBBLE_RADIUS * pixelsPerUnit,
        ),
      ),
    };
  }

  private makeCorpusRenderGraph(): MultiDirectedGraph {
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
  }

  private syncCorpusRadialDiagnostics(level: 0 | 1 | 2 | 3): void {
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
      diagnostics.hoverEffectsEnabled = String(
        GRAPH_HOVER_EFFECTS_ENABLED,
      );
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
      diagnostics.medianTreeDistance = String(
        model.layout.medianTreeDistance,
      );
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
    el.container.dataset.corpusRadialFull = String(
      this.corpusRadialShowAll,
    );
  }

  private refreshCorpusFullView(): void {
    const model = this.corpusRadialModel;
    const graph = this.graph;
    const renderer = this.renderer;
    if (!model || !graph || !renderer || !this.corpusRadialShowAll) return;
    const startedAt = performance.now();
    const projection = this.corpusRadialProjection();
    if (
      Math.abs(projection.projectedRadius - model.projectedRadius) >= 0.01
    ) {
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
  }

  private updateCorpusRadialDisclosure(force = false): void {
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
    const next = preservingExisting && !this.corpusRadialShowAll
      ? new Set(model.admitted)
      : new Set<number>();
    const margin = projectedRadius + 8;
    const points = new Map<number, { x: number; y: number; visible: boolean }>();
    const pointFor = (node: number): { x: number; y: number; visible: boolean } => {
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
  }

  setCorpusShowAll(enabled: boolean): void {
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
  }

  private corpusRadialTextLines(
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
  }

  private drawCorpusRadialOverlay(): void {
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
    ) return;
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
      ) continue;
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
  }

  private bindAtlasDetail(): void {
    if (!this.renderer) return;
    this.renderer.getCamera().on("updated", () => {
      if (this.atlasSettleTimer !== null) {
        window.clearTimeout(this.atlasSettleTimer);
      }
      this.atlasSettleTimer = window.setTimeout(() => {
        this.atlasSettleTimer = null;
        this.updateAtlasViewportDetail();
      }, 110);
      if (this.atlasDetailFrame !== null) return;
      this.atlasDetailFrame = window.requestAnimationFrame(() => {
        this.atlasDetailFrame = null;
        this.updateAtlasDetail();
      });
    });
    this.updateAtlasDetail();
    this.updateAtlasViewportDetail();
  }

  private updateAtlasDetail(): void {
    if (
      !this.renderer ||
      !this.graph ||
      this.payload?.layout.kind !== "atlas"
    ) return;
    const spacing = Number(this.payload.layout.medianSpacing || 1);
    const origin = this.renderer.graphToViewport({ x: 0, y: 0 });
    const spaced = this.renderer.graphToViewport({ x: spacing, y: 0 });
    const projectedSpacing = Math.hypot(
      spaced.x - origin.x,
      spaced.y - origin.y,
    );
    this.atlasProjectedSpacing = projectedSpacing;
    const stops = this.payload.layout.zoomStops || [4, 24, 56];
    const nextLevel: 0 | 1 | 2 | 3 =
      projectedSpacing < stops[0]
        ? 0
        : projectedSpacing < stops[1]
          ? 1
          : projectedSpacing < stops[2]
            ? 2
            : 3;
    if (this.atlasOverlayCanvas) {
      this.atlasOverlayCanvas.dataset.atlasLevel = String(nextLevel);
    }
    if (nextLevel === this.atlasLevel) return;
    this.atlasLevel = nextLevel;
    this.renderer.refresh();
  }

  private updateAtlasViewportDetail(): void {
    if (
      !this.renderer ||
      !this.graph ||
      this.payload?.layout.kind !== "atlas"
    ) return;
    const nextVisible = new Set<string>();
    const visiblePositions = new Map<
      string,
      { x: number; y: number }
    >();
    if (this.atlasLevel >= 1) {
      const { width, height } = this.renderer.getDimensions();
      const origin = this.renderer.graphToViewport({ x: 0, y: 0 });
      const unitX = this.renderer.graphToViewport({ x: 1, y: 0 });
      const unitY = this.renderer.graphToViewport({ x: 0, y: 1 });
      const xAxis = {
        x: unitX.x - origin.x,
        y: unitX.y - origin.y,
      };
      const yAxis = {
        x: unitY.x - origin.x,
        y: unitY.y - origin.y,
      };
      const margin = this.atlasLevel === 1 ? 24 : 80;
      for (const node of this.graph.nodes()) {
        if (Boolean(this.graph.getNodeAttribute(node, "atlasAnchor"))) {
          continue;
        }
        const attributes = this.graph.getNodeAttributes(node);
        const x = Number(attributes.x);
        const y = Number(attributes.y);
        const viewportX = origin.x + x * xAxis.x + y * yAxis.x;
        const viewportY = origin.y + x * xAxis.y + y * yAxis.y;
        if (
          viewportX >= -margin &&
          viewportX <= width + margin &&
          viewportY >= -margin &&
          viewportY <= height + margin
        ) {
          nextVisible.add(node);
          visiblePositions.set(node, {
            x: viewportX,
            y: viewportY,
          });
        }
      }
    }

    const nextContinuity = new Set<string>();
    const localEdges = new Set<string>();
    for (const node of nextVisible) {
      let hasLocalConnection = false;
      let strongestEdge: string | null = null;
      let strongestWeight = Number.NEGATIVE_INFINITY;
      for (const edge of this.graph.edges(node)) {
        if (!Boolean(this.graph.getEdgeAttribute(edge, "atlasRawEdge"))) {
          continue;
        }
        const [source, target] = this.graph.extremities(edge);
        const other = source === node ? target : source;
        if (nextVisible.has(other)) {
          hasLocalConnection = true;
          localEdges.add(edge);
          continue;
        }
        const weight = Number(
          this.graph.getEdgeAttribute(edge, "occurrenceCount") || 1,
        );
        if (weight > strongestWeight) {
          strongestWeight = weight;
          strongestEdge = edge;
        }
      }
      if (
        this.atlasLevel >= 2 &&
        !hasLocalConnection &&
        strongestEdge
      ) {
        nextContinuity.add(strongestEdge);
      }
    }
    const nextLabels = new Set<string>();
    const labelBoxes: Array<[number, number, number, number]> = [];
    const visibleNodeCircles = Array.from(nextVisible).map((node) => {
      const point = visiblePositions.get(node) as { x: number; y: number };
      const label = String(this.graph?.getNodeAttribute(node, "label") || "");
      const size =
        this.atlasLevel === 1
          ? Math.min(
              6,
              1.6 +
                Math.log2(
                  Number(this.graph?.getNodeAttribute(node, "frequency") || 1) +
                    1,
                ) *
                  0.42,
            )
          : Math.min(
              bubbleNodeSize(label),
              this.atlasLevel === 2
                ? Math.max(4, this.atlasProjectedSpacing * 0.15)
                : Math.max(7, this.atlasProjectedSpacing * 0.22),
            );
      return { node, x: point.x, y: point.y, size };
    });
    const labelCandidates = Array.from(nextVisible).sort((left, right) => {
      const priorityDifference =
        Number(this.graph?.getNodeAttribute(right, "atlasPriority") || 0) -
        Number(this.graph?.getNodeAttribute(left, "atlasPriority") || 0);
      return priorityDifference || left.localeCompare(right);
    });
    for (const node of labelCandidates) {
      if (nextLabels.size >= 80) break;
      const point = visiblePositions.get(node);
      if (!point) continue;
      const label = String(this.graph.getNodeAttribute(node, "label") || "");
      const nodeSize =
        this.atlasLevel === 1
          ? Math.min(
              6,
              1.6 +
                Math.log2(
                  Number(this.graph.getNodeAttribute(node, "frequency") || 1) +
                    1,
                ) *
                  0.42,
            )
          : Math.min(
              bubbleNodeSize(label),
              this.atlasLevel === 2
                ? Math.max(4, this.atlasProjectedSpacing * 0.15)
                : Math.max(7, this.atlasProjectedSpacing * 0.22),
            );
      const labelWidth = Math.min(220, label.length * 6.4 + 8);
      const box: [number, number, number, number] = [
        point.x + nodeSize + 3,
        point.y - 8,
        point.x + nodeSize + 3 + labelWidth,
        point.y + 8,
      ];
      if (
        box[2] < 0 ||
        box[0] > this.renderer.getDimensions().width ||
        box[3] < 0 ||
        box[1] > this.renderer.getDimensions().height ||
        labelBoxes.some(
          (other) =>
            box[0] < other[2] &&
            box[2] > other[0] &&
            box[1] < other[3] &&
            box[3] > other[1],
        ) ||
        visibleNodeCircles.some(
          (circle) =>
            circle.node !== node &&
            box[0] < circle.x + circle.size &&
            box[2] > circle.x - circle.size &&
            box[1] < circle.y + circle.size &&
            box[3] > circle.y - circle.size,
        )
      ) continue;
      labelBoxes.push(box);
      nextLabels.add(node);
    }
    const orderedLocalEdges = Array.from(localEdges).sort(
      (left, right) =>
        Number(
          this.graph?.getEdgeAttribute(right, "occurrenceCount") || 1,
        ) -
          Number(
            this.graph?.getEdgeAttribute(left, "occurrenceCount") || 1,
          ) ||
        left.localeCompare(right),
    );
    const nextRegionalEdges = new Set(orderedLocalEdges.slice(0, 32));
    const nextPredicateEdges = new Set<string>();
    const nextPredicateLabelPositions = new Map<
      string,
      { x: number; y: number; width: number; label: string }
    >();
    if (this.atlasLevel >= 3) {
      const predicateBoxes: Array<[number, number, number, number]> = [];
      for (const edge of orderedLocalEdges) {
        if (
          nextPredicateEdges.size >= 12 ||
          !this.atlasPredicateEdges.has(edge)
        ) {
          continue;
        }
        const [source, target] = this.graph.extremities(edge);
        const sourcePoint = visiblePositions.get(source);
        const targetPoint = visiblePositions.get(target);
        if (!sourcePoint || !targetPoint) continue;
        const label = String(
          this.graph.getEdgeAttribute(edge, "relationLabel") || "",
        );
        const width = Math.min(200, label.length * 6.2 + 12);
        let accepted:
          | { x: number; y: number; box: [number, number, number, number] }
          | undefined;
        for (const position of [0.68, 0.32, 0.5]) {
          const centerX =
            sourcePoint.x + (targetPoint.x - sourcePoint.x) * position;
          const centerY =
            sourcePoint.y + (targetPoint.y - sourcePoint.y) * position;
          const box: [number, number, number, number] = [
            centerX - width / 2,
            centerY - 7,
            centerX + width / 2,
            centerY + 7,
          ];
          if (
            box[0] < 3 ||
            box[2] > this.renderer.getDimensions().width - 3 ||
            box[1] < 3 ||
            box[3] > this.renderer.getDimensions().height - 3 ||
            labelBoxes.some(
              (other) =>
                box[0] < other[2] &&
                box[2] > other[0] &&
                box[1] < other[3] &&
                box[3] > other[1],
            ) ||
            predicateBoxes.some(
              (other) =>
                box[0] < other[2] &&
                box[2] > other[0] &&
                box[1] < other[3] &&
                box[3] > other[1],
            ) ||
            visibleNodeCircles.some(
              (circle) =>
                box[0] < circle.x + circle.size &&
                box[2] > circle.x - circle.size &&
                box[1] < circle.y + circle.size &&
                box[3] > circle.y - circle.size,
            )
          ) {
            continue;
          }
          accepted = { x: centerX, y: centerY, box };
          break;
        }
        if (!accepted) continue;
        predicateBoxes.push(accepted.box);
        nextPredicateEdges.add(edge);
        nextPredicateLabelPositions.set(edge, {
          x: accepted.x,
          y: accepted.y,
          width,
          label,
        });
      }
    }
    this.atlasPredicateLabelPositions = nextPredicateLabelPositions;
    if (
      sameSet(this.atlasVisibleNodes, nextVisible) &&
      sameSet(this.atlasContinuityEdges, nextContinuity) &&
      sameSet(this.atlasViewportLabels, nextLabels) &&
      sameSet(this.atlasViewportPredicateEdges, nextPredicateEdges) &&
      sameSet(this.atlasViewportRegionalEdges, nextRegionalEdges)
    ) {
      if (this.atlasLevel >= 3) this.renderer.refresh();
      return;
    }
    this.atlasVisibleNodes = nextVisible;
    this.atlasContinuityEdges = nextContinuity;
    this.atlasViewportLabels = nextLabels;
    this.atlasViewportPredicateEdges = nextPredicateEdges;
    this.atlasViewportRegionalEdges = nextRegionalEdges;
    if (this.atlasOverlayCanvas) {
      this.atlasOverlayCanvas.dataset.visibleNodes = String(
        nextVisible.size,
      );
      this.atlasOverlayCanvas.dataset.continuityEdges = String(
        nextContinuity.size,
      );
      this.atlasOverlayCanvas.dataset.nodeLabels = String(nextLabels.size);
      this.atlasOverlayCanvas.dataset.predicateLabels = String(
        nextPredicateEdges.size,
      );
      this.atlasOverlayCanvas.dataset.regionalEdges = String(
        nextRegionalEdges.size,
      );
    }
    this.renderer.refresh();
  }

  private bindAdaptiveDetail(): void {
    if (!this.renderer) return;
    this.renderer.getCamera().on("updated", () => {
      this.scheduleProgressiveDetail();
    });
    this.scheduleProgressiveDetail();
  }

  private scheduleProgressiveDetail(): void {
    if (this.progressiveDetailFrame !== null) return;
    this.progressiveDetailFrame = window.requestAnimationFrame(() => {
      this.progressiveDetailFrame = null;
      this.updateProgressiveDetail();
    });
  }

  private updateProgressiveDetail(): void {
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
      ) continue;
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
      visibleNodes.size <= edgeLabelCapacity
        ? visibleNodes
        : new Set<string>();
    if (
      sameSet(this.fullyLabeledNodes, nextFullyLabeled) &&
      sameSet(this.edgeLabeledNodes, nextEdgeLabeled)
    ) return;
    this.fullyLabeledNodes = nextFullyLabeled;
    this.edgeLabeledNodes = nextEdgeLabeled;
    this.renderer.refresh();
  }

  private runForceLayout(): void {
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
  }

  private reduceNode(node: string, data: Record<string, unknown>) {
    if (Boolean(data.thematicNode)) {
      const active =
        Number(
          (data.categoryDetail as ThematicEntityCategory)
            .activeMentionCount,
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
  }

  private reduceEdge(edge: string, data: Record<string, unknown>) {
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
        source === activeNode
          ? target
          : target === activeNode
            ? source
            : null;
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
      if (
        this.atlasLevel >= 1 ||
        !Boolean(data.atlasOverviewBundle)
      ) {
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
      if (
        this.atlasLevel === 0 &&
        !Boolean(data.atlasSmallComponent)
      ) {
        return { ...data, hidden: true };
      }
      if (
        this.atlasLevel === 1 &&
        !Boolean(data.atlasInternal) &&
        !Boolean(data.atlasSmallComponent)
      ) {
        return { ...data, hidden: true };
      }
      if (
        this.atlasLevel === 1 &&
        !this.atlasViewportRegionalEdges.has(edge)
      ) {
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
  }

  private bindRendererEvents(): void {
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
  }

  private updateHeader(): void {
    if (!this.payload) return;
    el.heading.textContent = this.payload.focus.label || "Knowledge graph";
    const claims = this.payload.claimCount.toLocaleString();
    if (this.thematicPayload) {
      const payload = this.thematicPayload;
      el.heading.textContent = payload.focus.label || "Thematic graph";
      if (new Set(["tag-filter", "filtered-tags"]).has(payload.focus.kind)) {
        const activeEntities = payload.entityCategories.filter(
          (category) => category.activeMentionCount > 0,
        ).length;
        const activeRelations = payload.relationCategories.filter(
          (category) => category.activeTripleCount > 0,
        ).length;
        const activePatterns = payload.patterns.filter(
          (pattern) => pattern[4] > 0,
        ).length;
        const tagCount = payload.focus.tagIds?.length || 0;
        const tagKind = payload.focus.kind === "filtered-tags"
          ? "low-level"
          : payload.focus.tagKind || "entity";
        el.counts.textContent =
          `${tagCount.toLocaleString()} selected ${tagKind} tags · ` +
          `${activeEntities.toLocaleString()} entity categories · ` +
          `${activeRelations.toLocaleString()} relation categories · ` +
          `${activePatterns.toLocaleString()} exact patterns · ` +
          `${payload.claimCount.toLocaleString()} triples`;
        return;
      }
      el.counts.textContent =
        `${payload.entityCategories.length.toLocaleString()} entity categories · ` +
        `${payload.relationCategories.length.toLocaleString()} relation categories · ` +
        `${payload.patterns.length.toLocaleString()} exact patterns · ` +
        `${payload.claimCount.toLocaleString()} active triples`;
      return;
    }
    el.counts.textContent =
      `${this.payload.nodes.length.toLocaleString()} nodes · ` +
      `${this.payload.edges.length.toLocaleString()} relations · ` +
      `${claims} claims${this.payload.truncated ? " · partial view" : ""}`;
  }

  private renderFocusDetail(): void {
    if (!this.payload) return;
    if (this.thematicPayload) {
      const payload = this.thematicPayload;
      clear(el.detail);
      const filteredTagSlice = payload.focus.kind === "filtered-tags";
      const semanticSlice = payload.focus.kind === "tag-filter" || filteredTagSlice;
      const tagSliceDescription = filteredTagSlice
        ? "Within each role, a triple matches any checked tag. Checked subject, relation, and object roles are intersected with the selected thematic buckets."
        : payload.focus.tagKind === "relation"
        ? "This scope includes a triple when its raw relation tag matches any checked tag. Every match is included, including one-off patterns."
        : "This scope includes a triple when its raw subject or object tag matches any checked tag. Every match is included, including one-off patterns.";
      const activeEntities = payload.entityCategories.filter(
        (category) => category.activeMentionCount > 0,
      ).length;
      const activeRelations = payload.relationCategories.filter(
        (category) => category.activeTripleCount > 0,
      ).length;
      const activePatterns = payload.patterns.filter(
        (pattern) => pattern[4] > 0,
      ).length;
      el.detail.appendChild(
        textElement(
          "h3",
          "graph-detail-title",
          payload.focus.label || "Thematic graph",
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-type",
          semanticSlice
            ? `semantic union · ${(payload.focus.tagIds?.length || 0).toLocaleString()} selected ${payload.focus.tagKind || "entity"} tags`
            : `${payload.entityCategories.length.toLocaleString()} entity categories · ` +
              `${payload.relationCategories.length.toLocaleString()} relation categories`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-summary",
          semanticSlice
            ? `${activeEntities.toLocaleString()} entity categories · ` +
              `${activeRelations.toLocaleString()} relation categories · ` +
              `${activePatterns.toLocaleString()} exact patterns · ` +
              `${payload.claimCount.toLocaleString()} unioned triples`
            : `${payload.patterns.length.toLocaleString()} exact entity → relation → entity patterns · ` +
              `${payload.claimCount.toLocaleString()} active triples`,
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          semanticSlice
            ? tagSliceDescription
            : "Every category and exact pattern is resident. Faint paths preserve the whole-corpus context; click a category to foreground all of its real paths.",
        ),
      );
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          "Click a second entity to focus the pair. Purple paths run out of the anchored group, teal paths run into it, and orange paths join two entities that are not the anchor.",
        ),
      );
      return;
    }
    const { focus } = this.payload;
    clear(el.detail);
    el.detail.appendChild(
      textElement("h3", "graph-detail-title", focus.label || "Graph"),
    );
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-type",
        focus.kind,
      ),
    );
    const shown = this.payload.claimCount.toLocaleString();
    const available = this.payload.availableCount;
    const description =
      available !== null && available !== this.payload.claimCount
        ? `Showing ${shown} of ${available.toLocaleString()} matching records.`
        : `${shown} matching claims in this view.`;
    el.detail.appendChild(
      textElement("div", "graph-detail-summary", description),
    );
    if (this.payload.truncated) {
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          "This view is capped to keep navigation responsive. The source RDF remains complete.",
        ),
      );
    }
    if (focus.kind === "relation") {
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          "Subjects are arranged in blue on the left and objects in green on the right; purple entities occur in both roles.",
        ),
      );
    }
    if (focus.kind === "page") {
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          "Connected claims are packed into compact groups so separate statements remain readable instead of drifting apart.",
        ),
      );
    }
    if (focus.kind === "range") {
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          "This view contains every distinct RDF claim appearing on any available page inside the selected inclusive range. It opens at readable detail; pan to explore or use Fit graph for the complete overview.",
        ),
      );
    }
    if (focus.kind === "entity") {
      this.appendEntityActions(focus.id);
    } else if (focus.kind === "relation") {
      this.appendRelationActions(focus.id);
    } else {
      el.detail.appendChild(
        textElement(
          "div",
          "graph-detail-status",
          "Click a node or directed edge to inspect and expand it.",
        ),
      );
    }
  }

  private appendEntityActions(entityId: string): void {
    const actions = document.createElement("div");
    actions.className = "graph-detail-actions";
    actions.append(
      actionButton("1-hop", "entity", `${entityId}:1`),
      actionButton("2-hop", "entity", `${entityId}:2`),
      actionButton("Gemini similar", "similar", `entity:${entityId}`),
    );
    el.detail.appendChild(actions);
  }

  private appendRelationActions(relationId: string): void {
    const actions = document.createElement("div");
    actions.className = "graph-detail-actions";
    actions.append(
      actionButton("Open relation", "relation", relationId),
      actionButton("Gemini similar", "similar", `relation:${relationId}`),
    );
    el.detail.appendChild(actions);
  }

  /**
   * A history layer is a copy of the state record plus the archived raw-view
   * fields. Listing the thematic fields by hand here is what let the snapshot
   * fall behind the state it was meant to capture: the pairing direction and
   * edge orientation were added to the selection but never to the snapshot,
   * so stepping back silently dropped them.
   */
  private navigationSnapshot(): GraphNavigationSnapshot {
    return {
      nav: cloneNavigationState(this.nav),
      selectedNeighbors: Array.from(this.selectedNeighbors),
      rawPrimaryEntity: this.rawPrimaryEntity,
      rawPairEntity: this.rawPairEntity,
      rawContextRelation: this.rawContextRelation,
      rawSelectedEdges: Array.from(this.rawSelectedEdges),
    };
  }

  private pushNavigationState(): void {
    this.navigationHistory.push(this.navigationSnapshot());
  }

  private restoreNavigationState(snapshot: GraphNavigationSnapshot): void {
    this.evidenceAbort?.abort();
    this.nav = cloneNavigationState(snapshot.nav);
    this.selectedNeighbors = new Set(snapshot.selectedNeighbors);
    this.rawPrimaryEntity = snapshot.rawPrimaryEntity;
    this.rawPairEntity = snapshot.rawPairEntity;
    this.rawContextRelation = snapshot.rawContextRelation;
    this.rawSelectedEdges = new Set(snapshot.rawSelectedEdges);
    this.thematicEvidencePattern = null;
    this.evidenceItems = [];
    this.evidenceHasMore = false;
    this.evidenceNextOffset = null;
    el.container.classList.toggle("node-hover", Boolean(this.selectedNode));
    if (this.thematicPayload) {
      this.setThematicSelection(this.thematicSelectedPatterns);
      // commitThematicSelection() republishes the restored state to all four
      // surfaces, so the panels cannot lag a step behind the canvas.
      this.commitThematicSelection(performance.now());
      if (!this.selectedNode) this.renderFocusDetail();
    } else if (this.rawPairEntity || this.rawContextRelation) {
      if (this.selectedEdge) {
        this.renderEdgeDetail();
        void this.loadEvidence(0);
      } else {
        this.renderRawNavigationDetail();
      }
    } else if (this.selectedEdge) {
      this.renderEdgeDetail();
      void this.loadEvidence(0);
    } else if (this.selectedNode) {
      this.renderRawEntityDetail(this.selectedNode);
    } else {
      this.renderFocusDetail();
    }
    this.scheduleProgressiveDetail();
    this.renderer?.refresh();
  }

  navigateBackOneLayer(): void {
    if (this.thematicEvidencePattern !== null) {
      this.evidenceAbort?.abort();
      this.thematicEvidencePattern = null;
      this.evidenceItems = [];
      this.evidenceHasMore = false;
      this.evidenceNextOffset = null;
      this.renderThematicSelection();
      return;
    }
    const previous = this.navigationHistory.pop();
    if (previous) {
      this.restoreNavigationState(previous);
      return;
    }
    this.clearSelection(false);
  }

  private rawEdgesForSelection(
    primary: string,
    pair: string | null,
    relationId: string | null,
  ): string[] {
    if (!this.graph) return [];
    return this.graph.edges().filter((edge) => {
      if (!this.graph) return false;
      const [source, target] = this.graph.extremities(edge);
      if (source !== primary) return false;
      if (pair !== null && target !== pair) return false;
      return (
        relationId === null ||
        String(this.graph.getEdgeAttribute(edge, "relationId")) === relationId
      );
    });
  }

  private applyRawNavigationSelection(edges: string[]): void {
    if (!this.graph || !this.rawPrimaryEntity) return;
    this.rawSelectedEdges = new Set(edges);
    this.selectedNode = this.rawPrimaryEntity;
    this.selectedEdge =
      this.rawPairEntity && this.rawContextRelation && edges.length === 1
        ? edges[0]
        : null;
    this.selectedNeighbors.clear();
    for (const edge of edges) {
      const [source, target] = this.graph.extremities(edge);
      if (source === this.rawPrimaryEntity) this.selectedNeighbors.add(target);
    }
    if (this.rawPairEntity) this.selectedNeighbors.add(this.rawPairEntity);
    el.detailPanel.open = true;
    el.container.classList.add("node-hover");
    if (this.selectedEdge) {
      this.evidenceItems = [];
      this.evidenceHasMore = false;
      this.evidenceNextOffset = null;
      this.renderEdgeDetail();
      void this.loadEvidence(0);
    } else {
      this.renderRawNavigationDetail();
    }
    this.scheduleProgressiveDetail();
    this.renderer?.refresh();
  }

  private renderRawNavigationDetail(): void {
    if (!this.graph || !this.rawPrimaryEntity) return;
    const sourceLabel = String(
      this.graph.getNodeAttribute(this.rawPrimaryEntity, "label"),
    );
    const targetLabel = this.rawPairEntity
      ? String(this.graph.getNodeAttribute(this.rawPairEntity, "label"))
      : "any object";
    const relations = new Map<
      string,
      { label: string; claims: number }
    >();
    let claims = 0;
    for (const edge of this.rawSelectedEdges) {
      const relationId = String(this.graph.getEdgeAttribute(edge, "relationId"));
      const count = Number(this.graph.getEdgeAttribute(edge, "occurrenceCount"));
      const current = relations.get(relationId) || {
        label: String(this.graph.getEdgeAttribute(edge, "relationLabel")),
        claims: 0,
      };
      current.claims += count;
      claims += count;
      relations.set(relationId, current);
    }
    clear(el.detail);
    const relationLabel = this.rawContextRelation
      ? relations.get(this.rawContextRelation)?.label || this.rawContextRelation
      : null;
    el.detail.appendChild(
      textElement(
        "h3",
        "graph-detail-title",
        relationLabel
          ? `${sourceLabel} → ${relationLabel} → ${targetLabel}`
          : `${sourceLabel} → ${targetLabel}`,
      ),
    );
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-type",
        this.rawPairEntity
          ? "ordered raw entity-pair focus"
          : "raw subject + relation focus",
      ),
    );
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-summary",
        `${relations.size.toLocaleString()} relation types · ${claims.toLocaleString()} visible claims`,
      ),
    );
    if (!relations.size) {
      el.detail.appendChild(
        textElement(
          "div",
          "graph-empty",
          "No directed subject → relation → object claims match this selection.",
        ),
      );
      return;
    }
    const actions = document.createElement("div");
    actions.className = "graph-detail-actions";
    for (const [relationId, relation] of Array.from(relations).sort(
      (left, right) => right[1].claims - left[1].claims,
    )) {
      actions.appendChild(
        actionButton(
          `${relation.label} (${relation.claims.toLocaleString()})`,
          "raw-context-relation",
          relationId,
        ),
      );
    }
    el.detail.appendChild(actions);
  }

  private renderRawEntityDetail(node: string): void {
    if (!this.graph || !this.graph.hasNode(node)) return;
    const data = this.graph.getNodeAttributes(node);
    clear(el.detail);
    el.detail.appendChild(
      textElement("h3", "graph-detail-title", String(data.label)),
    );
    el.detail.appendChild(
      textElement("div", "graph-detail-type", `raw entity · ${node}`),
    );
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-summary",
        `${Number(data.frequency).toLocaleString()} corpus occurrences · ` +
          `${Number(
            data.corpusDegree ?? this.graph.degree(node),
          ).toLocaleString()} corpus connections`,
      ),
    );
    this.appendEntityActions(node);
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-status",
        "Click an object entity to show every directed relation from this subject; click a relation label first to constrain that next object selection.",
      ),
    );
  }

  private selectNode(node: string): void {
    if (!this.graph) return;
    if (this.rawPrimaryEntity && this.rawPrimaryEntity !== node) {
      this.pushNavigationState();
      this.rawPairEntity = node;
      this.applyRawNavigationSelection(
        this.rawEdgesForSelection(
          this.rawPrimaryEntity,
          node,
          this.rawContextRelation,
        ),
      );
      return;
    }
    this.pushNavigationState();
    el.detailPanel.open = true;
    this.evidenceAbort?.abort();
    this.rawPrimaryEntity = node;
    this.rawPairEntity = null;
    this.rawContextRelation = null;
    this.rawSelectedEdges.clear();
    this.selectedNode = node;
    this.selectedEdge = null;
    const allNeighbors = this.graph.neighbors(node);
    const visibleNeighbors = allNeighbors;
    this.selectedNeighbors = new Set(visibleNeighbors);
    const data = this.graph.getNodeAttributes(node);
    this.updateHeader();

    clear(el.detail);
    el.detail.appendChild(
      textElement("h3", "graph-detail-title", String(data.label)),
    );
    el.detail.appendChild(
      textElement("div", "graph-detail-type", `raw entity · ${node}`),
    );
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-summary",
        `${Number(data.frequency).toLocaleString()} corpus occurrences · ` +
          `${Number(
            data.corpusDegree ?? this.graph.degree(node),
          ).toLocaleString()} corpus connections` +
          (allNeighbors.length > visibleNeighbors.length
            ? ` · showing top ${visibleNeighbors.length.toLocaleString()}`
            : ""),
      ),
    );
    this.appendEntityActions(node);
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-status",
        "Click an object entity to show every directed relation from this subject; click a relation label first to constrain that next object selection.",
      ),
    );
    if (this.payload?.layout.kind === "atlas") {
      this.renderer?.scheduleRefresh();
    } else {
      this.scheduleProgressiveDetail();
      this.renderer?.refresh();
    }
  }

  private selectEdge(edge: string): void {
    if (!this.graph) return;
    if (this.rawPrimaryEntity) {
      this.pushNavigationState();
      this.rawContextRelation = String(
        this.graph.getEdgeAttribute(edge, "relationId"),
      );
      this.applyRawNavigationSelection(
        this.rawEdgesForSelection(
          this.rawPrimaryEntity,
          this.rawPairEntity,
          this.rawContextRelation,
        ),
      );
      return;
    }
    this.pushNavigationState();
    el.detailPanel.open = true;
    this.selectedNode = null;
    this.selectedEdge = edge;
    this.selectedNeighbors.clear();
    this.evidenceItems = [];
    this.evidenceHasMore = false;
    this.evidenceNextOffset = null;
    this.renderEdgeDetail();
    if (this.payload?.layout.kind === "atlas") {
      this.renderer?.scheduleRefresh();
    } else {
      this.scheduleProgressiveDetail();
      this.renderer?.refresh();
    }
    void this.loadEvidence(0);
  }

  private renderEdgeDetail(errorMessage = ""): void {
    if (!this.graph || !this.selectedEdge) return;
    const data = this.graph.getEdgeAttributes(this.selectedEdge);
    const source = String(data.sourceId);
    const target = String(data.targetId);
    const sourceLabel = String(this.graph.getNodeAttribute(source, "label"));
    const targetLabel = String(this.graph.getNodeAttribute(target, "label"));
    clear(el.detail);
    el.detail.appendChild(
      textElement("h3", "graph-detail-title", String(data.relationLabel)),
    );
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-type",
        `raw directed relation · ${String(data.relationId)}`,
      ),
    );
    el.detail.appendChild(
      textElement(
        "div",
        "graph-detail-summary",
        `${sourceLabel} → ${targetLabel} · ` +
          `${Number(data.occurrenceCount).toLocaleString()} visible claims`,
      ),
    );
    this.appendRelationActions(String(data.relationId));

    if (this.evidenceItems.length) {
      for (const item of this.evidenceItems) {
        const claim = document.createElement("article");
        claim.className = "graph-claim";
        claim.appendChild(
          textElement(
            "div",
            "graph-claim-spo",
            `${sourceLabel} — ${String(data.relationLabel)} → ${targetLabel}`,
          ),
        );
        claim.appendChild(
          textElement("div", "graph-claim-my", item.sentenceMy),
        );
        claim.appendChild(
          textElement("div", "graph-claim-en", item.sentenceEn),
        );
        const link = document.createElement("a");
        link.className = "graph-page-link";
        link.href = `/chronicles/${item.volumeId}/${item.ownerPage}`;
        link.textContent =
          `${item.volumeId.toUpperCase()}, page ${item.ownerPage} · ` +
          `${item.sentenceId} #${item.ordinal}`;
        claim.appendChild(link);
        el.detail.appendChild(claim);
      }
    }
    if (this.evidenceLoading) {
      el.detail.appendChild(
        textElement("div", "graph-detail-status", "Loading source evidence…"),
      );
    } else if (errorMessage) {
      el.detail.appendChild(
        textElement("div", "graph-empty", errorMessage),
      );
    } else if (!this.evidenceItems.length) {
      el.detail.appendChild(
        textElement("div", "graph-empty", "No source evidence found."),
      );
    }
    if (this.evidenceHasMore && !this.evidenceLoading) {
      el.detail.appendChild(
        actionButton(
          "Load more evidence",
          "evidence-more",
          "",
          "graph-evidence-more",
        ),
      );
    }
  }

  private async loadEvidence(offset: number): Promise<void> {
    if (!this.graph || !this.selectedEdge) return;
    this.evidenceAbort?.abort();
    this.evidenceAbort = new AbortController();
    const request = this.evidenceAbort;
    const edge = this.selectedEdge;
    const data = this.graph.getEdgeAttributes(edge);
    const parameters = new URLSearchParams({
      source: String(data.sourceId),
      target: String(data.targetId),
      offset: String(offset),
      limit: "20",
    });
    parameters.set("relation", String(data.relationId));
    if (
      this.payload?.focus.kind === "claim" &&
      this.payload.focus.sentenceId &&
      this.payload.focus.ordinal
    ) {
      parameters.set("sentenceId", this.payload.focus.sentenceId);
      parameters.set("ordinal", String(this.payload.focus.ordinal));
    }
    this.evidenceLoading = true;
    this.renderEdgeDetail();
    try {
      const payload = await fetchJson<EvidencePayload>(
        `/api/graph/evidence?${parameters}`,
        request.signal,
      );
      if (
        request !== this.evidenceAbort ||
        edge !== this.selectedEdge
      ) return;
      this.evidenceItems.push(...payload.items);
      this.evidenceHasMore = payload.hasMore;
      this.evidenceNextOffset = payload.nextOffset;
      this.evidenceLoading = false;
      this.renderEdgeDetail();
    } catch (error) {
      if (request.signal.aborted) return;
      this.evidenceLoading = false;
      this.renderEdgeDetail(
        error instanceof Error ? error.message : String(error),
      );
    }
  }

  private clearSelection(clearHistory = true): void {
    this.evidenceAbort?.abort();
    if (clearHistory) this.navigationHistory = [];
    // A filtered-tag slice keeps the categories that produced it; anything
    // else resets the record outright.
    const keepCategories =
      this.thematicPayload?.focus.kind === "filtered-tags"
        ? this.nav.categories
        : { subjects: [], relations: [], objects: [] };
    this.nav = emptyNavigationState();
    this.nav.categories = keepCategories;
    this.rawPrimaryEntity = null;
    this.rawPairEntity = null;
    this.rawContextRelation = null;
    this.rawSelectedEdges.clear();
    this.hoveredNode = null;
    this.hoveredEdge = null;
    this.lineHoveredEdge = null;
    this.labelHoveredEdge = null;
    this.thematicHoveredRelationLabel = null;
    this.thematicHoveredPattern = null;
    this.thematicHoveredRelationPoint = null;
    el.container.classList.remove("node-hover");
    el.container.classList.remove("edge-label-hover");
    this.selectedNeighbors.clear();
    this.thematicEvidencePattern = null;
    this.publishNavigationToPanels();
    const semanticUnionRestored = this.activateSemanticTagUnion();
    if (this.thematicInspectorFrame !== null) {
      window.cancelAnimationFrame(this.thematicInspectorFrame);
      this.thematicInspectorFrame = null;
    }
    if (this.thematicOverlayCanvas) {
      if (!semanticUnionRestored) {
        this.thematicOverlayCanvas.dataset.selectedPatterns = "0";
        this.thematicOverlayCanvas.dataset.selectedRelations = "0";
        this.thematicOverlayCanvas.dataset.selectedCounterparts = "0";
        this.thematicOverlayCanvas.dataset.availableSelectionPatterns = "0";
        this.thematicOverlayCanvas.dataset.thematicFocusMode = "overview";
      }
      delete this.thematicOverlayCanvas.dataset.hoveredRelation;
      delete this.thematicOverlayCanvas.dataset.hoveredRelationTag;
      delete this.thematicOverlayCanvas.dataset.hoveredRelationLabel;
      delete this.thematicOverlayCanvas.dataset.hoveredPattern;
    }
    this.scheduleProgressiveDetail();
    this.renderer?.refresh();
    this.updateHeader();
    this.renderFocusDetail();
  }

  private async loadEntity(id: string, depth: number): Promise<void> {
    el.scope.value = "focus";
    await this.loadGraph(
      `/api/graph/topology/entity/${encodeURIComponent(id)}` +
        `?depth=${depth}&limit=${depth === 2 ? 500 : 300}`,
    );
  }

  private async filterByEntityCategory(id: string): Promise<void> {
    await this.ensureCategoryCatalog();
    this.setCategoryValues(el.entityCategories, [id]);
    this.setCategoryValues(el.objectCategories);
    this.setCategoryValues(el.relationCategories);
    const panel = el.browserSummary.closest("details");
    if (panel instanceof HTMLDetailsElement) panel.open = true;
    // applyThematicCategorySelection() commits, which republishes both panels;
    // refreshing the tag bucket here as well raced that and dropped whatever
    // tags were checked.
    this.applyThematicCategorySelection();
  }

  private async filterByRelationCategory(id: string): Promise<void> {
    await this.ensureCategoryCatalog();
    this.setCategoryValues(el.entityCategories);
    this.setCategoryValues(el.objectCategories);
    this.setCategoryValues(el.relationCategories, [id]);
    const panel = el.browserSummary.closest("details");
    if (panel instanceof HTMLDetailsElement) panel.open = true;
    this.applyThematicCategorySelection();
  }

  private centerNode(id: string): void {
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
        Math.min(state.ratio, state.ratio * projectedSpacing / 64),
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
  }

  // ------------------------------------------------------------------
  // Similar statement patterns
  //
  // The anchor is whatever subject / relation / object categories are locked
  // in right now, whether they were chosen in the selects or reached by
  // clicking through the graph. The server averages the raw tag vectors of
  // every claim under that anchor and ranks the other in-scope patterns
  // against it.
  // ------------------------------------------------------------------

  private similarPatternKey(item: SimilarPatternResult): string {
    return `${item.subjectId}|${item.relationId}|${item.objectId}`;
  }

  private themeSimilarityThreshold(): number {
    return Number(el.themeSimilarity.value) / 100;
  }

  syncSimilarPatternAnchor(): void {
    const subjects = this.selectedCategoryValues(el.entityCategories);
    const relations = this.selectedCategoryValues(el.relationCategories);
    const objects = this.selectedCategoryValues(el.objectCategories);
    const total = subjects.length + relations.length + objects.length;
    el.themeSimilar.disabled = total === 0;
    if (!total) {
      el.themeSimilarAnchor.textContent =
        "Apply or click a subject, relation, or object category to set the anchor pattern.";
      return;
    }
    const parts = [
      subjects.length ? `subject ${subjects.join(", ")}` : "",
      relations.length ? `relation ${relations.join(", ")}` : "",
      objects.length ? `object ${objects.join(", ")}` : "",
    ].filter(Boolean);
    el.themeSimilarAnchor.textContent = `Anchor: ${parts.join(" · ")}`;
  }

  clearSimilarPatterns(): void {
    this.similarPatternsAbort?.abort();
    this.similarPatterns = [];
    this.similarPatternsSelected.clear();
    clear(el.themeSimilarResults);
    el.themeSimilarNote.textContent = "";
    el.themeSimilarClear.disabled = true;
    el.themeSimilarApply.disabled = true;
  }

  async loadSimilarPatterns(): Promise<void> {
    if (!this.selectedCategoryValues(el.entityCategories).length &&
        !this.selectedCategoryValues(el.relationCategories).length &&
        !this.selectedCategoryValues(el.objectCategories).length) {
      return;
    }
    this.similarPatternsAbort?.abort();
    this.similarPatternsAbort = new AbortController();
    const request = this.similarPatternsAbort;
    const parameters = new URLSearchParams({
      minSimilarity: this.themeSimilarityThreshold().toFixed(2),
      limit: "60",
    });
    this.appendFilteredTagCategories(parameters);
    // The anchor is what is actually on screen, so a live tag filter narrows
    // the vector the search is built from.
    this.appendAppliedTagFilters(parameters);
    this.appendFilteredTagScope(parameters);
    el.themeSimilar.disabled = true;
    this.appendPairingDirection(parameters);
    el.themeSimilarNote.textContent =
      "Averaging the raw tag vectors under the anchor pattern…";
    try {
      const payload = await fetchJson<SimilarPatternsPayload>(
        `/api/graph/categories/patterns/similar?${parameters}`,
        request.signal,
      );
      if (request !== this.similarPatternsAbort) return;
      this.similarPatterns = payload.items;
      const visible = new Set(payload.items.map((item) => this.similarPatternKey(item)));
      this.similarPatternsSelected = new Set(
        [...this.similarPatternsSelected].filter((key) => visible.has(key)),
      );
      this.renderSimilarPatterns(payload);
    } catch (error) {
      if (request.signal.aborted) return;
      this.similarPatterns = [];
      clear(el.themeSimilarResults);
      el.themeSimilarNote.textContent =
        error instanceof Error ? error.message : String(error);
    } finally {
      if (request === this.similarPatternsAbort) {
        this.syncSimilarPatternAnchor();
      }
    }
  }

  private renderSimilarPatterns(payload: SimilarPatternsPayload): void {
    clear(el.themeSimilarResults);
    if (!payload.items.length) {
      el.themeSimilarResults.appendChild(
        textElement(
          "div",
          "graph-empty",
          "No other pattern reaches this similarity. Lower the slider.",
        ),
      );
    }
    for (const item of payload.items) {
      const key = this.similarPatternKey(item);
      const label = document.createElement("label");
      label.className = "graph-similar-pattern";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = key;
      checkbox.checked = this.similarPatternsSelected.has(key);
      const copy = document.createElement("span");
      copy.className = "graph-similar-pattern-copy";
      copy.appendChild(
        textElement(
          "span",
          "graph-result-label",
          `${item.subjectLabel} → ${item.relationLabel} → ${item.objectLabel}`,
        ),
      );
      copy.appendChild(
        textElement(
          "span",
          "graph-result-meta",
          `${(item.similarity * 100).toFixed(1)}% similar · ` +
            `${item.tripleCount.toLocaleString()} claims`,
        ),
      );
      label.append(checkbox, copy);
      el.themeSimilarResults.appendChild(label);
    }
    const narrowedBy = Object.entries(payload.narrowedBy || {}).map(
      ([role, labels]) => `${(labels || []).length} ${role} tag` +
        `${(labels || []).length === 1 ? "" : "s"}`,
    );
    el.themeSimilarNote.textContent =
      `${payload.totalCandidates.toLocaleString()} patterns are at least ` +
      `${el.themeSimilarity.value}% similar to the anchor's ` +
      `${payload.anchor.claimCount.toLocaleString()} claims across ` +
      `${payload.anchor.patternCount.toLocaleString()} anchor patterns. ` +
      `Comparing ${payload.comparedRoles.join(", ")}.` +
      (narrowedBy.length
        ? ` Anchor narrowed by the applied ${narrowedBy.join(" and ")}.`
        : "");
    el.themeSimilarClear.disabled = false;
    this.syncSimilarPatternSelection();
  }

  syncSimilarPatternSelection(): void {
    this.similarPatternsSelected = new Set(
      [...el.themeSimilarResults.querySelectorAll<HTMLInputElement>(
        'input[type="checkbox"]:checked',
      )].map((input) => input.value),
    );
    el.themeSimilarApply.disabled = this.similarPatternsSelected.size === 0;
  }

  applySimilarPatterns(): void {
    this.syncSimilarPatternSelection();
    if (!this.thematicPayload || !this.similarPatternsSelected.size) return;
    const entityIds = new Map(
      this.thematicPayload.entityCategories.map(
        (category, index) => [category.id, index] as const,
      ),
    );
    const relationIds = new Map(
      this.thematicPayload.relationCategories.map(
        (category, index) => [category.id, index] as const,
      ),
    );
    const wanted = new Set<string>();
    let missing = 0;
    for (const key of this.similarPatternsSelected) {
      const [subjectId, relationId, objectId] = key.split("|");
      const subject = entityIds.get(subjectId);
      const relation = relationIds.get(relationId);
      const object = entityIds.get(objectId);
      if (subject === undefined || relation === undefined || object === undefined) {
        missing += 1;
        continue;
      }
      wanted.add(`${subject}|${relation}|${object}`);
    }
    const added = this.thematicPayload.patterns.flatMap((pattern, index) =>
      wanted.has(`${pattern[0]}|${pattern[1]}|${pattern[2]}`) ? [index] : [],
    );
    if (!added.length) {
      el.themeSimilarNote.textContent =
        "None of the checked patterns are present in the current graph scope.";
      return;
    }
    this.pushNavigationState();
    const startedAt = performance.now();
    const union = new Set([...this.thematicSelectionCandidates, ...added]);
    this.thematicExactPatternFocus = null;
    this.thematicPairEntity = null;
    this.thematicContextRelation = null;
    this.thematicSelectionCandidates = [...union];
    this.setThematicSelection(
      this.filteredThematicPatterns(this.thematicSelectionCandidates),
    );
    const first = this.thematicPayload.patterns[added[0]];
    this.selectedNode = this.selectedNode || this.thematicEntityNodeId(
      this.thematicPayload.entityCategories[first[0]].id,
    );
    this.thematicPrimaryEntity = this.selectedNode;
    // The themes must describe the widened selection, not the anchor alone.
    const subjects = new Set<number>();
    const relations = new Set<number>();
    const objects = new Set<number>();
    for (const patternIndex of this.thematicSelectionCandidates) {
      const pattern = this.thematicPayload.patterns[patternIndex];
      if (!pattern) continue;
      subjects.add(pattern[0]);
      relations.add(pattern[1]);
      objects.add(pattern[2]);
    }
    this.syncThematicCategoryControls(
      [...subjects],
      [...relations],
      [...objects],
    );
    this.commitThematicSelection(startedAt);
    el.themeSimilarNote.textContent =
      `Added ${added.length.toLocaleString()} similar pattern` +
      `${added.length === 1 ? "" : "s"} to the selection` +
      `${missing ? `; ${missing} were outside the loaded categories` : ""}.`;
  }

  fitGraph(animated = true): void {
    if (!this.renderer) return;
    if (this.corpusRadialModel) {
      const rootCenter = this.cameraCenterForGraphPoint(
        { x: 0, y: 0 },
        1.08,
      );
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
      ].map((point) =>
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
      this.renderer.getCamera().animate(
        { x: center.x, y: center.y, ratio, angle: 0 },
        { duration: 450 },
      );
    }
    else {
      this.renderer.getCamera().setState({
        x: center.x,
        y: center.y,
        ratio,
        angle: 0,
      });
      requestAnimationFrame(() => this.renderer?.refresh());
    }
  }

  zoomIn(): void {
    this.zoomAtViewportCenter(true);
  }

  zoomOut(): void {
    this.zoomAtViewportCenter(false);
  }

  private zoomAtViewportCenter(zoomingIn: boolean): void {
    if (!this.renderer) return;
    const dimensions = this.renderer.getDimensions();
    const viewportPoint = {
      x: dimensions.width / 2,
      y: dimensions.height / 2,
    };
    const camera = this.renderer.getCamera();
    const zoomingRatio = Number(this.renderer.getSetting("zoomingRatio"));
    const ratio = camera.getBoundedRatio(
      camera.getState().ratio *
        (zoomingIn ? 1 / zoomingRatio : zoomingRatio),
    );
    if (this.corpusRadialModel) {
      const rootViewport = this.corpusRadialViewportPoint(
        this.corpusRadialModel.layout.dominantRoot,
      );
      const rootIsCentral =
        Math.hypot(
          rootViewport.x - viewportPoint.x,
          rootViewport.y - viewportPoint.y,
        ) < Math.min(dimensions.width, dimensions.height) * 0.2;
      camera.animate(
        this.renderer.getViewportZoomedState(
          rootIsCentral ? rootViewport : viewportPoint,
          ratio,
        ),
        { duration: 140 },
      );
      return;
    }
    camera.animate(
      this.renderer.getViewportZoomedState(viewportPoint, ratio),
      { duration: 220 },
    );
  }

  async open(options: OpenOptions): Promise<void> {
    this.setPage(
      options.volumeId || this.page.volumeId,
      Number(options.pageNumber || this.page.pageNumber),
    );
    this.clearSimilarPatterns();
    el.filteredTagsSimilarity.value = "92";
    el.filteredTagsSimilarityValue.value = "92%";
    el.themeSimilarity.value = "50";
    el.themeSimilarityValue.value = "50%";
    el.themeGranularityValue.value = `${el.themeGranularity.value}+`;
    el.reader.hidden = true;
    el.workspace.hidden = false;
    el.scope.value = "corpus";
    this.syncViewModeUi();
    await Promise.all([this.ensurePageIndex(), this.ensureCategoryCatalog()]);
    this.setCategoryValues(
      el.entityCategories,
      options.categories?.entities || [],
    );
    this.setCategoryValues(el.objectCategories);
    this.setCategoryValues(
      el.relationCategories,
      options.categories?.relations || [],
    );
    this.recordCategoryControls();
    this.resetFilteredTags(true);
    this.syncSimilarPatternAnchor();
    await this.loadScope();
  }

  close(): void {
    this.graphAbort?.abort();
    this.evidenceAbort?.abort();
    this.similarPatternsAbort?.abort();
    this.filteredTagsAbort?.abort();
    if (this.filteredTagsDebounce !== null) {
      window.clearTimeout(this.filteredTagsDebounce);
      this.filteredTagsDebounce = null;
    }
    if (this.filteredTagsSyncTimer !== null) {
      window.clearTimeout(this.filteredTagsSyncTimer);
      this.filteredTagsSyncTimer = null;
    }
    this.stopLayout();
    el.workspace.hidden = true;
    el.reader.hidden = false;
    // The outer reader owns the browser URL, so it receives one explicit
    // visibility event when the graph returns to the Chronicle page.
    window.dispatchEvent(new CustomEvent("chronicle-graph-closed"));
  }

  handleDetailAction(button: HTMLElement): void {
    const action = button.dataset.graphAction;
    const value = button.dataset.value || "";
    if (action === "entity") {
      const split = value.lastIndexOf(":");
      const id = value.slice(0, split);
      const depth = Number(value.slice(split + 1));
      void this.loadEntity(id, depth);
    } else if (action === "category-entity") {
      void this.filterByEntityCategory(value);
    } else if (action === "category-relation") {
      void this.filterByRelationCategory(value);
    } else if (action === "thematic-reset") {
      this.clearSelection();
      this.fitGraph();
    } else if (action === "thematic-entity-group") {
      this.selectThematicNode(this.thematicEntityNodeId(value));
    } else if (action === "thematic-relation-group") {
      this.selectThematicNode(this.thematicRelationNodeId(value));
    } else if (action === "thematic-back") {
      this.returnToForegroundedPatterns();
    } else if (action === "thematic-evidence") {
      // Opening one pattern's evidence focuses that pattern, so the graph and
      // the panels describe the claims the inspector is showing.
      this.focusThematicPattern(Number(value));
    } else if (action === "thematic-evidence-selected") {
      const select = document.getElementById(
        "graph-thematic-patterns",
      ) as HTMLSelectElement | null;
      if (select?.value) this.focusThematicPattern(Number(select.value));
    } else if (
      action === "thematic-evidence-more" &&
      this.thematicEvidencePattern !== null &&
      this.evidenceNextOffset !== null
    ) {
      void this.loadThematicPatternEvidence(
        this.thematicEvidencePattern,
        this.evidenceNextOffset,
      );
    } else if (action === "raw-context-relation" && this.graph) {
      const edge = this.graph.edges().find(
        (candidate) =>
          String(this.graph?.getEdgeAttribute(candidate, "relationId")) === value,
      );
      if (edge) this.selectEdge(edge);
    } else if (action === "relation") {
      el.scope.value = "focus";
      void this.loadGraph(
        `/api/graph/topology/relation/${encodeURIComponent(value)}?limit=300`,
      );
    } else if (action === "similar") {
      const split = value.indexOf(":");
      const kind = value.slice(0, split) as "entity" | "relation";
      const id = value.slice(split + 1);
      void this.anchorSimilarTagsOn(kind, id);
    } else if (
      action === "evidence-more" &&
      this.evidenceNextOffset !== null
    ) {
      void this.loadEvidence(this.evidenceNextOffset);
    }
  }

  /**
   * Open the tag panel with one raw tag already acting as the similarity
   * anchor. This is the entry point the inspector's "find similar" action uses
   * now that semantic search lives inside the tag bucket.
   */
  async anchorSimilarTagsOn(
    kind: "entity" | "relation",
    id: string,
  ): Promise<void> {
    const role: FilteredTagRole = kind === "relation" ? "relation" : "subject";
    for (const selected of this.filteredTagsSelected.values()) selected.clear();
    this.filteredTagsSelected.get(role)?.add(id);
    this.filteredTagsSimilarAnchors.set(role, [id]);
    el.filteredTagsPanel.open = true;
    this.populateFilteredTagRoles();
    el.filteredTagsRole.value = role;
    el.filteredTagsSearch.value = "";
    await this.refreshFilteredTags(false, 0);
  }
}

const controller = new GraphController();

el.detail.addEventListener("click", (event) => {
  const button = (event.target as Element).closest<HTMLElement>(
    "[data-graph-action]",
  );
  if (button) controller.handleDetailAction(button);
});
el.themeApply.addEventListener("click", () =>
  void controller.applyThemeFilters(),
);
el.themeClear.addEventListener("click", () =>
  void controller.clearThemeFilters(),
);
el.themeDirection.addEventListener("change", () => {
  controller.refreshCategoryFacets();
  controller.refreshThematicExplorationFilters();
});
el.themeGranularity.addEventListener("input", () =>
  controller.refreshThematicExplorationFilters(),
);
for (const container of [
  el.entityCategories,
  el.relationCategories,
  el.objectCategories,
]) {
  // `change` bubbles from each checkbox, so one listener covers the list.
  container.addEventListener("change", () => {
    controller.syncCategoryPickerFooter(container);
    controller.recordCategoryControls();
    controller.refreshCategoryFacets();
    controller.scheduleFilteredTagsSync();
  });
  const filter = document.getElementById(`${container.id}-filter`);
  if (filter instanceof HTMLInputElement) {
    filter.addEventListener("input", () =>
      controller.filterCategoryPicker(container, filter.value),
    );
  }
  const clear = document.getElementById(`${container.id}-clear`);
  if (clear instanceof HTMLButtonElement) {
    clear.addEventListener("click", () =>
      controller.clearCategoryPicker(container),
    );
  }
}
el.themeSimilarity.addEventListener("input", () => {
  el.themeSimilarityValue.value = `${el.themeSimilarity.value}%`;
});
el.themeSimilarity.addEventListener("change", () => {
  if (!el.themeSimilarClear.disabled) void controller.loadSimilarPatterns();
});
el.themeSimilar.addEventListener("click", () =>
  void controller.loadSimilarPatterns(),
);
el.themeSimilarClear.addEventListener("click", () =>
  controller.clearSimilarPatterns(),
);
el.themeSimilarResults.addEventListener("change", () =>
  controller.syncSimilarPatternSelection(),
);
el.themeSimilarApply.addEventListener("click", () =>
  controller.applySimilarPatterns(),
);
el.filteredTagsSimilarity.addEventListener("input", () => {
  el.filteredTagsSimilarityValue.value = `${el.filteredTagsSimilarity.value}%`;
  controller.scheduleSimilarTagThreshold();
});
el.filteredTagsSimilar.addEventListener("click", () =>
  void controller.findSimilarTags(),
);
el.filteredTagsSimilarClear.addEventListener("click", () =>
  void controller.clearSimilarTags(),
);
el.filteredTagsRole.addEventListener("change", () =>
  void controller.changeFilteredTagRole(),
);
el.filteredTagsSearch.addEventListener("input", () =>
  controller.scheduleFilteredTagSearch(),
);
el.filteredTagsSearch.addEventListener("keydown", (event) => {
  if (event.key === "Enter") void controller.refreshFilteredTags(false, 0);
});
el.filteredTagsSearchButton.addEventListener("click", () =>
  void controller.refreshFilteredTags(false, 0),
);
el.filteredTagsResults.addEventListener("change", (event) => {
  const input = event.target;
  if (input instanceof HTMLInputElement && input.type === "checkbox") {
    controller.handleFilteredTagChange(input);
  }
});
el.filteredTagsSelectPage.addEventListener("click", () =>
  controller.selectFilteredTagPage(),
);
el.filteredTagsClear.addEventListener("click", () =>
  controller.clearFilteredTagSelection(),
);
el.filteredTagsPrev.addEventListener("click", () =>
  controller.previousFilteredTagsPage(),
);
el.filteredTagsNext.addEventListener("click", () =>
  controller.nextFilteredTagsPage(),
);
el.filteredTagsApply.addEventListener("click", () =>
  void controller.applyFilteredTags(),
);
el.filteredTagsRelease.addEventListener("click", () =>
  void controller.releaseAppliedTagFilters(),
);
// The panels stay independently open: navigating the graph updates both, so
// forcing one shut would hide the live result the user is watching.
el.filteredTagsPanel.addEventListener("toggle", () => {
  if (el.filteredTagsPanel.open) controller.refreshFilteredTagsIfStale();
});
el.scope.addEventListener("change", () => void controller.loadScope());
el.pageVolume.addEventListener("change", () =>
  controller.handlePageVolumeChange(),
);
el.pageNumber.addEventListener("change", () =>
  controller.handlePageNumberChange(),
);
el.pagePrev.addEventListener("click", () =>
  void controller.moveSelectedPage(-1),
);
el.pageNext.addEventListener("click", () =>
  void controller.moveSelectedPage(1),
);
el.pageLoad.addEventListener("click", () =>
  void controller.loadSelectedPage(),
);
el.rangeStart.addEventListener("change", () =>
  controller.handleRangeChange("start"),
);
el.rangeEnd.addEventListener("change", () =>
  controller.handleRangeChange("end"),
);
el.rangeLoad.addEventListener("click", () =>
  void controller.loadSelectedRange(),
);
el.zoomOut.addEventListener("click", () => controller.zoomOut());
el.zoomIn.addEventListener("click", () => controller.zoomIn());
el.fit.addEventListener("click", () => controller.fitGraph());
el.showAll.addEventListener("change", () =>
  controller.setCorpusShowAll(el.showAll.checked),
);
el.close.addEventListener("click", () => controller.close());
window.addEventListener("keydown", (event) => {
  if (
    el.workspace.hidden ||
    event.ctrlKey ||
    event.metaKey ||
    event.altKey
  ) return;
  const target = event.target;
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    (target instanceof HTMLElement && target.isContentEditable)
  ) return;
  if (event.key === "Escape") {
    event.preventDefault();
    controller.navigateBackOneLayer();
  } else if (
    event.key === "+" ||
    event.key === "=" ||
    event.code === "NumpadAdd"
  ) {
    event.preventDefault();
    controller.zoomIn();
  } else if (
    event.key === "-" ||
    event.key === "_" ||
    event.code === "NumpadSubtract"
  ) {
    event.preventDefault();
    controller.zoomOut();
  } else if (event.key === "Escape") {
    controller.close();
  }
});

window.ChronicleGraph = {
  open: (options) => controller.open(options),
  close: () => controller.close(),
  setPage: (volumeId, pageNumber) => controller.setPage(volumeId, pageNumber),
  isOpen: () => !el.workspace.hidden,
};
