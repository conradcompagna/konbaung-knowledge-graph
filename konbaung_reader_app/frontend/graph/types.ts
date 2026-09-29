import { type CorpusRadialLayoutResult } from "../corpus_radial_layout";
import { type ThematicLayoutResult } from "../thematic_layout";

export type NodeTuple = [
  id: string,
  label: string,
  frequency: number,
  x: number | null,
  y: number | null,
  componentIndex?: number,
  communityIndex?: number,
  priority?: number,
];

export type EdgeTuple = [
  sourceIndex: number,
  targetIndex: number,
  relationId: string,
  label: string,
  occurrenceCount: number,
  bundleIndex?: number,
];

export type ComponentTuple = [
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

export type CommunityTuple = [
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

export type BundleTuple = [
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

export interface GraphFocus {
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

export interface TopologyPayload {
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

export interface CategoryDescriptor {
  id: string;
  tagId: string;
  label: string;
  definition: string;
  count: number;
  provisional: boolean;
  justification?: string;
  sourcePage?: string;
}

export interface ThematicEntityCategory extends CategoryDescriptor {
  mentionCount: number;
  activeMentionCount: number;
  counterpartCount: number;
  activeCounterpartCount: number;
  relationCount: number;
  activeRelationCount: number;
  patternCount: number;
  activePatternCount: number;
}

export interface ThematicRelationCategory extends CategoryDescriptor {
  tripleCount: number;
  activeTripleCount: number;
  entityCount: number;
  activeEntityCount: number;
  patternCount: number;
  activePatternCount: number;
}

export type ThematicPatternTuple = [
  sourceEntityIndex: number,
  relationIndex: number,
  targetEntityIndex: number,
  supportingTripleCount: number,
  activeTripleCount: number,
];

export type ThematicBundleTuple = [
  entityIndex: number,
  relationIndex: number,
  role: 0 | 1,
  supportingTripleCount: number,
  activeTripleCount: number,
  childPatternIndices: number[],
];

export interface ThematicTopologyPayload extends TopologyPayload {
  entityCategories: ThematicEntityCategory[];
  relationCategories: ThematicRelationCategory[];
  patterns: ThematicPatternTuple[];
  incidenceBundles: ThematicBundleTuple[];
}

export interface ThematicLayoutWorkerResponse {
  kind: "layout";
  requestId: number;
  result?: ThematicLayoutResult;
  error?: string;
}

export interface CategoryCatalogPayload {
  ok: boolean;
  entities: CategoryDescriptor[];
  relations: CategoryDescriptor[];
}

export interface EvidenceItem {
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

export interface EvidencePayload {
  ok: boolean;
  items: EvidenceItem[];
  hasMore: boolean;
  nextOffset: number | null;
}

export type FilteredTagRole = "subject" | "relation" | "object";

export interface FilteredTagResult {
  id: string | null;
  kind: "entity" | "relation";
  role: FilteredTagRole;
  label: string;
  frequency: number;
  similarity: number | null;
}

export interface SimilarPatternResult {
  subjectId: string;
  relationId: string;
  objectId: string;
  subjectLabel: string;
  relationLabel: string;
  objectLabel: string;
  tripleCount: number;
  similarity: number;
}

export interface SimilarPatternsPayload {
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

export interface FilteredTagsPayload {
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

export interface OpenOptions {
  volumeId?: string;
  pageNumber?: number;
  categories?: { entities?: string[]; relations?: string[] } | null;
}

export interface ChronicleVolume {
  id: string;
  label: string;
  availablePages: number[];
}

export interface ChronicleIndex {
  volumes: ChronicleVolume[];
}

export type ThematicDirection = "either" | "ab" | "ba";

/** The complete navigation state of the thematic graph. */
export interface ThematicNavigationState {
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

export interface GraphNavigationSnapshot {
  nav: ThematicNavigationState;
  selectedNeighbors: string[];
  rawPrimaryEntity: string | null;
  rawPairEntity: string | null;
  rawContextRelation: string | null;
  rawSelectedEdges: string[];
}

export interface CorpusRadialModel {
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

export interface CorpusRadialWorkerResponse {
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

export interface ThematicRelationLabelHit {
  node: string;
  patternIndex: number;
  x: number;
  y: number;
  halfWidth: number;
  halfHeight: number;
}
