import { type ThematicNavigationState } from "./types";

export function emptyNavigationState(): ThematicNavigationState {
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

export function cloneNavigationState(
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
