export const byId = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing graph UI element: ${id}`);
  return element as T;
};

export function ensureCorpusShowAllControl(): {
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

export const corpusShowAllControl = ensureCorpusShowAllControl();

export const el = {
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

export function clear(element: Element): void {
  element.replaceChildren();
}

export function textElement(
  tag: keyof HTMLElementTagNameMap,
  className: string,
  text: string,
): HTMLElement {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

export function actionButton(
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
