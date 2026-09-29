import type { GraphController } from "./controller";
import {
  type CategoryDescriptor,
  type ThematicPatternTuple,
  type ThematicTopologyPayload,
  type CategoryCatalogPayload,
  type ThematicDirection,
} from "./types";
import { byId, el, clear, textElement } from "./dom";
import { fetchJson } from "./utilities";

export const category_filters = {
  async ensureCategoryCatalog(this: GraphController): Promise<void> {
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
  },

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
  syncCategoryOptionCounts(
    this: GraphController,
    payload: ThematicTopologyPayload,
  ): void {
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
    const relations = indexOf(
      payload.relationCategories,
      el.relationCategories,
    );
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
  },

  /** Recompute the facets against whatever is checked right now. */
  refreshCategoryFacets(this: GraphController): void {
    if (this.thematicPayload) {
      this.syncCategoryOptionCounts(this.thematicPayload);
    }
  },

  /**
   * Copy the three category lists into the state record.
   *
   * The lists are the input device for the category part of the state, so an
   * edit there is written straight through. Without this, commitNavigation
   * republishing the record would revert checks a reader had made but not yet
   * applied, the moment anything else committed.
   */
  recordCategoryControls(this: GraphController): void {
    this.nav.categories = {
      subjects: this.selectedCategoryValues(el.entityCategories),
      relations: this.selectedCategoryValues(el.relationCategories),
      objects: this.selectedCategoryValues(el.objectCategories),
    };
  },

  categoryInputs(
    this: GraphController,
    container: HTMLElement,
  ): HTMLInputElement[] {
    return Array.from(
      container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'),
    );
  },

  selectedCategoryValues(
    this: GraphController,
    container: HTMLElement,
  ): string[] {
    return this.categoryInputs(container)
      .filter((input) => input.checked)
      .map((input) => input.value);
  },

  setCategoryValues(
    this: GraphController,
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
  },

  /** Keep each picker's "N checked" line and its Clear button honest. */
  syncCategoryPickerFooter(
    this: GraphController,
    container: HTMLElement,
  ): void {
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
  },

  clearCategoryPicker(this: GraphController, container: HTMLElement): void {
    for (const input of this.categoryInputs(container)) input.checked = false;
    this.syncCategoryPickerFooter(container);
    this.recordCategoryControls();
    this.refreshCategoryFacets();
    this.scheduleFilteredTagsSync();
  },

  /** Narrow a long category list to the typed term. */
  filterCategoryPicker(
    this: GraphController,
    container: HTMLElement,
    query: string,
  ): void {
    const needle = query.trim().toLowerCase();
    for (const row of container.querySelectorAll<HTMLElement>(
      ".graph-category-option",
    )) {
      row.hidden =
        Boolean(needle) && !(row.dataset.search || "").includes(needle);
    }
  },

  /**
   * Whether a pattern satisfies the two entity groups in each orientation.
   *
   * This mirrors the server's matching exactly: "forward" puts group A on the
   * pattern's subject side, "reverse" puts it on the object side. An empty
   * group matches anything. The graph highlight and the panels both go through
   * here, so they cannot drift apart the way they did when the highlight was
   * hard-coded to the subject/object order.
   */
  patternOrientations(
    this: GraphController,
    pattern: ThematicPatternTuple,
    groupA: Set<number>,
    groupB: Set<number>,
    direction: ThematicDirection,
  ): { forward: boolean; reverse: boolean } {
    const fits = (index: number, group: Set<number>): boolean =>
      group.size === 0 || group.has(index);
    return {
      forward:
        direction !== "ba" &&
        fits(pattern[0], groupA) &&
        fits(pattern[2], groupB),
      reverse:
        direction !== "ab" &&
        fits(pattern[2], groupA) &&
        fits(pattern[0], groupB),
    };
  },

  applyThematicCategorySelection(this: GraphController): void {
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
      (category, index) => (selectedSubjectIds.has(category.id) ? [index] : []),
    );
    const objectIndices = this.thematicPayload.entityCategories.flatMap(
      (category, index) => (selectedObjectIds.has(category.id) ? [index] : []),
    );
    const relationIndices = this.thematicPayload.relationCategories.flatMap(
      (category, index) =>
        selectedRelationIds.has(category.id) ? [index] : [],
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
    this.thematicPrimaryEntity =
      primaryIndex !== undefined
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
  },

  syncViewModeUi(this: GraphController): void {
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
  },

  async applyThemeFilters(this: GraphController): Promise<void> {
    if (this.thematicPayload?.focus.kind === "filtered-tags") {
      await this.loadGraph(this.thematicBaseUrl());
    } else {
      this.applyThematicCategorySelection();
    }
    this.clearSimilarPatterns();
    this.syncSimilarPatternAnchor();
    await this.refreshFilteredTags(true, 0);
  },

  async clearThemeFilters(this: GraphController): Promise<void> {
    const filteredTagScope =
      this.thematicPayload?.focus.kind === "filtered-tags"
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
  },
};
