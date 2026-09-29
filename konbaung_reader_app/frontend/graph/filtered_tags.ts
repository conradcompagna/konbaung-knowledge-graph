import type { GraphController } from "./controller";
import { type FilteredTagRole, type FilteredTagsPayload } from "./types";
import { el, clear, textElement } from "./dom";
import { fetchJson } from "./utilities";

export const filtered_tags = {
  activeFilteredTagRoles(this: GraphController): FilteredTagRole[] {
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
  },

  filteredTagRoleLabel(this: GraphController, role: FilteredTagRole): string {
    const select =
      role === "subject"
        ? el.entityCategories
        : role === "relation"
          ? el.relationCategories
          : el.objectCategories;
    const categories = this.selectedCategoryValues(select);
    const title =
      role === "subject"
        ? "Entity group A"
        : role === "object"
          ? "Entity group B"
          : "Relation";
    const narrowed = (
      ["subject", "relation", "object"] as FilteredTagRole[]
    ).some(
      (other) =>
        other !== role && (this.appliedTagFilters.get(other) || []).length,
    );
    const base = categories.length
      ? `${title} tags / ${categories.join(", ")}`
      : `${title} tags / every tag in scope`;
    return narrowed ? `${base} / narrowed by applied tags` : base;
  },

  /** Every role is always listed; unfiltered roles fall back to the scope. */
  populateFilteredTagRoles(this: GraphController): FilteredTagRole[] {
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
  },

  selectedFilteredTagCount(this: GraphController): number {
    return Array.from(this.filteredTagsSelected.values()).reduce(
      (total, selected) => total + selected.size,
      0,
    );
  },

  resetFilteredTags(this: GraphController, clearSearch: boolean = false): void {
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
  },

  /**
   * Identity of the tag list currently on screen. Reloading is only worth it
   * when this changes; without the check a stray sync would drop the reader
   * back onto page one of the bucket they were already paging through.
   */
  filteredTagsSignature(this: GraphController): string {
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
  },

  /** Reload only when the bucket differs from what is already rendered. */
  refreshFilteredTagsIfStale(this: GraphController): void {
    if (
      this.filteredTagsItems.length &&
      this.filteredTagsSignature() === this.filteredTagsBucketKey
    )
      return;
    void this.refreshFilteredTags(false, 0);
  },

  /**
   * Reload the open tag bucket after the graph selection changed. Debounced
   * because one click can move several category selects at once.
   */
  scheduleFilteredTagsSync(this: GraphController): void {
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
  },

  appendFilteredTagScope(
    this: GraphController,
    parameters: URLSearchParams,
  ): void {
    const scope =
      this.thematicPayload?.layout.scope ||
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
  },

  appendPairingDirection(
    this: GraphController,
    parameters: URLSearchParams,
  ): void {
    parameters.set("direction", this.thematicDirection());
  },

  appendFilteredTagCategories(
    this: GraphController,
    parameters: URLSearchParams,
    topology: boolean = false,
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
  },

  appendAppliedTagFilters(
    this: GraphController,
    parameters: URLSearchParams,
  ): void {
    for (const role of ["subject", "relation", "object"] as FilteredTagRole[]) {
      for (const id of this.appliedTagFilters.get(role) || []) {
        parameters.append(`${role}Tag`, id);
      }
    }
  },

  appliedTagFilterCount(this: GraphController): number {
    return [...this.appliedTagFilters.values()].reduce(
      (total, ids) => total + ids.length,
      0,
    );
  },

  clearAppliedTagFilters(this: GraphController): void {
    for (const role of this.appliedTagFilters.keys()) {
      this.appliedTagFilters.set(role, []);
    }
    el.filteredTagsRelease.hidden = true;
  },

  /** Drop the live tag filter, restore the base graph, and widen the facets. */
  async releaseAppliedTagFilters(this: GraphController): Promise<void> {
    if (!this.appliedTagFilterCount()) return;
    this.clearAppliedTagFilters();
    el.filteredTagsRelease.disabled = true;
    await this.loadGraph(this.thematicBaseUrl());
    el.filteredTagsRelease.disabled = false;
    await this.refreshFilteredTags(false, 0);
  },

  async refreshFilteredTags(
    this: GraphController,
    resetSelection: boolean = false,
    offset: number = 0,
  ): Promise<void> {
    this.populateFilteredTagRoles();
    if (resetSelection) {
      for (const selected of this.filteredTagsSelected.values())
        selected.clear();
      for (const role of this.filteredTagsSimilarAnchors.keys()) {
        this.filteredTagsSimilarAnchors.set(role, []);
      }
      for (const input of el.filteredTagsResults.querySelectorAll<HTMLInputElement>(
        'input[type="checkbox"]',
      ))
        input.checked = false;
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
  },

  renderFilteredTags(
    this: GraphController,
    payload: FilteredTagsPayload,
  ): void {
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
      const similarity =
        item.similarity === null || item.similarity === undefined
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
    const narrowing = (
      ["subject", "relation", "object"] as FilteredTagRole[]
    ).flatMap((role) => {
      const count = (payload.narrowedBy?.[role] || []).length;
      const name =
        role === "subject"
          ? "group A"
          : role === "object"
            ? "group B"
            : "relation";
      return count ? [`${count} ${name} tag${count === 1 ? "" : "s"}`] : [];
    });
    const slot =
      payload.role === "subject"
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
  },

  /**
   * Refresh the controls that depend on the checked set. This deliberately
   * does not re-derive that set from the DOM: the list re-renders whenever a
   * search, a page turn, or a graph click reloads the bucket, and rebuilding
   * the set from a freshly painted list would silently drop tags the reader
   * had checked on another page.
   */
  syncFilteredTagSelection(this: GraphController): void {
    const role = el.filteredTagsRole.value as FilteredTagRole;
    const selected = this.filteredTagsSelected.get(role);
    const count = this.selectedFilteredTagCount();
    el.filteredTagsApply.disabled = count === 0 || count > 100;
    el.filteredTagsClear.disabled = count === 0;
    el.filteredTagsSimilar.disabled = (selected?.size || 0) === 0;
    el.filteredTagsSelectionNote.textContent = count
      ? `${count.toLocaleString()} of 100 tags checked. Tags within a role use OR; checked roles combine with AND.`
      : "Up to 100 checked tags can be combined. Tags within one role are unioned; different roles intersect.";
  },

  /**
   * Switch buckets. The search box is cleared because a term that matched
   * subject tags almost never matches relation tags, and carrying it across
   * silently empties the new bucket.
   */
  async changeFilteredTagRole(this: GraphController): Promise<void> {
    el.filteredTagsSearch.value = "";
    await this.refreshFilteredTags(false, 0);
  },

  /** Rank the open bucket by similarity to the averaged checked vectors. */
  async findSimilarTags(this: GraphController): Promise<void> {
    this.syncFilteredTagSelection();
    const role = el.filteredTagsRole.value as FilteredTagRole;
    const checked = [...(this.filteredTagsSelected.get(role) || [])];
    if (!checked.length) return;
    this.filteredTagsSimilarAnchors.set(role, checked);
    el.filteredTagsSimilarNote.textContent =
      `Ranking this bucket against the mean vector of ${checked.length} checked ` +
      `tag${checked.length === 1 ? "" : "s"}. Checked tags stay at the top.`;
    await this.refreshFilteredTags(false, 0);
  },

  async clearSimilarTags(this: GraphController): Promise<void> {
    const role = el.filteredTagsRole.value as FilteredTagRole;
    if (!(this.filteredTagsSimilarAnchors.get(role) || []).length) return;
    this.filteredTagsSimilarAnchors.set(role, []);
    el.filteredTagsSimilarNote.textContent =
      "Check one or more tags, then rank this bucket by embedding similarity to their averaged vector.";
    await this.refreshFilteredTags(false, 0);
  },

  /** Re-rank at the new threshold when a similarity ordering is active. */
  scheduleSimilarTagThreshold(this: GraphController): void {
    const role = el.filteredTagsRole.value as FilteredTagRole;
    if (!(this.filteredTagsSimilarAnchors.get(role) || []).length) return;
    if (this.filteredTagsDebounce !== null) {
      window.clearTimeout(this.filteredTagsDebounce);
    }
    this.filteredTagsDebounce = window.setTimeout(() => {
      this.filteredTagsDebounce = null;
      void this.refreshFilteredTags(false, 0);
    }, 250);
  },

  selectFilteredTagPage(this: GraphController): void {
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
  },

  clearFilteredTagSelection(this: GraphController): void {
    for (const selected of this.filteredTagsSelected.values()) selected.clear();
    for (const input of el.filteredTagsResults.querySelectorAll<HTMLInputElement>(
      'input[type="checkbox"]',
    ))
      input.checked = false;
    this.syncFilteredTagSelection();
  },

  handleFilteredTagChange(
    this: GraphController,
    input: HTMLInputElement,
  ): void {
    const role = (input.dataset.role ||
      el.filteredTagsRole.value) as FilteredTagRole;
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
  },

  previousFilteredTagsPage(this: GraphController): void {
    void this.refreshFilteredTags(
      false,
      Math.max(0, this.filteredTagsOffset - 100),
    );
  },

  nextFilteredTagsPage(this: GraphController): void {
    if (!this.filteredTagsHasMore) return;
    void this.refreshFilteredTags(false, this.filteredTagsOffset + 100);
  },

  scheduleFilteredTagSearch(this: GraphController): void {
    if (this.filteredTagsDebounce !== null) {
      window.clearTimeout(this.filteredTagsDebounce);
    }
    this.filteredTagsDebounce = window.setTimeout(() => {
      this.filteredTagsDebounce = null;
      void this.refreshFilteredTags(false, 0);
    }, 250);
  },

  async applyFilteredTags(this: GraphController): Promise<void> {
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
  },

  /**
   * Open the tag panel with one raw tag already acting as the similarity
   * anchor. This is the entry point the inspector's "find similar" action uses
   * now that semantic search lives inside the tag bucket.
   */
  async anchorSimilarTagsOn(
    this: GraphController,
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
  },
};
