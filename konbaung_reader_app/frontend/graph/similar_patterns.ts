import type { GraphController } from "./controller";
import {
  type SimilarPatternResult,
  type SimilarPatternsPayload,
} from "./types";
import { el, clear, textElement } from "./dom";
import { fetchJson } from "./utilities";

export const similar_patterns = {
  // ------------------------------------------------------------------
  // Similar statement patterns
  //
  // The anchor is whatever subject / relation / object categories are locked
  // in right now, whether they were chosen in the selects or reached by
  // clicking through the graph. The server averages the raw tag vectors of
  // every claim under that anchor and ranks the other in-scope patterns
  // against it.
  // ------------------------------------------------------------------

  similarPatternKey(this: GraphController, item: SimilarPatternResult): string {
    return `${item.subjectId}|${item.relationId}|${item.objectId}`;
  },

  themeSimilarityThreshold(this: GraphController): number {
    return Number(el.themeSimilarity.value) / 100;
  },

  syncSimilarPatternAnchor(this: GraphController): void {
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
  },

  clearSimilarPatterns(this: GraphController): void {
    this.similarPatternsAbort?.abort();
    this.similarPatterns = [];
    this.similarPatternsSelected.clear();
    clear(el.themeSimilarResults);
    el.themeSimilarNote.textContent = "";
    el.themeSimilarClear.disabled = true;
    el.themeSimilarApply.disabled = true;
  },

  async loadSimilarPatterns(this: GraphController): Promise<void> {
    if (
      !this.selectedCategoryValues(el.entityCategories).length &&
      !this.selectedCategoryValues(el.relationCategories).length &&
      !this.selectedCategoryValues(el.objectCategories).length
    ) {
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
      const visible = new Set(
        payload.items.map((item) => this.similarPatternKey(item)),
      );
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
  },

  renderSimilarPatterns(
    this: GraphController,
    payload: SimilarPatternsPayload,
  ): void {
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
      ([role, labels]) =>
        `${(labels || []).length} ${role} tag` +
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
  },

  syncSimilarPatternSelection(this: GraphController): void {
    this.similarPatternsSelected = new Set(
      [
        ...el.themeSimilarResults.querySelectorAll<HTMLInputElement>(
          'input[type="checkbox"]:checked',
        ),
      ].map((input) => input.value),
    );
    el.themeSimilarApply.disabled = this.similarPatternsSelected.size === 0;
  },

  applySimilarPatterns(this: GraphController): void {
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
      if (
        subject === undefined ||
        relation === undefined ||
        object === undefined
      ) {
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
    this.selectedNode =
      this.selectedNode ||
      this.thematicEntityNodeId(
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
  },
};
