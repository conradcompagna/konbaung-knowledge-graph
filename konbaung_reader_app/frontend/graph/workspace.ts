import type { GraphController } from "./controller";
import { type OpenOptions } from "./types";
import { el } from "./dom";

export const workspace = {
  async open(this: GraphController, options: OpenOptions): Promise<void> {
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
  },

  close(this: GraphController): void {
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
  },

  handleDetailAction(this: GraphController, button: HTMLElement): void {
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
      const edge = this.graph
        .edges()
        .find(
          (candidate) =>
            String(this.graph?.getEdgeAttribute(candidate, "relationId")) ===
            value,
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
    } else if (action === "evidence-more" && this.evidenceNextOffset !== null) {
      void this.loadEvidence(this.evidenceNextOffset);
    }
  },
};
