import type { GraphController } from "./controller";
import { type EvidencePayload, type GraphNavigationSnapshot } from "./types";
import { emptyNavigationState, cloneNavigationState } from "./navigation_state";
import { el, clear, textElement, actionButton } from "./dom";
import { fetchJson } from "./utilities";

export const selection_inspector = {
  updateHeader(this: GraphController): void {
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
        const tagKind =
          payload.focus.kind === "filtered-tags"
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
  },

  renderFocusDetail(this: GraphController): void {
    if (!this.payload) return;
    if (this.thematicPayload) {
      const payload = this.thematicPayload;
      clear(el.detail);
      const filteredTagSlice = payload.focus.kind === "filtered-tags";
      const semanticSlice =
        payload.focus.kind === "tag-filter" || filteredTagSlice;
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
    el.detail.appendChild(textElement("div", "graph-detail-type", focus.kind));
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
  },

  appendEntityActions(this: GraphController, entityId: string): void {
    const actions = document.createElement("div");
    actions.className = "graph-detail-actions";
    actions.append(
      actionButton("1-hop", "entity", `${entityId}:1`),
      actionButton("2-hop", "entity", `${entityId}:2`),
      actionButton("Gemini similar", "similar", `entity:${entityId}`),
    );
    el.detail.appendChild(actions);
  },

  appendRelationActions(this: GraphController, relationId: string): void {
    const actions = document.createElement("div");
    actions.className = "graph-detail-actions";
    actions.append(
      actionButton("Open relation", "relation", relationId),
      actionButton("Gemini similar", "similar", `relation:${relationId}`),
    );
    el.detail.appendChild(actions);
  },

  /**
   * A history layer is a copy of the state record plus the archived raw-view
   * fields. Listing the thematic fields by hand here is what let the snapshot
   * fall behind the state it was meant to capture: the pairing direction and
   * edge orientation were added to the selection but never to the snapshot,
   * so stepping back silently dropped them.
   */
  navigationSnapshot(this: GraphController): GraphNavigationSnapshot {
    return {
      nav: cloneNavigationState(this.nav),
      selectedNeighbors: Array.from(this.selectedNeighbors),
      rawPrimaryEntity: this.rawPrimaryEntity,
      rawPairEntity: this.rawPairEntity,
      rawContextRelation: this.rawContextRelation,
      rawSelectedEdges: Array.from(this.rawSelectedEdges),
    };
  },

  pushNavigationState(this: GraphController): void {
    this.navigationHistory.push(this.navigationSnapshot());
  },

  restoreNavigationState(
    this: GraphController,
    snapshot: GraphNavigationSnapshot,
  ): void {
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
  },

  navigateBackOneLayer(this: GraphController): void {
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
  },

  rawEdgesForSelection(
    this: GraphController,
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
  },

  applyRawNavigationSelection(this: GraphController, edges: string[]): void {
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
  },

  renderRawNavigationDetail(this: GraphController): void {
    if (!this.graph || !this.rawPrimaryEntity) return;
    const sourceLabel = String(
      this.graph.getNodeAttribute(this.rawPrimaryEntity, "label"),
    );
    const targetLabel = this.rawPairEntity
      ? String(this.graph.getNodeAttribute(this.rawPairEntity, "label"))
      : "any object";
    const relations = new Map<string, { label: string; claims: number }>();
    let claims = 0;
    for (const edge of this.rawSelectedEdges) {
      const relationId = String(
        this.graph.getEdgeAttribute(edge, "relationId"),
      );
      const count = Number(
        this.graph.getEdgeAttribute(edge, "occurrenceCount"),
      );
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
  },

  renderRawEntityDetail(this: GraphController, node: string): void {
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
  },

  selectNode(this: GraphController, node: string): void {
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
  },

  selectEdge(this: GraphController, edge: string): void {
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
  },

  renderEdgeDetail(this: GraphController, errorMessage: string = ""): void {
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
      el.detail.appendChild(textElement("div", "graph-empty", errorMessage));
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
  },

  async loadEvidence(this: GraphController, offset: number): Promise<void> {
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
      if (request !== this.evidenceAbort || edge !== this.selectedEdge) return;
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
  },

  clearSelection(this: GraphController, clearHistory: boolean = true): void {
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
  },

  async loadEntity(
    this: GraphController,
    id: string,
    depth: number,
  ): Promise<void> {
    el.scope.value = "focus";
    await this.loadGraph(
      `/api/graph/topology/entity/${encodeURIComponent(id)}` +
        `?depth=${depth}&limit=${depth === 2 ? 500 : 300}`,
    );
  },

  async filterByEntityCategory(
    this: GraphController,
    id: string,
  ): Promise<void> {
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
  },

  async filterByRelationCategory(
    this: GraphController,
    id: string,
  ): Promise<void> {
    await this.ensureCategoryCatalog();
    this.setCategoryValues(el.entityCategories);
    this.setCategoryValues(el.objectCategories);
    this.setCategoryValues(el.relationCategories, [id]);
    const panel = el.browserSummary.closest("details");
    if (panel instanceof HTMLDetailsElement) panel.open = true;
    this.applyThematicCategorySelection();
  },
};
