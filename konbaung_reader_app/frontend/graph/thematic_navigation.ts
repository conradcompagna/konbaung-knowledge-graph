import type { ThematicRelationLabelHit } from "./types";
import type { GraphController } from "./controller";
import {
  type ThematicEntityCategory,
  type ThematicRelationCategory,
  type ThematicDirection,
} from "./types";
import { el, clear, textElement, actionButton } from "./dom";

export const thematic_navigation = {
  thematicDirection(this: GraphController): ThematicDirection {
    const direction = el.themeDirection.value;
    return direction === "ab" || direction === "ba" ? direction : "either";
  },

  filteredThematicPatterns(
    this: GraphController,
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
  },

  refreshThematicExplorationFilters(this: GraphController): void {
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
    )
      return;
    this.setThematicSelection(
      this.filteredThematicPatterns(this.thematicSelectionCandidates),
    );
    // Changing the threshold or the pairing direction changes what the graph
    // shows, so it travels the same publish path as any other change.
    this.commitThematicSelection(performance.now());
  },

  setThematicSelection(
    this: GraphController,
    patternIndices: Iterable<number>,
  ): void {
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
        this.thematicEntityNodeId(payload.entityCategories[pattern[0]].id),
      );
      this.selectedNeighbors.add(
        this.thematicEntityNodeId(payload.entityCategories[pattern[2]].id),
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
  },

  appendThematicPatternList(
    this: GraphController,
    patternIndices: Iterable<number>,
  ): void {
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
  },

  selectThematicNode(
    this: GraphController,
    node: string,
    _pairFocus: boolean = false,
  ): void {
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
      this.thematicOrientationAnchors =
        kind === "entity"
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
  },

  /**
   * Leave an evidence list and return to the foregrounded pattern set.
   *
   * This steps back out of the layer that focusing a pattern pushed, so the
   * graph and both panels widen with the inspector instead of the inspector
   * alone changing.
   */
  returnToForegroundedPatterns(this: GraphController): void {
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
  },

  /**
   * Narrow the whole workspace to one exact pattern, then open its evidence.
   *
   * Inspecting a pattern is a navigation move like any other, so it goes
   * through the state record and commitThematicSelection() rather than only
   * swapping the inspector's contents.
   */
  focusThematicPattern(this: GraphController, patternIndex: number): void {
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
    this.syncThematicCategoryControls([pattern[0]], [pattern[1]], [pattern[2]]);
    this.setThematicSelection([patternIndex]);
    this.commitThematicSelection(startedAt);
    void this.loadThematicPatternEvidence(patternIndex);
  },

  selectThematicRelationLabel(
    this: GraphController,
    hit: ThematicRelationLabelHit,
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
      this.thematicSelectionCandidates = this.thematicPatternsForCombination(
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
      this.thematicSelectionCandidates = this.thematicPatternsForCombination(
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
  },

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
  publishNavigationToPanels(this: GraphController): void {
    this.setCategoryValues(el.entityCategories, this.nav.categories.subjects);
    this.setCategoryValues(
      el.relationCategories,
      this.nav.categories.relations,
    );
    this.setCategoryValues(el.objectCategories, this.nav.categories.objects);
    this.refreshCategoryFacets();
    this.scheduleFilteredTagsSync();
  },

  commitThematicSelection(this: GraphController, startedAt: number): void {
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
  },

  renderThematicSelection(this: GraphController): void {
    if (!this.graph || !this.thematicPayload || !this.selectedNode) return;
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
      const relation = payload.relationCategories[this.thematicContextRelation];
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
        textElement("div", "graph-detail-type", "entity-pair + relation focus"),
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
      const relation = payload.relationCategories[this.thematicContextRelation];
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
        textElement("div", "graph-detail-type", "entity + relation focus"),
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
        textElement("div", "graph-detail-type", "ordered entity-pair focus"),
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
      actions.append(
        actionButton("Return to complete graph", "thematic-reset", ""),
      );
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
      for (const patternIndex of this.thematicSelectedPatterns) {
        const pattern = payload.patterns[patternIndex];
        // The selected patterns already reflect the ab/ba/either filter.
        entities.add(pattern[0]);
        entities.add(pattern[2]);
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
  },
};
