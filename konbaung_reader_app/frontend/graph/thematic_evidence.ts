import type { GraphController } from "./controller";
import { type EvidencePayload } from "./types";
import { el, clear, textElement, actionButton } from "./dom";
import { fetchJson } from "./utilities";

export const thematic_evidence = {
  renderThematicEvidence(
    this: GraphController,
    errorMessage: string = "",
  ): void {
    if (!this.thematicPayload || this.thematicEvidencePattern === null) return;
    const pattern = this.thematicPayload.patterns[this.thematicEvidencePattern];
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
      textElement("div", "graph-detail-type", "exact thematic pattern"),
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
      claim.appendChild(textElement("div", "graph-claim-my", item.sentenceMy));
      claim.appendChild(textElement("div", "graph-claim-en", item.sentenceEn));
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
          "thematic-evidence-more",
          "",
          "graph-evidence-more",
        ),
      );
    }
  },

  async loadThematicPatternEvidence(
    this: GraphController,
    patternIndex: number,
    offset: number = 0,
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
      )
        return;
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
  },
};
