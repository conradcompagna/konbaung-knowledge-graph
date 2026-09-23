import type { GraphController } from "./controller";
import { type TopologyPayload, type ChronicleIndex } from "./types";
import { el, clear, textElement } from "./dom";
import { fetchJson } from "./utilities";

export const page_navigation = {
  setPage(this: GraphController, volumeId: string, pageNumber: number): void {
    this.page = { volumeId, pageNumber: Number(pageNumber) };
    this.updatePageScopeLabel();
    if (this.pageVolumes.size) {
      this.populatePageSelectors(volumeId, Number(pageNumber));
    }
  },

  updatePageScopeLabel(this: GraphController): void {
    const { volumeId, pageNumber } = this.page;
    const pageOption = el.scope.querySelector<HTMLOptionElement>(
      'option[value="page"]',
    );
    if (pageOption) {
      pageOption.textContent = `Current page · ${volumeId.toUpperCase()} ${pageNumber}`;
    }
  },

  async ensurePageIndex(this: GraphController): Promise<void> {
    if (this.pageVolumes.size) return;
    if (this.pageIndexPromise) return this.pageIndexPromise;
    this.pageIndexPromise = (async () => {
      const response = await fetch("/api/chronicles/index", {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`Page index request failed (${response.status})`);
      }
      const index = (await response.json()) as ChronicleIndex;
      this.pageVolumes = new Map(
        index.volumes.map((volume) => [
          volume.id,
          {
            ...volume,
            availablePages: volume.availablePages.map(Number),
          },
        ]),
      );
      this.populatePageSelectors(this.page.volumeId, this.page.pageNumber);
    })();
    try {
      await this.pageIndexPromise;
    } finally {
      this.pageIndexPromise = null;
    }
  },

  populatePageSelect(
    this: GraphController,
    select: HTMLSelectElement,
    pages: number[],
    selectedPage: number,
  ): void {
    select.replaceChildren(
      ...pages.map((page) => {
        const option = document.createElement("option");
        option.value = String(page);
        option.textContent = String(page);
        return option;
      }),
    );
    select.value = String(selectedPage);
  },

  populatePageSelectors(
    this: GraphController,
    volumeId: string,
    preferredPage?: number,
  ): void {
    const volume = this.pageVolumes.get(volumeId);
    if (!volume || !volume.availablePages.length) return;
    const pages = volume.availablePages;
    const target = Number(preferredPage || pages[0]);
    const selectedPage = pages.includes(target)
      ? target
      : pages.reduce((closest, page) =>
          Math.abs(page - target) < Math.abs(closest - target) ? page : closest,
        );
    el.pageVolume.value = volumeId;
    this.populatePageSelect(el.pageNumber, pages, selectedPage);
    this.populatePageSelect(el.rangeStart, pages, selectedPage);
    this.populatePageSelect(el.rangeEnd, pages, selectedPage);
    el.pageStatus.textContent = `${volume.label} has ${pages.length.toLocaleString()} available pages.`;
    this.updatePageNavigation();
  },

  updatePageNavigation(this: GraphController): void {
    const pages =
      this.pageVolumes.get(el.pageVolume.value)?.availablePages || [];
    const index = pages.indexOf(Number(el.pageNumber.value));
    el.pagePrev.disabled = index <= 0;
    el.pageNext.disabled = index < 0 || index >= pages.length - 1;
  },

  normalizeRange(this: GraphController, changed: "start" | "end"): void {
    const start = Number(el.rangeStart.value);
    const end = Number(el.rangeEnd.value);
    if (start <= end) return;
    if (changed === "start") el.rangeEnd.value = String(start);
    else el.rangeStart.value = String(end);
  },

  handlePageVolumeChange(this: GraphController): void {
    this.populatePageSelectors(el.pageVolume.value);
  },

  handlePageNumberChange(this: GraphController): void {
    this.updatePageNavigation();
  },

  handleRangeChange(this: GraphController, changed: "start" | "end"): void {
    this.normalizeRange(changed);
  },

  async loadSelectedPage(this: GraphController): Promise<void> {
    const volumeId = el.pageVolume.value;
    const pageNumber = Number(el.pageNumber.value);
    if (!pageNumber) return;
    this.setPage(volumeId, pageNumber);
    el.scope.value = "page";
    el.pagesPanel.open = false;
    await this.loadScope();
  },

  async moveSelectedPage(
    this: GraphController,
    direction: -1 | 1,
  ): Promise<void> {
    const pages =
      this.pageVolumes.get(el.pageVolume.value)?.availablePages || [];
    const index = pages.indexOf(Number(el.pageNumber.value));
    const nextPage = pages[index + direction];
    if (nextPage === undefined) return;
    el.pageNumber.value = String(nextPage);
    this.updatePageNavigation();
    await this.loadSelectedPage();
  },

  async loadSelectedRange(this: GraphController): Promise<void> {
    const volumeId = el.pageVolume.value;
    const startPage = Number(el.rangeStart.value);
    const endPage = Number(el.rangeEnd.value);
    if (!startPage || !endPage || startPage > endPage) return;
    const selectedPages =
      this.pageVolumes
        .get(volumeId)
        ?.availablePages.filter((page) => page >= startPage && page <= endPage)
        .length || 0;
    this.page = { volumeId, pageNumber: startPage };
    this.updatePageScopeLabel();
    const rangeOption = el.scope.querySelector<HTMLOptionElement>(
      'option[value="range"]',
    );
    if (rangeOption) {
      rangeOption.textContent = `${volumeId.toUpperCase()} ${startPage}–${endPage}`;
    }
    el.pageStatus.textContent = `Loading all triples from ${selectedPages.toLocaleString()} available pages…`;
    el.scope.value = "range";
    el.pagesPanel.open = false;
    const path = `/api/graph/categories/topology/range/${volumeId}/${startPage}/${endPage}`;
    await this.loadGraph(path);
    if (this.activeFilteredTagRoles().length) {
      await this.refreshFilteredTags(true, 0);
    }
    el.pageStatus.textContent = `${selectedPages.toLocaleString()} available pages selected.`;
  },

  graphUrlForScope(this: GraphController): string {
    if (el.scope.value === "page") {
      return `/api/graph/categories/topology/page/${this.page.volumeId}/${this.page.pageNumber}`;
    }
    return `/api/graph/categories/topology/overview/${el.scope.value}`;
  },

  thematicBaseUrl(this: GraphController): string {
    const scope = this.thematicPayload?.layout.scope || "corpus";
    const focus = this.thematicPayload?.focus;
    if (scope === "page") {
      const volumeId = focus?.volumeId || this.page.volumeId;
      const pageNumber = focus?.pageNumber || this.page.pageNumber;
      return `/api/graph/categories/topology/page/${volumeId}/${pageNumber}`;
    }
    if (scope === "range") {
      const volumeId = focus?.volumeId || el.pageVolume.value;
      const startPage = focus?.startPage || Number(el.rangeStart.value);
      const endPage = focus?.endPage || Number(el.rangeEnd.value);
      return `/api/graph/categories/topology/range/${volumeId}/${startPage}/${endPage}`;
    }
    return `/api/graph/categories/topology/overview/${scope}`;
  },

  async loadScope(this: GraphController): Promise<void> {
    if (el.scope.value === "focus" || el.scope.value === "range") return;
    await this.loadGraph(this.graphUrlForScope());
    if (this.activeFilteredTagRoles().length) {
      await this.refreshFilteredTags(true, 0);
    }
  },

  /** Resolves true only when this request's payload was actually installed. */
  async loadGraph(this: GraphController, url: string): Promise<boolean> {
    this.graphAbort?.abort();
    this.evidenceAbort?.abort();
    this.graphAbort = new AbortController();
    const request = this.graphAbort;
    this.stopLayout();
    this.setLoading(true);
    try {
      const payload = await fetchJson<TopologyPayload>(url, request.signal);
      if (request !== this.graphAbort) return false;
      await this.installPayload(payload);
      this.scheduleFilteredTagsSync();
      return true;
    } catch (error) {
      if (request.signal.aborted) return false;
      clear(el.detail);
      el.detail.appendChild(
        textElement(
          "div",
          "graph-empty",
          error instanceof Error ? error.message : String(error),
        ),
      );
      return false;
    } finally {
      if (request === this.graphAbort && !this.readableLayoutPending) {
        this.setLoading(false);
      }
    }
  },
};
