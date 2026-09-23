import { GraphController } from "./graph/controller";
import { el } from "./graph/dom";

const controller = new GraphController();

el.detail.addEventListener("click", (event) => {
  const button = (event.target as Element).closest<HTMLElement>(
    "[data-graph-action]",
  );
  if (button) controller.handleDetailAction(button);
});
el.themeApply.addEventListener(
  "click",
  () => void controller.applyThemeFilters(),
);
el.themeClear.addEventListener(
  "click",
  () => void controller.clearThemeFilters(),
);
el.themeDirection.addEventListener("change", () => {
  controller.refreshCategoryFacets();
  controller.refreshThematicExplorationFilters();
});
el.themeGranularity.addEventListener("input", () =>
  controller.refreshThematicExplorationFilters(),
);
for (const container of [
  el.entityCategories,
  el.relationCategories,
  el.objectCategories,
]) {
  // `change` bubbles from each checkbox, so one listener covers the list.
  container.addEventListener("change", () => {
    controller.syncCategoryPickerFooter(container);
    controller.recordCategoryControls();
    controller.refreshCategoryFacets();
    controller.scheduleFilteredTagsSync();
  });
  const filter = document.getElementById(`${container.id}-filter`);
  if (filter instanceof HTMLInputElement) {
    filter.addEventListener("input", () =>
      controller.filterCategoryPicker(container, filter.value),
    );
  }
  const clear = document.getElementById(`${container.id}-clear`);
  if (clear instanceof HTMLButtonElement) {
    clear.addEventListener("click", () =>
      controller.clearCategoryPicker(container),
    );
  }
}
el.themeSimilarity.addEventListener("input", () => {
  el.themeSimilarityValue.value = `${el.themeSimilarity.value}%`;
});
el.themeSimilarity.addEventListener("change", () => {
  if (!el.themeSimilarClear.disabled) void controller.loadSimilarPatterns();
});
el.themeSimilar.addEventListener(
  "click",
  () => void controller.loadSimilarPatterns(),
);
el.themeSimilarClear.addEventListener("click", () =>
  controller.clearSimilarPatterns(),
);
el.themeSimilarResults.addEventListener("change", () =>
  controller.syncSimilarPatternSelection(),
);
el.themeSimilarApply.addEventListener("click", () =>
  controller.applySimilarPatterns(),
);
el.filteredTagsSimilarity.addEventListener("input", () => {
  el.filteredTagsSimilarityValue.value = `${el.filteredTagsSimilarity.value}%`;
  controller.scheduleSimilarTagThreshold();
});
el.filteredTagsSimilar.addEventListener(
  "click",
  () => void controller.findSimilarTags(),
);
el.filteredTagsSimilarClear.addEventListener(
  "click",
  () => void controller.clearSimilarTags(),
);
el.filteredTagsRole.addEventListener(
  "change",
  () => void controller.changeFilteredTagRole(),
);
el.filteredTagsSearch.addEventListener("input", () =>
  controller.scheduleFilteredTagSearch(),
);
el.filteredTagsSearch.addEventListener("keydown", (event) => {
  if (event.key === "Enter") void controller.refreshFilteredTags(false, 0);
});
el.filteredTagsSearchButton.addEventListener(
  "click",
  () => void controller.refreshFilteredTags(false, 0),
);
el.filteredTagsResults.addEventListener("change", (event) => {
  const input = event.target;
  if (input instanceof HTMLInputElement && input.type === "checkbox") {
    controller.handleFilteredTagChange(input);
  }
});
el.filteredTagsSelectPage.addEventListener("click", () =>
  controller.selectFilteredTagPage(),
);
el.filteredTagsClear.addEventListener("click", () =>
  controller.clearFilteredTagSelection(),
);
el.filteredTagsPrev.addEventListener("click", () =>
  controller.previousFilteredTagsPage(),
);
el.filteredTagsNext.addEventListener("click", () =>
  controller.nextFilteredTagsPage(),
);
el.filteredTagsApply.addEventListener(
  "click",
  () => void controller.applyFilteredTags(),
);
el.filteredTagsRelease.addEventListener(
  "click",
  () => void controller.releaseAppliedTagFilters(),
);
// The panels stay independently open: navigating the graph updates both, so
// forcing one shut would hide the live result the user is watching.
el.filteredTagsPanel.addEventListener("toggle", () => {
  if (el.filteredTagsPanel.open) controller.refreshFilteredTagsIfStale();
});
el.scope.addEventListener("change", () => void controller.loadScope());
el.pageVolume.addEventListener("change", () =>
  controller.handlePageVolumeChange(),
);
el.pageNumber.addEventListener("change", () =>
  controller.handlePageNumberChange(),
);
el.pagePrev.addEventListener(
  "click",
  () => void controller.moveSelectedPage(-1),
);
el.pageNext.addEventListener(
  "click",
  () => void controller.moveSelectedPage(1),
);
el.pageLoad.addEventListener("click", () => void controller.loadSelectedPage());
el.rangeStart.addEventListener("change", () =>
  controller.handleRangeChange("start"),
);
el.rangeEnd.addEventListener("change", () =>
  controller.handleRangeChange("end"),
);
el.rangeLoad.addEventListener(
  "click",
  () => void controller.loadSelectedRange(),
);
el.zoomOut.addEventListener("click", () => controller.zoomOut());
el.zoomIn.addEventListener("click", () => controller.zoomIn());
el.fit.addEventListener("click", () => controller.fitGraph());
el.showAll.addEventListener("change", () =>
  controller.setCorpusShowAll(el.showAll.checked),
);
el.close.addEventListener("click", () => controller.close());
window.addEventListener("keydown", (event) => {
  if (el.workspace.hidden || event.ctrlKey || event.metaKey || event.altKey)
    return;
  const target = event.target;
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    (target instanceof HTMLElement && target.isContentEditable)
  )
    return;
  if (event.key === "Escape") {
    event.preventDefault();
    controller.navigateBackOneLayer();
  } else if (
    event.key === "+" ||
    event.key === "=" ||
    event.code === "NumpadAdd"
  ) {
    event.preventDefault();
    controller.zoomIn();
  } else if (
    event.key === "-" ||
    event.key === "_" ||
    event.code === "NumpadSubtract"
  ) {
    event.preventDefault();
    controller.zoomOut();
  } else if (event.key === "Escape") {
    controller.close();
  }
});

window.ChronicleGraph = {
  open: (options) => controller.open(options),
  close: () => controller.close(),
  setPage: (volumeId, pageNumber) => controller.setPage(volumeId, pageNumber),
  isOpen: () => !el.workspace.hidden,
};
