const { test, expect } = require("playwright/test");

const baseUrl = process.env.GRAPH_TEST_URL || "http://127.0.0.1:5077";

/**
 * The theme category lists are checkbox groups, not <select multiple>: click
 * toggles, so bundling and unbundling both work with plain clicks.
 */
async function checkCategories(page, listId, ids) {
  const list = page.locator(listId);
  for (const input of await list.locator('input[type="checkbox"]:checked').all()) {
    await input.uncheck();
  }
  for (const id of ids) {
    await list.locator(`input[value="${id}"]`).check();
  }
}

async function checkedCategories(page, listId) {
  return page
    .locator(`${listId} input[type="checkbox"]:checked`)
    .evaluateAll((inputs) => inputs.map((input) => input.value));
}

/** Panels are independent now, so open one without toggling it shut. */
async function openPanel(page, panelId) {
  const panel = page.locator(panelId);
  if (!(await panel.evaluate((node) => node.open))) {
    await panel.locator("summary").click();
  }
}

test("reader shows axial categories and opens the thematic graph", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(`${baseUrl}/chronicles/vol1/47`);
  await expect(page.locator("#triple-list")).toContainText(
    "E01 · SovereignAndReigningMonarch",
  );
  await expect(page.locator("#triple-list")).toContainText(
    "R01 · GenealogicalDynasticAndHistoricalLegitimation",
  );
  await page.screenshot({
    path: "test-artifacts/axial-page-annotations.png",
    fullPage: true,
  });

  await page.locator("#graph-toggle").click();
  await expect(page.locator(".graph-view-indicator")).toHaveCount(0);
  await expect(page.locator("#graph-view-mode")).toHaveCount(0);
  if ((await page.locator("#graph-scope").inputValue()) !== "corpus") {
    await page.locator("#graph-scope").selectOption("corpus");
  }
  await expect(page.locator("#graph-scope")).toHaveValue("corpus");
  await openPanel(page, "#graph-browser-panel");
  await expect(page.locator("#graph-theme-direction")).toBeVisible();
  await expect(page.locator("#graph-theme-granularity")).toHaveValue("5");
  await openPanel(page, "#graph-filtered-tags-panel");
  await expect(page.locator("#graph-filtered-tags-search")).toBeVisible();
  // Both panels stay open together so graph clicks update them in place.
  await expect(page.locator("#graph-theme-direction")).toBeVisible();
  const overlay = page.locator('canvas[data-ready="true"]');
  await expect(overlay).toHaveAttribute("data-entity-nodes", "52", {
    timeout: 15_000,
  });
  await expect(overlay).toHaveAttribute("data-relation-nodes", "81");
  await expect(overlay).toHaveAttribute("data-rendered-graph-nodes", "52");
  await expect(overlay).toHaveAttribute("data-rendered-relation-nodes", "0");
  await expect(overlay).toHaveAttribute("data-patterns", "6050");
  await expect(overlay).toHaveAttribute("data-pattern-edges", "6050");
  await expect(overlay).toHaveAttribute("data-overlaps", "0");
  await expect(overlay).toHaveAttribute("data-visual-overlaps", "0");
  await expect(overlay).toHaveAttribute("data-visible-tags", "52");
  await expect(overlay).toHaveAttribute("data-visible-direct-edge-labels", "0");
  await expect(overlay).toHaveAttribute("data-foreground-relation-tags", "0");
  await expect(overlay).toHaveAttribute("data-ghost-relation-tags", "81");
  await expect(overlay).toHaveAttribute("data-entity-size-ratio", "10.000");
  const coreMedians = await overlay.evaluate((canvas) => ({
    high: Number(canvas.dataset.coreMedianHigh),
    low: Number(canvas.dataset.coreMedianLow),
  }));
  expect(coreMedians.high).toBeLessThan(coreMedians.low);
  await expect(page.locator("#graph-heading")).toContainText(
    "Thematic graph · Full corpus",
  );
  await expect(page.locator("#graph-counts")).toContainText(
    "52 entity categories",
  );
  await expect(page.locator("#graph-counts")).toContainText(
    "81 relation categories",
  );
  await expect(page.locator("#graph-counts")).toContainText(
    "6,050 exact patterns",
  );
  await expect(page.locator("#graph-theme-controls")).not.toHaveAttribute("hidden", "");
  await expect(page.locator("#graph-workspace")).not.toContainText(/[ÂÃâ]/);
  await expect(page.locator(".graph-filtered-tags-controls")).toHaveCount(1);
  await expect(page.locator("#graph-entity-categories .graph-category-option")).toHaveCount(52);
  await expect(page.locator("#graph-object-categories .graph-category-option")).toHaveCount(52);
  await expect(page.locator("#graph-relation-categories .graph-category-option")).toHaveCount(81);
  await page.screenshot({
    path: "test-artifacts/direct-thematic-network-overview.png",
  });

  const overviewLabels = Number(
    await overlay.getAttribute("data-full-entity-labels"),
  );
  await page.locator("#graph-zoom-in").click();
  await page.waitForTimeout(300);
  await page.locator("#graph-zoom-in").click();
  await expect
    .poll(async () =>
      Number(await overlay.getAttribute("data-full-entity-labels")),
    )
    .toBeGreaterThan(overviewLabels);
  await page.locator("#graph-fit").click();
  await expect(overlay).toHaveAttribute("data-visual-overlaps", "0");
  await page.waitForTimeout(500);

  const point = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.e01X),
    y: Number(canvas.dataset.e01Y),
  }));
  const bounds = await page.locator("#graph-canvas").boundingBox();
  expect(Math.abs(point.x - bounds.width / 2)).toBeLessThan(30);
  expect(Math.abs(point.y - bounds.height / 2)).toBeLessThan(30);
  await page.mouse.click(bounds.x + point.x, bounds.y + point.y);
  await expect(overlay).toHaveAttribute("data-available-selection-patterns", "1494");
  await expect(overlay).toHaveAttribute("data-minimum-pattern-instances", "5");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "442");
  await expect(overlay).toHaveAttribute("data-selected-relations", "66");
  await expect(overlay).toHaveAttribute("data-selected-counterparts", "46");
  const directLabelCounts = await overlay.evaluate((canvas) => ({
    visible: Number(canvas.dataset.visibleDirectEdgeLabels),
    hidden: Number(canvas.dataset.hiddenDirectEdgeLabels),
  }));
  expect(directLabelCounts.visible).toBeGreaterThan(0);
  expect(directLabelCounts.visible + directLabelCounts.hidden).toBe(442);
  await page.locator("#graph-theme-direction").selectOption("ab");
  await expect(overlay).toHaveAttribute("data-direction", "ab");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "356");
  await page.locator("#graph-theme-direction").selectOption("ba");
  await expect(overlay).toHaveAttribute("data-direction", "ba");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "91");
  await page.locator("#graph-theme-direction").selectOption("either");
  await page.locator("#graph-theme-granularity").fill("1");
  await expect(overlay).toHaveAttribute("data-minimum-pattern-instances", "1");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1494");
  await expect(overlay).toHaveAttribute("data-selected-relations", "78");
  await expect(overlay).toHaveAttribute("data-selected-counterparts", "52");
  await expect(page.locator("#graph-detail-content")).toContainText(
    "1,494 exact patterns",
  );
  const patternCounts = await page
    .locator("#graph-thematic-patterns option")
    .evaluateAll((options) =>
      options.map((option) => Number(option.dataset.patternCount)),
    );
  expect(patternCounts).toEqual(
    [...patternCounts].sort((left, right) => right - left),
  );

  const firstEntityRelationPoint = await overlay.evaluate((canvas) => ({
    id: canvas.dataset.firstRelationLabelId,
    x: Number(canvas.dataset.firstRelationLabelX),
    y: Number(canvas.dataset.firstRelationLabelY),
  }));
  const relationBounds = await page.locator("#graph-canvas").boundingBox();
  await page.mouse.move(
    relationBounds.x + firstEntityRelationPoint.x,
    relationBounds.y + firstEntityRelationPoint.y,
  );
  await expect(overlay).toHaveAttribute(
    "data-hovered-relation",
    firstEntityRelationPoint.id,
  );
  await expect(overlay).not.toHaveAttribute("data-hovered-relation-label", "");
  await expect(page.locator("#graph-canvas")).toHaveClass(
    /edge-label-hover/,
  );
  await page.mouse.click(
    relationBounds.x + firstEntityRelationPoint.x,
    relationBounds.y + firstEntityRelationPoint.y,
  );
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    "entity-relation",
  );
  await expect(overlay).toHaveAttribute("data-selected-relations", "1");
  await expect(page.locator("#graph-detail-content")).toContainText(
    "entity + relation focus",
  );

  await page
    .locator('[data-graph-action="thematic-relation-group"]')
    .click();
  await expect(overlay).toHaveAttribute("data-thematic-focus-mode", "relation");
  await expect(page.locator("#graph-detail-content")).toContainText(
    "relation category",
  );
  const relationInstancePoint = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.firstRelationLabelX),
    y: Number(canvas.dataset.firstRelationLabelY),
  }));
  await page.mouse.click(
    relationBounds.x + relationInstancePoint.x,
    relationBounds.y + relationInstancePoint.y,
  );
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    "exact-pattern",
  );
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1");
  await expect(page.locator("#graph-detail-content")).toContainText(
    "specific source",
  );

  await page.mouse.click(bounds.x + point.x, bounds.y + point.y);
  await expect(overlay).toHaveAttribute("data-thematic-focus-mode", "entity");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1494");

  const pairPoint = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.e24X),
    y: Number(canvas.dataset.e24Y),
  }));
  // Relation pills take the click where they overlap a node, so aim at a part
  // of the node no pill is covering. Recomputed at click time because each
  // selection re-lays the labels.
  const clearPointOnE24 = () =>
    overlay.evaluate((canvas) => {
      const centre = {
        x: Number(canvas.dataset.e24X),
        y: Number(canvas.dataset.e24Y),
      };
      const labels = (canvas.dataset.relationLabelPoints || "")
        .split(";").filter(Boolean)
        .map((entry) => entry.split(",").map(Number))
        .map(([pattern, x, y]) => ({ x, y }));
      const circle = (canvas.dataset.entityCircles || "")
        .split(";").filter(Boolean)
        .map((entry) => entry.split(",").map(Number))
        .map(([x, y, radius]) => ({ x, y, radius }))
        .find((item) => Math.hypot(item.x - centre.x, item.y - centre.y) < 2);
      const clear = (point) =>
        labels.every(
          (label) =>
            Math.abs(label.x - point.x) > 19 || Math.abs(label.y - point.y) > 12,
        );
      if (clear(centre) || !circle) return centre;
      for (let step = 0.25; step <= 0.9; step += 0.1) {
        for (let turn = 0; turn < 16; turn += 1) {
          const angle = (turn / 16) * Math.PI * 2;
          const candidate = {
            x: centre.x + Math.cos(angle) * circle.radius * step,
            y: centre.y + Math.sin(angle) * circle.radius * step,
          };
          if (clear(candidate)) return candidate;
        }
      }
      return centre;
    });
  const pairBounds = await page.locator("#graph-canvas").boundingBox();
  const orderedRelationPoint = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.firstRelationLabelX),
    y: Number(canvas.dataset.firstRelationLabelY),
  }));
  await page.mouse.click(
    pairBounds.x + orderedRelationPoint.x,
    pairBounds.y + orderedRelationPoint.y,
  );
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    "entity-relation",
  );
  const pairClickPoint = await clearPointOnE24();
  await page.mouse.click(
    pairBounds.x + pairClickPoint.x,
    pairBounds.y + pairClickPoint.y,
  );
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    "entity-pair-relation",
  );
  const emptyStagePoint = {
    x: pairBounds.x + pairBounds.width - 4,
    y: pairBounds.y + pairBounds.height - 4,
  };
  const panStart = {
    x: pairBounds.x + pairBounds.width - 24,
    y: pairBounds.y + pairBounds.height - 24,
  };
  const panEnd = { x: panStart.x - 140, y: panStart.y - 90 };
  await page.mouse.move(panStart.x, panStart.y);
  await page.mouse.down();
  await page.mouse.move(panEnd.x, panEnd.y, { steps: 8 });
  await page.mouse.up();
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    "entity-pair-relation",
  );
  const pairPointAfterDrag = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.e24X),
    y: Number(canvas.dataset.e24Y),
  }));
  expect(
    Math.hypot(
      pairPointAfterDrag.x - pairPoint.x,
      pairPointAfterDrag.y - pairPoint.y,
    ),
  ).toBeGreaterThan(40);
  await page.mouse.move(emptyStagePoint.x, emptyStagePoint.y);
  await expect(overlay).not.toHaveAttribute("data-hovered-relation", /.+/);
  const graphMouseCanvas = page.locator("#graph-canvas canvas.sigma-mouse");
  await graphMouseCanvas.click({
    position: { x: pairBounds.width - 4, y: pairBounds.height - 4 },
    force: true,
  });
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    "entity-relation",
  );
  await graphMouseCanvas.click({
    position: { x: pairBounds.width - 4, y: pairBounds.height - 4 },
    force: true,
  });
  await expect(overlay).toHaveAttribute("data-thematic-focus-mode", "entity");
  const pairPointForSelection = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.e24X),
    y: Number(canvas.dataset.e24Y),
  }));
  await page.mouse.click(
    pairBounds.x + pairPointForSelection.x,
    pairBounds.y + pairPointForSelection.y,
  );
  await expect(overlay).toHaveAttribute("data-selected-patterns", "58");
  await expect(overlay).toHaveAttribute("data-selected-relations", "51");
  await expect(overlay).toHaveAttribute("data-visible-direct-edge-labels", "58");
  await expect(page.locator("#graph-detail-content")).toContainText(
    "58 exact patterns",
  );
  await expect(page.locator(".graph-category-mechanism")).toHaveCount(0);
  await expect(page.locator("#graph-thematic-patterns option")).toHaveCount(58);
  expect(await checkedCategories(page, "#graph-entity-categories")).toEqual(["E01"]);
  expect(await checkedCategories(page, "#graph-object-categories")).toEqual(["E24"]);
  // Inspecting a pattern is a navigation move: it narrows the graph to that
  // one pattern as well as opening its evidence.
  await page
    .locator('[data-graph-action="thematic-evidence-selected"]')
    .click();
  await expect(page.locator("#graph-detail-content")).toContainText(
    "exact thematic pattern",
  );
  await expect(page.locator("#graph-detail-content .graph-claim"))
    .not.toHaveCount(0);
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1");
  // "Back to foregrounded patterns" steps back out of the layer that focusing
  // the pattern pushed, so the graph widens with the inspector.
  await page.locator('[data-graph-action="thematic-back"]').click();
  await expect(overlay).toHaveAttribute("data-selected-patterns", "58");

  const exactPairRelationPoint = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.firstRelationLabelX),
    y: Number(canvas.dataset.firstRelationLabelY),
  }));
  await page.mouse.click(
    relationBounds.x + exactPairRelationPoint.x,
    relationBounds.y + exactPairRelationPoint.y,
  );
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    "exact-pattern",
  );
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1");
  await expect(overlay).toHaveAttribute("data-selected-relations", "1");
  await expect(page.locator("#graph-detail-content")).toContainText(
    "specific source",
  );

  await graphMouseCanvas.click({
    position: { x: pairBounds.width - 4, y: pairBounds.height - 4 },
    force: true,
  });
  await expect(overlay).toHaveAttribute("data-thematic-focus-mode", "entity-pair");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "58");
  await graphMouseCanvas.click({
    position: { x: pairBounds.width - 4, y: pairBounds.height - 4 },
    force: true,
  });
  await expect(overlay).toHaveAttribute("data-thematic-focus-mode", "entity");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1494");

  await page.locator("#graph-theme-granularity").fill("5");
  await page.locator("#graph-theme-direction").selectOption("either");
  await checkCategories(page, "#graph-entity-categories", ["E01"]);
  await checkCategories(page, "#graph-object-categories", []);
  await checkCategories(page, "#graph-relation-categories", []);
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute("data-thematic-focus-mode", "entity");
  // Group A alone, with group B left empty, is everything E01 takes part in
  // whichever side it was written on -- the same 442 as clicking its node.
  await expect(overlay).toHaveAttribute("data-selected-patterns", "442");
  expect(
    Number(await overlay.getAttribute("data-visible-direct-edge-labels")),
  ).toBeGreaterThan(0);
  await expect(
    page.locator('[data-graph-action="category-entity"]'),
  ).toHaveCount(0);

  await checkCategories(page, "#graph-entity-categories", []);
  await checkCategories(page, "#graph-object-categories", []);
  await checkCategories(page, "#graph-relation-categories", ["R05"]);
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute("data-thematic-focus-mode", "relation");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "21");
  await expect(overlay).toHaveAttribute(
    "data-visible-direct-edge-labels",
    "21",
  );
  await checkCategories(page, "#graph-entity-categories", ["E01"]);
  await page.locator('[data-graph-action="category-relation"]').click();
  await expect(overlay).toHaveAttribute("data-selected-patterns", "21");
  expect(
    await page
      .locator('#graph-entity-categories input[type="checkbox"]:checked')
      .count(),
  ).toBe(0);

  await checkCategories(page, "#graph-entity-categories", ["E01"]);
  await checkCategories(page, "#graph-object-categories", []);
  await checkCategories(page, "#graph-relation-categories", ["R05"]);
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    "entity-relation",
  );
  // 13 patterns put E01 on the subject side of R05 and one puts it on the
  // object side; the undirected pairing foregrounds all 14.
  await expect(overlay).toHaveAttribute("data-selected-patterns", "14");
  await expect(overlay).toHaveAttribute("data-selected-relations", "1");
  await expect(overlay).toHaveAttribute(
    "data-visible-direct-edge-labels",
    "14",
  );

  await checkCategories(page, "#graph-entity-categories", ["E01"]);
  await checkCategories(page, "#graph-object-categories", ["E24"]);
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    "entity-pair-relation",
  );
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1");
  await expect(overlay).toHaveAttribute("data-visible-direct-edge-labels", "1");

  await page.locator("#graph-theme-direction").selectOption("ab");
  await expect(overlay).toHaveAttribute("data-direction", "ab");
  await page.locator("#graph-theme-direction").selectOption("either");

  // Tag lookup now lives in the tag bucket, which stays scoped to the theme
  // categories the graph navigation just locked in.
  await openPanel(page, "#graph-filtered-tags-panel");
  await page.locator("#graph-filtered-tags-search").fill("King");
  await page.locator("#graph-filtered-tags-search-button").click();
  await expect(
    page.locator("#graph-filtered-tags-results .graph-result-label").first(),
  ).toContainText("King", { timeout: 15_000 });
  await expect(page.locator(".graph-view-indicator")).toHaveCount(0);
  await expect(page.locator(".graph-filtered-tags-controls")).toHaveCount(1);
  await expect(page.locator("#graph-theme-controls")).not.toHaveAttribute("hidden", "");
  expect(errors).toEqual([]);
});
