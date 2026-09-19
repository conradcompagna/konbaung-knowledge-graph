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

async function openGraph(page) {
  await page.goto(`${baseUrl}/chronicles/vol1/47`);
  await page.locator("#graph-toggle").click();
  await expect(page.locator('canvas[data-ready="true"]')).toHaveCount(1, {
    timeout: 30000,
  });
}

test("tag buckets and theme categories follow graph navigation", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);

  // Semantic search no longer has its own accordion.
  await expect(page.locator("#graph-semantic-panel")).toHaveCount(0);

  // With no categories applied the bucket still lists every in-scope tag,
  // which is what replaces the old standalone semantic search entry point.
  await page.locator("#graph-filtered-tags-summary").click();
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "1-100 of 23,890 tags",
    { timeout: 30000 },
  );

  // Navigating the graph by clicking a node re-derives the theme selects and
  // reloads the open tag bucket without closing and reopening the panel.
  await page.locator("#graph-browser-summary").click();
  await checkCategories(page, "#graph-entity-categories", ["E01"]);
  await page.locator("#graph-theme-apply").click();
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 623 tags",
    { timeout: 30000 },
  );
  await expect(page.locator("#graph-filtered-tags-role")).toHaveValue("subject");
  await expect(page.locator("#graph-theme-similar-anchor")).toContainText(
    "subject E01",
  );

  // Pagination walks forward and back rather than sticking on page one.
  await page.locator("#graph-filtered-tags-next").click();
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "101-200 of 623 tags",
  );
  await page.locator("#graph-filtered-tags-next").click();
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "201-300 of 623 tags",
  );
  await page.locator("#graph-filtered-tags-prev").click();
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "101-200 of 623 tags",
  );
  expect(errors).toEqual([]);
});

test("checked tags drive an averaged embedding ranking", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await checkCategories(page, "#graph-entity-categories", ["E01"]);
  await page.locator("#graph-theme-apply").click();
  await page.locator("#graph-filtered-tags-summary").click();
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 623 tags",
    { timeout: 30000 },
  );

  // Check "King" and "Alaungpaya" and rank the bucket by their mean vector.
  const rows = page.locator("#graph-filtered-tags-results input");
  await rows.nth(0).check();
  await rows.nth(1).check();
  await page.locator("#graph-filtered-tags-similar").click();
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "similar to the averaged vector of 2 checked tags",
    { timeout: 30000 },
  );
  const labels = page.locator("#graph-filtered-tags-results .graph-result-label");
  await expect(labels.nth(0)).toHaveText("King");
  await expect(labels.nth(1)).toHaveText("Alaungpaya");
  await expect(
    page.locator("#graph-filtered-tags-results .graph-result-meta").nth(2),
  ).toContainText("% similar");
  // Both anchors stay checked, so the graph filter is still applicable.
  await expect(page.locator("#graph-filtered-tags-selection-note")).toContainText(
    "2 of 100 tags checked",
  );

  await page.locator("#graph-filtered-tags-similar-clear").click();
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "ranked by frequency",
    { timeout: 30000 },
  );
  expect(errors).toEqual([]);
});

test("similar statement patterns extend the locked triple selection", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await checkCategories(page, "#graph-entity-categories", ["E01"]);
  await checkCategories(page, "#graph-relation-categories", ["R01"]);
  await page.locator("#graph-theme-apply").click();

  await page.locator("#graph-theme-similarity").fill("50");
  await page.locator("#graph-theme-similar").click();
  await expect(page.locator("#graph-theme-similar-note")).toContainText(
    "patterns are at least 50% similar",
    { timeout: 60000 },
  );
  const patterns = page.locator("#graph-theme-similar-results input");
  expect(await patterns.count()).toBeGreaterThan(0);
  await expect(
    page.locator("#graph-theme-similar-results .graph-result-label").first(),
  ).toContainText("→");

  const overlay = page.locator('canvas[data-ready="true"]');
  const before = Number(await overlay.getAttribute("data-selected-patterns"));
  await patterns.nth(0).check();
  await patterns.nth(1).check();
  await page.locator("#graph-theme-similar-apply").click();
  await expect(page.locator("#graph-theme-similar-note")).toContainText(
    "to the selection",
  );
  await expect
    .poll(async () => Number(await overlay.getAttribute("data-selected-patterns")))
    .toBeGreaterThan(before);
  expect(errors).toEqual([]);
});

test("clicking the graph live-updates both panels without reopening them", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-filtered-tags-summary").click();
  // Both panels stay open at once.
  await expect(page.locator("#graph-browser-panel")).toHaveAttribute("open", "");
  await expect(page.locator("#graph-filtered-tags-panel")).toHaveAttribute("open", "");
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 23,890 tags",
    { timeout: 30000 },
  );

  // Click the E01 node on the canvas; the theme selects, the tag bucket, and
  // the similar-pattern anchor all follow that click while the panels stay put.
  const overlay = page.locator('canvas[data-ready="true"]');
  await page.locator("#graph-fit").click();
  await page.waitForTimeout(600);
  const point = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.e01X),
    y: Number(canvas.dataset.e01Y),
  }));
  const bounds = await page.locator("#graph-canvas").boundingBox();
  await page.mouse.click(bounds.x + point.x, bounds.y + point.y);

  expect(await checkedCategories(page, "#graph-entity-categories")).toEqual(["E01"]);
  await expect(page.locator("#graph-theme-similar-anchor")).toContainText(
    "subject E01",
    { timeout: 15000 },
  );
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 623 tags",
    { timeout: 30000 },
  );
  await expect(page.locator("#graph-filtered-tags-role")).toHaveValue("subject");
  await expect(page.locator("#graph-browser-panel")).toHaveAttribute("open", "");
  await expect(page.locator("#graph-filtered-tags-panel")).toHaveAttribute("open", "");
  expect(errors).toEqual([]);
});

test("applied tag filters progressively narrow the other role buckets", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-filtered-tags-summary").click();

  // Pin three spellings of one king as the subject.
  await page.locator("#graph-filtered-tags-search").fill("Alaung");
  await page.locator("#graph-filtered-tags-search-button").click();
  const rows = page.locator("#graph-filtered-tags-results input");
  // Wait for the searched list, not merely for some list, before checking.
  await expect(
    page.locator("#graph-filtered-tags-results .graph-result-label").first(),
  ).toHaveText("Alaungpaya", { timeout: 30000 });
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "group A tags",
  );
  for (const index of [0, 1, 2]) await rows.nth(index).check();
  await expect(page.locator("#graph-filtered-tags-selection-note")).toContainText(
    "3 of 100 tags checked",
  );
  await page.locator("#graph-filtered-tags-apply").click();
  await expect(page.locator("#graph-counts")).toContainText(
    "3 selected low-level tags",
    { timeout: 30000 },
  );

  // The relation and object buckets now describe only that king's claims.
  await page.locator("#graph-filtered-tags-role").selectOption("relation");
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 855 tags",
    { timeout: 30000 },
  );
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "Narrowed by the applied 3 group A tags",
  );
  await page.locator("#graph-filtered-tags-role").selectOption("object");
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 926 tags",
    { timeout: 30000 },
  );

  // The subject facet stays open so another spelling can still be added.
  await page.locator("#graph-filtered-tags-role").selectOption("subject");
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 23,890 tags",
    { timeout: 30000 },
  );

  // Narrow a second step: add a relation tag on top of the subject filter.
  await page.locator("#graph-filtered-tags-role").selectOption("relation");
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 855 tags",
    { timeout: 30000 },
  );
  const relations = page.locator("#graph-filtered-tags-results input");
  await relations.nth(0).check();
  await page.locator("#graph-filtered-tags-apply").click();
  await expect(page.locator("#graph-counts")).toContainText(
    "4 selected low-level tags",
    { timeout: 30000 },
  );
  await page.locator("#graph-filtered-tags-role").selectOption("object");
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "Narrowed by the applied 3 group A tags and 1 relation tag",
    { timeout: 30000 },
  );
  const narrowedObjects = await page
    .locator("#graph-filtered-tags-page")
    .textContent();
  expect(Number(narrowedObjects.match(/of ([\d,]+) tags/)[1].replace(/,/g, "")))
    .toBeLessThan(926);

  // Releasing the filter restores the base graph and the full buckets.
  await page.locator("#graph-filtered-tags-release").click();
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 23,890 tags",
    { timeout: 30000 },
  );
  await expect(page.locator("#graph-filtered-tags-note")).not.toContainText(
    "Narrowed by",
  );
  expect(errors).toEqual([]);
});

test("entity groups bundle several categories on each side of a relation", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();

  // Kings + heirs + royal women on one side, monks + monastic orders on the
  // other, joined by religious patronage.
  await checkCategories(page, "#graph-entity-categories", ["E01", "E02", "E03"]);
  await checkCategories(page, "#graph-relation-categories", ["R05"]);
  await checkCategories(page, "#graph-object-categories", ["E20", "E21"]);
  await expect(page.locator("#graph-theme-direction")).toHaveValue("either");
  await page.locator("#graph-theme-apply").click();
  await page.locator("#graph-filtered-tags-summary").click();

  // Both bundles are reachable, and the buckets are named for the groups.
  await expect(page.locator("#graph-filtered-tags-role")).toHaveValue("subject");
  await expect(
    page.locator("#graph-filtered-tags-role option").first(),
  ).toContainText("Entity group A tags / E01, E02, E03");
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "group A tags in the applied categories",
    { timeout: 30000 },
  );
  const eitherWay = await page.locator("#graph-filtered-tags-page").textContent();
  const eitherCount = Number(eitherWay.match(/of ([\d,]+) tags/)[1].replace(/,/g, ""));
  expect(eitherCount).toBeGreaterThan(0);

  // Restricting the direction is a strict subset of the undirected pairing.
  await page.locator("#graph-theme-direction").selectOption("ab");
  await page.locator("#graph-theme-apply").click();
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "group A tags",
    { timeout: 30000 },
  );
  await expect
    .poll(async () => {
      const text = await page.locator("#graph-filtered-tags-page").textContent();
      return Number(text.match(/of ([\d,]+) tags/)[1].replace(/,/g, ""));
    })
    .toBeLessThanOrEqual(eitherCount);
  expect(errors).toEqual([]);
});

test("theme category lists re-derive their counts from the loaded slice", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();

  // Full corpus: every category is reachable.
  const disabled = page.locator(
    '#graph-entity-categories input[type="checkbox"]:disabled',
  );
  await expect(disabled).toHaveCount(0);
  const row = page.locator("#graph-entity-categories .graph-category-option")
    .filter({ has: page.locator('input[value="E01"]') });
  const corpusLabel = await row.locator(".graph-category-option-count").textContent();

  // A single page contains only a handful, and the counts follow the slice.
  await page.locator("#graph-scope").selectOption("page");
  await expect
    .poll(async () => await disabled.count(), { timeout: 30000 })
    .toBeGreaterThan(0);
  const pageLabel = await row.locator(".graph-category-option-count").textContent();
  expect(pageLabel).not.toBe(corpusLabel);
  expect(errors).toEqual([]);
});

test("category lists multi-select and deselect with plain clicks", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();

  const list = page.locator("#graph-entity-categories");
  const row = (id) =>
    list.locator(".graph-category-option").filter({
      has: page.locator(`input[value="${id}"]`),
    });

  // Three plain clicks accumulate rather than replacing one another.
  for (const id of ["E01", "E02", "E03"]) await row(id).click();
  expect(await checkedCategories(page, "#graph-entity-categories"))
    .toEqual(["E01", "E02", "E03"]);
  await expect(page.locator("#graph-entity-categories-count"))
    .toHaveText("3 checked · combined with OR");

  // A plain click on a checked row removes just that one.
  await row("E02").click();
  expect(await checkedCategories(page, "#graph-entity-categories"))
    .toEqual(["E01", "E03"]);

  // Relation categories behave the same way; several can be combined.
  for (const id of ["R05", "R15"]) {
    await page
      .locator("#graph-relation-categories .graph-category-option")
      .filter({ has: page.locator(`input[value="${id}"]`) })
      .click();
  }
  expect(await checkedCategories(page, "#graph-relation-categories"))
    .toEqual(["R05", "R15"]);

  // The filter box narrows a long list without disturbing what is checked.
  await page.locator("#graph-relation-categories-filter").fill("patronage");
  const visible = page.locator(
    "#graph-relation-categories .graph-category-option:visible",
  );
  expect(await visible.count()).toBeGreaterThan(0);
  expect(await visible.count()).toBeLessThan(81);
  expect(await checkedCategories(page, "#graph-relation-categories"))
    .toEqual(["R05", "R15"]);
  await page.locator("#graph-relation-categories-filter").fill("");

  // Each list clears on its own, leaving the others alone.
  await page.locator("#graph-relation-categories-clear").click();
  expect(await checkedCategories(page, "#graph-relation-categories")).toEqual([]);
  expect(await checkedCategories(page, "#graph-entity-categories"))
    .toEqual(["E01", "E03"]);
  expect(errors).toEqual([]);
});

test("the graph honours the pairing direction, not the written subject order", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');
  const click = async (list, id) =>
    page.locator(`${list} .graph-category-option`)
      .filter({ has: page.locator(`input[value="${id}"]`) }).click();

  // Sovereigns and Brahmins: the corpus writes 14 patterns one way and 11 the
  // other, so an undirected pairing must foreground all 25.
  await click("#graph-entity-categories", "E01");
  await click("#graph-object-categories", "E22");
  const patterns = async (direction) => {
    await page.locator("#graph-theme-direction").selectOption(direction);
    await page.locator("#graph-theme-apply").click();
    await expect
      .poll(async () => await overlay.getAttribute("data-direction"))
      .toBe(direction);
    return Number(await overlay.getAttribute("data-selected-patterns"));
  };
  expect(await patterns("either")).toBe(25);
  expect(await patterns("ab")).toBe(14);
  // The reverse orientation is where Brahmins are the acting subject; before
  // the pairing rework the graph returned nothing at all for it.
  expect(await patterns("ba")).toBe(11);
  expect(errors).toEqual([]);
});

test("theme lists narrow to what is reachable under the other slots", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  const enabled = (list) =>
    page.locator(`${list} input[type="checkbox"]:not(:disabled)`).count();
  const click = async (list, id) =>
    page.locator(`${list} .graph-category-option`)
      .filter({ has: page.locator(`input[value="${id}"]`) }).click();

  expect(await enabled("#graph-relation-categories")).toBe(81);
  await click("#graph-entity-categories", "E01");
  await click("#graph-object-categories", "E22");

  // Exactly the 20 relations that hold between sovereigns and Brahmins.
  await expect
    .poll(async () => await enabled("#graph-relation-categories"))
    .toBe(20);
  const relations = await page
    .locator('#graph-relation-categories input[type="checkbox"]:not(:disabled)')
    .evaluateAll((inputs) => inputs.map((input) => input.value));
  expect(relations).toContain("R05");
  expect(relations).not.toContain("R54");

  // A slot never constrains itself, so a third entity can still be added.
  expect(await enabled("#graph-entity-categories")).toBeGreaterThan(1);

  // Unchecking widens the list back out.
  await click("#graph-object-categories", "E22");
  await expect
    .poll(async () => await enabled("#graph-relation-categories"))
    .toBeGreaterThan(20);
  expect(errors).toEqual([]);
});

test("edge colour separates the two orientations of a pairing", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');
  const click = async (list, id) =>
    page.locator(`${list} .graph-category-option`)
      .filter({ has: page.locator(`input[value="${id}"]`) }).click();

  await click("#graph-entity-categories", "E01");
  await click("#graph-object-categories", "E22");
  await page.locator("#graph-theme-apply").click();

  // Both slots filled used to paint every path the same flat "pair" colour,
  // which hid the direction exactly when it matters most.
  await expect(overlay).toHaveAttribute("data-selected-patterns", "25");
  await expect(overlay).toHaveAttribute("data-outgoing-paths", "14");
  await expect(overlay).toHaveAttribute("data-incoming-paths", "11");
  await expect(page.locator("#graph-edge-legend")).toContainText(
    "purple = anchor is the subject",
  );

  // Restricting the direction leaves only that orientation coloured.
  await page.locator("#graph-theme-direction").selectOption("ab");
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute("data-outgoing-paths", "14");
  await expect(overlay).toHaveAttribute("data-incoming-paths", "0");
  await page.locator("#graph-theme-direction").selectOption("ba");
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute("data-outgoing-paths", "0");
  await expect(overlay).toHaveAttribute("data-incoming-paths", "11");
  expect(errors).toEqual([]);
});

test("both orientations of a pairing share one non-overlapping label cluster", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1500, height: 1000 });
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');
  const click = async (list, id) =>
    page.locator(`${list} .graph-category-option`)
      .filter({ has: page.locator(`input[value="${id}"]`) }).click();

  // Princes and court officials are written both ways round, 15 and 16 times.
  // Each orientation used to build its own label grid on the shared midpoint,
  // so the two grids landed on top of one another.
  await click("#graph-entity-categories", "E02");
  await click("#graph-object-categories", "E05");
  await page.locator("#graph-theme-apply").click();

  await expect(overlay).toHaveAttribute("data-selected-patterns", "31");
  await expect(overlay).toHaveAttribute("data-outgoing-paths", "15");
  await expect(overlay).toHaveAttribute("data-incoming-paths", "16");
  // Every label is drawn, and none of them overlap another.
  await expect(overlay).toHaveAttribute("data-visible-direct-edge-labels", "31");
  await expect(overlay).toHaveAttribute("data-relation-relation-overlaps", "0");
  // ...and no two curves of the pairing resolve to the same control point,
  // which is what drew one orientation's lines on top of the other's.
  await expect(overlay).toHaveAttribute("data-coincident-paths", "0");
  expect(errors).toEqual([]);
});

test("no pairing draws overlapping lines or labels in either orientation", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1500, height: 1000 });
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');
  const uncheckAll = async (list) => {
    for (const input of await page.locator(`${list} input:checked`).all()) {
      await input.uncheck();
    }
  };

  // Pairings that the corpus writes in both directions, of varying density.
  for (const [a, b] of [["E02", "E05"], ["E01", "E24"], ["E01", "E02"], ["E01", "E05"]]) {
    await uncheckAll("#graph-entity-categories");
    await uncheckAll("#graph-object-categories");
    await page.locator(`#graph-entity-categories input[value="${a}"]`).check();
    await page.locator(`#graph-object-categories input[value="${b}"]`).check();
    await page.locator("#graph-theme-apply").click();
    await expect
      .poll(async () => await overlay.getAttribute("data-coincident-paths"))
      .toBe("0");
    await expect(overlay).toHaveAttribute("data-relation-relation-overlaps", "0");
    // Relation pills can still touch an entity circle at some zooms; that is
    // a separate, pre-existing concern and is not asserted here.
    // Every selected pattern is still drawn; nothing was dropped to fit.
    const selected = await overlay.getAttribute("data-selected-patterns");
    await expect(overlay).toHaveAttribute("data-visible-direct-edge-labels", selected);
  }
  expect(errors).toEqual([]);
});

test("one navigation state drives themes, tags, canvas and inspector alike", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1500, height: 1000 });
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-filtered-tags-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');
  const relationFacet = () =>
    page.locator('#graph-relation-categories input:not(:disabled)').count();

  // Entry point 1: the theme lists.
  await page.locator('#graph-entity-categories input[value="E01"]').check();
  await page.locator('#graph-object-categories input[value="E22"]').check();
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute("data-selected-patterns", "25");
  await expect.poll(relationFacet).toBe(20);
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 20 tags",
    { timeout: 30000 },
  );
  await expect(page.locator("#graph-detail-content")).toContainText(
    "25 exact patterns",
  );

  // Entry point 2: the canvas. Every surface follows the click.
  await page.locator("#graph-fit").click();
  await page.waitForTimeout(700);
  const point = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.e01X),
    y: Number(canvas.dataset.e01Y),
  }));
  const bounds = await page.locator("#graph-canvas").boundingBox();
  await page.mouse.click(bounds.x + point.x, bounds.y + point.y);
  await expect
    .poll(async () => await checkedCategories(page, "#graph-object-categories"))
    .toEqual([]);
  await expect.poll(relationFacet).toBe(78);
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 623 tags",
    { timeout: 30000 },
  );

  // Entry point 3: the inspector's own navigation action.
  await page
    .locator('#graph-detail-content button[data-graph-action="thematic-reset"]')
    .first()
    .click();
  await expect(overlay).toHaveAttribute("data-selected-patterns", "0");
  await expect
    .poll(async () => await checkedCategories(page, "#graph-entity-categories"))
    .toEqual([]);
  await expect.poll(relationFacet).toBe(81);
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 23,890 tags",
    { timeout: 30000 },
  );
  expect(errors).toEqual([]);
});

test("history layers restore the whole state, including pairing direction", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');

  await page.locator('#graph-entity-categories input[value="E01"]').check();
  await page.locator('#graph-object-categories input[value="E22"]').check();
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute("data-selected-patterns", "25");
  await expect(overlay).toHaveAttribute("data-outgoing-paths", "14");
  await expect(overlay).toHaveAttribute("data-incoming-paths", "11");

  // Drill in, then step back out. The snapshot used to list its fields by
  // hand and had fallen behind, dropping the orientation on the way back.
  await page.locator("#graph-fit").click();
  await page.waitForTimeout(700);
  const point = await overlay.evaluate((canvas) => ({
    x: Number(canvas.dataset.e01X),
    y: Number(canvas.dataset.e01Y),
  }));
  const bounds = await page.locator("#graph-canvas").boundingBox();
  await page.mouse.click(bounds.x + point.x, bounds.y + point.y);
  // Every pattern E01 takes part in, at a minimum-instance threshold of 1.
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1494");

  await page.locator("#graph-canvas").press("Escape");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "25");
  await expect(overlay).toHaveAttribute("data-outgoing-paths", "14");
  await expect(overlay).toHaveAttribute("data-incoming-paths", "11");
  expect(await checkedCategories(page, "#graph-entity-categories")).toEqual(["E01"]);
  expect(await checkedCategories(page, "#graph-object-categories")).toEqual(["E22"]);
  expect(errors).toEqual([]);
});

test("hovering a relation pill lights only its own direction", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1500, height: 1000 });
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');

  // Court officials and regalia carry R24 in both directions.
  await page.locator('#graph-entity-categories input[value="E05"]').check();
  await page.locator('#graph-object-categories input[value="E31"]').check();
  await page.locator("#graph-theme-apply").click();
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-fit").click();
  await page.waitForTimeout(900);

  const points = (await overlay.getAttribute("data-relation-label-points"))
    .split(";")
    .map((entry) => entry.split(",").map(Number))
    .map(([pattern, x, y]) => ({ pattern, x, y }));
  expect(points.length).toBeGreaterThan(1);
  const bounds = await page.locator("#graph-canvas").boundingBox();

  // Hovering two different pills reports two different patterns, so the two
  // directions of one relation category are no longer highlighted together.
  const seen = new Set();
  for (const point of points.slice(0, 2)) {
    await page.mouse.move(bounds.x + point.x, bounds.y + point.y);
    await expect
      .poll(async () => await overlay.getAttribute("data-hovered-pattern"))
      .toBe(String(point.pattern));
    seen.add(await overlay.getAttribute("data-hovered-pattern"));
  }
  expect(seen.size).toBe(2);
  expect(errors).toEqual([]);
});

test("inspecting a pattern narrows every surface, and the anchor follows the tag filter", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1500, height: 1000 });
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');
  await page.locator('#graph-entity-categories input[value="E05"]').check();
  await page.locator('#graph-object-categories input[value="E31"]').check();
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute("data-selected-patterns", "19");
  const entityFacetBefore = await page
    .locator('#graph-entity-categories input:not(:disabled)')
    .count();

  // The active share is always spelled out, including for fully active rows.
  await expect(page.locator("#graph-thematic-patterns option").first())
    .toContainText("32 of 32 triples in view");

  // Inspect is a navigation move: every surface narrows to that one pattern.
  await page
    .locator('#graph-detail-content button[data-graph-action="thematic-evidence-selected"]')
    .click();
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1");
  expect(await checkedCategories(page, "#graph-relation-categories")).toEqual(["R24"]);
  expect(await checkedCategories(page, "#graph-object-categories")).toEqual(["E31"]);
  // The relation list itself stays open -- a slot never constrains its own
  // facet -- but the entity facet narrows to what that one relation reaches.
  await expect
    .poll(async () =>
      page.locator('#graph-entity-categories input:not(:disabled)').count(),
    )
    .toBeLessThan(entityFacetBefore);
  expect(errors).toEqual([]);
});

test("every inspector button moves all four surfaces together", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1500, height: 1000 });
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-filtered-tags-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');
  const entityFacet = () =>
    page.locator('#graph-entity-categories input:not(:disabled)').count();

  await page.locator('#graph-entity-categories input[value="E05"]').check();
  await page.locator('#graph-object-categories input[value="E31"]').check();
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute("data-selected-patterns", "19");
  const wideFacet = await entityFacet();
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "of 78 tags",
    { timeout: 30000 },
  );

  // Each of these three buttons is a navigation move, and each must carry the
  // themes, the tag bucket and the canvas with it.
  const narrowed = async (label) => {
    await expect(overlay).toHaveAttribute("data-selected-patterns", "1");
    expect(await checkedCategories(page, "#graph-relation-categories")).toEqual(
      ["R24"],
    );
    await expect.poll(entityFacet).toBeLessThan(wideFacet);
    await expect(page.locator("#graph-filtered-tags-page")).toContainText(
      "of 31 tags",
      { timeout: 30000 },
    );
  };
  const widened = async () => {
    await expect(overlay).toHaveAttribute("data-selected-patterns", "19");
    expect(await checkedCategories(page, "#graph-relation-categories")).toEqual([]);
    await expect.poll(entityFacet).toBe(wideFacet);
    await expect(page.locator("#graph-filtered-tags-page")).toContainText(
      "of 78 tags",
      { timeout: 30000 },
    );
  };

  // A per-pattern evidence row.
  await page
    .locator('#graph-detail-content button[data-graph-action="thematic-evidence"]')
    .first()
    .click();
  await narrowed();

  // "Back to foregrounded patterns" widens everything, not just the inspector.
  await page.locator('[data-graph-action="thematic-back"]').click();
  await widened();

  // "Inspect selected pattern and evidence", then back again.
  await page.locator('[data-graph-action="thematic-evidence-selected"]').click();
  await narrowed();
  await page.locator('[data-graph-action="thematic-back"]').click();
  await widened();
  expect(errors).toEqual([]);
});

test("a relation pill lying on an entity circle takes the click", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1500, height: 1000 });
  await openGraph(page);
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-theme-granularity").fill("1");
  const overlay = page.locator('canvas[data-ready="true"]');

  await page.locator('#graph-entity-categories input[value="E02"]').check();
  await page.locator('#graph-object-categories input[value="E05"]').check();
  await page.locator("#graph-theme-apply").click();
  await expect(overlay).toHaveAttribute("data-selected-patterns", "31");
  await page.locator("#graph-browser-summary").click();
  await page.locator("#graph-fit").click();
  await page.waitForTimeout(800);

  // Labels stay in the slot the layout gave them, so some do land on a node.
  // Find one that genuinely overlaps a circle.
  const overlapping = await overlay.evaluate((canvas) => {
    const labels = (canvas.dataset.relationLabelPoints || "")
      .split(";").filter(Boolean)
      .map((entry) => entry.split(",").map(Number))
      .map(([pattern, x, y]) => ({ pattern, x, y }));
    const circles = (canvas.dataset.entityCircles || "")
      .split(";").filter(Boolean)
      .map((entry) => entry.split(",").map(Number))
      .map(([x, y, radius]) => ({ x, y, radius }));
    return labels.find((label) =>
      circles.some((circle) => {
        const dx = Math.max(Math.abs(circle.x - label.x) - 15, 0);
        const dy = Math.max(Math.abs(circle.y - label.y) - 7.5, 0);
        return Math.hypot(dx, dy) < circle.radius;
      }),
    ) || null;
  });
  expect(overlapping).not.toBeNull();

  // Clicking it must select that relation, not the node underneath it.
  const bounds = await page.locator("#graph-canvas").boundingBox();
  await page.mouse.click(bounds.x + overlapping.x, bounds.y + overlapping.y);
  await expect(overlay).toHaveAttribute(
    "data-thematic-focus-mode",
    /relation|exact-pattern/,
  );
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1");
  expect(errors).toEqual([]);
});
