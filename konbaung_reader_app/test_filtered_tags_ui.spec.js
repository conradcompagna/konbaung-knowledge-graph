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

test("filtered tags expose frequency-ranked raw tags and narrow the thematic graph", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto(`${baseUrl}/chronicles/vol1/47`);
  await page.locator("#graph-toggle").click();
  await expect(page.locator('canvas[data-ready="true"]')).toHaveCount(1, {
    timeout: 15000,
  });
  await openPanel(page, "#graph-browser-panel");

  // One selected role has hundreds of low-level tags, but only 100 are shown.
  await checkCategories(page, "#graph-entity-categories", ["E01"]);
  await page.locator("#graph-theme-apply").click();
  await openPanel(page, "#graph-filtered-tags-panel");
  // All three buckets are always listed; unfiltered roles fall back to scope.
  await expect(page.locator("#graph-filtered-tags-role option")).toHaveCount(3);
  await expect(page.locator("#graph-filtered-tags-role")).toHaveValue("subject");
  await expect(page.locator("#graph-filtered-tags-results input")).toHaveCount(100);
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "1-100 of 623 tags",
  );
  await expect(page.locator("#graph-filtered-tags-results .graph-result-label").first()).toHaveText("King");

  // Adding relation and object buckets exposes three role-specific tag lists.
  await openPanel(page, "#graph-browser-panel");
  await checkCategories(page, "#graph-relation-categories", ["R01"]);
  await checkCategories(page, "#graph-object-categories", ["E04"]);
  await page.locator("#graph-theme-apply").click();
  await openPanel(page, "#graph-filtered-tags-panel");
  await expect(page.locator("#graph-filtered-tags-role option")).toHaveCount(3);
  await expect(page.locator("#graph-filtered-tags-role")).toHaveValue("subject");

  await page.locator("#graph-filtered-tags-search").fill("Alaung");
  await page.locator("#graph-filtered-tags-search-button").click();
  const matches = page.locator("#graph-filtered-tags-results input");
  // Three under either-direction pairing: "King Pyusawhti to Alaungmintaya"
  // reaches group A from the object side of its claim.
  await expect(matches).toHaveCount(3);
  await expect(page.locator("#graph-filtered-tags-results")).toContainText(
    "Alaungmintaya",
  );
  await expect(page.locator("#graph-filtered-tags-results")).toContainText(
    "Alaungpaya",
  );
  await matches.nth(0).check();
  await matches.nth(1).check();
  await expect(page.locator("#graph-filtered-tags-selection-note")).toContainText(
    "2 of 100 tags checked",
  );

  await page.locator("#graph-filtered-tags-apply").click();
  await expect(page.locator("#graph-heading")).toContainText("2 filtered tags");
  await expect(page.locator("#graph-counts")).toContainText(
    "2 selected low-level tags",
  );
  await expect(page.locator("#graph-counts")).toContainText("7 triples");
  expect(await checkedCategories(page, "#graph-entity-categories")).toEqual(["E01"]);
  expect(await checkedCategories(page, "#graph-relation-categories")).toEqual(["R01"]);
  expect(await checkedCategories(page, "#graph-object-categories")).toEqual(["E04"]);
  const overlay = page.locator('canvas[data-ready="true"]');
  await expect(overlay).toHaveAttribute("data-scope-tag-count", "2");
  await expect(overlay).toHaveAttribute("data-active-patterns", "1");
  await expect(overlay).toHaveAttribute("data-selected-patterns", "1");

  // Clearing themes also removes the low-level scope and restores the base graph.
  await openPanel(page, "#graph-browser-panel");
  await page.locator("#graph-theme-clear").click();
  await expect(page.locator("#graph-heading")).toContainText("Full corpus");
  await expect(page.locator("#graph-counts")).toContainText(
    "27,129 active triples",
  );
  expect(await checkedCategories(page, "#graph-entity-categories")).toEqual([]);
  expect(await checkedCategories(page, "#graph-relation-categories")).toEqual([]);
  expect(await checkedCategories(page, "#graph-object-categories")).toEqual([]);
  expect(errors).toEqual([]);
});
