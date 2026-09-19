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

// Semantic search is no longer its own accordion: the same anchor-and-neighbors
// flow now runs inside the filtered tag bucket, which additionally supports
// averaging several anchors together.

test("semantic search selects embedding neighbors and loads a thematic slice", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto(`${baseUrl}/chronicles/vol1/47`);
  await page.locator("#graph-toggle").click();
  await expect(page.locator('canvas[data-ready="true"]')).toHaveCount(1, {
    timeout: 30000,
  });
  await openPanel(page, "#graph-filtered-tags-panel");

  // With no categories applied the bucket is every subject tag in scope, so a
  // free-text lookup works exactly as the old standalone search did.
  await page.locator("#graph-filtered-tags-search").fill("Hsinbyushin");
  await page.locator("#graph-filtered-tags-search-button").click();
  const results = page.locator("#graph-filtered-tags-results input");
  await expect(results.first()).toBeVisible({ timeout: 30000 });

  await results.first().check();
  await page.locator("#graph-filtered-tags-similarity").fill("90");
  await expect(page.locator("#graph-filtered-tags-similarity-value")).toHaveText(
    "90%",
  );
  // Widen back to the whole bucket; the checked anchor survives the reload.
  await page.locator("#graph-filtered-tags-search").fill("");
  await page.locator("#graph-filtered-tags-search-button").click();
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "1-100 of 23,890 tags",
    { timeout: 30000 },
  );
  await expect(page.locator("#graph-filtered-tags-selection-note")).toContainText(
    "1 of 100 tags checked",
  );
  await page.locator("#graph-filtered-tags-similar").click();
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "similar to the averaged vector of 1 checked tag",
    { timeout: 30000 },
  );

  // The anchor stays checked and leads the similarity ordering.
  await expect(results.first()).toBeChecked();
  await expect(
    page.locator("#graph-filtered-tags-results .graph-result-meta").nth(1),
  ).toContainText("% similar");

  // Adding a neighbor and filtering builds the same union tag slice.
  await results.nth(1).check();
  await expect(page.locator("#graph-filtered-tags-selection-note")).toContainText(
    "2 of 100 tags checked",
  );
  await page.locator("#graph-filtered-tags-apply").click();
  await expect(page.locator(".graph-view-indicator")).toHaveCount(0);
  await expect(page.locator("#graph-counts")).toContainText(
    "2 selected low-level tags",
  );
  expect(errors).toEqual([]);
});

test("averaging several checked tags narrows the ranking to their shared sense", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto(`${baseUrl}/chronicles/vol1/47`);
  await page.locator("#graph-toggle").click();
  await expect(page.locator('canvas[data-ready="true"]')).toHaveCount(1, {
    timeout: 30000,
  });
  await openPanel(page, "#graph-browser-panel");
  await checkCategories(page, "#graph-entity-categories", ["E01"]);
  await checkCategories(page, "#graph-relation-categories", ["R01"]);
  await page.locator("#graph-theme-apply").click();
  await openPanel(page, "#graph-filtered-tags-panel");
  // Group A tags are AND-intersected with the applied relation category, and
  // counted on whichever side of the claim group A matched.
  await expect(page.locator("#graph-filtered-tags-page")).toContainText(
    "1-46 of 46 tags",
    { timeout: 30000 },
  );

  // "King" alone and "King" + "Alaungpaya" produce different orderings: the
  // averaged vector pulls the ranking toward the named monarch.
  const rows = page.locator("#graph-filtered-tags-results input");
  const labels = page.locator("#graph-filtered-tags-results .graph-result-label");
  // This bucket is small, so drop the threshold far enough to keep a tail to
  // compare the two rankings against.
  await page.locator("#graph-filtered-tags-similarity").fill("80");
  await rows.nth(0).check();
  await page.locator("#graph-filtered-tags-similar").click();
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "averaged vector of 1 checked tag",
    { timeout: 30000 },
  );
  const singleAnchor = await labels.nth(2).textContent();

  await rows.nth(1).check();
  await page.locator("#graph-filtered-tags-similar").click();
  await expect(page.locator("#graph-filtered-tags-note")).toContainText(
    "averaged vector of 2 checked tags",
    { timeout: 30000 },
  );
  await expect(labels.nth(2)).not.toHaveText(singleAnchor);
  expect(errors).toEqual([]);
});
