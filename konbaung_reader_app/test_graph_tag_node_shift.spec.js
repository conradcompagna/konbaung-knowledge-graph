const { test, expect } = require("playwright/test");

const SCOPES = [
  { label: "single page", start: null, end: null },
  { label: "3 pages", start: "47", end: "49" },
  { label: "10 pages", start: "47", end: "56" },
  { label: "30 pages", start: "47", end: "76" },
];

function readMetrics(page) {
  return page.locator("#graph-canvas canvas").evaluateAll((canvases) => {
    const canvas = canvases.find(
      (candidate) => candidate.dataset.hardCollisionFree !== undefined,
    );
    if (!canvas) return null;
    return {
      revision: Number(canvas.dataset.layoutRevision),
      nodeOverlaps: Number(canvas.dataset.nodeOverlaps),
      labelNodeOverlaps: Number(canvas.dataset.labelNodeOverlaps),
      labelEndpointOverlaps: Number(canvas.dataset.labelEndpointOverlaps),
      renderedEdges: Number(canvas.dataset.renderedEdges),
    };
  });
}

test("post-layout shifts keep relation tags off nodes", async ({ page }) => {
  test.setTimeout(300_000);
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("http://127.0.0.1:5077/chronicles/vol1/57");
  await page.locator("#graph-toggle").click();

  let revision = -1;
  for (const scope of SCOPES) {
    if (scope.start) {
      await page.locator("#graph-pages-panel").evaluate((panel) => {
        panel.open = true;
      });
      await page.locator("#graph-range-start").selectOption(scope.start);
      await page.locator("#graph-range-end").selectOption(scope.end);
      await page.locator("#graph-range-load").click();
    }
    await expect
      .poll(async () => (await readMetrics(page))?.revision ?? -1, {
        timeout: 120_000,
      })
      .toBeGreaterThan(revision);
    const metrics = await readMetrics(page);
    revision = metrics.revision;
    console.log(
      `${scope.label}: edges=${metrics.renderedEdges}` +
        ` node=${metrics.nodeOverlaps}` +
        ` tag-node=${metrics.labelNodeOverlaps}` +
        ` tag-endpoint=${metrics.labelEndpointOverlaps}`,
    );
    expect(metrics, scope.label).toMatchObject({
      nodeOverlaps: 0,
      labelNodeOverlaps: 0,
      labelEndpointOverlaps: 0,
    });
  }
});
