const { test } = require("playwright/test");
test("characterise residual overlap on a single page", async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("http://127.0.0.1:5077/chronicles/vol1/57");
  await page.locator("#graph-toggle").click();
  await page.waitForTimeout(4000);
  const out = await page.locator("#graph-canvas canvas").evaluateAll((cs) => {
    const c = cs.find((x) => x.dataset.hardCollisionFree !== undefined);
    return c ? {
      nodeOverlaps: c.dataset.nodeOverlaps,
      edgeNodeOverlaps: c.dataset.edgeNodeOverlaps,
      labelNodeOverlaps: c.dataset.labelNodeOverlaps,
      labelEndpointOverlaps: c.dataset.labelEndpointOverlaps,
      labelEdgeOverlaps: c.dataset.labelEdgeOverlaps,
      labelOverlaps: c.dataset.labelOverlaps,
      labelEdgeConflictPairs: c.dataset.labelEdgeConflictPairs,
    } : null;
  });
  console.log("\nBREAKDOWN " + JSON.stringify(out, null, 1) + "\n");
});
