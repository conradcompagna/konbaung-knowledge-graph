const { test, expect } = require("playwright/test");

// Exercises the constructive layout across widening page ranges. The point is
// twofold: prove that no forbidden collision survives at any scope, and record
// how long the layout actually takes now that nothing iterates.

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
    const number = (key) => Number(canvas.dataset[key]);
    return {
      free: canvas.dataset.hardCollisionFree,
      passes: number("hardCollisionPasses"),
      nodeOverlaps: number("nodeOverlaps"),
      edgeNodeOverlaps: number("edgeNodeOverlaps"),
      labelNodeOverlaps: number("labelNodeOverlaps"),
      labelEndpointOverlaps: number("labelEndpointOverlaps"),
      labelEdgeOverlaps: number("labelEdgeOverlaps"),
      labelOverlaps: number("labelOverlaps"),
      forbiddenCollisions: number("forbiddenCollisions"),
      offCenterEdgeLabels: number("offCenterEdgeLabels"),
      minimumEdgeLabelRatio: number("minimumEdgeLabelRatio"),
      minimumEdgeLabelClearance: number("minimumEdgeLabelClearance"),
      medianEdgeLabelClearance: number("medianEdgeLabelClearance"),
      renderedEdges: number("renderedEdges"),
      layoutRevision: number("layoutRevision"),
      solveScale: Number(canvas.dataset.layoutSolveScale),
      solveSteps: number("layoutSolveSteps"),
      verified: canvas.dataset.layoutVerified,
    };
  });
}

// The previous graph leaves hardCollisionFree=true on its canvas, so polling
// that alone returns the old graph's numbers. Wait for the layout revision to
// move on before trusting anything.
async function waitForLayout(page, afterRevision) {
  await expect
    .poll(
      async () => {
        const metrics = await readMetrics(page);
        if (!metrics || metrics.free !== "true") return -1;
        return metrics.layoutRevision;
      },
      { timeout: 120_000 },
    )
    .toBeGreaterThan(afterRevision);
  return readMetrics(page);
}

test("constructive layout stays collision free as the scope widens", async ({
  page,
}) => {
  test.setTimeout(300_000);
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("http://127.0.0.1:5077/chronicles/vol1/57");
  await page.locator("#graph-toggle").click();

  const report = [];
  let revision = -1;
  for (const scope of SCOPES) {
    const started = Date.now();
    if (scope.start) {
      await page.locator("#graph-pages-panel").evaluate((panel) => {
        panel.open = true;
      });
      await page.locator("#graph-range-start").selectOption(scope.start);
      await page.locator("#graph-range-end").selectOption(scope.end);
      await page.locator("#graph-range-load").click();
    }
    const metrics = await waitForLayout(page, revision);
    revision = metrics.layoutRevision;
    const elapsed = Date.now() - started;

    console.log(
      `${scope.label.padEnd(12)} edges=${String(metrics.renderedEdges).padStart(4)}` +
        ` passes=${metrics.passes}` +
        ` forbidden=${metrics.forbiddenCollisions}` +
        ` minBareLine=${metrics.minimumEdgeLabelClearance.toFixed(1)}px` +
        ` medBareLine=${metrics.medianEdgeLabelClearance.toFixed(1)}px` +
        ` solveScale=${metrics.solveScale.toFixed(2)}` +
        ` verified=${metrics.verified}` +
        ` solveSteps=${metrics.solveSteps}` +
        ` ${elapsed}ms`,
    );
    expect(metrics, `${scope.label}: metrics`).toMatchObject({
      nodeOverlaps: 0,
      edgeNodeOverlaps: 0,
      labelNodeOverlaps: 0,
      labelEndpointOverlaps: 0,
      labelEdgeOverlaps: 0,
      labelOverlaps: 0,
      forbiddenCollisions: 0,
      offCenterEdgeLabels: 0,
      passes: 0,
    });
    // drawCustomEdge draws a 12px arrowhead at the target end. Requiring
    // more than that on the tightest edge in the graph is what keeps the
    // direction of every relation readable rather than hidden under its tag.
    expect(
      metrics.minimumEdgeLabelClearance,
      `${scope.label}: bare line beside the tightest tag`,
    ).toBeGreaterThan(14);

    await page.screenshot({
      path: `test-artifacts/constructive-${scope.label.replace(/\s+/g, "-")}.png`,
      fullPage: false,
    });
  }
  console.log("\n" + report.join("\n") + "\n");
});
