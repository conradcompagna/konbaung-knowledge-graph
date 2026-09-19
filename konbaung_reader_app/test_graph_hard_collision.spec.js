const { test, expect } = require("playwright/test");

test.use({ channel: "chrome" });

async function waitForCollisionFreeGraph(page) {
  await expect
    .poll(
      async () =>
        page.locator("#graph-canvas canvas").evaluateAll((canvases) => {
          const canvas = canvases.find(
            (candidate) =>
              candidate.dataset.hardCollisionFree !== undefined,
          );
          return canvas?.dataset.hardCollisionFree || "";
        }),
      { timeout: 120_000 },
    )
    .toBe("true");

  return page.locator("#graph-canvas canvas").evaluateAll((canvases) => {
    const canvas = canvases.find(
      (candidate) =>
        candidate.dataset.hardCollisionFree !== undefined,
    );
    if (!canvas) return null;
    return {
      nodeOverlaps: Number(canvas.dataset.nodeOverlaps),
      edgeNodeOverlaps: Number(canvas.dataset.edgeNodeOverlaps),
      labelNodeOverlaps: Number(canvas.dataset.labelNodeOverlaps),
      labelEndpointOverlaps: Number(
        canvas.dataset.labelEndpointOverlaps,
      ),
      labelEdgeOverlaps: Number(canvas.dataset.labelEdgeOverlaps),
      labelOverlaps: Number(canvas.dataset.labelOverlaps),
      forbiddenCollisions: Number(canvas.dataset.forbiddenCollisions),
      renderedEdges: Number(canvas.dataset.renderedEdges),
      offCenterEdgeLabels: Number(canvas.dataset.offCenterEdgeLabels),
      minimumEdgeLabelRatio: Number(
        canvas.dataset.minimumEdgeLabelRatio,
      ),
      medianEdgeLabelRatio: Number(canvas.dataset.medianEdgeLabelRatio),
      passes: Number(canvas.dataset.hardCollisionPasses),
    };
  });
}

test("page and ten-page range reject every forbidden graph collision", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("http://127.0.0.1:5077/chronicles/vol1/57");
  await page.locator("#graph-toggle").click();

  const pageMetrics = await waitForCollisionFreeGraph(page);
  expect(pageMetrics).toMatchObject({
    nodeOverlaps: 0,
    edgeNodeOverlaps: 0,
    labelNodeOverlaps: 0,
    labelEndpointOverlaps: 0,
    labelEdgeOverlaps: 0,
    labelOverlaps: 0,
    forbiddenCollisions: 0,
    offCenterEdgeLabels: 0,
  });
  expect(pageMetrics.renderedEdges).toBeGreaterThan(0);

  await page.locator("#graph-zoom-in").click();
  await page.waitForTimeout(300);
  const zoomedRenderedEdges = await page
    .locator("#graph-canvas canvas")
    .evaluateAll((canvases) => {
      const canvas = canvases.find(
        (candidate) =>
          candidate.dataset.hardCollisionFree !== undefined,
      );
      return Number(canvas?.dataset.renderedEdges);
    });
  expect(zoomedRenderedEdges).toBe(pageMetrics.renderedEdges);

  await page.screenshot({
    path: "test-artifacts/page57-hard-collision-compact.png",
    fullPage: false,
  });

  await page.locator("#graph-pages-panel").evaluate((panel) => {
    panel.open = true;
  });
  await page.locator("#graph-range-start").selectOption("47");
  await page.locator("#graph-range-end").selectOption("56");
  await page.locator("#graph-range-load").click();

  const rangeMetrics = await waitForCollisionFreeGraph(page);
  expect(rangeMetrics).toMatchObject({
    nodeOverlaps: 0,
    edgeNodeOverlaps: 0,
    labelNodeOverlaps: 0,
    labelEndpointOverlaps: 0,
    labelEdgeOverlaps: 0,
    labelOverlaps: 0,
    forbiddenCollisions: 0,
    offCenterEdgeLabels: 0,
  });
});
