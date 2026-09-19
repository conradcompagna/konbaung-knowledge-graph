const { test, expect } = require("playwright/test");

function readCorpusMetrics(page) {
  return page.locator("#graph-canvas").evaluate((container) => {
    const canvas = [...container.querySelectorAll("canvas")].find(
      (candidate) => candidate.dataset.layoutReady === "true",
    );
    if (!canvas) return null;
    const number = (key) => Number(canvas.dataset[key]);
    return {
      hash: canvas.dataset.layoutHash,
      root: canvas.dataset.dominantRoot,
      overlapCount: number("overlapCount"),
      totalNodes: number("totalNodes"),
      totalTreeEdges: number("totalTreeEdges"),
      rootCount: number("rootCount"),
      predicateLabels: number("predicateLabels"),
      allCoordinatesFinite: canvas.dataset.allCoordinatesFinite,
      fullGraph: canvas.dataset.fullGraph,
      level: number("detailLevel"),
      nodes: number("admittedNodes"),
      edges: number("admittedEdges"),
      pixelsPerUnit: number("pixelsPerUnit"),
      layoutMs: number("layoutDurationMs"),
      disclosureMs: number("disclosureDurationMs"),
      minimumComponentGap: number("minimumComponentGap"),
      hoverEffectsEnabled: canvas.dataset.hoverEffectsEnabled,
    };
  });
}

test("full corpus radial forest is deterministic, collision free, and cumulative", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("http://127.0.0.1:5077/chronicles/vol1/57");
  await page.locator("#graph-toggle").click();
  await page.locator("#graph-scope").selectOption("corpus");
  await expect
    .poll(async () => (await readCorpusMetrics(page))?.hash || "", {
      timeout: 120_000,
    })
    .not.toBe("");

  const initial = await readCorpusMetrics(page);
  expect(initial.root).toBe("King");
  expect(initial.overlapCount).toBe(0);
  expect(initial.totalNodes).toBe(23_890);
  expect(initial.totalTreeEdges).toBe(
    initial.totalNodes - initial.rootCount,
  );
  expect(initial.predicateLabels).toBe(0);
  expect(initial.allCoordinatesFinite).toBe("true");
  expect(initial.level).toBe(0);
  expect(initial.nodes).toBeGreaterThanOrEqual(10);
  expect(initial.nodes).toBeLessThanOrEqual(80);
  expect(initial.edges).toBeLessThan(initial.nodes);
  expect(initial.layoutMs).toBeLessThan(5_000);
  expect(initial.minimumComponentGap).toBeGreaterThanOrEqual(319.99);
  expect(initial.hoverEffectsEnabled).toBe("false");

  const fullStarted = Date.now();
  await page.locator("#graph-show-all").check();
  await expect
    .poll(async () => (await readCorpusMetrics(page))?.nodes || 0, {
      timeout: 120_000,
    })
    .toBe(initial.totalNodes);
  const full = await readCorpusMetrics(page);
  expect(full.edges).toBe(initial.totalTreeEdges);
  expect(full.fullGraph).toBe("true");
  expect(Date.now() - fullStarted).toBeLessThan(5_000);

  await page.locator("#graph-show-all").uncheck();
  await expect
    .poll(async () => (await readCorpusMetrics(page))?.nodes || 0, {
      timeout: 120_000,
    })
    .toBeLessThan(100);
  const restored = await readCorpusMetrics(page);
  expect(restored.fullGraph).toBe("false");

  const counts = [{ nodes: restored.nodes, edges: restored.edges }];
  for (let step = 0; step < 24; step += 1) {
    await page.locator("#graph-zoom-in").click();
    await page.waitForTimeout(430);
    const metrics = await readCorpusMetrics(page);
    counts.push({ nodes: metrics.nodes, edges: metrics.edges });
    expect(metrics.nodes).toBeGreaterThanOrEqual(counts[counts.length - 2].nodes);
    expect(metrics.edges).toBeGreaterThanOrEqual(counts[counts.length - 2].edges);
    expect(metrics.disclosureMs).toBeLessThan(150);
    if (metrics.level === 3) break;
  }
  const detailed = await readCorpusMetrics(page);
  expect(detailed.level).toBe(3);
  expect(detailed.pixelsPerUnit).toBeGreaterThanOrEqual(1);
  expect(detailed.nodes).toBeGreaterThan(initial.nodes);

  const canvasBox = await page.locator("#graph-canvas").boundingBox();
  await page.mouse.move(canvasBox.x + canvasBox.width / 2, canvasBox.y + canvasBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    canvasBox.x + canvasBox.width / 2 + 180,
    canvasBox.y + canvasBox.height / 2 + 80,
    { steps: 8 },
  );
  await page.mouse.up();
  await page.waitForTimeout(180);
  const panned = await readCorpusMetrics(page);
  expect(panned.nodes).toBeGreaterThanOrEqual(detailed.nodes);
  expect(panned.edges).toBeGreaterThanOrEqual(detailed.edges);

  await page.locator("#graph-zoom-out").click();
  await page.waitForTimeout(240);
  const zoomedOut = await readCorpusMetrics(page);
  expect(zoomedOut.nodes).toBeLessThanOrEqual(panned.nodes);
  expect(zoomedOut.edges).toBeLessThanOrEqual(panned.edges);

  const repeat = await page.context().newPage();
  await repeat.setViewportSize({ width: 1280, height: 720 });
  await repeat.goto("http://127.0.0.1:5077/chronicles/vol1/57");
  await repeat.locator("#graph-toggle").click();
  await repeat.locator("#graph-scope").selectOption("corpus");
  await expect
    .poll(async () => (await readCorpusMetrics(repeat))?.hash || "", {
      timeout: 120_000,
    })
    .toBe(initial.hash);
  const repeated = await readCorpusMetrics(repeat);
  expect(repeated.root).toBe(initial.root);
  expect(repeated.overlapCount).toBe(0);
  await repeat.close();

  await page.screenshot({
    path: "test-artifacts/corpus-radial-progressive.png",
  });
});
