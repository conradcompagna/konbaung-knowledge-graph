const { test } = require("node:test");
const assert = require("node:assert/strict");
const { MultiDirectedGraph } = require("graphology");
const { graphFixture } = require("./graph_fixture.cjs");

test("controllers own independent navigation state and retain class method semantics", () => {
  const { controller, GraphController } = graphFixture();
  const second = new GraphController();
  controller.selectedNode = "entity-a";
  controller.thematicSelectedPatterns.add(7);
  assert.equal(controller.nav.selectedNode, "entity-a");
  assert.equal(second.selectedNode, null);
  assert.equal(second.thematicSelectedPatterns.size, 0);
  assert.equal(Object.hasOwn(controller, "setPage"), false);
  assert.equal(Object.getOwnPropertyDescriptor(GraphController.prototype, "setPage").enumerable, false);
});

test("navigation snapshots isolate nested selections from later edits", () => {
  const { controller } = graphFixture();
  controller.nav.categories.subjects.push("E01");
  controller.nav.selectedPatterns.add(3);
  controller.selectedNeighbors.add("entity-b");
  const snapshot = controller.navigationSnapshot();
  controller.nav.categories.subjects.push("E02");
  controller.nav.selectedPatterns.clear();
  controller.selectedNeighbors.clear();
  assert.deepEqual(Array.from(snapshot.nav.categories.subjects), ["E01"]);
  assert.deepEqual(Array.from(snapshot.nav.selectedPatterns), [3]);
  assert.deepEqual(Array.from(snapshot.selectedNeighbors), ["entity-b"]);
});

test("page-range controls preserve the boundary the user just changed", () => {
  const { controller, el } = graphFixture();
  el.rangeStart.value = "8"; el.rangeEnd.value = "4";
  controller.normalizeRange("start");
  assert.equal(el.rangeEnd.value, "8");
  el.rangeEnd.value = "3";
  controller.normalizeRange("end");
  assert.equal(el.rangeStart.value, "3");
});

test("thematic filtering composes direction, frequency, and selected-group policy", () => {
  const { controller, el } = graphFixture();
  controller.graph = new MultiDirectedGraph();
  controller.graph.addNode("primary", { categoryIndex: 0 });
  controller.thematicPrimaryEntity = "primary";
  controller.thematicPayload = { patterns: [[0, 0, 1, 0, 5], [1, 0, 0, 0, 6], [0, 0, 2, 0, 1]] };
  el.themeGranularity.value = "2"; el.themeDirection.value = "ab";
  assert.deepEqual(Array.from(controller.filteredThematicPatterns([0, 1, 2])), [0]);
  el.themeDirection.value = "ba";
  assert.deepEqual(Array.from(controller.filteredThematicPatterns([0, 1, 2])), [1]);
  controller.thematicGroupSelection = true;
  assert.deepEqual(Array.from(controller.filteredThematicPatterns([0, 1, 2])), [0, 1]);
});

test("tag filters preserve separate subject, predicate, and object query fields", () => {
  const { controller, el } = graphFixture();
  controller.appliedTagFilters.set("subject", ["s1", "s2"]);
  controller.appliedTagFilters.set("relation", ["r1"]);
  controller.appliedTagFilters.set("object", ["o1"]);
  const query = new URLSearchParams();
  controller.appendAppliedTagFilters(query);
  assert.equal(controller.appliedTagFilterCount(), 4);
  assert.deepEqual(query.getAll("subjectTag"), ["s1", "s2"]);
  assert.equal(query.get("relationTag"), "r1");
  assert.equal(query.get("objectTag"), "o1");
  controller.clearAppliedTagFilters();
  assert.equal(controller.appliedTagFilterCount(), 0);
  assert.equal(el.filteredTagsRelease.hidden, true);
});

test("a pan-release suppression is consumed once", () => {
  const { controller } = graphFixture();
  controller.suppressGraphClickUntil = 200;
  assert.equal(controller.consumeGraphClickSuppression(), true);
  assert.equal(controller.consumeGraphClickSuppression(), false);
});

test("layout workers retain request identity and transfer graph arrays", async () => {
  const { controller, workers } = graphFixture();
  const pending = controller.runCorpusRadialWorker({ nodes: [["a", "A", 2], ["b", "B", 1]], edges: [[0, 1]] });
  const worker = workers[0];
  assert.deepEqual(Array.from(worker.message.input.ids), ["a", "b"]);
  assert.deepEqual(Array.from(worker.message.input.frequencies), [2, 1]);
  assert.equal(worker.transfer.length, 3);
  worker.reply({ kind: "layout", requestId: -1, result: {} });
  assert.equal(worker.terminated, undefined);
  const result = { positions: [[0, 0], [1, 1]] };
  worker.reply({ kind: "layout", requestId: worker.message.requestId, result });
  assert.equal(await pending, result);
  assert.equal(worker.terminated, true);
  assert.equal(controller.corpusRadialWorker, null);
});

test("a superseded layout cannot replace the active request", async () => {
  const { controller, workers } = graphFixture();
  const pending = controller.runCorpusRadialWorker({ nodes: [], edges: [] });
  const rejection = assert.rejects(pending, { name: "AbortError" });
  controller.corpusRadialRequest += 1;
  workers[0].reply({ kind: "layout", requestId: workers[0].message.requestId, result: {} });
  await rejection;
  assert.equal(workers[0].terminated, true);
});
