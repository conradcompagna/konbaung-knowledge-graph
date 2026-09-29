import type { GraphController } from "./controller";
import { MultiDirectedGraph } from "graphology";
import Sigma from "sigma";
import { EdgeLineProgram } from "sigma/rendering";
import { type CorpusRadialLayoutResult } from "../corpus_radial_layout";
import CorpusRadialWorker from "../corpus_radial_worker?worker";
import {
  type TopologyPayload,
  type CorpusRadialModel,
  type CorpusRadialWorkerResponse,
} from "./types";
import { el, clear } from "./dom";
import { LargeArrowProgram, atlasNodeColor } from "./utilities";

export const corpus_radial_model = {
  runCorpusRadialWorker(
    this: GraphController,
    payload: TopologyPayload,
  ): Promise<CorpusRadialLayoutResult> {
    const requestId = ++this.corpusRadialRequest;
    const frequencies = new Float64Array(payload.nodes.length);
    const sources = new Int32Array(payload.edges.length);
    const targets = new Int32Array(payload.edges.length);
    const ids = payload.nodes.map((node, index) => {
      frequencies[index] = Number(node[2] || 0);
      return node[0];
    });
    payload.edges.forEach((edge, index) => {
      sources[index] = edge[0];
      targets[index] = edge[1];
    });
    const worker = new CorpusRadialWorker({
      name: "corpus-radial-layout",
    });
    this.corpusRadialWorker = worker;
    return new Promise((resolve, reject) => {
      const finish = (): void => {
        worker.terminate();
        if (this.corpusRadialWorker === worker) {
          this.corpusRadialWorker = null;
        }
      };
      worker.addEventListener(
        "message",
        (event: MessageEvent<CorpusRadialWorkerResponse>) => {
          const response = event.data;
          if (response.kind !== "layout" || response.requestId !== requestId)
            return;
          finish();
          if (requestId !== this.corpusRadialRequest) {
            reject(new DOMException("Superseded corpus layout", "AbortError"));
            return;
          }
          if (response.error || !response.result) {
            reject(new Error(response.error || "Corpus radial layout failed."));
            return;
          }
          resolve(response.result);
        },
      );
      worker.addEventListener("error", (event) => {
        finish();
        reject(new Error(event.message || "Corpus radial worker failed."));
      });
      worker.postMessage(
        {
          kind: "layout",
          requestId,
          input: { ids, frequencies, sources, targets },
        },
        [frequencies.buffer, sources.buffer, targets.buffer],
      );
    });
  },

  graphClickIsSuppressed(this: GraphController): boolean {
    return performance.now() < this.suppressGraphClickUntil;
  },

  consumeGraphClickSuppression(this: GraphController): boolean {
    if (!this.graphClickIsSuppressed()) return false;
    this.suppressGraphClickUntil = 0;
    return true;
  },

  makeCorpusRadialModel(
    this: GraphController,
    payload: TopologyPayload,
    layout: CorpusRadialLayoutResult,
  ): CorpusRadialModel {
    const nodeCount = payload.nodes.length;
    const rankPosition = new Int32Array(nodeCount);
    layout.ranked.forEach((node, rank) => {
      rankPosition[node] = rank;
    });
    const markAncestors = (flag: Uint8Array, node: number): void => {
      let current = node;
      while (current >= 0 && !flag[current]) {
        flag[current] = 1;
        current = layout.parent[current];
      }
    };
    const buildLevel = (count: number, previous?: Uint8Array): Uint8Array => {
      const flag = previous ? previous.slice() : new Uint8Array(nodeCount);
      for (
        let rank = 0;
        rank < Math.min(count, layout.ranked.length);
        rank += 1
      ) {
        markAncestors(flag, layout.ranked[rank]);
      }
      return flag;
    };
    const level0 = buildLevel(Math.min(20, nodeCount));
    const level1 = buildLevel(Math.ceil(nodeCount * 0.01), level0);
    const level2 = buildLevel(Math.ceil(nodeCount * 0.1), level1);
    const order = (nodes: number[]): number[] =>
      nodes.sort(
        (left, right) =>
          layout.depth[left] - layout.depth[right] ||
          rankPosition[left] - rankPosition[right] ||
          payload.nodes[left][0].localeCompare(payload.nodes[right][0]),
      );
    const additions: [number[], number[], number[]] = [
      order(Array.from(layout.ranked).filter((node) => level0[node] === 1)),
      order(
        Array.from(layout.ranked).filter(
          (node) => level1[node] === 1 && level0[node] === 0,
        ),
      ),
      order(
        Array.from(layout.ranked).filter(
          (node) => level2[node] === 1 && level1[node] === 0,
        ),
      ),
    ];
    return {
      payload,
      layout,
      rankPosition,
      levelFlags: [level0, level1, level2],
      levelAdditions: additions,
      admitted: new Set<number>(),
      pixelsPerUnit: 0,
      projectedRadius: 4,
      lastDisclosureMs: 0,
      lastCenterX: Number.NaN,
      lastCenterY: Number.NaN,
    };
  },

  addCorpusRadialNode(
    this: GraphController,
    graph: MultiDirectedGraph,
    model: CorpusRadialModel,
    nodeIndex: number,
  ): void {
    const node = model.payload.nodes[nodeIndex];
    const nodeId = node[0];
    if (graph.hasNode(nodeId)) return;
    graph.addNode(nodeId, {
      label: node[1],
      frequency: node[2],
      x: model.layout.x[nodeIndex],
      y: model.layout.y[nodeIndex],
      size: model.projectedRadius,
      color: atlasNodeColor(Number(node[6] || 0)),
      forceLabel: false,
      highlighted: false,
      zIndex: 1,
      type: "circle",
      corpusRadial: true,
      corpusIndex: nodeIndex,
      corpusDegree: model.layout.degree[nodeIndex],
    });
  },

  addCorpusRadialEdge(
    this: GraphController,
    graph: MultiDirectedGraph,
    model: CorpusRadialModel,
    childIndex: number,
  ): void {
    const sourceEdgeIndex = model.layout.parentEdge[childIndex];
    if (sourceEdgeIndex < 0) return;
    const edge = model.payload.edges[sourceEdgeIndex];
    if (!edge) return;
    const sourceId = model.payload.nodes[edge[0]][0];
    const targetId = model.payload.nodes[edge[1]][0];
    if (!graph.hasNode(sourceId) || !graph.hasNode(targetId)) return;
    const edgeId = `corpus-tree-${childIndex}`;
    if (graph.hasEdge(edgeId)) return;
    graph.addDirectedEdgeWithKey(edgeId, sourceId, targetId, {
      label: null,
      relationLabel: edge[3],
      relationId: edge[2],
      occurrenceCount: edge[4],
      sourceId,
      targetId,
      size: Math.min(2.8, 0.65 + Math.log2(edge[4] + 1) * 0.32),
      color: "#9a7caf",
      type: "arrow",
      forceLabel: false,
      zIndex: 0,
      corpusRadialTree: true,
      corpusChildIndex: childIndex,
    });
  },

  async installCorpusRadialPayload(
    this: GraphController,
    payload: TopologyPayload,
  ): Promise<void> {
    this.destroyRenderer();
    this.payload = payload;
    this.selectedNode = null;
    this.selectedEdge = null;
    this.rawPrimaryEntity = null;
    this.rawPairEntity = null;
    this.rawContextRelation = null;
    this.rawSelectedEdges.clear();
    this.navigationHistory = [];
    this.hoveredNode = null;
    this.hoveredEdge = null;
    this.lineHoveredEdge = null;
    this.labelHoveredEdge = null;
    this.selectedNeighbors.clear();
    this.evidenceItems = [];
    el.container.style.visibility = "hidden";
    this.setLoading(
      true,
      `Laying out ${payload.nodes.length.toLocaleString()} corpus entities\u2026`,
    );

    const layout = await this.runCorpusRadialWorker(payload);
    for (let node = 0; node < payload.nodes.length; node += 1) {
      if (
        !Number.isFinite(layout.x[node]) ||
        !Number.isFinite(layout.y[node])
      ) {
        throw new Error(
          `Corpus radial layout produced a non-finite coordinate at ${node}.`,
        );
      }
    }
    const dominantRoot = payload.nodes[layout.dominantRoot];
    if (!dominantRoot || dominantRoot[1] !== "King") {
      throw new Error(
        `Expected King as the dominant corpus root; got ${dominantRoot?.[1] || "none"}.`,
      );
    }
    const model = this.makeCorpusRadialModel(payload, layout);
    this.corpusRadialModel = model;
    el.showAllControl.hidden = false;
    el.showAll.checked = false;
    const graph = new MultiDirectedGraph();
    graph.setAttribute("corpusRadial", true);
    graph.setAttribute("uniformScaling", true);
    graph.setAttribute("hardCollisionFree", layout.overlapCount === 0);
    graph.setAttribute("layoutHash", layout.layoutHash);
    this.graph = graph;
    this.graphBBox = {
      x: [layout.bounds[0], layout.bounds[2]],
      y: [layout.bounds[1], layout.bounds[3]],
    };
    this.addCorpusRadialNode(graph, model, layout.dominantRoot);
    model.admitted.add(layout.dominantRoot);

    this.renderer = new Sigma(graph, el.container, {
      edgeProgramClasses: {
        arrow: LargeArrowProgram,
        line: EdgeLineProgram,
      },
      defaultEdgeType: "arrow",
      enableEdgeEvents: true,
      hideEdgesOnMove: false,
      hideLabelsOnMove: true,
      renderLabels: false,
      renderEdgeLabels: false,
      defaultDrawNodeHover: () => undefined,
      minEdgeThickness: 0.45,
      stagePadding: 28,
      zIndex: true,
      minCameraRatio: 0.0000001,
      maxCameraRatio: 10000000,
      zoomToSizeRatioFunction: () => 1,
      cameraPanBoundaries: null,
      nodeReducer: (node, data) => this.reduceNode(node, data),
      edgeReducer: (edge, data) => this.reduceEdge(edge, data),
    });
    this.renderer.setCustomBBox(this.graphBBox);
    this.corpusRadialOverlayCanvas = this.renderer.createCanvas(
      "corpusRadialLabels",
      {
        beforeLayer: "mouse",
        style: { pointerEvents: "none" },
      },
    );
    this.renderer.on("afterRender", () => this.drawCorpusRadialOverlay());
    this.bindRendererEvents();
    this.bindCorpusRadialDisclosure();
    this.updateHeader();
    this.renderFocusDetail();
    el.scope.value = "corpus";

    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => {
        if (!this.renderer || this.corpusRadialModel !== model) {
          resolve();
          return;
        }
        this.renderer.resize(true);
        this.fitGraph(false);
        requestAnimationFrame(() => {
          if (this.corpusRadialModel === model) {
            this.updateCorpusRadialDisclosure(true);
            el.container.style.visibility = "";
            this.setLoading(false);
          }
          resolve();
        });
      });
    });
  },
};
