/// <reference lib="webworker" />

import {
  calculateCorpusRadialLayout,
  type CorpusRadialLayoutInput,
  type CorpusRadialLayoutResult,
} from "./corpus_radial_layout";

interface LayoutRequest {
  kind: "layout";
  requestId: number;
  input: CorpusRadialLayoutInput;
}

interface LayoutResponse {
  kind: "layout";
  requestId: number;
  result?: CorpusRadialLayoutResult;
  error?: string;
}

const worker = self as DedicatedWorkerGlobalScope;

worker.addEventListener("message", (event: MessageEvent<LayoutRequest>) => {
  const request = event.data;
  if (request.kind !== "layout") return;
  try {
    const result = calculateCorpusRadialLayout(request.input);
    const response: LayoutResponse = {
      kind: "layout",
      requestId: request.requestId,
      result,
    };
    worker.postMessage(response, [
      result.x.buffer,
      result.y.buffer,
      result.parent.buffer,
      result.parentEdge.buffer,
      result.depth.buffer,
      result.degree.buffer,
      result.component.buffer,
      result.subtreeSize.buffer,
      result.ranked.buffer,
      result.roots.buffer,
    ]);
  } catch (error) {
    const response: LayoutResponse = {
      kind: "layout",
      requestId: request.requestId,
      error: error instanceof Error ? error.message : String(error),
    };
    worker.postMessage(response);
  }
});
