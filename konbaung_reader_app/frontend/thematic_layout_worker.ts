/// <reference lib="webworker" />

import {
  calculateThematicLayout,
  type ThematicLayoutInput,
  type ThematicLayoutResult,
} from "./thematic_layout";

interface LayoutRequest {
  kind: "layout";
  requestId: number;
  input: ThematicLayoutInput;
}

interface LayoutResponse {
  kind: "layout";
  requestId: number;
  result?: ThematicLayoutResult;
  error?: string;
}

const worker = self as DedicatedWorkerGlobalScope;

worker.addEventListener("message", (event: MessageEvent<LayoutRequest>) => {
  const request = event.data;
  if (request.kind !== "layout") return;
  try {
    const result = calculateThematicLayout(request.input);
    const response: LayoutResponse = {
      kind: "layout",
      requestId: request.requestId,
      result,
    };
    worker.postMessage(response, [result.x.buffer, result.y.buffer]);
  } catch (error) {
    const response: LayoutResponse = {
      kind: "layout",
      requestId: request.requestId,
      error: error instanceof Error ? error.message : String(error),
    };
    worker.postMessage(response);
  }
});
