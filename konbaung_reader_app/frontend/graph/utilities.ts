import { createEdgeArrowProgram } from "sigma/rendering";

export const GRAPH_HOVER_EFFECTS_ENABLED = false;

export const LargeArrowProgram = createEdgeArrowProgram({
  lengthToThicknessRatio: 5.2,
  widenessToThicknessRatio: 4,
});

export async function fetchJson<T>(
  url: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(url, {
    signal,
    headers: { Accept: "application/json" },
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(
      payload.error || `Graph request failed (${response.status})`,
    );
  }
  return payload as T;
}

export function nodeSize(frequency: number): number {
  return Math.min(11, 2.2 + Math.log2(frequency + 1) * 0.62);
}

export function bubbleNodeSize(label: string): number {
  const charactersPerLine = 10;
  const lines = Math.ceil(label.length / charactersPerLine);
  const textWidth = Math.min(charactersPerLine, label.length) * 4.3;
  const textHeight = lines * 8.5;
  return Math.min(
    36,
    Math.max(23, Math.ceil(Math.hypot(textWidth / 2, textHeight / 2) + 6)),
  );
}

export const atlasPalette = [
  "#226f8a",
  "#0f8978",
  "#6f4a96",
  "#9a6534",
  "#376c51",
  "#8a4967",
  "#4f6fa5",
  "#75712b",
];

export function atlasNodeColor(communityIndex: number): string {
  const mixed = Math.imul(communityIndex + 1, 2654435761) >>> 0;
  return atlasPalette[mixed % atlasPalette.length];
}

export function sameSet(left: Set<string>, right: Set<string>): boolean {
  if (left.size !== right.size) return false;
  for (const value of left) {
    if (!right.has(value)) return false;
  }
  return true;
}
