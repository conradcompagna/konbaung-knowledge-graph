import type { GraphController } from "./controller";
import { nodeSize, bubbleNodeSize, sameSet } from "./utilities";

export const atlas_view = {
  drawAtlasRegions(this: GraphController): void {
    if (
      !this.renderer ||
      !this.atlasOverlayCanvas ||
      this.payload?.layout.kind !== "atlas"
    )
      return;
    const canvas = this.atlasOverlayCanvas;
    const { width, height } = this.renderer.getDimensions();
    const pixelRatio = window.devicePixelRatio || 1;
    const targetWidth = Math.round(width * pixelRatio);
    const targetHeight = Math.round(height * pixelRatio);
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    const mouse = this.renderer.getMouseCaptor();
    if (
      this.renderer.getCamera().isAnimated() ||
      mouse.isMoving ||
      mouse.draggedEvents > 0 ||
      mouse.currentWheelDirection !== 0
    )
      return;
    if (this.atlasLevel >= 2) {
      if (this.atlasLevel >= 3) {
        context.font = "600 10px Inter, Segoe UI, sans-serif";
        context.textAlign = "center";
        context.textBaseline = "middle";
        for (const [edge, placement] of this.atlasPredicateLabelPositions) {
          if (edge === this.selectedEdge || edge === this.hoveredEdge) continue;
          context.beginPath();
          context.roundRect(
            placement.x - placement.width / 2,
            placement.y - 7,
            placement.width,
            14,
            3,
          );
          context.fillStyle = "rgba(250, 247, 252, 0.98)";
          context.fill();
          context.strokeStyle = "#d4c1df";
          context.lineWidth = 0.7;
          context.stroke();
          context.fillStyle = "#5c2b5d";
          context.fillText(placement.label, placement.x, placement.y + 0.4);
        }
      }
      this.drawAtlasOffscreenMarkers(context, width, height);
      return;
    }

    const origin = this.renderer.graphToViewport({ x: 0, y: 0 });
    const unitX = this.renderer.graphToViewport({ x: 1, y: 0 });
    const unitY = this.renderer.graphToViewport({ x: 0, y: 1 });
    const xAxis = {
      x: unitX.x - origin.x,
      y: unitX.y - origin.y,
    };
    const yAxis = {
      x: unitY.x - origin.x,
      y: unitY.y - origin.y,
    };
    const toViewport = (x: number, y: number) => ({
      x: origin.x + x * xAxis.x + y * yAxis.x,
      y: origin.y + x * xAxis.y + y * yAxis.y,
    });
    const labelBoxes: Array<[number, number, number, number]> = [];
    let labelsDrawn = 0;
    const drawRegion = (
      label: string,
      nodeCount: number,
      minimumX: number,
      minimumY: number,
      maximumX: number,
      maximumY: number,
      community: boolean,
    ) => {
      const topLeft = toViewport(minimumX, minimumY);
      const bottomRight = toViewport(maximumX, maximumY);
      const left = Math.min(topLeft.x, bottomRight.x);
      const right = Math.max(topLeft.x, bottomRight.x);
      const top = Math.min(topLeft.y, bottomRight.y);
      const bottom = Math.max(topLeft.y, bottomRight.y);
      if (right < 0 || left > width || bottom < 0 || top > height) return;
      const boxWidth = right - left;
      const boxHeight = bottom - top;
      if (boxWidth < 2 || boxHeight < 2) return;
      context.strokeStyle = community
        ? "rgba(111,74,150,0.12)"
        : "rgba(65,82,96,0.18)";
      context.lineWidth = community ? 0.7 : 1;
      context.setLineDash(community ? [4, 4] : []);
      context.strokeRect(left, top, boxWidth, boxHeight);
      if (labelsDrawn >= 36 || nodeCount < 2 || boxWidth < 88 || boxHeight < 28)
        return;
      label = label.replaceAll("_", " ");
      context.font = "600 10px Inter, Segoe UI, sans-serif";
      const labelWidth = Math.min(240, context.measureText(label).width + 12);
      const labelBox: [number, number, number, number] = [
        left + 5,
        top + 5,
        left + 5 + labelWidth,
        top + 23,
      ];
      if (
        labelBoxes.some(
          (box) =>
            labelBox[0] < box[2] &&
            labelBox[2] > box[0] &&
            labelBox[1] < box[3] &&
            labelBox[3] > box[1],
        )
      )
        return;
      labelBoxes.push(labelBox);
      context.setLineDash([]);
      context.fillStyle = "rgba(248,249,250,0.88)";
      context.fillRect(
        labelBox[0],
        labelBox[1],
        labelBox[2] - labelBox[0],
        labelBox[3] - labelBox[1],
      );
      context.fillStyle = "#52616c";
      context.textBaseline = "middle";
      context.fillText(
        label,
        labelBox[0] + 6,
        (labelBox[1] + labelBox[3]) / 2,
        labelWidth - 12,
      );
      labelsDrawn += 1;
    };
    for (const component of this.payload.components || []) {
      drawRegion(
        component[1],
        component[2],
        component[4],
        component[5],
        component[6],
        component[7],
        false,
      );
    }
    if (this.atlasLevel === 1) {
      for (const community of this.payload.communities || []) {
        if (community[2] < 3) continue;
        drawRegion(
          community[1],
          community[2],
          community[3],
          community[4],
          community[5],
          community[6],
          true,
        );
      }
    }
    context.setLineDash([]);
    this.drawAtlasOffscreenMarkers(context, width, height);
  },

  drawAtlasOffscreenMarkers(
    this: GraphController,
    context: CanvasRenderingContext2D,
    width: number,
    height: number,
  ): void {
    if (!this.renderer || !this.graph) return;
    const activeNode = this.selectedNode || this.hoveredNode;
    if (!activeNode || !this.graph.hasNode(activeNode)) return;
    const activeAttributes = this.graph.getNodeAttributes(activeNode);
    const activePoint = this.renderer.graphToViewport({
      x: Number(activeAttributes.x),
      y: Number(activeAttributes.y),
    });
    if (
      activePoint.x < 0 ||
      activePoint.x > width ||
      activePoint.y < 0 ||
      activePoint.y > height
    )
      return;

    const inset = 11;
    const buckets = new Map<
      string,
      {
        x: number;
        y: number;
        angle: number;
        outgoing: boolean;
        count: number;
      }
    >();
    for (const edge of this.graph.edges(activeNode)) {
      if (!Boolean(this.graph.getEdgeAttribute(edge, "atlasRawEdge"))) {
        continue;
      }
      const [source, target] = this.graph.extremities(edge);
      if (source === target) continue;
      const other = source === activeNode ? target : source;
      const attributes = this.graph.getNodeAttributes(other);
      const otherPoint = this.renderer.graphToViewport({
        x: Number(attributes.x),
        y: Number(attributes.y),
      });
      if (
        otherPoint.x >= inset &&
        otherPoint.x <= width - inset &&
        otherPoint.y >= inset &&
        otherPoint.y <= height - inset
      )
        continue;
      const deltaX = otherPoint.x - activePoint.x;
      const deltaY = otherPoint.y - activePoint.y;
      const candidates: number[] = [];
      if (deltaX > 0) {
        candidates.push((width - inset - activePoint.x) / deltaX);
      } else if (deltaX < 0) {
        candidates.push((inset - activePoint.x) / deltaX);
      }
      if (deltaY > 0) {
        candidates.push((height - inset - activePoint.y) / deltaY);
      } else if (deltaY < 0) {
        candidates.push((inset - activePoint.y) / deltaY);
      }
      const scale = Math.min(
        ...candidates.filter((value) => value >= 0 && value <= 1),
      );
      if (!Number.isFinite(scale)) continue;
      const x = activePoint.x + deltaX * scale;
      const y = activePoint.y + deltaY * scale;
      const outgoing = source === activeNode;
      const side =
        x <= inset + 0.5
          ? "left"
          : x >= width - inset - 0.5
            ? "right"
            : y <= inset + 0.5
              ? "top"
              : "bottom";
      const along = side === "left" || side === "right" ? y : x;
      const key = `${side}:${Math.round(along / 28)}:${outgoing ? 1 : 0}`;
      const existing = buckets.get(key);
      if (existing) {
        existing.count += 1;
      } else {
        buckets.set(key, {
          x,
          y,
          angle: Math.atan2(deltaY, deltaX),
          outgoing,
          count: 1,
        });
      }
    }

    context.setLineDash([]);
    for (const marker of buckets.values()) {
      context.save();
      context.translate(marker.x, marker.y);
      context.rotate(marker.angle);
      context.beginPath();
      context.moveTo(7, 0);
      context.lineTo(-4, -4.5);
      context.lineTo(-4, 4.5);
      context.closePath();
      context.fillStyle = marker.outgoing ? "#75419a" : "#15968a";
      if (marker.outgoing) context.fill();
      else {
        context.lineWidth = 2;
        context.strokeStyle = "#15968a";
        context.stroke();
      }
      context.restore();
      if (marker.count > 1) {
        context.font = "600 9px Inter, Segoe UI, sans-serif";
        context.textBaseline = "middle";
        context.fillStyle = marker.outgoing ? "#66348c" : "#087f75";
        context.fillText(
          marker.count.toLocaleString(),
          Math.min(width - 34, Math.max(3, marker.x + 7)),
          Math.min(height - 8, Math.max(8, marker.y)),
        );
      }
    }
  },

  bindAtlasDetail(this: GraphController): void {
    if (!this.renderer) return;
    this.renderer.getCamera().on("updated", () => {
      if (this.atlasSettleTimer !== null) {
        window.clearTimeout(this.atlasSettleTimer);
      }
      this.atlasSettleTimer = window.setTimeout(() => {
        this.atlasSettleTimer = null;
        this.updateAtlasViewportDetail();
      }, 110);
      if (this.atlasDetailFrame !== null) return;
      this.atlasDetailFrame = window.requestAnimationFrame(() => {
        this.atlasDetailFrame = null;
        this.updateAtlasDetail();
      });
    });
    this.updateAtlasDetail();
    this.updateAtlasViewportDetail();
  },

  updateAtlasDetail(this: GraphController): void {
    if (!this.renderer || !this.graph || this.payload?.layout.kind !== "atlas")
      return;
    const spacing = Number(this.payload.layout.medianSpacing || 1);
    const origin = this.renderer.graphToViewport({ x: 0, y: 0 });
    const spaced = this.renderer.graphToViewport({ x: spacing, y: 0 });
    const projectedSpacing = Math.hypot(
      spaced.x - origin.x,
      spaced.y - origin.y,
    );
    this.atlasProjectedSpacing = projectedSpacing;
    const stops = this.payload.layout.zoomStops || [4, 24, 56];
    const nextLevel: 0 | 1 | 2 | 3 =
      projectedSpacing < stops[0]
        ? 0
        : projectedSpacing < stops[1]
          ? 1
          : projectedSpacing < stops[2]
            ? 2
            : 3;
    if (this.atlasOverlayCanvas) {
      this.atlasOverlayCanvas.dataset.atlasLevel = String(nextLevel);
    }
    if (nextLevel === this.atlasLevel) return;
    this.atlasLevel = nextLevel;
    this.renderer.refresh();
  },

  updateAtlasViewportDetail(this: GraphController): void {
    if (!this.renderer || !this.graph || this.payload?.layout.kind !== "atlas")
      return;
    const nextVisible = new Set<string>();
    const visiblePositions = new Map<string, { x: number; y: number }>();
    if (this.atlasLevel >= 1) {
      const { width, height } = this.renderer.getDimensions();
      const origin = this.renderer.graphToViewport({ x: 0, y: 0 });
      const unitX = this.renderer.graphToViewport({ x: 1, y: 0 });
      const unitY = this.renderer.graphToViewport({ x: 0, y: 1 });
      const xAxis = {
        x: unitX.x - origin.x,
        y: unitX.y - origin.y,
      };
      const yAxis = {
        x: unitY.x - origin.x,
        y: unitY.y - origin.y,
      };
      const margin = this.atlasLevel === 1 ? 24 : 80;
      for (const node of this.graph.nodes()) {
        if (Boolean(this.graph.getNodeAttribute(node, "atlasAnchor"))) {
          continue;
        }
        const attributes = this.graph.getNodeAttributes(node);
        const x = Number(attributes.x);
        const y = Number(attributes.y);
        const viewportX = origin.x + x * xAxis.x + y * yAxis.x;
        const viewportY = origin.y + x * xAxis.y + y * yAxis.y;
        if (
          viewportX >= -margin &&
          viewportX <= width + margin &&
          viewportY >= -margin &&
          viewportY <= height + margin
        ) {
          nextVisible.add(node);
          visiblePositions.set(node, {
            x: viewportX,
            y: viewportY,
          });
        }
      }
    }

    const nextContinuity = new Set<string>();
    const localEdges = new Set<string>();
    for (const node of nextVisible) {
      let hasLocalConnection = false;
      let strongestEdge: string | null = null;
      let strongestWeight = Number.NEGATIVE_INFINITY;
      for (const edge of this.graph.edges(node)) {
        if (!Boolean(this.graph.getEdgeAttribute(edge, "atlasRawEdge"))) {
          continue;
        }
        const [source, target] = this.graph.extremities(edge);
        const other = source === node ? target : source;
        if (nextVisible.has(other)) {
          hasLocalConnection = true;
          localEdges.add(edge);
          continue;
        }
        const weight = Number(
          this.graph.getEdgeAttribute(edge, "occurrenceCount") || 1,
        );
        if (weight > strongestWeight) {
          strongestWeight = weight;
          strongestEdge = edge;
        }
      }
      if (this.atlasLevel >= 2 && !hasLocalConnection && strongestEdge) {
        nextContinuity.add(strongestEdge);
      }
    }
    const nextLabels = new Set<string>();
    const labelBoxes: Array<[number, number, number, number]> = [];
    const visibleNodeCircles = Array.from(nextVisible).map((node) => {
      const point = visiblePositions.get(node) as { x: number; y: number };
      const label = String(this.graph?.getNodeAttribute(node, "label") || "");
      const size =
        this.atlasLevel === 1
          ? Math.min(
              6,
              1.6 +
                Math.log2(
                  Number(this.graph?.getNodeAttribute(node, "frequency") || 1) +
                    1,
                ) *
                  0.42,
            )
          : Math.min(
              bubbleNodeSize(label),
              this.atlasLevel === 2
                ? Math.max(4, this.atlasProjectedSpacing * 0.15)
                : Math.max(7, this.atlasProjectedSpacing * 0.22),
            );
      return { node, x: point.x, y: point.y, size };
    });
    const labelCandidates = Array.from(nextVisible).sort((left, right) => {
      const priorityDifference =
        Number(this.graph?.getNodeAttribute(right, "atlasPriority") || 0) -
        Number(this.graph?.getNodeAttribute(left, "atlasPriority") || 0);
      return priorityDifference || left.localeCompare(right);
    });
    for (const node of labelCandidates) {
      if (nextLabels.size >= 80) break;
      const point = visiblePositions.get(node);
      if (!point) continue;
      const label = String(this.graph.getNodeAttribute(node, "label") || "");
      const nodeSize =
        this.atlasLevel === 1
          ? Math.min(
              6,
              1.6 +
                Math.log2(
                  Number(this.graph.getNodeAttribute(node, "frequency") || 1) +
                    1,
                ) *
                  0.42,
            )
          : Math.min(
              bubbleNodeSize(label),
              this.atlasLevel === 2
                ? Math.max(4, this.atlasProjectedSpacing * 0.15)
                : Math.max(7, this.atlasProjectedSpacing * 0.22),
            );
      const labelWidth = Math.min(220, label.length * 6.4 + 8);
      const box: [number, number, number, number] = [
        point.x + nodeSize + 3,
        point.y - 8,
        point.x + nodeSize + 3 + labelWidth,
        point.y + 8,
      ];
      if (
        box[2] < 0 ||
        box[0] > this.renderer.getDimensions().width ||
        box[3] < 0 ||
        box[1] > this.renderer.getDimensions().height ||
        labelBoxes.some(
          (other) =>
            box[0] < other[2] &&
            box[2] > other[0] &&
            box[1] < other[3] &&
            box[3] > other[1],
        ) ||
        visibleNodeCircles.some(
          (circle) =>
            circle.node !== node &&
            box[0] < circle.x + circle.size &&
            box[2] > circle.x - circle.size &&
            box[1] < circle.y + circle.size &&
            box[3] > circle.y - circle.size,
        )
      )
        continue;
      labelBoxes.push(box);
      nextLabels.add(node);
    }
    const orderedLocalEdges = Array.from(localEdges).sort(
      (left, right) =>
        Number(this.graph?.getEdgeAttribute(right, "occurrenceCount") || 1) -
          Number(this.graph?.getEdgeAttribute(left, "occurrenceCount") || 1) ||
        left.localeCompare(right),
    );
    const nextRegionalEdges = new Set(orderedLocalEdges.slice(0, 32));
    const nextPredicateEdges = new Set<string>();
    const nextPredicateLabelPositions = new Map<
      string,
      { x: number; y: number; width: number; label: string }
    >();
    if (this.atlasLevel >= 3) {
      const predicateBoxes: Array<[number, number, number, number]> = [];
      for (const edge of orderedLocalEdges) {
        if (
          nextPredicateEdges.size >= 12 ||
          !this.atlasPredicateEdges.has(edge)
        ) {
          continue;
        }
        const [source, target] = this.graph.extremities(edge);
        const sourcePoint = visiblePositions.get(source);
        const targetPoint = visiblePositions.get(target);
        if (!sourcePoint || !targetPoint) continue;
        const label = String(
          this.graph.getEdgeAttribute(edge, "relationLabel") || "",
        );
        const width = Math.min(200, label.length * 6.2 + 12);
        let accepted:
          | { x: number; y: number; box: [number, number, number, number] }
          | undefined;
        for (const position of [0.68, 0.32, 0.5]) {
          const centerX =
            sourcePoint.x + (targetPoint.x - sourcePoint.x) * position;
          const centerY =
            sourcePoint.y + (targetPoint.y - sourcePoint.y) * position;
          const box: [number, number, number, number] = [
            centerX - width / 2,
            centerY - 7,
            centerX + width / 2,
            centerY + 7,
          ];
          if (
            box[0] < 3 ||
            box[2] > this.renderer.getDimensions().width - 3 ||
            box[1] < 3 ||
            box[3] > this.renderer.getDimensions().height - 3 ||
            labelBoxes.some(
              (other) =>
                box[0] < other[2] &&
                box[2] > other[0] &&
                box[1] < other[3] &&
                box[3] > other[1],
            ) ||
            predicateBoxes.some(
              (other) =>
                box[0] < other[2] &&
                box[2] > other[0] &&
                box[1] < other[3] &&
                box[3] > other[1],
            ) ||
            visibleNodeCircles.some(
              (circle) =>
                box[0] < circle.x + circle.size &&
                box[2] > circle.x - circle.size &&
                box[1] < circle.y + circle.size &&
                box[3] > circle.y - circle.size,
            )
          ) {
            continue;
          }
          accepted = { x: centerX, y: centerY, box };
          break;
        }
        if (!accepted) continue;
        predicateBoxes.push(accepted.box);
        nextPredicateEdges.add(edge);
        nextPredicateLabelPositions.set(edge, {
          x: accepted.x,
          y: accepted.y,
          width,
          label,
        });
      }
    }
    this.atlasPredicateLabelPositions = nextPredicateLabelPositions;
    if (
      sameSet(this.atlasVisibleNodes, nextVisible) &&
      sameSet(this.atlasContinuityEdges, nextContinuity) &&
      sameSet(this.atlasViewportLabels, nextLabels) &&
      sameSet(this.atlasViewportPredicateEdges, nextPredicateEdges) &&
      sameSet(this.atlasViewportRegionalEdges, nextRegionalEdges)
    ) {
      if (this.atlasLevel >= 3) this.renderer.refresh();
      return;
    }
    this.atlasVisibleNodes = nextVisible;
    this.atlasContinuityEdges = nextContinuity;
    this.atlasViewportLabels = nextLabels;
    this.atlasViewportPredicateEdges = nextPredicateEdges;
    this.atlasViewportRegionalEdges = nextRegionalEdges;
    if (this.atlasOverlayCanvas) {
      this.atlasOverlayCanvas.dataset.visibleNodes = String(nextVisible.size);
      this.atlasOverlayCanvas.dataset.continuityEdges = String(
        nextContinuity.size,
      );
      this.atlasOverlayCanvas.dataset.nodeLabels = String(nextLabels.size);
      this.atlasOverlayCanvas.dataset.predicateLabels = String(
        nextPredicateEdges.size,
      );
      this.atlasOverlayCanvas.dataset.regionalEdges = String(
        nextRegionalEdges.size,
      );
    }
    this.renderer.refresh();
  },
};
