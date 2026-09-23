import type { GraphController } from "./controller";
import { type CategoryDescriptor } from "./types";

export const thematic_rendering = {
  wrapThematicLabel(
    this: GraphController,
    context: CanvasRenderingContext2D,
    label: string,
    maximumWidth: number,
    maximumLines: number,
  ): string[] | null {
    if (maximumLines < 1 || maximumWidth <= 0) return null;
    const words = label
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replace(/[_-]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .split(" ");
    const lines: string[] = [];
    let line = "";
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (context.measureText(candidate).width <= maximumWidth) {
        line = candidate;
        continue;
      }
      if (line) {
        lines.push(line);
        line = "";
      }
      if (context.measureText(word).width <= maximumWidth) {
        line = word;
        continue;
      }
      let fragment = "";
      for (const character of word) {
        const next = fragment + character;
        if (fragment && context.measureText(next).width > maximumWidth) {
          lines.push(fragment);
          fragment = character;
        } else {
          fragment = next;
        }
      }
      line = fragment;
      if (lines.length > maximumLines) return null;
    }
    if (line) lines.push(line);
    return lines.length <= maximumLines ? lines : null;
  },

  drawThematicDirectEdges(
    this: GraphController,
    context: CanvasRenderingContext2D,
    width: number,
    height: number,
  ): Array<{
    kind: "rectangle";
    x: number;
    y: number;
    halfWidth: number;
    halfHeight: number;
  }> {
    if (!this.renderer || !this.graph || !this.thematicPayload) return [];
    this.thematicRelationLabelHits = [];
    const canvas = this.thematicOverlayCanvas;
    const hoveredRelationIndex = this.thematicHoveredRelationLabel
      ? this.thematicRelationIndex(this.thematicHoveredRelationLabel)
      : null;
    const hoveredPattern = this.thematicHoveredPattern;
    if (!this.thematicSelectionCandidates.length) {
      if (canvas) {
        canvas.dataset.visibleDirectEdgeLabels = "0";
        canvas.dataset.hiddenDirectEdgeLabels = "0";
        canvas.dataset.outgoingPaths = "0";
        canvas.dataset.incomingPaths = "0";
        canvas.dataset.coincidentPaths = "0";
        canvas.dataset.foregroundRelationTags = "0";
        canvas.dataset.ghostRelationTags = String(
          this.thematicPayload.relationCategories.length,
        );
      }
      return [];
    }

    type DirectPath = {
      patternIndex: number;
      relationIndex: number;
      pairKey: string;
      count: number;
      startX: number;
      startY: number;
      controlX: number;
      controlY: number;
      endX: number;
      endY: number;
      labelX: number;
      labelY: number;
      color: string;
    };
    const groups = new Map<string, number[]>();
    for (const patternIndex of this.thematicSelectedPatterns) {
      const pattern = this.thematicPayload.patterns[patternIndex];
      if (!pattern) continue;
      // Keyed on the unordered pair, matching parallelEdgeOffsets() in
      // collision_labels.ts. Keying on the written subject/object order split
      // one pairing into two groups that then laid themselves out
      // independently on a shared midpoint.
      const key =
        pattern[0] <= pattern[2]
          ? `${pattern[0]}:${pattern[2]}`
          : `${pattern[2]}:${pattern[0]}`;
      const group = groups.get(key) || [];
      group.push(patternIndex);
      groups.set(key, group);
    }
    const paths: DirectPath[] = [];
    const quadraticPoint = (
      start: number,
      control: number,
      end: number,
      t: number,
    ): number => {
      const inverse = 1 - t;
      return (
        inverse * inverse * start + 2 * inverse * t * control + t * t * end
      );
    };
    for (const [pairKey, group] of groups) {
      group.sort((left, right) => {
        const leftPattern = this.thematicPayload?.patterns[left];
        const rightPattern = this.thematicPayload?.patterns[right];
        if (!leftPattern || !rightPattern) return left - right;
        return (
          leftPattern[1] - rightPattern[1] ||
          rightPattern[4] - leftPattern[4] ||
          left - right
        );
      });
      group.forEach((patternIndex, index) => {
        const pattern = this.thematicPayload?.patterns[patternIndex];
        if (!pattern) return;
        const sourceNode = this.thematicEntityNodeId(
          this.thematicPayload?.entityCategories[pattern[0]].id || "",
        );
        const targetNode = this.thematicEntityNodeId(
          this.thematicPayload?.entityCategories[pattern[2]].id || "",
        );
        if (
          !this.graph?.hasNode(sourceNode) ||
          !this.graph.hasNode(targetNode)
        ) {
          return;
        }
        const sourceAttributes = this.graph.getNodeAttributes(sourceNode);
        const targetAttributes = this.graph.getNodeAttributes(targetNode);
        const source = this.renderer?.graphToViewport({
          x: Number(sourceAttributes.x),
          y: Number(sourceAttributes.y),
        });
        const target = this.renderer?.graphToViewport({
          x: Number(targetAttributes.x),
          y: Number(targetAttributes.y),
        });
        if (!source || !target) return;
        const sourceRadius =
          this.renderer?.scaleSize(Number(sourceAttributes.size)) || 0;
        const targetRadius =
          this.renderer?.scaleSize(Number(targetAttributes.size)) || 0;
        const centeredIndex = index - (group.length - 1) / 2;
        let startX = source.x;
        let startY = source.y;
        let endX = target.x;
        let endY = target.y;
        let controlX = (source.x + target.x) / 2;
        let controlY = (source.y + target.y) / 2;
        if (sourceNode === targetNode) {
          const angle = pattern[1] * 2.399963 + centeredIndex * 0.19;
          const spread = sourceRadius + 24 + Math.abs(centeredIndex) * 2.8;
          startX = source.x + Math.cos(angle - 0.46) * (sourceRadius + 1);
          startY = source.y + Math.sin(angle - 0.46) * (sourceRadius + 1);
          endX = source.x + Math.cos(angle + 0.46) * (sourceRadius + 1);
          endY = source.y + Math.sin(angle + 0.46) * (sourceRadius + 1);
          controlX = source.x + Math.cos(angle) * spread * 2;
          controlY = source.y + Math.sin(angle) * spread * 2;
        } else {
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const distance = Math.max(1, Math.hypot(dx, dy));
          const unitX = dx / distance;
          const unitY = dy / distance;
          startX += unitX * (sourceRadius + 2);
          startY += unitY * (sourceRadius + 2);
          endX -= unitX * (targetRadius + 7);
          endY -= unitY * (targetRadius + 7);
          const curveStep = Math.max(
            4.5,
            Math.min(9, 150 / Math.max(1, group.length)),
          );
          // Same canonicalDirection trick parallelEdgeOffsets() uses. The
          // perpendicular comes from this claim's own subject-to-object
          // vector, which reverses for the opposite orientation, so without
          // the sign a forward and a reverse path with mirrored offsets
          // resolve to one control point and trace the same curve twice.
          const canonicalSide = pattern[0] <= pattern[2] ? 1 : -1;
          const offset = centeredIndex * curveStep * canonicalSide;
          controlX = (startX + endX) / 2 - unitY * offset;
          controlY = (startY + endY) / 2 + unitX * offset;
        }
        let labelX: number;
        let labelY: number;
        if (
          this.thematicPairEntity &&
          sourceNode !== targetNode &&
          group.length > 1
        ) {
          const columns = Math.min(8, Math.ceil(Math.sqrt(group.length * 1.4)));
          const rows = Math.ceil(group.length / columns);
          const column = index % columns;
          const row = Math.floor(index / columns);
          // Centre the grid on the two node centres rather than on this
          // path's own trimmed endpoints. The trim differs per orientation
          // (and with each node's radius), so deriving the centre per path
          // gave the pairing's two orientations two offset grids.
          const gridCenterX = (source.x + target.x) / 2;
          const gridCenterY = (source.y + target.y) / 2;
          labelX = gridCenterX + (column - (columns - 1) / 2) * 34;
          labelY = gridCenterY + (row - (rows - 1) / 2) * 19;
          controlX = labelX * 2 - (startX + endX) / 2;
          controlY = labelY * 2 - (startY + endY) / 2;
        } else {
          const labelT =
            group.length === 1
              ? 0.5
              : 0.25 + 0.5 * ((index + 0.5) / group.length);
          labelX = quadraticPoint(startX, controlX, endX, labelT);
          labelY = quadraticPoint(startY, controlY, endY, labelT);
        }
        // Orientation wins over the pair highlight. Flat orange for every
        // paired edge threw away the one thing the reader is looking for once
        // both slots are filled: which side is acting on which.
        const anchors = this.thematicOrientationAnchors;
        const color = anchors.has(pattern[0])
          ? "#8c45b5"
          : anchors.has(pattern[2])
            ? "#168f91"
            : this.thematicPairEntity
              ? "#d9792b"
              : this.thematicPrimaryEntity === sourceNode
                ? "#8c45b5"
                : this.thematicPrimaryEntity === targetNode
                  ? "#168f91"
                  : "#70419a";
        paths.push({
          patternIndex,
          relationIndex: pattern[1],
          pairKey,
          count: pattern[4],
          startX,
          startY,
          controlX,
          controlY,
          endX,
          endY,
          labelX,
          labelY,
          color,
        });
      });
    }

    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";
    for (const path of paths) {
      const relationHovered = hoveredPattern === path.patternIndex;
      const anotherRelationHovered =
        hoveredPattern !== null && !relationHovered;
      context.beginPath();
      context.moveTo(path.startX, path.startY);
      context.quadraticCurveTo(
        path.controlX,
        path.controlY,
        path.endX,
        path.endY,
      );
      context.globalAlpha = relationHovered
        ? 0.98
        : anotherRelationHovered
          ? 0.14
          : 0.58;
      context.strokeStyle = path.color;
      context.lineWidth =
        Math.min(3.2, 0.75 + Math.log1p(path.count) * 0.32) +
        (relationHovered ? 1.6 : 0);
      context.stroke();
      const angle = Math.atan2(
        path.endY - path.controlY,
        path.endX - path.controlX,
      );
      context.beginPath();
      context.moveTo(path.endX, path.endY);
      context.lineTo(
        path.endX - Math.cos(angle - 0.48) * 7,
        path.endY - Math.sin(angle - 0.48) * 7,
      );
      context.lineTo(
        path.endX - Math.cos(angle + 0.48) * 7,
        path.endY - Math.sin(angle + 0.48) * 7,
      );
      context.closePath();
      context.fillStyle = path.color;
      context.fill();
    }
    context.globalAlpha = 1;

    const labelHalfWidth = 15;
    const labelHalfHeight = 7.5;
    const occupied: Array<[number, number, number, number]> = [];
    for (const category of this.thematicPayload.entityCategories) {
      if (this.isSemanticTagSlice() && category.activeMentionCount <= 0) {
        continue;
      }
      const node = this.thematicEntityNodeId(category.id);
      if (!this.graph.hasNode(node)) continue;
      const attributes = this.graph.getNodeAttributes(node);
      const point = this.renderer.graphToViewport({
        x: Number(attributes.x),
        y: Number(attributes.y),
      });
      const radius = this.renderer.scaleSize(Number(attributes.size)) + 2;
      occupied.push([
        point.x - radius,
        point.y - radius,
        point.x + radius,
        point.y + radius,
      ]);
    }
    const selectedRelationFocus = this.selectedNode
      ? this.thematicRelationIndex(this.selectedNode)
      : null;
    const forceAllDirectLabels =
      Boolean(this.thematicPairEntity) ||
      this.thematicContextRelation !== null ||
      selectedRelationFocus !== null;
    const labels = paths
      .slice()
      .sort(
        (left, right) =>
          right.count - left.count ||
          left.relationIndex - right.relationIndex ||
          left.patternIndex - right.patternIndex,
      );
    const footprints: Array<{
      kind: "rectangle";
      x: number;
      y: number;
      halfWidth: number;
      halfHeight: number;
    }> = [];
    const visibleRelations = new Set<number>();
    let hoveredAnchor: DirectPath | null = null;
    for (const path of labels) {
      let box: [number, number, number, number] = [
        path.labelX - labelHalfWidth,
        path.labelY - labelHalfHeight,
        path.labelX + labelHalfWidth,
        path.labelY + labelHalfHeight,
      ];
      if (box[2] < 0 || box[0] > width || box[3] < 0 || box[1] > height)
        continue;
      let collision = occupied.some(
        (other) =>
          box[0] < other[2] + 2 &&
          box[2] > other[0] - 2 &&
          box[1] < other[3] + 2 &&
          box[3] > other[1] - 2,
      );
      if (collision && forceAllDirectLabels && !this.thematicPairEntity) {
        for (const alternativeT of [0.35, 0.65, 0.2, 0.8, 0.1, 0.9]) {
          const alternativeX = quadraticPoint(
            path.startX,
            path.controlX,
            path.endX,
            alternativeT,
          );
          const alternativeY = quadraticPoint(
            path.startY,
            path.controlY,
            path.endY,
            alternativeT,
          );
          const alternativeBox: [number, number, number, number] = [
            alternativeX - labelHalfWidth,
            alternativeY - labelHalfHeight,
            alternativeX + labelHalfWidth,
            alternativeY + labelHalfHeight,
          ];
          const alternativeCollision = occupied.some(
            (other) =>
              alternativeBox[0] < other[2] + 2 &&
              alternativeBox[2] > other[0] - 2 &&
              alternativeBox[1] < other[3] + 2 &&
              alternativeBox[3] > other[1] - 2,
          );
          if (alternativeCollision) continue;
          path.labelX = alternativeX;
          path.labelY = alternativeY;
          box = alternativeBox;
          collision = false;
          break;
        }
      }
      if (collision && !forceAllDirectLabels) continue;
      occupied.push(box);
      const relation =
        this.thematicPayload.relationCategories[path.relationIndex];
      if (!relation) continue;
      const relationHovered = hoveredPattern === path.patternIndex;
      if (
        relationHovered &&
        (!hoveredAnchor ||
          !this.thematicHoveredRelationPoint ||
          Math.hypot(
            path.labelX - this.thematicHoveredRelationPoint.x,
            path.labelY - this.thematicHoveredRelationPoint.y,
          ) <
            Math.hypot(
              hoveredAnchor.labelX - this.thematicHoveredRelationPoint.x,
              hoveredAnchor.labelY - this.thematicHoveredRelationPoint.y,
            ))
      ) {
        hoveredAnchor = path;
      }
      context.beginPath();
      context.roundRect(
        box[0],
        box[1],
        labelHalfWidth * 2,
        labelHalfHeight * 2,
        7,
      );
      context.fillStyle = relationHovered ? "#9a4fc4" : "#6a3290";
      context.fill();
      context.lineWidth = relationHovered ? 2.2 : 1.1;
      context.strokeStyle = relationHovered
        ? "rgba(255, 222, 112, 1)"
        : "rgba(244, 201, 93, 0.9)";
      context.stroke();
      context.font = "800 8px Inter, Segoe UI, sans-serif";
      context.fillStyle = "#ffffff";
      context.fillText(relation.tagId, path.labelX, path.labelY + 0.5);
      const relationNode = this.thematicRelationNodeId(relation.id);
      this.thematicRelationLabelHits.push({
        node: relationNode,
        patternIndex: path.patternIndex,
        x: path.labelX,
        y: path.labelY,
        halfWidth: labelHalfWidth,
        halfHeight: labelHalfHeight,
      });
      footprints.push({
        kind: "rectangle",
        x: path.labelX,
        y: path.labelY,
        halfWidth: labelHalfWidth,
        halfHeight: labelHalfHeight,
      });
      visibleRelations.add(path.relationIndex);
      if (canvas && !canvas.dataset.firstRelationLabelX) {
        canvas.dataset.firstRelationLabelId = relation.id;
        canvas.dataset.firstRelationLabelX = path.labelX.toFixed(2);
        canvas.dataset.firstRelationLabelY = path.labelY.toFixed(2);
      }
      if (relation.id === "R01" && canvas && !canvas.dataset.r01X) {
        canvas.dataset.r01X = path.labelX.toFixed(2);
        canvas.dataset.r01Y = path.labelY.toFixed(2);
      }
    }
    if (hoveredAnchor && hoveredRelationIndex !== null) {
      const relation =
        this.thematicPayload.relationCategories[hoveredRelationIndex];
      if (relation) {
        const tooltip = `${relation.tagId} · ${relation.label}`;
        context.font = "650 11px Inter, Segoe UI, sans-serif";
        const tooltipWidth = Math.min(
          380,
          Math.max(120, context.measureText(tooltip).width + 20),
        );
        const tooltipHeight = 25;
        const tooltipLeft = Math.max(
          6,
          Math.min(
            width - tooltipWidth - 6,
            hoveredAnchor.labelX - tooltipWidth / 2,
          ),
        );
        const preferredTop =
          hoveredAnchor.labelY - labelHalfHeight - tooltipHeight - 7;
        const tooltipTop =
          preferredTop >= 6
            ? preferredTop
            : hoveredAnchor.labelY + labelHalfHeight + 7;
        context.beginPath();
        context.roundRect(
          tooltipLeft,
          tooltipTop,
          tooltipWidth,
          tooltipHeight,
          6,
        );
        context.fillStyle = "rgba(29, 22, 34, 0.96)";
        context.fill();
        context.lineWidth = 1.2;
        context.strokeStyle = "rgba(154, 79, 196, 0.95)";
        context.stroke();
        context.fillStyle = "#ffffff";
        context.textAlign = "left";
        context.fillText(
          tooltip,
          tooltipLeft + 10,
          tooltipTop + tooltipHeight / 2 + 0.5,
          tooltipWidth - 20,
        );
        context.textAlign = "center";
      }
    }
    context.restore();
    if (canvas) {
      canvas.dataset.visibleDirectEdgeLabels = String(footprints.length);
      canvas.dataset.hiddenDirectEdgeLabels = String(
        Math.max(0, paths.length - footprints.length),
      );
      // Curves of one pairing that still resolve to the same control point,
      // i.e. that would be drawn one on top of the other.
      const byPair = new Map<string, DirectPath[]>();
      for (const path of paths) {
        const list = byPair.get(path.pairKey);
        if (list) list.push(path);
        else byPair.set(path.pairKey, [path]);
      }
      let coincidentPaths = 0;
      for (const list of byPair.values()) {
        for (let left = 0; left < list.length; left += 1) {
          for (let right = left + 1; right < list.length; right += 1) {
            if (
              Math.hypot(
                list[left].controlX - list[right].controlX,
                list[left].controlY - list[right].controlY,
              ) < 1.5
            ) {
              coincidentPaths += 1;
            }
          }
        }
      }
      canvas.dataset.coincidentPaths = String(coincidentPaths);
      // Label anchor points, so a test can hover a specific direction of a
      // relation rather than whichever pill happens to be first.
      // Entity circle geometry, so a test can aim at a pill that really does
      // sit on a node and prove the click resolves to the pill.
      canvas.dataset.entityCircles = this.thematicPayload.entityCategories
        .flatMap((category) => {
          const node = this.thematicEntityNodeId(category.id);
          if (!this.graph?.hasNode(node)) return [];
          const attributes = this.graph.getNodeAttributes(node);
          const point = this.renderer?.graphToViewport({
            x: Number(attributes.x),
            y: Number(attributes.y),
          });
          if (!point) return [];
          const radius = this.renderer?.scaleSize(Number(attributes.size)) || 0;
          return [
            `${point.x.toFixed(1)},${point.y.toFixed(1)},${radius.toFixed(1)}`,
          ];
        })
        .join(";");
      canvas.dataset.relationLabelPoints = this.thematicRelationLabelHits
        .slice(0, 80)
        .map(
          (hit) =>
            `${hit.patternIndex},${hit.x.toFixed(1)},${hit.y.toFixed(1)}`,
        )
        .join(";");
      // Direction of each drawn path, by the colour it was given.
      canvas.dataset.outgoingPaths = String(
        paths.filter((path) => path.color === "#8c45b5").length,
      );
      canvas.dataset.incomingPaths = String(
        paths.filter((path) => path.color === "#168f91").length,
      );
      canvas.dataset.foregroundRelationTags = String(visibleRelations.size);
      canvas.dataset.ghostRelationTags = String(
        this.thematicPayload.relationCategories.length - visibleRelations.size,
      );
    }
    return footprints;
  },

  drawThematicOverlay(this: GraphController): void {
    if (
      !this.renderer ||
      !this.graph ||
      !this.thematicPayload ||
      !this.thematicOverlayCanvas
    )
      return;
    const canvas = this.thematicOverlayCanvas;
    const context = canvas.getContext("2d");
    if (!context) return;
    const { width, height } = this.renderer.getDimensions();
    const dpr = window.devicePixelRatio || 1;
    const targetWidth = Math.max(1, Math.round(width * dpr));
    const targetHeight = Math.max(1, Math.round(height * dpr));
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    context.textAlign = "center";
    context.textBaseline = "middle";
    delete canvas.dataset.r01X;
    delete canvas.dataset.r01Y;
    delete canvas.dataset.firstRelationLabelId;
    delete canvas.dataset.firstRelationLabelX;
    delete canvas.dataset.firstRelationLabelY;
    const footprints: Array<
      | { kind: "circle"; x: number; y: number; radius: number }
      | {
          kind: "rectangle";
          x: number;
          y: number;
          halfWidth: number;
          halfHeight: number;
        }
    > = this.drawThematicDirectEdges(context, width, height);
    let visibleEntityTags = 0;
    let fullEntityLabels = 0;

    for (const category of this.thematicPayload.entityCategories) {
      if (this.isSemanticTagSlice() && category.activeMentionCount <= 0) {
        continue;
      }
      const node = this.thematicEntityNodeId(category.id);
      if (!this.graph.hasNode(node)) continue;
      const renderedRadius = this.renderer.scaleSize(
        Number(this.graph.getNodeAttribute(node, "size")),
      );
      const point = this.renderer.graphToViewport({
        x: Number(this.graph.getNodeAttribute(node, "x")),
        y: Number(this.graph.getNodeAttribute(node, "y")),
      });
      if (category.id === "E01" || category.id === "E24") {
        const key = category.id.toLowerCase();
        canvas.dataset[`${key}X`] = point.x.toFixed(2);
        canvas.dataset[`${key}Y`] = point.y.toFixed(2);
      }
      if (
        point.x + renderedRadius < 0 ||
        point.y + renderedRadius < 0 ||
        point.x - renderedRadius > width ||
        point.y - renderedRadius > height
      )
        continue;
      const selected =
        node === this.selectedNode ||
        node === this.thematicPrimaryEntity ||
        node === this.thematicPairEntity;
      context.font = "800 9px Inter, Segoe UI, sans-serif";
      const tagFits =
        context.measureText(category.tagId).width + 5 <= renderedRadius * 2;
      if (tagFits) {
        context.font = "600 7px Inter, Segoe UI, sans-serif";
        const labelLines = this.wrapThematicLabel(
          context,
          category.label,
          renderedRadius * 1.55,
          Math.max(0, Math.floor((renderedRadius * 1.45 - 12) / 8)),
        );
        if (labelLines) {
          const totalHeight = 10 + labelLines.length * 8;
          let lineY = point.y - totalHeight / 2 + 5;
          context.font = "800 9px Inter, Segoe UI, sans-serif";
          context.fillStyle = selected ? "#ffffff" : "#f6fffc";
          context.fillText(category.tagId, point.x, lineY);
          lineY += 10;
          context.font = "600 7px Inter, Segoe UI, sans-serif";
          context.fillStyle = selected ? "#f3eaff" : "#dff8ef";
          for (const line of labelLines) {
            context.fillText(line, point.x, lineY);
            lineY += 8;
          }
          fullEntityLabels += 1;
        } else {
          context.font = "800 9px Inter, Segoe UI, sans-serif";
          context.fillStyle = selected ? "#ffffff" : "#f6fffc";
          context.fillText(category.tagId, point.x, point.y);
        }
        visibleEntityTags += 1;
      }
      footprints.push({
        kind: "circle",
        x: point.x,
        y: point.y,
        radius: renderedRadius,
      });
    }

    let visualOverlaps = 0;
    let entityEntityOverlaps = 0;
    let entityRelationOverlaps = 0;
    let relationRelationOverlaps = 0;
    const clearance = 0.25;
    for (let left = 0; left < footprints.length; left += 1) {
      for (let right = left + 1; right < footprints.length; right += 1) {
        const first = footprints[left];
        const second = footprints[right];
        if (first.kind === "circle" && second.kind === "circle") {
          if (
            Math.hypot(first.x - second.x, first.y - second.y) <
            first.radius + second.radius + clearance
          ) {
            visualOverlaps += 1;
            entityEntityOverlaps += 1;
          }
          continue;
        }
        if (first.kind === "rectangle" && second.kind === "rectangle") {
          if (
            Math.abs(first.x - second.x) <
              first.halfWidth + second.halfWidth + clearance &&
            Math.abs(first.y - second.y) <
              first.halfHeight + second.halfHeight + clearance
          ) {
            visualOverlaps += 1;
            relationRelationOverlaps += 1;
          }
          continue;
        }
        const circle = (first.kind === "circle" ? first : second) as Extract<
          (typeof footprints)[number],
          { kind: "circle" }
        >;
        const rectangle = (
          first.kind === "rectangle" ? first : second
        ) as Extract<(typeof footprints)[number], { kind: "rectangle" }>;
        const dx = Math.max(
          Math.abs(circle.x - rectangle.x) - rectangle.halfWidth,
          0,
        );
        const dy = Math.max(
          Math.abs(circle.y - rectangle.y) - rectangle.halfHeight,
          0,
        );
        if (Math.hypot(dx, dy) < circle.radius + clearance) {
          visualOverlaps += 1;
          entityRelationOverlaps += 1;
        }
      }
    }
    canvas.dataset.visibleTags = String(footprints.length);
    canvas.dataset.visibleEntityTags = String(visibleEntityTags);
    canvas.dataset.fullEntityLabels = String(fullEntityLabels);
    canvas.dataset.visualOverlaps = String(visualOverlaps);
    canvas.dataset.entityEntityOverlaps = String(entityEntityOverlaps);
    canvas.dataset.entityRelationOverlaps = String(entityRelationOverlaps);
    canvas.dataset.relationRelationOverlaps = String(relationRelationOverlaps);

    const detailNode = this.hoveredNode || this.selectedNode;
    if (detailNode && this.graph.hasNode(detailNode)) {
      const data = this.graph.getNodeAttributes(detailNode);
      if (data.thematicKind === "relation") return;
      const category = data.categoryDetail as CategoryDescriptor | null;
      if (category) {
        const point = this.renderer.graphToViewport({
          x: Number(data.x),
          y: Number(data.y),
        });
        context.font = "600 11px Inter, Segoe UI, sans-serif";
        const label = `${category.tagId} · ${category.label}`;
        const labelWidth = Math.min(320, context.measureText(label).width + 18);
        const left = Math.min(
          width - labelWidth - 6,
          Math.max(6, point.x - labelWidth / 2),
        );
        const top = Math.min(height - 29, Math.max(6, point.y + 21));
        context.beginPath();
        context.roundRect(left, top, labelWidth, 23, 6);
        context.fillStyle = "rgba(26, 29, 36, 0.92)";
        context.fill();
        context.textAlign = "left";
        context.fillStyle = "#ffffff";
        context.fillText(label, left + 9, top + 12);
        context.textAlign = "center";
      }
    }
  },
};
