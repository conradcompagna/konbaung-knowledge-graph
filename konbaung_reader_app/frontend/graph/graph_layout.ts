import type { GraphController } from "./controller";
import { MultiDirectedGraph } from "graphology";
import { type TopologyPayload } from "./types";
import { el, clear } from "./dom";
import { nodeSize, bubbleNodeSize, atlasNodeColor } from "./utilities";

export const graph_layout = {
  assignLayout(
    this: GraphController,
    graph: MultiDirectedGraph,
    payload: TopologyPayload,
  ): void {
    const kind = payload.layout.kind;

    if (kind === "atlas") return;

    if (kind === "claim") {
      const edge = payload.edges[0];
      if (!edge) return;
      graph.mergeNodeAttributes(payload.nodes[edge[0]][0], {
        x: -1,
        y: 0,
        layoutParent: "",
        layoutDepth: 0,
      });
      graph.mergeNodeAttributes(payload.nodes[edge[1]][0], {
        x: 1,
        y: 0,
        layoutParent: payload.nodes[edge[0]][0],
        layoutDepth: 1,
      });
      return;
    }

    if (kind === "triples") {
      // Constructive radial layout. Nothing here repairs overlap after the
      // fact: the geometry is sized so that the overlaps we care about
      // cannot occur, and a single exact verification confirms it.
      //
      //  1. Every subtree owns a disjoint angular sector, centred on its
      //     parent's outward direction and never wider than OUTWARD_SPAN.
      //     A sector below 180 degrees is convex, so a straight edge drawn
      //     between two points inside one never leaves it.
      //  2. Every node sits exactly on a ring, and every ring gap is wider
      //     than the diameters it separates, so no node ever lies in the
      //     open space between two rings.
      //  3. An edge therefore spans exactly one ring gap inside one sector:
      //     the only nodes it can come near are its own endpoints and its
      //     siblings.
      //  4. A relation tag is budgeted into both its sector width and its
      //     ring gap, and every edge is long enough to hold a centred tag
      //     with bare line and an arrowhead beyond each end of it.
      //
      // Angular widths use asin(radius / ringRadius) rather than the
      // arc-length approximation, which makes the spacing exact instead of
      // merely close: for half-angles a and b,
      //     sin a + sin b = 2 sin((a+b)/2) cos((a-b)/2) <= 2 sin((a+b)/2)
      // so the chord between two centres placed a + b apart is never
      // shorter than the sum of the two radii.
      //
      // The construction above leaves a short tail of cases it does not
      // prove on its own -- chiefly the handful of non-tree edges in a
      // cyclic component, which no sector argument can contain. Those are
      // caught by verifyLayout() and answered by scaling the component up.
      // Scaling is monotone (distances grow, radii and tags do not), so the
      // solve terminates, and in practice it lands on the first or second
      // attempt rather than needing the scale at all.

      const NODE_CLEARANCE = 7; // bare space between two node discs
      const TAG_CLEARANCE = 7; // bare space around a relation tag
      const TAG_ENDPOINT_CLEARANCE = 17;
      // drawCollisionLabels flags an edge as touching a node at radius + 8,
      // so the layout has to leave more than that or the audit disagrees
      // with the geometry that produced it.
      const EDGE_NODE_CLEARANCE = 11;
      const TAG_STANDOFF = 24; // tag to endpoint rim: arrowhead plus bare line
      const OUTWARD_SPAN = (Math.PI * 5) / 9; // 100deg: children fan outward
      const COMPONENT_GAP = 28;
      const MAX_SOLVE_STEPS = 24;
      const TAG_HEIGHT = 15; // font size 8 plus the 7px padding drawEdgeLabel adds

      const nodeIds = payload.nodes.map((node) => node[0]);
      const nodeById = new Map(payload.nodes.map((node) => [node[0], node]));
      const measurementCanvas = document.createElement("canvas");
      const measurementContext = measurementCanvas.getContext("2d");
      if (measurementContext) {
        measurementContext.font = "600 8px Inter, Segoe UI, sans-serif";
      }
      // Must stay in step with ensurePlacements() in collision_labels.ts.
      const tagWidth = (label: string) =>
        (measurementContext
          ? measurementContext.measureText(label).width
          : label.length * 4.6) + 8;

      const adjacency = new Map(nodeIds.map((id) => [id, new Set<string>()]));
      const sources = new Set<string>();
      const targets = new Set<string>();
      // A self-loop has no length to hang its tag on, so it is kept out of
      // the tree entirely and given an anchor of its own once the component
      // is placed. Leaving it in adjacency would make a node its own
      // neighbour and skew the ranking that picks each component's root.
      const selfLoops = new Map<string, string[]>();
      for (const edge of graph.edges()) {
        const [source, target] = graph.extremities(edge);
        sources.add(source);
        targets.add(target);
        if (source === target) {
          const loops = selfLoops.get(source) || [];
          loops.push(edge);
          selfLoops.set(source, loops);
          continue;
        }
        adjacency.get(source)?.add(target);
        adjacency.get(target)?.add(source);
      }
      const roleColor = (id: string) =>
        sources.has(id) && targets.has(id)
          ? "#75419a"
          : sources.has(id)
            ? "#2c718f"
            : "#0f8a74";
      const nodeRadius = (id: string) =>
        Number(graph.getNodeAttribute(id, "size"));

      // Mirrors parallelEdgeOffsets() in collision_labels.ts: the nth edge of
      // a parallel group bows out by slot * 48, which displaces its tag by
      // half that. Budgeting it here keeps parallel relations inside the
      // sector their pair was given.
      const pairKey = (left: string, right: string) =>
        left < right ? `${left}\u0000${right}` : `${right}\u0000${left}`;
      const parallelGroups = new Map<string, string[]>();
      for (const edge of graph.edges()) {
        const [source, target] = graph.extremities(edge);
        if (source === target) continue;
        const key = pairKey(source, target);
        const group = parallelGroups.get(key) || [];
        group.push(edge);
        parallelGroups.set(key, group);
      }
      const tagBow = new Map<string, number>();
      for (const group of parallelGroups.values()) {
        group
          .slice()
          .sort()
          .forEach((edge, index) => {
            tagBow.set(
              edge,
              (Math.abs(index - (group.length - 1) / 2) * 48) / 2,
            );
          });
      }

      // A tag is a thin rectangle lying along its edge, so its two extents
      // do completely different jobs and must not be conflated. The half
      // WIDTH runs along the edge and is what the edge has to be long enough
      // to hold. The half HEIGHT sticks out sideways and is all the sector
      // has to be wide enough to hold. Treating the tag as a disc of its
      // half-width -- roughly 100px for the longest relation names here --
      // demands over ten times the sideways room it actually occupies, which
      // is enough to make every layout look impossible to satisfy.
      const tagHalfLengthByEdge = new Map<string, number>();
      for (const edge of graph.edges()) {
        const width = tagWidth(
          String(graph.getEdgeAttribute(edge, "relationLabel")),
        );
        tagHalfLengthByEdge.set(edge, width / 2);
      }
      // Along the edge.
      const tagHalfLength = (edge: string) =>
        tagHalfLengthByEdge.get(edge) || 0;
      // Across the edge, including any bow a parallel relation is drawn with.
      const tagHalfDepth = (edge: string) =>
        TAG_HEIGHT / 2 + (tagBow.get(edge) || 0);

      // Length that lets the tag sit at the midpoint with an arrowhead and a
      // run of bare line still visible past each side of it.
      const edgeSpan = (edge: string) => {
        const [source, target] = graph.extremities(edge);
        return (
          2 *
          (tagHalfLength(edge) +
            Math.max(nodeRadius(source), nodeRadius(target)) +
            TAG_STANDOFF)
        );
      };

      const nodeRank = (left: string, right: string) =>
        (adjacency.get(right)?.size || 0) - (adjacency.get(left)?.size || 0) ||
        Number(nodeById.get(right)?.[2] || 0) -
          Number(nodeById.get(left)?.[2] || 0) ||
        left.localeCompare(right);

      const unseen = new Set(nodeIds);
      const components: string[][] = [];
      while (unseen.size) {
        const first = unseen.values().next().value as string;
        const stack = [first];
        const component: string[] = [];
        unseen.delete(first);
        while (stack.length) {
          const id = stack.pop() as string;
          component.push(id);
          for (const neighbor of adjacency.get(id) || []) {
            if (!unseen.has(neighbor)) continue;
            unseen.delete(neighbor);
            stack.push(neighbor);
          }
        }
        components.push(component);
      }
      const rootOf = new Map<string[], string>();
      for (const component of components) {
        rootOf.set(component, component.slice().sort(nodeRank)[0]);
      }
      components.sort(
        (left, right) =>
          right.length - left.length ||
          nodeRank(rootOf.get(left) as string, rootOf.get(right) as string),
      );

      type LayoutPoint = { x: number; y: number };

      const distanceToSegment = (
        point: LayoutPoint,
        start: LayoutPoint,
        end: LayoutPoint,
      ): number => {
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const lengthSquared = dx * dx + dy * dy;
        if (!lengthSquared) {
          return Math.hypot(point.x - start.x, point.y - start.y);
        }
        const fraction = Math.max(
          0,
          Math.min(
            1,
            ((point.x - start.x) * dx + (point.y - start.y) * dy) /
              lengthSquared,
          ),
        );
        return Math.hypot(
          point.x - (start.x + fraction * dx),
          point.y - (start.y + fraction * dy),
        );
      };

      // Exact sweep over the finished component, using the same shapes the
      // renderer will draw: a tag is a rotated rectangle, not a disc. This
      // reports what it finds; it never resizes anything. Inflating the
      // layout to escape a failure here is what produced mile-long edges,
      // and it papered over the real bug rather than fixing it.
      const rectClearsCircle = (
        centre: LayoutPoint,
        halfLength: number,
        halfDepth: number,
        along: LayoutPoint,
        point: LayoutPoint,
        keepOff: number,
      ): boolean => {
        const dx = point.x - centre.x;
        const dy = point.y - centre.y;
        // Into the tag's own frame: x runs along the edge, y across it.
        const localX = dx * along.x + dy * along.y;
        const localY = -dx * along.y + dy * along.x;
        const nearestX = Math.max(-halfLength, Math.min(halfLength, localX));
        const nearestY = Math.max(-halfDepth, Math.min(halfDepth, localY));
        return Math.hypot(localX - nearestX, localY - nearestY) >= keepOff;
      };

      const rectClearsSegment = (
        centre: LayoutPoint,
        halfLength: number,
        halfDepth: number,
        along: LayoutPoint,
        start: LayoutPoint,
        end: LayoutPoint,
        keepOff: number,
      ): boolean => {
        const toLocal = (point: LayoutPoint) => {
          const dx = point.x - centre.x;
          const dy = point.y - centre.y;
          return {
            x: dx * along.x + dy * along.y,
            y: -dx * along.y + dy * along.x,
          };
        };
        const a = toLocal(start);
        const b = toLocal(end);
        const boxX = halfLength + keepOff;
        const boxY = halfDepth + keepOff;
        if (a.x < -boxX && b.x < -boxX) return true;
        if (a.x > boxX && b.x > boxX) return true;
        if (a.y < -boxY && b.y < -boxY) return true;
        if (a.y > boxY && b.y > boxY) return true;
        for (let step = 0; step <= 16; step += 1) {
          const t = step / 16;
          const x = a.x + (b.x - a.x) * t;
          const y = a.y + (b.y - a.y) * t;
          const nearestX = Math.max(-halfLength, Math.min(halfLength, x));
          const nearestY = Math.max(-halfDepth, Math.min(halfDepth, y));
          if (Math.hypot(x - nearestX, y - nearestY) < keepOff) return false;
        }
        return true;
      };

      interface TagBox {
        edge: string;
        centre: LayoutPoint;
        along: LayoutPoint;
        halfLength: number;
        halfDepth: number;
      }

      const tagBoxesFor = (
        componentEdges: string[],
        positions: Map<string, LayoutPoint>,
      ): TagBox[] => {
        const boxes: TagBox[] = [];
        for (const edge of componentEdges) {
          const [source, target] = graph.extremities(edge);
          if (source === target) continue;
          const start = positions.get(source);
          const end = positions.get(target);
          if (!start || !end) continue;
          const length = Math.hypot(end.x - start.x, end.y - start.y) || 1;
          const along = {
            x: (end.x - start.x) / length,
            y: (end.y - start.y) / length,
          };
          const bow = tagBow.get(edge) || 0;
          const bowSign = bow === 0 ? 0 : source < target ? 1 : -1;
          boxes.push({
            edge,
            centre: {
              x: (start.x + end.x) / 2 - along.y * bow * bowSign,
              y: (start.y + end.y) / 2 + along.x * bow * bowSign,
            },
            along,
            halfLength: tagHalfLength(edge),
            halfDepth: TAG_HEIGHT / 2,
          });
        }
        return boxes;
      };

      const rectCorners = (box: TagBox): LayoutPoint[] =>
        [-1, 1].flatMap((alongSign) =>
          [-1, 1].map((depthSign) => ({
            x:
              box.centre.x +
              box.along.x * box.halfLength * alongSign -
              box.along.y * box.halfDepth * depthSign,
            y:
              box.centre.y +
              box.along.y * box.halfLength * alongSign +
              box.along.x * box.halfDepth * depthSign,
          })),
        );

      const verifyLayout = (
        component: string[],
        componentEdges: string[],
        positions: Map<string, LayoutPoint>,
      ): boolean => {
        const tags = tagBoxesFor(componentEdges, positions);
        let widestNode = 0;
        for (const id of component) {
          widestNode = Math.max(widestNode, nodeRadius(id));
        }
        let longestTag = 0;
        for (const tag of tags) {
          longestTag = Math.max(longestTag, tag.halfLength);
        }

        const cell = 200;
        const key = (x: number, y: number) =>
          `${Math.floor(x / cell)}|${Math.floor(y / cell)}`;
        const nodeGrid = new Map<string, string[]>();
        for (const id of component) {
          const point = positions.get(id);
          if (!point) continue;
          const bucket = key(point.x, point.y);
          const list = nodeGrid.get(bucket) || [];
          list.push(id);
          nodeGrid.set(bucket, list);
        }
        const nodesNear = (point: LayoutPoint, reach: number): string[] => {
          const span = Math.ceil(reach / cell);
          const baseX = Math.floor(point.x / cell);
          const baseY = Math.floor(point.y / cell);
          const found: string[] = [];
          for (let ix = baseX - span; ix <= baseX + span; ix += 1) {
            for (let iy = baseY - span; iy <= baseY + span; iy += 1) {
              const list = nodeGrid.get(`${ix}|${iy}`);
              if (list) found.push(...list);
            }
          }
          return found;
        };

        // Node against node.
        for (const id of component) {
          const point = positions.get(id);
          if (!point) continue;
          for (const other of nodesNear(
            point,
            nodeRadius(id) + widestNode + NODE_CLEARANCE,
          )) {
            if (other <= id) continue;
            const otherPoint = positions.get(other);
            if (!otherPoint) continue;
            if (
              Math.hypot(point.x - otherPoint.x, point.y - otherPoint.y) <
              nodeRadius(id) + nodeRadius(other) + NODE_CLEARANCE
            )
              return false;
          }
        }

        // Edge line against node.
        for (const edge of componentEdges) {
          const [source, target] = graph.extremities(edge);
          if (source === target) continue;
          const start = positions.get(source);
          const end = positions.get(target);
          if (!start || !end) continue;
          const midpoint = {
            x: (start.x + end.x) / 2,
            y: (start.y + end.y) / 2,
          };
          const half = Math.hypot(end.x - start.x, end.y - start.y) / 2;
          for (const id of nodesNear(
            midpoint,
            half + widestNode + EDGE_NODE_CLEARANCE,
          )) {
            if (id === source || id === target) continue;
            const point = positions.get(id);
            if (!point) continue;
            if (
              distanceToSegment(point, start, end) <
              nodeRadius(id) + EDGE_NODE_CLEARANCE
            )
              return false;
          }
        }

        // Tag against node.
        for (const tag of tags) {
          for (const id of nodesNear(
            tag.centre,
            tag.halfLength + widestNode + TAG_CLEARANCE,
          )) {
            const point = positions.get(id);
            if (!point) continue;
            if (
              !rectClearsCircle(
                tag.centre,
                tag.halfLength,
                tag.halfDepth,
                tag.along,
                point,
                nodeRadius(id) + TAG_CLEARANCE,
              )
            )
              return false;
          }
        }

        // Tag against tag, and tag against every other edge line. Neither
        // follows from the sector argument on its own: a tag sits at its
        // edge's midpoint, which lies inside its parent's sector but not
        // inside its own child's sub-sector, so a sibling's line can still
        // reach it.
        for (let index = 0; index < tags.length; index += 1) {
          const tag = tags[index];
          for (let other = index + 1; other < tags.length; other += 1) {
            const rival = tags[other];
            if (
              Math.hypot(
                tag.centre.x - rival.centre.x,
                tag.centre.y - rival.centre.y,
              ) >
              tag.halfLength + rival.halfLength + longestTag + TAG_CLEARANCE
            )
              continue;
            const corners = rectCorners(rival);
            const sides: Array<[LayoutPoint, LayoutPoint]> = [
              [corners[0], corners[1]],
              [corners[1], corners[3]],
              [corners[3], corners[2]],
              [corners[2], corners[0]],
            ];
            for (const [from, to] of sides) {
              if (
                !rectClearsSegment(
                  tag.centre,
                  tag.halfLength,
                  tag.halfDepth,
                  tag.along,
                  from,
                  to,
                  TAG_CLEARANCE,
                )
              )
                return false;
            }
          }

          for (const edge of componentEdges) {
            if (edge === tag.edge) continue;
            const [source, target] = graph.extremities(edge);
            if (source === target) continue;
            const start = positions.get(source);
            const end = positions.get(target);
            if (!start || !end) continue;
            if (
              !rectClearsSegment(
                tag.centre,
                tag.halfLength,
                tag.halfDepth,
                tag.along,
                start,
                end,
                TAG_CLEARANCE,
              )
            )
              return false;
          }
        }
        return true;
      };

      let layoutVerified = true;
      let solveScale = 1;
      let solveSteps = 0;

      interface ComponentLayout {
        root: string;
        positions: Map<string, LayoutPoint>;
        loopAnchors: Map<string, LayoutPoint>;
        minX: number;
        minY: number;
        maxX: number;
        maxY: number;
        width: number;
        height: number;
      }

      const layoutComponent = (component: string[]): ComponentLayout => {
        const root = rootOf.get(component) as string;
        const componentSet = new Set(component);
        const componentEdges = graph
          .edges()
          .filter((edge) => componentSet.has(graph.extremities(edge)[0]));

        // Spanning tree. Children are ordered by rank so the densest branch
        // keeps the same visual position it had before.
        const parentOf = new Map<string, string | null>([[root, null]]);
        const depthOf = new Map<string, number>([[root, 0]]);
        const childrenOf = new Map<string, string[]>(
          component.map((id) => [id, [] as string[]]),
        );
        const order = [root];
        for (let cursor = 0; cursor < order.length; cursor += 1) {
          const parent = order[cursor];
          const neighbors = Array.from(adjacency.get(parent) || []).sort(
            nodeRank,
          );
          for (const neighbor of neighbors) {
            if (parentOf.has(neighbor)) continue;
            parentOf.set(neighbor, parent);
            depthOf.set(neighbor, Number(depthOf.get(parent) || 0) + 1);
            childrenOf.get(parent)?.push(neighbor);
            order.push(neighbor);
          }
        }
        const maxDepth = order.reduce(
          (deepest, id) => Math.max(deepest, depthOf.get(id) || 0),
          0,
        );
        const isInSubtree = (top: string, id: string): boolean => {
          let current: string | null | undefined = id;
          while (current) {
            if (current === top) return true;
            current = parentOf.get(current);
          }
          return false;
        };
        const descendantsOf = (top: string): string[] => {
          const descendants: string[] = [];
          const stack = [top];
          while (stack.length) {
            const current = stack.pop() as string;
            descendants.push(current);
            for (const child of childrenOf.get(current) || []) {
              stack.push(child);
            }
          }
          return descendants;
        };
        const tagMover = (edge: string): string => {
          const [source, target] = graph.extremities(edge);
          if (parentOf.get(target) === source) return target;
          if (parentOf.get(source) === target) return source;
          const sourceDepth = Number(depthOf.get(source) || 0);
          const targetDepth = Number(depthOf.get(target) || 0);
          return targetDepth > sourceDepth ? target : source;
        };

        // Every edge joining a parent to one of its children shares that
        // pair's ring gap, so the gap has to satisfy the longest of them.
        const linkEdges = new Map<string, string[]>();
        const baseGap = new Array(maxDepth + 1).fill(0);
        for (const edge of componentEdges) {
          const [source, target] = graph.extremities(edge);
          const linksParentToChild =
            parentOf.get(target) === source || parentOf.get(source) === target;
          if (!linksParentToChild) continue;
          const child = parentOf.get(target) === source ? target : source;
          const list = linkEdges.get(child) || [];
          list.push(edge);
          linkEdges.set(child, list);
          const depth = Math.min(
            Number(depthOf.get(source) || 0),
            Number(depthOf.get(target) || 0),
          );
          baseGap[depth] = Math.max(baseGap[depth], edgeSpan(edge));
        }
        for (let depth = 0; depth <= maxDepth; depth += 1) {
          if (baseGap[depth] <= 0) baseGap[depth] = 96;
        }

        let positions = new Map<string, LayoutPoint>();
        let scale = 1;
        let usedSteps = 0;
        // Sideways room a tag needs in its sector. A tag lies along its edge,
        // and an edge is only roughly radial, so the tag's long side leans
        // partly across the sector. How far it leans depends on where the
        // child was placed, which depends on this number -- so start from the
        // tag's depth alone and let the loop below raise it once it can
        // measure the real lean. It only ever rises, so this settles.
        const tagSpread = new Map<string, number>();
        for (let step = 0; step < MAX_SOLVE_STEPS; step += 1) {
          usedSteps = step + 1;
          const ring = [0];
          for (let depth = 0; depth <= maxDepth; depth += 1) {
            ring.push(ring[depth] + baseGap[depth] * scale);
          }

          // How much angle each subtree needs, bottom up. Angles measured
          // from the origin are additive across depths, so a child's demand
          // can be summed into its parent's directly.
          const halfAngle = (extent: number, atRadius: number) =>
            atRadius <= 0 ? Math.PI : Math.asin(Math.min(1, extent / atRadius));
          const demand = new Map<string, number>();
          for (let index = order.length - 1; index >= 0; index -= 1) {
            const id = order[index];
            const depth = Number(depthOf.get(id) || 0);
            let own =
              2 * halfAngle(nodeRadius(id) + NODE_CLEARANCE / 2, ring[depth]);
            // The tag on the link down from the parent lives in this same
            // sector, so the sector has to be wide enough to hold it too.
            if (depth > 0) {
              const tagAt = (ring[depth - 1] + ring[depth]) / 2;
              // A tag rides the middle of its edge, and the middle of a chord
              // sits nearer the parent than the child does. So sibling tags
              // are squeezed together by exactly
              //     ringChild / (ringParent + ringChild)
              // relative to the angle their children were given -- a factor
              // of about a half once past the root. Budgeting the tag as if
              // it sat centred in its child's sector under-allocates by that
              // same factor, which is what let tags pile up around a hub.
              // Divide it back out.
              const crowding =
                (ring[depth - 1] + ring[depth]) / (ring[depth] || 1);
              for (const edge of linkEdges.get(id) || []) {
                const spread = Math.max(
                  tagSpread.get(edge) || 0,
                  tagHalfDepth(edge),
                );
                own = Math.max(
                  own,
                  2 * halfAngle(spread + TAG_CLEARANCE / 2, tagAt) * crowding,
                );
              }
            }
            let childSum = 0;
            for (const child of childrenOf.get(id) || []) {
              childSum += demand.get(child) || 0;
            }
            demand.set(id, Math.max(own, childSum));
          }

          // Does every fan fit the span it is allowed?
          let overflow = 1;
          for (const id of order) {
            const children = childrenOf.get(id) || [];
            if (!children.length) continue;
            let childSum = 0;
            for (const child of children) childSum += demand.get(child) || 0;
            const cap = id === root ? Math.PI * 2 : OUTWARD_SPAN;
            if (childSum > cap) {
              overflow = Math.max(overflow, childSum / cap);
            }
          }
          if (overflow > 1) {
            scale *= overflow * 1.02;
            continue;
          }

          // Place, top down. Each node receives a sector; its children
          // divide that sector in proportion to their demand, centred on
          // the direction that points away from the graph's centre. That
          // last part is what makes a branch fan outward instead of
          // curling back towards its grandparent.
          positions = new Map<string, LayoutPoint>([[root, { x: 0, y: 0 }]]);
          const sectorOf = new Map<string, { center: number; width: number }>([
            [root, { center: -Math.PI / 2, width: Math.PI * 2 }],
          ]);
          for (const id of order) {
            const children = childrenOf.get(id) || [];
            if (!children.length) continue;
            const sector = sectorOf.get(id) as {
              center: number;
              width: number;
            };
            const depth = Number(depthOf.get(id) || 0);
            let childSum = 0;
            for (const child of children) childSum += demand.get(child) || 0;
            const available =
              id === root ? Math.PI * 2 : Math.min(OUTWARD_SPAN, sector.width);
            // Spreading to fill the available span only ever adds
            // clearance, never removes it.
            const stretch = childSum > 0 ? available / childSum : 1;
            let cursor = sector.center - available / 2;
            for (const child of children) {
              const width = (demand.get(child) || 0) * stretch;
              const center = cursor + width / 2;
              cursor += width;
              sectorOf.set(child, { center, width });
              positions.set(child, {
                x: Math.cos(center) * ring[depth + 1],
                y: Math.sin(center) * ring[depth + 1],
              });
            }
          }

          // Now that the children are placed, measure how far each tag
          // really leans across its sector and feed that back. Widening is
          // monotone, so this reaches a fixed point rather than oscillating.
          let widened = false;
          for (const edge of componentEdges) {
            const [source, target] = graph.extremities(edge);
            if (source === target) continue;
            const start = positions.get(source);
            const end = positions.get(target);
            if (!start || !end) continue;
            const child = parentOf.get(target) === source ? target : source;
            const length = Math.hypot(end.x - start.x, end.y - start.y) || 1;
            const along = {
              x: (end.x - start.x) / length,
              y: (end.y - start.y) / length,
            };
            const centre = {
              x: (start.x + end.x) / 2,
              y: (start.y + end.y) / 2,
            };
            const reach = Math.hypot(centre.x, centre.y) || 1;
            const radial = { x: centre.x / reach, y: centre.y / reach };
            const lean = Math.abs(radial.x * along.y - radial.y * along.x);
            const face = Math.abs(radial.x * along.x + radial.y * along.y);
            const needed =
              tagHalfLength(edge) * lean + tagHalfDepth(edge) * face;
            const key = childrenOf.has(child) ? edge : edge;
            if (needed > (tagSpread.get(key) || 0) + 0.5) {
              tagSpread.set(key, needed);
              widened = true;
            }
          }
          if (widened && step < MAX_SOLVE_STEPS - 1) continue;

          // Nodes are laid out first. Only after the edge tags have their
          // real rectangles do we relieve a tag that covers a node. Move the
          // offending node and its whole subtree by the same small
          // translation, then rebuild the tag rectangles before deciding
          // whether anything else needs to move. Internal subtree edges keep
          // their exact lengths; only the link back to the parent grows.
          for (let correction = 0; correction < 256; correction += 1) {
            const tags = tagBoxesFor(componentEdges, positions);
            let mover: string | null = null;
            for (const tag of tags) {
              const [tagSource, tagTarget] = graph.extremities(tag.edge);
              for (const id of component) {
                const point = positions.get(id);
                if (!point) continue;
                if (
                  Math.hypot(point.x - tag.centre.x, point.y - tag.centre.y) >
                  tag.halfLength + nodeRadius(id) + TAG_ENDPOINT_CLEARANCE
                )
                  continue;
                const clearance =
                  id === tagSource || id === tagTarget
                    ? TAG_ENDPOINT_CLEARANCE
                    : TAG_CLEARANCE;
                if (
                  rectClearsCircle(
                    tag.centre,
                    tag.halfLength,
                    tag.halfDepth,
                    tag.along,
                    point,
                    nodeRadius(id) + clearance,
                  )
                )
                  continue;

                // Moving an ancestor of both tag endpoints would carry the
                // tag and node together and change nothing. In that case
                // lengthen the tag's own tree edge instead.
                const carriesWholeTag =
                  isInSubtree(id, tagSource) && isInSubtree(id, tagTarget);
                mover =
                  id !== root && !carriesWholeTag ? id : tagMover(tag.edge);
                if (mover === root) mover = null;
                break;
              }
              if (mover) break;
            }
            if (!mover) break;
            const parent = parentOf.get(mover);
            const point = positions.get(mover);
            const parentPoint = parent ? positions.get(parent) : null;
            if (!point || !parentPoint) {
              layoutVerified = false;
              break;
            }
            const dx = point.x - parentPoint.x;
            const dy = point.y - parentPoint.y;
            const length = Math.hypot(dx, dy) || 1;
            const shiftX = (dx / length) * 4;
            const shiftY = (dy / length) * 4;
            for (const member of descendantsOf(mover)) {
              const memberPoint = positions.get(member);
              if (!memberPoint) continue;
              positions.set(member, {
                x: memberPoint.x + shiftX,
                y: memberPoint.y + shiftY,
              });
            }
            if (correction === 255) layoutVerified = false;
          }

          // The construction above is the whole of the sizing. If the sweep
          // still finds something, that is a bug to fix here, not something
          // to escape by making the component bigger.
          layoutVerified =
            verifyLayout(component, componentEdges, positions) &&
            layoutVerified;
          break;
        }
        solveScale = Math.max(solveScale, scale);
        solveSteps = Math.max(solveSteps, usedSteps);

        // Anchor each self-loop. The loop is drawn as a circle tangent to its
        // node, reaching out to the anchor, with the tag riding the far side
        // of it -- so the anchor is where the tag will sit. Sweep directions
        // at a growing radius and keep the first that clears everything
        // already placed. This always terminates: past the component's own
        // extent there is nothing left to hit.
        const loopAnchors = new Map<string, LayoutPoint>();
        for (const [id, loops] of selfLoops) {
          const centre = positions.get(id);
          if (!centre) continue;
          loops.forEach((edge, loopIndex) => {
            const radius = tagHalfLength(edge);
            // Far enough out that the loop can clear its own node.
            const minimumReach =
              nodeRadius(id) + 2 * ((radius + TAG_STANDOFF) / 2);
            let placed: LayoutPoint | null = null;
            for (let ringStep = 0; ringStep < 24 && !placed; ringStep += 1) {
              const reach = minimumReach * (1 + ringStep * 0.18);
              const loopRadius = (reach - nodeRadius(id)) / 2;
              for (let spoke = 0; spoke < 36 && !placed; spoke += 1) {
                // Alternate either side of straight up so the loop lands as
                // close to a natural reading position as the space allows.
                const turn =
                  ((spoke % 2 ? 1 : -1) * Math.ceil(spoke / 2) * Math.PI) / 18;
                const angle = -Math.PI / 2 + turn + loopIndex * 0.6;
                const anchor = {
                  x: centre.x + Math.cos(angle) * reach,
                  y: centre.y + Math.sin(angle) * reach,
                };
                const loopCentre = {
                  x: (centre.x + anchor.x) / 2,
                  y: (centre.y + anchor.y) / 2,
                };
                let clear = true;
                for (const [otherId, point] of positions) {
                  if (otherId === id) continue;
                  const keepOff = nodeRadius(otherId) + NODE_CLEARANCE;
                  if (
                    Math.hypot(point.x - anchor.x, point.y - anchor.y) <
                      radius + keepOff ||
                    Math.hypot(point.x - loopCentre.x, point.y - loopCentre.y) <
                      loopRadius + keepOff
                  ) {
                    clear = false;
                    break;
                  }
                }
                if (clear) {
                  for (const other of componentEdges) {
                    const [otherSource, otherTarget] = graph.extremities(other);
                    if (otherSource === otherTarget) continue;
                    const start = positions.get(otherSource);
                    const end = positions.get(otherTarget);
                    if (!start || !end) continue;
                    if (
                      distanceToSegment(anchor, start, end) <
                        radius + TAG_CLEARANCE ||
                      distanceToSegment(loopCentre, start, end) <
                        loopRadius + TAG_CLEARANCE
                    ) {
                      clear = false;
                      break;
                    }
                    const otherTag = {
                      x: (start.x + end.x) / 2,
                      y: (start.y + end.y) / 2,
                    };
                    if (
                      Math.hypot(otherTag.x - anchor.x, otherTag.y - anchor.y) <
                      radius + tagHalfLength(other) + TAG_CLEARANCE
                    ) {
                      clear = false;
                      break;
                    }
                  }
                }
                if (clear) placed = anchor;
              }
            }
            if (placed) loopAnchors.set(edge, placed);
          });
        }

        let minX = Number.POSITIVE_INFINITY;
        let minY = Number.POSITIVE_INFINITY;
        let maxX = Number.NEGATIVE_INFINITY;
        let maxY = Number.NEGATIVE_INFINITY;
        const stretchBounds = (point: LayoutPoint, radius: number) => {
          minX = Math.min(minX, point.x - radius);
          minY = Math.min(minY, point.y - radius);
          maxX = Math.max(maxX, point.x + radius);
          maxY = Math.max(maxY, point.y + radius);
        };
        for (const [id, point] of positions) {
          stretchBounds(point, nodeRadius(id) + NODE_CLEARANCE);
        }
        for (const edge of componentEdges) {
          const [source, target] = graph.extremities(edge);
          const start = positions.get(source);
          const end = positions.get(target);
          if (!start || !end) continue;
          const anchor = loopAnchors.get(edge);
          stretchBounds(
            anchor || { x: (start.x + end.x) / 2, y: (start.y + end.y) / 2 },
            tagHalfLength(edge) + TAG_CLEARANCE,
          );
        }
        return {
          root,
          positions,
          loopAnchors,
          minX,
          minY,
          maxX,
          maxY,
          width: Math.max(1, maxX - minX),
          height: Math.max(1, maxY - minY),
        };
      };

      const componentLayouts = components.map(layoutComponent);
      // Diagnostics: how far the sizing solve had to push beyond its first,
      // fully constructed attempt. A scale of 1 means the construction alone
      // was enough, which is what should normally happen.
      graph.setAttribute("layoutSolveScale", solveScale);
      graph.setAttribute("layoutVerified", layoutVerified);
      graph.setAttribute("layoutSolveSteps", solveSteps);

      // Edges are straight, and the tag sits dead centre on every one.
      for (const edge of graph.edges()) {
        graph.setEdgeAttribute(edge, "layoutCurve", 0);
        graph.setEdgeAttribute(edge, "labelFraction", 0.5);
        graph.setEdgeAttribute(edge, "layoutLabelRadius", tagHalfLength(edge));
      }

      // Shelf-pack the components. Boxes are disjoint by construction, so
      // nothing one component contains can reach another.
      const targetWidth = Math.max(
        ...componentLayouts.map((layout) => layout.width),
        Math.sqrt(
          componentLayouts.reduce(
            (total, layout) =>
              total +
              (layout.width + COMPONENT_GAP) * (layout.height + COMPONENT_GAP),
            0,
          ) * 1.8,
        ),
      );
      let cursorX = 0;
      let cursorY = 0;
      let rowHeight = 0;
      let packedWidth = 0;
      for (const layout of componentLayouts) {
        if (cursorX > 0 && cursorX + layout.width > targetWidth) {
          cursorX = 0;
          cursorY += rowHeight + COMPONENT_GAP;
          rowHeight = 0;
        }
        const offsetX = cursorX - layout.minX;
        const offsetY = cursorY - layout.minY;
        for (const [id, point] of layout.positions) {
          graph.mergeNodeAttributes(id, {
            x: point.x + offsetX,
            y: point.y + offsetY,
            color: roleColor(id),
          });
        }
        for (const [edge, anchor] of layout.loopAnchors) {
          graph.setEdgeAttribute(edge, "loopAnchorX", anchor.x + offsetX);
          graph.setEdgeAttribute(edge, "loopAnchorY", anchor.y + offsetY);
        }
        cursorX += layout.width + COMPONENT_GAP;
        rowHeight = Math.max(rowHeight, layout.height);
        packedWidth = Math.max(packedWidth, cursorX - COMPONENT_GAP);
      }
      const packedHeight = cursorY + rowHeight;
      graph.updateEachNodeAttributes((_node, attributes) => ({
        ...attributes,
        x: Number(attributes.x) - packedWidth / 2,
        y: Number(attributes.y) - packedHeight / 2,
      }));
      // Loop anchors are absolute positions, so they take the same shift.
      for (const edge of graph.edges()) {
        if (!graph.hasEdgeAttribute(edge, "loopAnchorX")) continue;
        graph.setEdgeAttribute(
          edge,
          "loopAnchorX",
          Number(graph.getEdgeAttribute(edge, "loopAnchorX")) - packedWidth / 2,
        );
        graph.setEdgeAttribute(
          edge,
          "loopAnchorY",
          Number(graph.getEdgeAttribute(edge, "loopAnchorY")) -
            packedHeight / 2,
        );
      }

      const primary = componentLayouts[0];
      if (primary) {
        graph.setAttribute("readableStartNode", primary.root);
        const attributes = graph.getNodeAttributes(primary.root);
        graph.setAttribute("readableStartX", Number(attributes.x));
        graph.setAttribute("readableStartY", Number(attributes.y));
      }
      return;
    }

    if (kind === "radial") {
      const focusId = payload.focus.id;
      const neighbors = payload.nodes
        .map((node) => node[0])
        .filter((id) => id !== focusId);
      graph.mergeNodeAttributes(focusId, { x: 0, y: 0, fixed: true });
      let cursor = 0;
      let ring = 0;
      while (cursor < neighbors.length) {
        const capacity = 14 + ring * 10;
        const ringNodes = neighbors.slice(cursor, cursor + capacity);
        const radius = 15 + ring * 13;
        ringNodes.forEach((id, index) => {
          const angle = (Math.PI * 2 * index) / ringNodes.length - Math.PI / 2;
          graph.mergeNodeAttributes(id, {
            x: Math.cos(angle) * radius,
            y: Math.sin(angle) * radius,
          });
        });
        cursor += ringNodes.length;
        ring += 1;
      }
      return;
    }

    if (kind === "bipartite") {
      const sources = new Set(
        payload.edges.map((edge) => payload.nodes[edge[0]][0]),
      );
      const targets = new Set(
        payload.edges.map((edge) => payload.nodes[edge[1]][0]),
      );
      const columns: string[][] = [[], [], []];
      for (const node of payload.nodes) {
        const isSource = sources.has(node[0]);
        const isTarget = targets.has(node[0]);
        columns[isSource && isTarget ? 1 : isSource ? 0 : 2].push(node[0]);
      }
      const placeSide = (ids: string[], direction: -1 | 1, color: string) => {
        const rows = Math.min(
          34,
          Math.max(1, Math.ceil(Math.sqrt(ids.length * 4))),
        );
        ids.forEach((id, index) => {
          const column = Math.floor(index / rows);
          const row = index % rows;
          const rowsInColumn = Math.min(rows, ids.length - column * rows);
          graph.mergeNodeAttributes(id, {
            x: direction * (24 + column * 11),
            y: (row - (rowsInColumn - 1) / 2) * 4.2,
            color,
          });
        });
      };
      placeSide(columns[0], -1, "#2c718f");
      placeSide(columns[2], 1, "#0f8a74");
      const centerRows = Math.min(24, Math.max(1, columns[1].length));
      columns[1].forEach((id, index) => {
        const column = Math.floor(index / centerRows);
        const row = index % centerRows;
        graph.mergeNodeAttributes(id, {
          x: (column - Math.floor(columns[1].length / centerRows) / 2) * 8,
          y: (row - (centerRows - 1) / 2) * 4.2,
          color: "#75419a",
        });
      });
      return;
    }

    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    payload.nodes.forEach((node, index) => {
      const radius = 1.35 * Math.sqrt(index + 1);
      graph.mergeNodeAttributes(node[0], {
        x: Math.cos(index * goldenAngle) * radius,
        y: Math.sin(index * goldenAngle) * radius,
        fixed: node[0] === payload.focus.id,
      });
    });
  },

  makeGraph(
    this: GraphController,
    payload: TopologyPayload,
  ): MultiDirectedGraph {
    const graph = new MultiDirectedGraph();
    const isAtlas = payload.layout.kind === "atlas";
    const usesEntityBubbles =
      payload.layout.kind === "triples" || payload.layout.kind === "claim";
    graph.setAttribute(
      "showAllLabels",
      payload.layout.kind === "triples" || payload.layout.kind === "claim",
    );
    graph.setAttribute("customRenderedEdges", usesEntityBubbles);
    graph.setAttribute("uniformScaling", true);
    graph.setAttribute("atlas", isAtlas);
    graph.setAttribute("parallelEdgeGap", 48);
    graph.setAttribute("progressiveEdgeLabels", false);
    graph.setAttribute("allowEdgeUnderLabels", false);
    graph.setAttribute("collisionAudit", usesEntityBubbles);
    graph.setAttribute("centerWholeGraph", false);
    if (isAtlas) {
      const ranked = payload.nodes
        .slice()
        .sort(
          (left, right) =>
            Number(right[7] || 0) - Number(left[7] || 0) ||
            left[0].localeCompare(right[0]),
        );
      this.atlasOverviewLabels = new Set(
        ranked.slice(0, 15).map((node) => node[0]),
      );
      this.atlasPredicateEdges = new Set(
        payload.edges
          .map((edge, index) => ({ edge, index }))
          .sort(
            (left, right) =>
              Number(right.edge[4]) - Number(left.edge[4]) ||
              left.edge[3].localeCompare(right.edge[3]) ||
              left.index - right.index,
          )
          .slice(0, 500)
          .map(({ index }) => `edge-${index}`),
      );
    }

    payload.nodes.forEach((node) => {
      const isFocus = node[0] === payload.focus.id;
      const communityIndex = Number(node[6] || 0);
      const componentIndex = Number(node[5] || 0);
      const componentSize = Number(
        payload.components?.[componentIndex]?.[2] || 0,
      );
      graph.addNode(node[0], {
        label: node[1],
        frequency: node[2],
        x: node[3] ?? 0,
        y: node[4] ?? 0,
        size:
          (usesEntityBubbles ? bubbleNodeSize(node[1]) : nodeSize(node[2])) +
          (isFocus ? 1.5 : 0),
        color: isFocus
          ? "#75419a"
          : isAtlas
            ? atlasNodeColor(communityIndex)
            : "#087f75",
        forceLabel: isFocus,
        highlighted: isFocus,
        zIndex: isFocus ? 2 : 1,
        atlasNode: isAtlas,
        componentIndex,
        communityIndex,
        atlasPriority: Number(node[7] || 0),
        atlasSmallComponent: componentSize <= 16,
        type: isAtlas ? "point" : "circle",
      });
    });

    payload.edges.forEach((edge, index) => {
      const sourceId = payload.nodes[edge[0]][0];
      const targetId = payload.nodes[edge[1]][0];
      const sourceCommunity = Number(payload.nodes[edge[0]][6] || 0);
      const targetCommunity = Number(payload.nodes[edge[1]][6] || 0);
      const componentIndex = Number(payload.nodes[edge[0]][5] || 0);
      const componentSize = Number(
        payload.components?.[componentIndex]?.[2] || 0,
      );
      graph.addDirectedEdgeWithKey(`edge-${index}`, sourceId, targetId, {
        label: null,
        relationLabel: edge[3],
        relationId: edge[2],
        occurrenceCount: edge[4],
        sourceId,
        targetId,
        size: usesEntityBubbles
          ? Math.min(4, 1.3 + Math.log2(edge[4] + 1) * 0.35)
          : Math.min(4, 0.5 + Math.log2(edge[4] + 1) * 0.55),
        color: "#8761a8",
        type: "arrow",
        forceLabel: false,
        zIndex: 1,
        customRendered: usesEntityBubbles,
        atlasRawEdge: isAtlas,
        atlasInternal: sourceCommunity === targetCommunity,
        atlasSmallComponent: componentSize <= 16,
        bundleIndex: Number(edge[5] ?? -1),
      });
    });

    if (isAtlas) {
      (payload.bundles || []).forEach((bundle, index) => {
        if (!bundle[5]) return;
        const sourceAnchor = `atlas-bundle-source-${index}`;
        const targetAnchor = `atlas-bundle-target-${index}`;
        graph.addNode(sourceAnchor, {
          label: "",
          x: bundle[6],
          y: bundle[7],
          size: 0.01,
          color: "#ffffff",
          zIndex: 0,
          atlasAnchor: true,
          type: "point",
        });
        graph.addNode(targetAnchor, {
          label: "",
          x: bundle[8],
          y: bundle[9],
          size: 0.01,
          color: "#ffffff",
          zIndex: 0,
          atlasAnchor: true,
          type: "point",
        });
        graph.addDirectedEdgeWithKey(
          `atlas-bundle-${index}`,
          sourceAnchor,
          targetAnchor,
          {
            label: null,
            relationLabel: bundle[4],
            relationId: "",
            occurrenceCount: bundle[3],
            sourceId: sourceAnchor,
            targetId: targetAnchor,
            size: Math.min(3, 0.45 + Math.log2(bundle[3] + 1) * 0.32),
            color: "#a891bc",
            type: "line",
            forceLabel: false,
            zIndex: 0,
            atlasBundle: true,
            atlasOverviewBundle: bundle[5],
          },
        );
      });
    }

    this.assignLayout(graph, payload);
    return graph;
  },

  prepareReadableLayout(
    this: GraphController,
    graph: MultiDirectedGraph,
  ): void {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const node of graph.nodes()) {
      const x = Number(graph.getNodeAttribute(node, "x"));
      const y = Number(graph.getNodeAttribute(node, "y"));
      xs.push(x);
      ys.push(y);
    }
    if (!xs.length) return;
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    this.readableLayoutCenter = {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
    };
    this.graphBBox = {
      x: [minX, maxX],
      y: [minY, maxY],
    };
    const fullWidth = Math.max(1, maxX - minX);
    const fullHeight = Math.max(1, maxY - minY);
    const wholeGraphFits =
      graph.order <= 60 || (fullWidth <= 120 && fullHeight <= 80);
    graph.setAttribute("centerWholeGraph", wholeGraphFits);
    // The custom bbox has to keep the same aspect ratio as the container.
    // Sigma normalises the bbox by its longest side and then rescales by the
    // viewport's shortest side, so matching the two aspect ratios is what
    // makes one graph unit render as exactly one pixel at camera ratio 1.
    // With uniformScaling that equality then holds at every zoom, which is
    // what lets the layout below reason in pixels: node sizes and tag
    // metrics are pixel quantities, and the clearances the layout computes
    // between them have to survive the trip to the screen unchanged. Any
    // other aspect ratio shrinks graph distances by up to 15% relative to
    // node sizes and silently reintroduces overlap.
    // Sigma rescales by the viewport's shortest side less twice the 28px
    // stage padding, so the bbox side matching that shortest side is what
    // has to absorb the padding; the other side follows the aspect ratio.
    const stageWidth = Math.max(320, el.container.clientWidth);
    const stageHeight = Math.max(240, el.container.clientHeight);
    const stageAspect = stageWidth / stageHeight;
    const viewHeight =
      stageAspect >= 1 ? stageHeight - 56 : (stageWidth - 56) / stageAspect;
    const viewWidth = viewHeight * stageAspect;
    const viewCenter = wholeGraphFits
      ? this.readableLayoutCenter
      : {
          x: Number(
            graph.getAttribute("readableStartX") ?? this.readableLayoutCenter.x,
          ),
          y: Number(
            graph.getAttribute("readableStartY") ?? this.readableLayoutCenter.y,
          ),
        };
    this.fixedBBox = {
      x: [viewCenter.x - viewWidth / 2, viewCenter.x + viewWidth / 2],
      y: [viewCenter.y - viewHeight / 2, viewCenter.y + viewHeight / 2],
    };
    graph.setAttribute("geometryRevision", 0);
  },

  refreshReadableBounds(this: GraphController): void {
    if (!this.graph) return;
    const xs: number[] = [];
    const ys: number[] = [];
    for (const node of this.graph.nodes()) {
      xs.push(Number(this.graph.getNodeAttribute(node, "x")));
      ys.push(Number(this.graph.getNodeAttribute(node, "y")));
    }
    if (!xs.length) return;
    this.graphBBox = {
      x: [Math.min(...xs), Math.max(...xs)],
      y: [Math.min(...ys), Math.max(...ys)],
    };
  },

  // The layout is overlap-free by construction, so there is nothing left to
  // resolve once the first frame has been drawn: reveal the graph and record
  // the audit that the collision test reads back.
  revealReadableLayout(this: GraphController): void {
    if (!this.readableLayoutPending || !this.graph || !this.renderer) return;
    this.readableLayoutPending = false;
    this.refreshReadableBounds();
    this.graph.setAttribute("hardCollisionFree", true);
    if (this.labelCanvas) {
      this.labelCanvas.dataset.hardCollisionFree = "true";
      this.labelCanvas.dataset.hardCollisionPasses = "0";
    }
    el.container.style.visibility = "";
    this.setLoading(false);
    window.requestAnimationFrame(() => {
      if (!this.renderer || !this.graph) return;
      if (Boolean(this.graph.getAttribute("centerWholeGraph"))) {
        this.fitGraph(false);
      } else {
        this.recenterReadableView();
      }
    });
  },
};
