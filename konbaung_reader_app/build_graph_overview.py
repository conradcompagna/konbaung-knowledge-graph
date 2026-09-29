from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

try:
    from .graph_schema import (
        EMBEDDING_ROOT,
        GRAPH_MANIFEST,
        GRAPH_ROOT,
        OCCURRENCES_PATH,
        read_jsonl,
    )
except ImportError:
    from graph_schema import (
        EMBEDDING_ROOT,
        GRAPH_MANIFEST,
        GRAPH_ROOT,
        OCCURRENCES_PATH,
        read_jsonl,
    )


OVERVIEW_PATH = GRAPH_ROOT / "overview.json"
ATLAS_SCHEMA_VERSION = 3
ATLAS_SCOPES = ("corpus", "vol1", "vol2", "vol3")
SCOPE_LABELS = {
    "corpus": "Full corpus",
    "vol1": "Volume I",
    "vol2": "Volume II",
    "vol3": "Volume III",
}
LEAF_COMMUNITY_SIZE = 180
NODE_GAP = 7.0
COMMUNITY_GAP = 34.0
COMPONENT_GAP = 140.0
SPRING_ITERATIONS = 70


def bubble_radius(label: str) -> float:
    characters_per_line = 10
    lines = math.ceil(len(label) / characters_per_line)
    text_width = min(characters_per_line, len(label)) * 4.3
    text_height = lines * 8.5
    return float(
        min(
            36,
            max(
                23,
                math.ceil(math.hypot(text_width / 2, text_height / 2) + 6),
            ),
        )
    )


def _weighted_graph(
    node_ids: list[str],
    edges: list[dict[str, Any]],
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        if source == target:
            continue
        weight = 1.0 + math.log2(int(edge["occurrenceCount"]) + 1)
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
        else:
            graph.add_edge(source, target, weight=weight)
    return graph


def _fallback_chunks(
    graph: nx.Graph,
    node_ids: set[str],
) -> list[set[str]]:
    remaining = set(node_ids)
    chunks: list[set[str]] = []
    while remaining:
        seed = min(
            remaining,
            key=lambda node_id: (-graph.degree(node_id), node_id),
        )
        queue = [seed]
        remaining.remove(seed)
        chunk: set[str] = set()
        while queue and len(chunk) < LEAF_COMMUNITY_SIZE:
            current = queue.pop(0)
            chunk.add(current)
            neighbors = sorted(
                (neighbor for neighbor in graph.neighbors(current) if neighbor in remaining),
                key=lambda node_id: (-graph.degree(node_id), node_id),
            )
            for neighbor in neighbors:
                if len(chunk) + len(queue) >= LEAF_COMMUNITY_SIZE:
                    break
                remaining.remove(neighbor)
                queue.append(neighbor)
        if len(chunk) < LEAF_COMMUNITY_SIZE and remaining:
            fillers = sorted(
                remaining,
                key=lambda node_id: (-graph.degree(node_id), node_id),
            )[: LEAF_COMMUNITY_SIZE - len(chunk)]
            for filler in fillers:
                remaining.remove(filler)
                chunk.add(filler)
        chunks.append(chunk)
    return chunks


def _partition_component(
    graph: nx.Graph,
    node_ids: set[str],
    *,
    depth: int = 0,
) -> list[set[str]]:
    if len(node_ids) <= LEAF_COMMUNITY_SIZE:
        return [node_ids]
    return _fallback_chunks(graph, node_ids)


def _hub_ring_positions(
    graph: nx.Graph,
    node_ids: list[str],
    radii: dict[str, float],
    frequencies: dict[str, int],
) -> dict[str, np.ndarray]:
    hub = min(
        node_ids,
        key=lambda node_id: (
            -graph.degree(node_id),
            -frequencies[node_id],
            node_id,
        ),
    )
    positions = {hub: np.asarray([0.0, 0.0], dtype=np.float64)}
    others = sorted(
        (node_id for node_id in node_ids if node_id != hub),
        key=lambda node_id: (
            -graph.degree(node_id),
            -frequencies[node_id],
            node_id,
        ),
    )
    if not others:
        return positions
    maximum_radius = max(radii[node_id] for node_id in node_ids)
    ring_radius = radii[hub] + maximum_radius + NODE_GAP
    cursor = 0
    ring = 0
    while cursor < len(others):
        circumference = math.tau * ring_radius
        capacity = max(
            6,
            int(circumference // (maximum_radius * 2 + NODE_GAP)),
        )
        ring_nodes = others[cursor : cursor + capacity]
        for index, node_id in enumerate(ring_nodes):
            angle = math.tau * index / len(ring_nodes) - math.pi / 2
            positions[node_id] = np.asarray(
                [
                    math.cos(angle) * ring_radius,
                    math.sin(angle) * ring_radius,
                ],
                dtype=np.float64,
            )
        cursor += len(ring_nodes)
        ring += 1
        ring_radius += maximum_radius * 2 + NODE_GAP + ring * 1.5
    return positions


def _clearance_scale(
    positions: np.ndarray,
    radii: np.ndarray,
    gap: float,
) -> float:
    if len(positions) < 2:
        return 1.0
    maximum_distance = float(np.max(radii) * 2 + gap)
    pairs = cKDTree(positions).query_pairs(
        maximum_distance,
        output_type="ndarray",
    )
    if not len(pairs):
        return 1.0
    differences = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    distances = np.sqrt(np.einsum("ij,ij->i", differences, differences))
    minimum_distances = radii[pairs[:, 0]] + radii[pairs[:, 1]] + gap
    valid = distances > 0.000001
    if not np.all(valid):
        for pair_index in np.flatnonzero(~valid):
            left, right = pairs[pair_index]
            angle = ((left * 37 + right * 101) % 360) * math.pi / 180
            positions[right] += (
                math.cos(angle) * 0.01,
                math.sin(angle) * 0.01,
            )
        return _clearance_scale(positions, radii, gap)
    return max(1.0, float(np.max(minimum_distances / distances)) * 1.015)


def _layout_leaf(
    graph: nx.Graph,
    node_ids: list[str],
    radii: dict[str, float],
    frequencies: dict[str, int],
    seed: int,
) -> tuple[dict[str, np.ndarray], tuple[float, float, float, float]]:
    if len(node_ids) == 1:
        positions = {
            node_ids[0]: np.asarray([0.0, 0.0], dtype=np.float64),
        }
    else:
        maximum_degree = max(graph.degree(node_id) for node_id in node_ids)
        if maximum_degree >= max(8, int(len(node_ids) * 0.28)):
            positions = _hub_ring_positions(
                graph,
                node_ids,
                radii,
                frequencies,
            )
        else:
            node_set = set(node_ids)
            subgraph = nx.Graph()
            subgraph.add_nodes_from(node_ids)
            for source in node_ids:
                for target in sorted(graph.neighbors(source)):
                    if target not in node_set or source >= target:
                        continue
                    subgraph.add_edge(
                        source,
                        target,
                        **graph[source][target],
                    )
            spring = nx.spring_layout(
                subgraph,
                k=2.2 / math.sqrt(len(node_ids)),
                iterations=SPRING_ITERATIONS,
                weight="weight",
                scale=max(80.0, math.sqrt(len(node_ids)) * 62.0),
                seed=seed,
                method="force",
            )
            positions = {
                node_id: np.asarray(spring[node_id], dtype=np.float64) for node_id in node_ids
            }
    position_array = np.asarray([positions[node_id] for node_id in node_ids])
    radius_array = np.asarray([radii[node_id] for node_id in node_ids])
    position_array *= _clearance_scale(position_array, radius_array, NODE_GAP)
    position_array -= np.mean(position_array, axis=0)
    for node_id, position in zip(node_ids, position_array, strict=True):
        positions[node_id] = position
    minimum = np.min(position_array - radius_array[:, None], axis=0)
    maximum = np.max(position_array + radius_array[:, None], axis=0)
    return positions, (
        float(minimum[0]),
        float(minimum[1]),
        float(maximum[0]),
        float(maximum[1]),
    )


def _layout_component(
    graph: nx.Graph,
    component_ids: set[str],
    radii: dict[str, float],
    frequencies: dict[str, int],
    component_index: int,
) -> dict[str, Any]:
    leaves = _partition_component(graph, component_ids)
    leaves.sort(key=lambda value: (-len(value), min(value)))
    local_layouts: list[dict[str, Any]] = []
    node_to_leaf: dict[str, int] = {}
    for leaf_index, leaf in enumerate(leaves):
        node_ids = sorted(
            leaf,
            key=lambda node_id: (
                -graph.degree(node_id),
                -frequencies[node_id],
                node_id,
            ),
        )
        positions, bounds = _layout_leaf(
            graph,
            node_ids,
            radii,
            frequencies,
            seed=1009 + component_index * 997 + leaf_index,
        )
        for node_id in node_ids:
            node_to_leaf[node_id] = leaf_index
        local_layouts.append(
            {
                "ids": node_ids,
                "positions": positions,
                "bounds": bounds,
                "width": bounds[2] - bounds[0],
                "height": bounds[3] - bounds[1],
            }
        )

    if len(leaves) == 1:
        centers = np.zeros((1, 2), dtype=np.float64)
    else:
        quotient = nx.Graph()
        quotient.add_nodes_from(range(len(leaves)))
        for source, target, attributes in graph.subgraph(component_ids).edges(data=True):
            left = node_to_leaf[source]
            right = node_to_leaf[target]
            if left == right:
                continue
            weight = float(attributes.get("weight", 1.0))
            if quotient.has_edge(left, right):
                quotient[left][right]["weight"] += weight
            else:
                quotient.add_edge(left, right, weight=weight)
        weighted_degree = {
            node: sum(
                float(attributes.get("weight", 1.0))
                for _, _, attributes in quotient.edges(node, data=True)
            )
            for node in quotient
        }
        root = min(
            quotient,
            key=lambda node: (
                -weighted_degree[node],
                -len(local_layouts[node]["ids"]),
                node,
            ),
        )
        topology_order: list[int] = []
        remaining = set(quotient)
        queue = [root]
        remaining.remove(root)
        while queue:
            current = queue.pop(0)
            topology_order.append(current)
            neighbors = sorted(
                (neighbor for neighbor in quotient.neighbors(current) if neighbor in remaining),
                key=lambda neighbor: (
                    -float(quotient[current][neighbor].get("weight", 1.0)),
                    -weighted_degree[neighbor],
                    neighbor,
                ),
            )
            for neighbor in neighbors:
                remaining.remove(neighbor)
                queue.append(neighbor)
        topology_order.extend(sorted(remaining))

        if quotient.degree(root) >= max(8, int(len(leaves) * 0.28)):
            layout_centers = np.asarray(
                [
                    [
                        (layout["bounds"][0] + layout["bounds"][2]) / 2,
                        (layout["bounds"][1] + layout["bounds"][3]) / 2,
                    ]
                    for layout in local_layouts
                ],
                dtype=np.float64,
            )
            layout_radii = np.asarray(
                [math.hypot(layout["width"], layout["height"]) / 2 for layout in local_layouts],
                dtype=np.float64,
            )
            desired_centers = np.zeros(
                (len(local_layouts), 2),
                dtype=np.float64,
            )
            ring_order = [leaf_index for leaf_index in topology_order if leaf_index != root]
            cursor = 0
            previous_outer = float(layout_radii[root])
            while cursor < len(ring_order):
                maximum_remaining = max(float(layout_radii[index]) for index in ring_order[cursor:])
                ring_radius = previous_outer + maximum_remaining + COMMUNITY_GAP
                ring: list[int] = []
                spans: list[float] = []
                span_total = 0.0
                while cursor < len(ring_order):
                    leaf_index = ring_order[cursor]
                    span = 2 * math.asin(
                        min(
                            0.98,
                            (float(layout_radii[leaf_index]) + COMMUNITY_GAP / 2) / ring_radius,
                        )
                    )
                    if ring and span_total + span > math.pi * 1.75:
                        break
                    ring.append(leaf_index)
                    spans.append(span)
                    span_total += span
                    cursor += 1
                spacing = (math.tau - span_total) / len(ring)
                angle = -math.pi / 2
                for leaf_index, span in zip(ring, spans, strict=True):
                    angle += span / 2
                    desired_centers[leaf_index] = [
                        math.cos(angle) * ring_radius,
                        math.sin(angle) * ring_radius,
                    ]
                    angle += span / 2 + spacing
                previous_outer = ring_radius + max(float(layout_radii[index]) for index in ring)
            centers = desired_centers - layout_centers
        else:
            total_area = sum(
                (layout["width"] + COMMUNITY_GAP) * (layout["height"] + COMMUNITY_GAP)
                for layout in local_layouts
            )
            target_width = max(
                max(layout["width"] for layout in local_layouts),
                math.sqrt(total_area * 1.25),
            )
            offsets: list[np.ndarray | None] = [None] * len(local_layouts)
            cursor_x = 0.0
            cursor_y = 0.0
            row_height = 0.0
            packed_width = 0.0
            for leaf_index in topology_order:
                layout = local_layouts[leaf_index]
                width = float(layout["width"])
                height = float(layout["height"])
                if cursor_x and cursor_x + width > target_width:
                    cursor_x = 0.0
                    cursor_y += row_height + COMMUNITY_GAP
                    row_height = 0.0
                minimum_x, minimum_y, _, _ = layout["bounds"]
                offsets[leaf_index] = np.asarray(
                    [cursor_x - minimum_x, cursor_y - minimum_y],
                    dtype=np.float64,
                )
                cursor_x += width + COMMUNITY_GAP
                packed_width = max(
                    packed_width,
                    cursor_x - COMMUNITY_GAP,
                )
                row_height = max(row_height, height)
            packed_height = cursor_y + row_height
            packed_center = np.asarray(
                [packed_width / 2, packed_height / 2],
                dtype=np.float64,
            )
            centers = np.zeros((len(leaves), 2), dtype=np.float64)
            for leaf_index in range(len(local_layouts)):
                centers[leaf_index] = offsets[leaf_index] - packed_center

    positions: dict[str, np.ndarray] = {}
    communities: list[dict[str, Any]] = []
    for leaf_index, layout in enumerate(local_layouts):
        offset = centers[leaf_index]
        ids = layout["ids"]
        for node_id in ids:
            positions[node_id] = layout["positions"][node_id] + offset
        points = np.asarray([positions[node_id] for node_id in ids])
        radius_array = np.asarray([radii[node_id] for node_id in ids])
        minimum = np.min(points - radius_array[:, None], axis=0)
        maximum = np.max(points + radius_array[:, None], axis=0)
        label_nodes = sorted(
            ids,
            key=lambda node_id: (
                -frequencies[node_id],
                -graph.degree(node_id),
                node_id,
            ),
        )[:2]
        communities.append(
            {
                "labelNodes": label_nodes,
                "ids": ids,
                "bounds": (
                    float(minimum[0]),
                    float(minimum[1]),
                    float(maximum[0]),
                    float(maximum[1]),
                ),
                "center": (
                    float((minimum[0] + maximum[0]) / 2),
                    float((minimum[1] + maximum[1]) / 2),
                ),
            }
        )
    points = np.asarray([positions[node_id] for node_id in component_ids])
    radius_array = np.asarray([radii[node_id] for node_id in component_ids])
    minimum = np.min(points - radius_array[:, None], axis=0)
    maximum = np.max(points + radius_array[:, None], axis=0)
    return {
        "positions": positions,
        "communities": communities,
        "nodeToLeaf": node_to_leaf,
        "bounds": (
            float(minimum[0]),
            float(minimum[1]),
            float(maximum[0]),
            float(maximum[1]),
        ),
        "width": float(maximum[0] - minimum[0]),
        "height": float(maximum[1] - minimum[1]),
    }


def _pack_components(layouts: list[dict[str, Any]]) -> None:
    if not layouts:
        return

    def pack_group(
        group: list[dict[str, Any]],
    ) -> tuple[float, float]:
        total_area = sum(
            (layout["width"] + COMPONENT_GAP) * (layout["height"] + COMPONENT_GAP)
            for layout in group
        )
        target_width = max(
            max(layout["width"] for layout in group),
            math.sqrt(total_area * 1.25),
        )
        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0
        packed_width = 0.0
        for layout in group:
            width = float(layout["width"])
            height = float(layout["height"])
            if cursor_x and cursor_x + width > target_width:
                cursor_x = 0.0
                cursor_y += row_height + COMPONENT_GAP
                row_height = 0.0
            minimum_x, minimum_y, _, _ = layout["bounds"]
            layout["offset"] = (
                cursor_x - minimum_x,
                cursor_y - minimum_y,
            )
            cursor_x += width + COMPONENT_GAP
            packed_width = max(
                packed_width,
                cursor_x - COMPONENT_GAP,
            )
            row_height = max(row_height, height)
        return packed_width, cursor_y + row_height

    if len(layouts) == 1:
        minimum_x, minimum_y, maximum_x, maximum_y = layouts[0]["bounds"]
        layouts[0]["offset"] = (
            -(minimum_x + maximum_x) / 2,
            -(minimum_y + maximum_y) / 2,
        )
        return

    main = layouts[0]
    minimum_x, minimum_y, maximum_x, maximum_y = main["bounds"]
    main["offset"] = (
        -(minimum_x + maximum_x) / 2,
        -(minimum_y + maximum_y) / 2,
    )
    _, island_height = pack_group(layouts[1:])
    island_origin = np.asarray(
        [
            float(main["width"]) / 2 + COMPONENT_GAP,
            -island_height / 2,
        ],
        dtype=np.float64,
    )
    for layout in layouts[1:]:
        layout["offset"] = tuple(np.asarray(layout["offset"], dtype=np.float64) + island_origin)

    packed_bounds = []
    for layout in layouts:
        offset = np.asarray(layout["offset"], dtype=np.float64)
        minimum_x, minimum_y, maximum_x, maximum_y = layout["bounds"]
        packed_bounds.append(
            (
                minimum_x + offset[0],
                minimum_y + offset[1],
                maximum_x + offset[0],
                maximum_y + offset[1],
            )
        )
    overall_center = np.asarray(
        [
            (
                min(bounds[0] for bounds in packed_bounds)
                + max(bounds[2] for bounds in packed_bounds)
            )
            / 2,
            (
                min(bounds[1] for bounds in packed_bounds)
                + max(bounds[3] for bounds in packed_bounds)
            )
            / 2,
        ],
        dtype=np.float64,
    )
    for layout in layouts:
        layout["offset"] = tuple(np.asarray(layout["offset"], dtype=np.float64) - overall_center)


def _community_port(
    community: list[Any],
    target_x: float,
    target_y: float,
) -> tuple[float, float]:
    minimum_x = float(community[3])
    minimum_y = float(community[4])
    maximum_x = float(community[5])
    maximum_y = float(community[6])
    center_x = float(community[7])
    center_y = float(community[8])
    delta_x = target_x - center_x
    delta_y = target_y - center_y
    candidates: list[float] = []
    if delta_x > 0:
        candidates.append((maximum_x - center_x) / delta_x)
    elif delta_x < 0:
        candidates.append((minimum_x - center_x) / delta_x)
    if delta_y > 0:
        candidates.append((maximum_y - center_y) / delta_y)
    elif delta_y < 0:
        candidates.append((minimum_y - center_y) / delta_y)
    scale = min((value for value in candidates if value >= 0), default=0.0)
    return (
        center_x + delta_x * scale,
        center_y + delta_y * scale,
    )


def _build_source_records() -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, Counter[str]],
]:
    node_records = list(read_jsonl(EMBEDDING_ROOT / "node_records.jsonl"))
    relation_records = list(read_jsonl(EMBEDDING_ROOT / "edge_records.jsonl"))
    node_by_tag = {record["tag"]: record for record in node_records}
    relation_by_tag = {record["tag"]: record for record in relation_records}
    node_volumes: dict[str, Counter[str]] = {
        record["baseKey"]: Counter() for record in node_records
    }
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for occurrence in read_jsonl(OCCURRENCES_PATH):
        volume_id = occurrence["volumeId"]
        subject = node_by_tag[occurrence["subject"]]["baseKey"]
        relation = relation_by_tag[occurrence["predicate"]]["baseKey"]
        object_ = node_by_tag[occurrence["object"]]["baseKey"]
        node_volumes[subject][volume_id] += 1
        node_volumes[object_][volume_id] += 1
        key = (subject, relation, object_)
        edge = edges.setdefault(
            key,
            {
                "source": subject,
                "target": object_,
                "relationId": relation,
                "label": occurrence["predicate"],
                "occurrenceCount": 0,
                "volumes": Counter(),
            },
        )
        edge["occurrenceCount"] += 1
        edge["volumes"][volume_id] += 1
    return node_records, edges, node_volumes


def _scope_source(
    scope: str,
    node_records: list[dict[str, Any]],
    all_edges: dict[tuple[str, str, str], dict[str, Any]],
    node_volumes: dict[str, Counter[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if scope == "corpus":
        return node_records, list(all_edges.values())
    nodes = [record for record in node_records if scope in node_volumes[record["baseKey"]]]
    edges = [
        {
            **edge,
            "occurrenceCount": int(edge["volumes"][scope]),
        }
        for edge in all_edges.values()
        if scope in edge["volumes"]
    ]
    return nodes, edges


def _build_scope_atlas(
    scope: str,
    node_records: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    node_ids = [record["baseKey"] for record in node_records]
    labels = {record["baseKey"]: record["tag"] for record in node_records}
    frequencies = {record["baseKey"]: int(record["frequency"]) for record in node_records}
    radii = {node_id: bubble_radius(labels[node_id]) for node_id in node_ids}
    graph = _weighted_graph(node_ids, edges)
    components = [set(component) for component in nx.connected_components(graph)]
    components.sort(key=lambda value: (-len(value), min(value)))
    component_layouts = [
        _layout_component(
            graph,
            component,
            radii,
            frequencies,
            component_index,
        )
        for component_index, component in enumerate(components)
    ]
    _pack_components(component_layouts)

    positions: dict[str, tuple[float, float]] = {}
    node_component: dict[str, int] = {}
    node_community: dict[str, int] = {}
    component_rows: list[list[Any]] = []
    community_rows: list[list[Any]] = []
    next_community_index = 0
    for component_index, (component, layout) in enumerate(
        zip(components, component_layouts, strict=True)
    ):
        offset = np.asarray(layout["offset"], dtype=np.float64)
        component_community_start = next_community_index
        for local_index, community in enumerate(layout["communities"]):
            global_index = next_community_index + local_index
            for node_id in community["ids"]:
                node_community[node_id] = global_index
            minimum_x, minimum_y, maximum_x, maximum_y = community["bounds"]
            center_x, center_y = community["center"]
            community_rows.append(
                [
                    component_index,
                    " / ".join(labels[node_id] for node_id in community["labelNodes"]),
                    len(community["ids"]),
                    round(minimum_x + offset[0], 4),
                    round(minimum_y + offset[1], 4),
                    round(maximum_x + offset[0], 4),
                    round(maximum_y + offset[1], 4),
                    round(center_x + offset[0], 4),
                    round(center_y + offset[1], 4),
                ]
            )
        next_community_index += len(layout["communities"])
        for node_id in component:
            point = layout["positions"][node_id] + offset
            positions[node_id] = (float(point[0]), float(point[1]))
            node_component[node_id] = component_index
        points = np.asarray([positions[node_id] for node_id in component])
        radius_array = np.asarray([radii[node_id] for node_id in component])
        minimum = np.min(points - radius_array[:, None], axis=0)
        maximum = np.max(points + radius_array[:, None], axis=0)
        ranked = sorted(
            component,
            key=lambda node_id: (
                -frequencies[node_id],
                -graph.degree(node_id),
                node_id,
            ),
        )
        component_edge_count = graph.subgraph(component).number_of_edges()
        component_rows.append(
            [
                f"component-{component_index + 1}",
                " / ".join(labels[node_id] for node_id in ranked[:2]),
                len(component),
                component_edge_count,
                round(float(minimum[0]), 4),
                round(float(minimum[1]), 4),
                round(float(maximum[0]), 4),
                round(float(maximum[1]), 4),
                component_community_start,
                len(layout["communities"]),
            ]
        )

    ordered_nodes = sorted(
        node_records,
        key=lambda record: (
            node_component[record["baseKey"]],
            node_community[record["baseKey"]],
            -frequencies[record["baseKey"]],
            labels[record["baseKey"]],
            record["baseKey"],
        ),
    )
    node_indexes = {record["baseKey"]: index for index, record in enumerate(ordered_nodes)}
    node_rows: list[list[Any]] = []
    for record in ordered_nodes:
        node_id = record["baseKey"]
        priority = math.log2(frequencies[node_id] + 1) * 2.0 + math.log2(graph.degree(node_id) + 1)
        node_rows.append(
            [
                node_id,
                labels[node_id],
                frequencies[node_id],
                round(positions[node_id][0], 4),
                round(positions[node_id][1], 4),
                node_component[node_id],
                node_community[node_id],
                round(priority, 4),
            ]
        )

    bundle_records: dict[tuple[int, int], dict[str, Any]] = {}
    for edge in edges:
        source_community = node_community[edge["source"]]
        target_community = node_community[edge["target"]]
        if source_community == target_community:
            continue
        key = (source_community, target_community)
        bundle = bundle_records.setdefault(
            key,
            {
                "edgeCount": 0,
                "claimCount": 0,
                "relations": Counter(),
            },
        )
        bundle["edgeCount"] += 1
        bundle["claimCount"] += int(edge["occurrenceCount"])
        bundle["relations"][edge["label"]] += int(edge["occurrenceCount"])
    ordered_bundles = sorted(
        bundle_records.items(),
        key=lambda item: (
            -item[1]["claimCount"],
            item[0][0],
            item[0][1],
        ),
    )
    bundle_indexes = {key: index for index, (key, _) in enumerate(ordered_bundles)}
    quotient = nx.Graph()
    for (source_community, target_community), record in ordered_bundles:
        pair = tuple(sorted((source_community, target_community)))
        if quotient.has_edge(*pair):
            quotient[pair[0]][pair[1]]["weight"] += record["claimCount"]
        else:
            quotient.add_edge(
                pair[0],
                pair[1],
                weight=record["claimCount"],
            )
    overview_pairs = {
        tuple(sorted((source, target)))
        for source, target in nx.maximum_spanning_edges(
            quotient,
            data=False,
        )
    }
    overview_keys: set[tuple[int, int]] = set()
    for pair in overview_pairs:
        candidates = [
            (key, record) for key, record in ordered_bundles if tuple(sorted(key)) == pair
        ]
        overview_keys.add(
            min(
                candidates,
                key=lambda item: (
                    -item[1]["claimCount"],
                    item[0][0],
                    item[0][1],
                ),
            )[0]
        )

    bundle_rows: list[list[Any]] = []
    for (source_community, target_community), record in ordered_bundles:
        source_row = community_rows[source_community]
        target_row = community_rows[target_community]
        source_port = _community_port(
            source_row,
            float(target_row[7]),
            float(target_row[8]),
        )
        target_port = _community_port(
            target_row,
            float(source_row[7]),
            float(source_row[8]),
        )
        bundle_rows.append(
            [
                source_community,
                target_community,
                record["edgeCount"],
                record["claimCount"],
                record["relations"].most_common(1)[0][0],
                (source_community, target_community) in overview_keys,
                round(source_port[0], 4),
                round(source_port[1], 4),
                round(target_port[0], 4),
                round(target_port[1], 4),
            ]
        )

    edge_rows = []
    for edge in sorted(
        edges,
        key=lambda item: (
            -int(item["occurrenceCount"]),
            item["label"],
            item["source"],
            item["target"],
        ),
    ):
        key = (
            node_community[edge["source"]],
            node_community[edge["target"]],
        )
        edge_rows.append(
            [
                node_indexes[edge["source"]],
                node_indexes[edge["target"]],
                edge["relationId"],
                edge["label"],
                int(edge["occurrenceCount"]),
                bundle_indexes.get(key, -1),
            ]
        )

    coordinate_array = np.asarray(
        [[row[3], row[4]] for row in node_rows],
        dtype=np.float64,
    )
    radius_array = np.asarray(
        [radii[row[0]] for row in node_rows],
        dtype=np.float64,
    )
    minimum = np.min(coordinate_array - radius_array[:, None], axis=0)
    maximum = np.max(coordinate_array + radius_array[:, None], axis=0)
    nearest = (
        cKDTree(coordinate_array).query(coordinate_array, k=2)[0][:, 1]
        if len(coordinate_array) > 1
        else np.asarray([1.0])
    )
    claim_count = sum(int(edge["occurrenceCount"]) for edge in edges)
    return {
        "ok": True,
        "schemaVersion": ATLAS_SCHEMA_VERSION,
        "focus": {
            "kind": scope,
            "id": scope,
            "label": SCOPE_LABELS[scope],
        },
        "layout": {
            "kind": "atlas",
            "scope": scope,
            "bounds": [
                round(float(minimum[0]), 4),
                round(float(minimum[1]), 4),
                round(float(maximum[0]), 4),
                round(float(maximum[1]), 4),
            ],
            "zoomStops": [4, 24, 56],
            "medianSpacing": round(float(np.median(nearest)), 4),
            "leafCommunitySize": LEAF_COMMUNITY_SIZE,
            "method": (
                "scope-specific connected-component packing with bounded "
                "topology-walked regions and collision-cleared local layouts"
            ),
        },
        "nodes": node_rows,
        "edges": edge_rows,
        "components": component_rows,
        "communities": community_rows,
        "bundles": bundle_rows,
        "claimCount": claim_count,
        "availableCount": claim_count,
        "truncated": False,
        "limit": None,
    }


def _write_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    gzip_path = path.with_suffix(path.suffix + ".gz")
    compressed = gzip.compress(encoded, compresslevel=9, mtime=0)
    gzip_path.write_bytes(compressed)
    return {
        "path": path.name,
        "gzipPath": gzip_path.name,
        "bytes": len(encoded),
        "gzipBytes": len(compressed),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "gzipSha256": hashlib.sha256(compressed).hexdigest(),
        "nodes": len(payload["nodes"]),
        "edges": len(payload["edges"]),
        "components": len(payload["components"]),
        "communities": len(payload["communities"]),
        "bundles": len(payload["bundles"]),
        "claimCount": payload["claimCount"],
        "bounds": payload["layout"]["bounds"],
        "medianSpacing": payload["layout"]["medianSpacing"],
    }


def build_overview(output_path: Path = OVERVIEW_PATH) -> dict[str, Any]:
    node_records, all_edges, node_volumes = _build_source_records()
    scopes: dict[str, Any] = {}
    for scope in ATLAS_SCOPES:
        scope_nodes, scope_edges = _scope_source(
            scope,
            node_records,
            all_edges,
            node_volumes,
        )
        payload = _build_scope_atlas(scope, scope_nodes, scope_edges)
        scopes[scope] = _write_payload(
            output_path.with_name(f"atlas_{scope}.json"),
            payload,
        )
    return {
        "schemaVersion": ATLAS_SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "scopes": scopes,
    }


def main() -> None:
    metadata = build_overview()
    manifest = json.loads(GRAPH_MANIFEST.read_text(encoding="utf-8"))
    manifest["overview"] = metadata
    GRAPH_MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
