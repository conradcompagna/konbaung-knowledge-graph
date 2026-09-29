import gzip
import hashlib
import json
import math
import unittest

from scipy.spatial import cKDTree

from build_graph_overview import NODE_GAP, bubble_radius
from graph_schema import GRAPH_MANIFEST, GRAPH_ROOT


EXPECTED = {
    "corpus": (23890, 26867, 27129, 2466),
    "vol1": (7566, 8096, 8147, 941),
    "vol2": (8438, 9071, 9179, 828),
    "vol3": (9295, 9720, 9803, 1136),
}


class GraphAtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(GRAPH_MANIFEST.read_text(encoding="utf-8"))
        cls.payloads = {
            scope: json.loads((GRAPH_ROOT / f"atlas_{scope}.json").read_text(encoding="utf-8"))
            for scope in EXPECTED
        }

    def test_artifact_hashes_match_deterministic_stored_bytes(self):
        for scope in EXPECTED:
            with self.subTest(scope=scope):
                record = self.manifest["overview"]["scopes"][scope]
                plain = (GRAPH_ROOT / record["path"]).read_bytes()
                compressed = (GRAPH_ROOT / record["gzipPath"]).read_bytes()
                self.assertEqual(
                    hashlib.sha256(plain).hexdigest(),
                    record["sha256"],
                )
                self.assertEqual(
                    hashlib.sha256(compressed).hexdigest(),
                    record["gzipSha256"],
                )
                self.assertEqual(gzip.decompress(compressed), plain)

    def test_every_scope_preserves_raw_graph_structure(self):
        for scope, expected in EXPECTED.items():
            with self.subTest(scope=scope):
                payload = self.payloads[scope]
                nodes, edges, claims, components = expected
                self.assertEqual(payload["schemaVersion"], 3)
                self.assertEqual(payload["layout"]["kind"], "atlas")
                self.assertEqual(len(payload["nodes"]), nodes)
                self.assertEqual(len(payload["edges"]), edges)
                self.assertEqual(payload["claimCount"], claims)
                self.assertEqual(len(payload["components"]), components)
                self.assertEqual(
                    len({row[0] for row in payload["nodes"]}),
                    nodes,
                )
                self.assertTrue(
                    all(0 <= edge[0] < nodes and 0 <= edge[1] < nodes for edge in payload["edges"])
                )

    def test_every_node_has_a_renderable_connection(self):
        for scope, payload in self.payloads.items():
            with self.subTest(scope=scope):
                degree = [0] * len(payload["nodes"])
                for edge in payload["edges"]:
                    degree[edge[0]] += 1
                    degree[edge[1]] += 1
                self.assertNotIn(0, degree)

    def test_component_membership_matches_raw_edges(self):
        for scope, payload in self.payloads.items():
            with self.subTest(scope=scope):
                parents = list(range(len(payload["nodes"])))

                def find(index):
                    while parents[index] != index:
                        parents[index] = parents[parents[index]]
                        index = parents[index]
                    return index

                def union(left, right):
                    left_root = find(left)
                    right_root = find(right)
                    if left_root != right_root:
                        parents[right_root] = left_root

                for edge in payload["edges"]:
                    union(edge[0], edge[1])
                raw_components = {}
                for index, node in enumerate(payload["nodes"]):
                    raw_components.setdefault(find(index), set()).add(node[5])
                self.assertEqual(
                    len(raw_components),
                    len(payload["components"]),
                )
                self.assertTrue(
                    all(len(component_ids) == 1 for component_ids in raw_components.values())
                )

    def test_detailed_node_circles_do_not_overlap(self):
        for scope, payload in self.payloads.items():
            with self.subTest(scope=scope):
                nodes = payload["nodes"]
                coordinates = [(float(row[3]), float(row[4])) for row in nodes]
                radii = [bubble_radius(row[1]) for row in nodes]
                maximum = max(radii) * 2 + NODE_GAP
                pairs = cKDTree(coordinates).query_pairs(maximum)
                for left, right in pairs:
                    distance = math.dist(coordinates[left], coordinates[right])
                    self.assertGreaterEqual(
                        distance + 0.02,
                        radii[left] + radii[right] + NODE_GAP,
                    )

    def test_component_and_community_bounds_do_not_overlap(self):
        for scope, payload in self.payloads.items():
            with self.subTest(scope=scope):
                for rows in (
                    [row[4:8] for row in payload["components"]],
                    [row[3:7] for row in payload["communities"]],
                ):
                    ordered = sorted(rows, key=lambda row: (row[0], row[1]))
                    active = []
                    for bounds in ordered:
                        active = [other for other in active if other[2] > bounds[0]]
                        for other in active:
                            overlaps_y = bounds[1] < other[3] and bounds[3] > other[1]
                            self.assertFalse(overlaps_y)
                        active.append(bounds)

    def test_overview_bundles_are_a_topology_backbone_with_ports(self):
        for scope, payload in self.payloads.items():
            with self.subTest(scope=scope):
                overview = [row for row in payload["bundles"] if row[5]]
                expected = sum(max(0, component[9] - 1) for component in payload["components"])
                self.assertEqual(len(overview), expected)
                for bundle in payload["bundles"]:
                    source = payload["communities"][bundle[0]]
                    target = payload["communities"][bundle[1]]
                    for community, x, y in (
                        (source, bundle[6], bundle[7]),
                        (target, bundle[8], bundle[9]),
                    ):
                        self.assertGreaterEqual(x + 0.02, community[3])
                        self.assertLessEqual(x - 0.02, community[5])
                        self.assertGreaterEqual(y + 0.02, community[4])
                        self.assertLessEqual(y - 0.02, community[6])
                        self.assertTrue(
                            math.isclose(x, community[3], abs_tol=0.02)
                            or math.isclose(x, community[5], abs_tol=0.02)
                            or math.isclose(y, community[4], abs_tol=0.02)
                            or math.isclose(y, community[6], abs_tol=0.02)
                        )

    def test_volume_layout_is_not_filtered_corpus_coordinates(self):
        corpus_positions = {row[0]: (row[3], row[4]) for row in self.payloads["corpus"]["nodes"]}
        for scope in ("vol1", "vol2", "vol3"):
            positions = {row[0]: (row[3], row[4]) for row in self.payloads[scope]["nodes"]}
            self.assertTrue(
                any(positions[node_id] != corpus_positions[node_id] for node_id in positions)
            )


if __name__ == "__main__":
    unittest.main()
