from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from konbaung_reader_app.app import app
from konbaung_reader_app.graph_store import ChronicleGraphStore


ALAUNGPAYA = "e_0f7888dbf7609aba28afb99b26d299ac"
RECEIVED_TITLE_FROM_KING = "e_6264bec8f8cf37a17e5ea94eeaa60a44"


class ChronicleGraphDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = ChronicleGraphStore()

    def test_authoritative_rdf_counts(self) -> None:
        stats = self.graph.stats()
        self.assertEqual(stats["statements"], 698_710)
        self.assertEqual(stats["counts"]["claims"], 27_129)
        self.assertEqual(stats["counts"]["directStatements"], 26_867)
        self.assertEqual(stats["counts"]["entities"], 23_890)
        self.assertEqual(stats["counts"]["relations"], 11_886)
        self.assertEqual(stats["counts"]["sentences"], 11_222)
        self.assertEqual(stats["counts"]["pages"], 1_215)

    def test_page_graph_preserves_v3_claims(self) -> None:
        payload = self.graph.page_graph("vol1", 47)
        self.assertEqual(payload["claimCount"], 16)
        self.assertEqual(len(payload["edges"]), 16)
        self.assertEqual(len(payload["nodes"]), 20)

    def test_materialized_volume_and_corpus_graphs(self) -> None:
        expected = {
            "vol1": (7_566, 8_096, 8_147),
            "vol2": (8_438, 9_071, 9_179),
            "vol3": (9_295, 9_720, 9_803),
            "corpus": (23_890, 26_867, 27_129),
        }
        for scope, (nodes, edges, claims) in expected.items():
            with self.subTest(scope=scope):
                payload = self.graph.overview_graph(scope)
                self.assertTrue(payload["fixedLayout"])
                self.assertEqual(len(payload["nodes"]), nodes)
                self.assertEqual(len(payload["edges"]), edges)
                self.assertEqual(payload["claimCount"], claims)
                self.assertIn("x", payload["nodes"][0])
                self.assertIn("y", payload["nodes"][0])

    def test_corpus_layout_has_no_overlapping_nodes(self) -> None:
        payload = self.graph.overview_graph("corpus")
        coordinates = np.asarray(
            [(node["x"], node["y"]) for node in payload["nodes"]],
            dtype=np.float64,
        )
        frequencies = np.asarray(
            [node["frequency"] for node in payload["nodes"]],
            dtype=np.float64,
        )
        radii = np.minimum(
            32.0,
            3.0 + np.power(np.log2(frequencies + 1.0), 1.55) * 0.55,
        )
        gap = float(payload["layout"]["collisionGap"])
        pairs = cKDTree(coordinates).query_pairs(
            float(np.max(radii) * 2 + gap),
            output_type="ndarray",
        )
        differences = coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]]
        distances = np.sqrt(np.einsum("ij,ij->i", differences, differences))
        required = radii[pairs[:, 0]] + radii[pairs[:, 1]] + gap
        self.assertFalse(np.any(distances < required))

    def test_claim_retains_sentence_and_source_navigation(self) -> None:
        payload = self.graph.claim_graph("vol1_s000001", 1)
        self.assertEqual(payload["claimCount"], 1)
        claim = payload["edges"][0]["claims"][0]
        self.assertEqual(claim["sentenceId"], "vol1_s000001")
        self.assertEqual(claim["volumeId"], "vol1")
        self.assertEqual(claim["ownerPage"], 47)
        self.assertTrue(claim["sentenceMy"])
        self.assertTrue(claim["sentenceEn"])

    def test_gemini_sort_uses_raw_same_kind_vectors(self) -> None:
        payload = self.graph.search(
            sort="gemini",
            anchor_kind="entity",
            anchor_id=ALAUNGPAYA,
            limit=5,
        )
        self.assertEqual(payload["anchor"]["label"], "Alaungpaya")
        self.assertEqual(payload["results"][0]["label"], "alaungpaya")
        self.assertGreater(payload["results"][0]["similarity"], 0.98)
        self.assertTrue(all(item["kind"] == "entity" for item in payload["results"]))

    def test_neighborhood_and_relation_queries_are_bounded(self) -> None:
        entity = self.graph.entity_graph(ALAUNGPAYA, depth=2, limit=250)
        relation = self.graph.relation_graph(RECEIVED_TITLE_FROM_KING, limit=20)
        self.assertLessEqual(entity["claimCount"], 250)
        self.assertEqual(relation["claimCount"], 20)


class ChronicleGraphApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = app.test_client()

    def test_graph_endpoints(self) -> None:
        urls = (
            "/api/graph/stats",
            "/api/graph/page/vol1/47",
            "/api/graph/overview/vol1",
            "/api/graph/overview/corpus",
            f"/api/graph/entity/{ALAUNGPAYA}?depth=1&limit=50",
            f"/api/graph/relation/{RECEIVED_TITLE_FROM_KING}?limit=20",
            "/api/graph/claim/vol1_s000001/1",
            (
                "/api/graph/search?kind=entity&sort=gemini"
                f"&anchorKind=entity&anchorId={ALAUNGPAYA}&limit=5"
            ),
        )
        for url in urls:
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.get_json()["ok"])

    def test_invalid_graph_identifier_is_rejected(self) -> None:
        response = self.client.get("/api/graph/entity/not-an-id")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
