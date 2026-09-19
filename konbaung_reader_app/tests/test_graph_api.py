import unittest

from app import app


def first_edge(payload):
    edge = payload["edges"][0]
    return (
        payload["nodes"][edge[0]][0],
        edge[2],
        payload["nodes"][edge[1]][0],
    )


class GraphApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_topology_is_compact_and_contains_no_evidence(self):
        response = self.client.get("/api/graph/topology/page/vol1/47")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["schemaVersion"], 2)
        self.assertEqual(payload["layout"]["kind"], "triples")
        self.assertTrue(payload["nodes"])
        self.assertTrue(payload["edges"])
        self.assertTrue(all(len(node) == 5 for node in payload["nodes"]))
        self.assertTrue(all(len(edge) == 5 for edge in payload["edges"]))
        serialized = response.get_data(as_text=True)
        self.assertNotIn('"claims"', serialized)
        self.assertNotIn("sentenceMy", serialized)
        self.assertNotIn("sentenceEn", serialized)

    def test_overview_uses_the_prebuilt_atlas_schema(self):
        response = self.client.get("/api/graph/topology/overview/corpus")
        self.addCleanup(response.close)
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["schemaVersion"], 3)
        self.assertEqual(payload["layout"]["kind"], "atlas")
        self.assertEqual(len(payload["nodes"]), 23890)
        self.assertEqual(len(payload["edges"]), 26867)
        self.assertEqual(len(payload["components"]), 2466)
        self.assertTrue(payload["communities"])
        self.assertTrue(payload["bundles"])
        self.assertTrue(all(len(node) == 8 for node in payload["nodes"]))
        self.assertTrue(all(len(edge) == 6 for edge in payload["edges"]))
        self.assertLess(len(response.data), 7_000_000)
        self.assertTrue(response.headers["ETag"])

    def test_overview_serves_the_precompressed_artifact(self):
        response = self.client.get(
            "/api/graph/topology/overview/corpus",
            headers={"Accept-Encoding": "gzip"},
        )
        self.addCleanup(response.close)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Encoding"], "gzip")
        self.assertEqual(response.headers["Vary"], "Accept-Encoding")
        self.assertLess(len(response.data), 2_600_000)

    def test_page_range_returns_complete_deduplicated_topology(self):
        first_page = self.client.get(
            "/api/graph/topology/page/vol2/235"
        ).get_json()
        second_page = self.client.get(
            "/api/graph/topology/page/vol2/236"
        ).get_json()
        response = self.client.get(
            "/api/graph/topology/range/vol2/235/236"
        )
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["focus"]["kind"], "range")
        self.assertEqual(payload["focus"]["volumeId"], "vol2")
        self.assertEqual(payload["focus"]["startPage"], 235)
        self.assertEqual(payload["focus"]["endPage"], 236)
        self.assertEqual(payload["layout"]["kind"], "triples")
        self.assertFalse(payload["truncated"])
        self.assertIsNone(payload["limit"])
        self.assertEqual(payload["claimCount"], payload["availableCount"])
        self.assertLessEqual(
            payload["claimCount"],
            first_page["claimCount"] + second_page["claimCount"],
        )
        self.assertTrue(all(node[3] is None for node in payload["nodes"]))
        self.assertTrue(all(node[4] is None for node in payload["nodes"]))

    def test_page_range_rejects_reversed_bounds(self):
        response = self.client.get(
            "/api/graph/topology/range/vol2/236/235"
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("end page", response.get_json()["error"])

    def test_evidence_is_exact_and_paginated(self):
        topology = self.client.get(
            "/api/graph/topology/entity/e_d02ca5be0460043fd02e891bbe69b007"
            "?depth=1&limit=30"
        ).get_json()
        source, relation, target = first_edge(topology)
        response = self.client.get(
            "/api/graph/evidence",
            query_string={
                "source": source,
                "relation": relation,
                "target": target,
                "limit": 1,
            },
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(
            set(payload["items"][0]),
            {
                "claimId",
                "sentenceId",
                "sentenceMy",
                "sentenceEn",
                "volumeId",
                "ownerPage",
                "ordinal",
            },
        )

    def test_claim_topology_can_reload_its_exact_evidence(self):
        page = self.client.get("/api/graph/topology/page/vol1/47").get_json()
        source, relation, target = first_edge(page)
        evidence = self.client.get(
            "/api/graph/evidence",
            query_string={
                "source": source,
                "relation": relation,
                "target": target,
                "limit": 1,
            },
        ).get_json()["items"][0]
        claim = self.client.get(
            f"/api/graph/topology/claim/{evidence['sentenceId']}/{evidence['ordinal']}"
        ).get_json()
        self.assertEqual(claim["focus"]["kind"], "claim")
        self.assertEqual(claim["layout"]["kind"], "claim")
        self.assertEqual(claim["claimCount"], 1)

        exact = self.client.get(
            "/api/graph/evidence",
            query_string={
                "source": claim["nodes"][claim["edges"][0][0]][0],
                "relation": claim["edges"][0][2],
                "target": claim["nodes"][claim["edges"][0][1]][0],
                "sentenceId": evidence["sentenceId"],
                "ordinal": evidence["ordinal"],
            },
        ).get_json()
        self.assertEqual(len(exact["items"]), 1)
        self.assertEqual(exact["items"][0]["claimId"], evidence["claimId"])

    def test_focused_layout_policies_are_explicit(self):
        entity_id = "e_d02ca5be0460043fd02e891bbe69b007"
        one_hop = self.client.get(
            f"/api/graph/topology/entity/{entity_id}?depth=1&limit=30"
        ).get_json()
        two_hop = self.client.get(
            f"/api/graph/topology/entity/{entity_id}?depth=2&limit=80"
        ).get_json()
        self.assertEqual(one_hop["layout"]["kind"], "radial")
        self.assertEqual(two_hop["layout"]["kind"], "force")
        self.assertEqual(two_hop["focus"]["depth"], 2)
        self.assertGreater(two_hop["claimCount"], one_hop["claimCount"])

        relation_id = one_hop["edges"][0][2]
        relation = self.client.get(
            f"/api/graph/topology/relation/{relation_id}?limit=30"
        ).get_json()
        self.assertEqual(relation["layout"]["kind"], "bipartite")

    def test_retired_evidence_heavy_routes_are_gone(self):
        self.assertEqual(self.client.get("/api/graph/page/vol1/47").status_code, 404)
        self.assertEqual(
            self.client.get("/api/graph/overview/corpus").status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
