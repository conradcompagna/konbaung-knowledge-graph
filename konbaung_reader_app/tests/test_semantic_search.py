import unittest

from app import app


class SemanticSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def candidate(self):
        response = self.client.get(
            "/api/graph/search",
            query_string={
                "q": "Hsinbyushin",
                "kind": "entity",
                "sort": "frequency",
                "limit": 1,
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.get_json()["results"][0]

    def test_semantic_neighbors_respect_threshold_and_limit(self):
        anchor = self.candidate()
        response = self.client.get(
            "/api/graph/search",
            query_string={
                "kind": "entity",
                "sort": "gemini",
                "anchorKind": "entity",
                "anchorId": anchor["id"],
                "minSimilarity": 0.90,
                "limit": 99,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["anchor"]["id"], anchor["id"])
        self.assertEqual(payload["anchor"]["frequency"], anchor["frequency"])
        self.assertEqual(payload["minimumSimilarity"], 0.9)
        self.assertLessEqual(len(payload["results"]), 99)
        self.assertTrue(all(item["similarity"] >= 0.90 for item in payload["results"]))
        self.assertTrue(all(item["kind"] == "entity" for item in payload["results"]))

    def test_selected_tags_return_the_complete_union_graph(self):
        anchor = self.candidate()
        neighbors = self.client.get(
            "/api/graph/search",
            query_string={
                "kind": "entity",
                "sort": "gemini",
                "anchorKind": "entity",
                "anchorId": anchor["id"],
                "minSimilarity": 0.90,
                "limit": 1,
            },
        ).get_json()["results"]
        selected = [anchor["id"], neighbors[0]["id"]]
        response = self.client.get(
            "/api/graph/topology/tags",
            query_string=[("kind", "entity"), *(("id", value) for value in selected)],
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["focus"]["kind"], "tag-filter")
        self.assertEqual(payload["focus"]["tagIds"], selected)
        self.assertFalse(payload["truncated"])
        self.assertIsNone(payload["limit"])
        self.assertEqual(payload["claimCount"], payload["availableCount"])
        node_ids = {node[0] for node in payload["nodes"]}
        self.assertTrue(set(selected).issubset(node_ids))

    def test_selected_tags_return_a_thematic_slice_with_scoped_evidence(self):
        anchor = self.candidate()
        neighbor = self.client.get(
            "/api/graph/search",
            query_string={
                "kind": "entity",
                "sort": "gemini",
                "anchorKind": "entity",
                "anchorId": anchor["id"],
                "minSimilarity": 0.90,
                "limit": 1,
            },
        ).get_json()["results"][0]
        selected = [anchor["id"], neighbor["id"]]
        query = [("kind", "entity")]
        query.extend(("id", value) for value in selected)

        response = self.client.get(
            "/api/graph/categories/topology/tags",
            query_string=query,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["categoryMode"])
        self.assertEqual(payload["layout"], {"kind": "thematic", "scope": "corpus"})
        self.assertEqual(payload["focus"]["kind"], "tag-filter")
        self.assertEqual(payload["focus"]["tagIds"], selected)
        self.assertEqual(len(payload["entityCategories"]), 52)
        self.assertEqual(len(payload["relationCategories"]), 81)
        self.assertEqual(len(payload["patterns"]), 6050)
        self.assertEqual(sum(pattern[4] for pattern in payload["patterns"]), 115)
        self.assertEqual(payload["claimCount"], 115)

        pattern = next(pattern for pattern in payload["patterns"] if pattern[4])
        evidence_query = [
            ("source", payload["entityCategories"][pattern[0]]["id"]),
            ("target", payload["entityCategories"][pattern[2]]["id"]),
            ("relation", payload["relationCategories"][pattern[1]]["id"]),
            ("scope", "corpus"),
            ("tagKind", "entity"),
        ]
        evidence_query.extend(("tagId", value) for value in selected)
        evidence = self.client.get(
            "/api/graph/categories/evidence",
            query_string=evidence_query,
        ).get_json()
        selected_labels = set(payload["focus"]["tagLabels"])
        self.assertTrue(evidence["items"])
        self.assertTrue(
            all(
                item["subject"] in selected_labels or item["object"] in selected_labels
                for item in evidence["items"]
            )
        )

    def test_twenty_embedding_tags_are_combined_as_an_or_union(self):
        anchor = self.client.get(
            "/api/graph/search",
            query_string={
                "q": "Brahmin",
                "kind": "entity",
                "sort": "frequency",
                "limit": 1,
            },
        ).get_json()["results"][0]
        neighbors = self.client.get(
            "/api/graph/search",
            query_string={
                "kind": "entity",
                "sort": "gemini",
                "anchorKind": "entity",
                "anchorId": anchor["id"],
                "minSimilarity": 0.90,
                "limit": 99,
            },
        ).get_json()["results"]
        selected = [anchor["id"]] + [item["id"] for item in neighbors[:19]]
        query = [("kind", "entity")]
        query.extend(("id", value) for value in selected)

        raw = self.client.get(
            "/api/graph/topology/tags",
            query_string=query,
        ).get_json()
        thematic = self.client.get(
            "/api/graph/categories/topology/tags",
            query_string=query,
        ).get_json()

        self.assertEqual(thematic["focus"]["tagIds"], selected)
        self.assertEqual(thematic["claimCount"], raw["claimCount"])
        self.assertEqual(thematic["claimCount"], 74)
        self.assertEqual(sum(pattern[4] for pattern in thematic["patterns"]), 74)
        self.assertEqual(
            sum(pattern[4] > 0 for pattern in thematic["patterns"]),
            39,
        )

    def test_semantic_search_rejects_invalid_threshold(self):
        anchor = self.candidate()
        response = self.client.get(
            "/api/graph/search",
            query_string={
                "kind": "entity",
                "sort": "gemini",
                "anchorKind": "entity",
                "anchorId": anchor["id"],
                "minSimilarity": 1.1,
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("similarity", response.get_json()["error"].lower())


if __name__ == "__main__":
    unittest.main()
