import unittest

from app import app


class AxialCategoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    def test_v3_is_default_and_page_annotations_include_full_categories(self):
        index = self.client.get("/api/chronicles/index").get_json()
        self.assertEqual(index["defaultTripleVariant"], "v3")

        page = self.client.get("/api/chronicles/page/vol1/47").get_json()
        self.assertEqual(page["tripleVariant"]["id"], "v3")
        axial = page["annotations"][0]["axial"]
        self.assertEqual(axial["status"], "accepted")
        self.assertEqual(axial["subject"]["tagId"], "E01")
        self.assertEqual(
            axial["subject"]["label"],
            "SovereignAndReigningMonarch",
        )
        self.assertTrue(axial["relation"]["definition"])
        self.assertTrue(axial["object"]["label"])

    def test_closed_schema_replaces_every_provisional_category(self):
        catalog = self.client.get("/api/graph/categories/catalog").get_json()
        self.assertEqual(len(catalog["entities"]), 52)
        self.assertEqual(len(catalog["relations"]), 81)
        self.assertFalse(any(category["provisional"] for category in catalog["entities"]))
        self.assertFalse(any(category["provisional"] for category in catalog["relations"]))

        page = self.client.get("/api/chronicles/page/vol1/216?triples=v3").get_json()
        collapsed = [
            endpoint["id"]
            for annotation in page["annotations"]
            for endpoint in (
                annotation["axial"]["subject"],
                annotation["axial"]["relation"],
                annotation["axial"]["object"],
            )
            if endpoint
        ]
        self.assertIn("E37", collapsed)
        self.assertFalse(any(tag.startswith(("NE", "NR")) for tag in collapsed))

        diorama_page = self.client.get("/api/chronicles/page/vol2/217?triples=v3").get_json()
        diorama_tags = {
            endpoint["id"]
            for annotation in diorama_page["annotations"]
            for endpoint in (
                annotation["axial"]["subject"],
                annotation["axial"]["relation"],
                annotation["axial"]["object"],
            )
            if endpoint
        }
        self.assertIn("E31", diorama_tags)
        self.assertIn("R65", diorama_tags)
        self.assertFalse(any(tag.startswith(("NE", "NR")) for tag in diorama_tags))

    def test_chunk_retried_page_categories_are_resolved(self):
        page = self.client.get("/api/chronicles/page/vol1/206?triples=v3").get_json()
        self.assertEqual(
            page["diagnostics"]["axialUnresolvedAnnotationCount"],
            0,
        )
        axial = page["annotations"][0]["axial"]
        self.assertEqual(axial["status"], "accepted")
        self.assertIsNotNone(axial["subject"])
        self.assertIsNone(axial["reason"])

    def test_category_overview_is_complete_thematic_metagraph(self):
        overview = self.client.get("/api/graph/categories/topology/overview/corpus").get_json()
        self.assertTrue(overview["categoryMode"])
        self.assertEqual(overview["schemaVersion"], 5)
        self.assertEqual(overview["layout"]["kind"], "thematic")
        self.assertEqual(len(overview["entityCategories"]), 52)
        self.assertEqual(len(overview["relationCategories"]), 81)
        self.assertEqual(len(overview["patterns"]), 6050)
        self.assertEqual(len(overview["incidenceBundles"]), 3279)
        self.assertEqual(
            sum(bundle[2] == 0 for bundle in overview["incidenceBundles"]),
            1433,
        )
        self.assertEqual(
            sum(bundle[2] == 1 for bundle in overview["incidenceBundles"]),
            1846,
        )
        self.assertEqual(overview["claimCount"], 27129)
        self.assertEqual(overview["entityMentionCount"], 54258)
        self.assertEqual(
            sum(category["mentionCount"] for category in overview["entityCategories"]),
            54258,
        )
        self.assertEqual(
            sum(pattern[3] for pattern in overview["patterns"]),
            27129,
        )
        self.assertFalse(overview["truncated"])
        self.assertIsNone(overview["limit"])
        self.assertNotIn("sentenceMy", str(overview))

        e01_index = next(
            index
            for index, category in enumerate(overview["entityCategories"])
            if category["id"] == "E01"
        )
        e24_index = next(
            index
            for index, category in enumerate(overview["entityCategories"])
            if category["id"] == "E24"
        )
        e01 = overview["entityCategories"][e01_index]
        self.assertEqual(e01["patternCount"], 1494)
        self.assertEqual(e01["relationCount"], 78)
        self.assertEqual(e01["counterpartCount"], 52)
        self.assertEqual(
            sum(
                pattern[0] == e01_index and pattern[2] == e24_index
                for pattern in overview["patterns"]
            ),
            48,
        )

    def test_category_filters_foreground_without_removing_topology(self):
        filtered = self.client.get(
            "/api/graph/categories/topology/overview/corpus",
            query_string={"entity": "E01", "relation": "R01"},
        ).get_json()
        self.assertEqual(filtered["filters"]["entities"], ["E01"])
        self.assertEqual(filtered["filters"]["relations"], ["R01"])
        self.assertEqual(len(filtered["entityCategories"]), 52)
        self.assertEqual(len(filtered["relationCategories"]), 81)
        self.assertEqual(len(filtered["patterns"]), 6050)
        self.assertEqual(len(filtered["incidenceBundles"]), 3279)
        self.assertEqual(filtered["claimCount"], 119)
        self.assertEqual(sum(pattern[4] for pattern in filtered["patterns"]), 119)
        self.assertTrue(any(pattern[4] == 0 for pattern in filtered["patterns"]))

    def test_redone_dense_page_contributes_every_triple_to_category_graph(self):
        topology = self.client.get("/api/graph/categories/topology/page/vol2/32").get_json()
        self.assertEqual(topology["claimCount"], 165)
        self.assertEqual(topology["entityMentionCount"], 330)
        self.assertEqual(len(topology["patterns"]), 6050)
        self.assertEqual(sum(pattern[4] for pattern in topology["patterns"]), 165)
        self.assertFalse(topology["truncated"])

    def test_category_evidence_returns_raw_triple_and_sentence(self):
        response = self.client.get(
            "/api/graph/categories/evidence",
            query_string={
                "source": "E12",
                "target": "E02",
                "relation": "R01",
                "limit": 1,
            },
        )
        self.assertEqual(response.status_code, 200)
        item = response.get_json()["items"][0]
        self.assertTrue(item["subject"])
        self.assertTrue(item["predicate"])
        self.assertTrue(item["object"])
        self.assertTrue(item["sentenceMy"])
        self.assertTrue(item["sentenceEn"])
        self.assertEqual(item["relationCategory"]["id"], "R01")


if __name__ == "__main__":
    unittest.main()
