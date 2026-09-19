import unittest

from app import app, axial_graph, graph


class FilteredTagsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        cls.client = app.test_client()

    @staticmethod
    def theme_query(role="subject", **extra):
        query = [
            ("role", role),
            ("subject", "E01"),
            ("relation", "R01"),
            ("object", "E04"),
        ]
        query.extend((name, str(value)) for name, value in extra.items())
        return query

    def test_role_tags_are_frequency_ranked_bounded_and_searchable(self):
        response = self.client.get(
            "/api/graph/categories/tags",
            query_string=self.theme_query(limit=100),
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["role"], "subject")
        self.assertEqual(payload["pagination"]["limit"], 100)
        self.assertLessEqual(len(payload["items"]), 100)
        self.assertEqual(payload["filters"]["subjects"], ["E01"])
        self.assertEqual(payload["filters"]["relations"], ["R01"])
        self.assertEqual(payload["filters"]["objects"], ["E04"])
        frequencies = [item["frequency"] for item in payload["items"]]
        self.assertEqual(frequencies, sorted(frequencies, reverse=True))
        self.assertTrue(all(item["role"] == "subject" for item in payload["items"]))
        self.assertTrue(all(item["kind"] == "entity" for item in payload["items"]))
        self.assertTrue(all(item["id"].startswith("e_") for item in payload["items"]))

        searched = self.client.get(
            "/api/graph/categories/tags",
            query_string=self.theme_query(q="alaung", limit=100),
        ).get_json()
        self.assertTrue(searched["items"])
        self.assertTrue(
            all("alaung" in item["label"].casefold() for item in searched["items"])
        )

    def test_tag_pages_are_stable_and_never_exceed_one_hundred(self):
        first = self.client.get(
            "/api/graph/categories/tags",
            query_string={"role": "subject", "subject": "E01", "limit": 100},
        ).get_json()
        second = self.client.get(
            "/api/graph/categories/tags",
            query_string={
                "role": "subject",
                "subject": "E01",
                "offset": 100,
                "limit": 1000,
            },
        ).get_json()
        self.assertEqual(len(first["items"]), 100)
        self.assertEqual(len(second["items"]), 100)
        self.assertTrue(first["pagination"]["hasMore"])
        self.assertEqual(second["pagination"]["offset"], 100)
        self.assertFalse(
            {item["id"] for item in first["items"]}
            & {item["id"] for item in second["items"]}
        )

    def test_checked_tags_union_within_role_and_intersect_across_roles(self):
        subject_page = self.client.get(
            "/api/graph/categories/tags",
            query_string=self.theme_query(limit=2),
        ).get_json()
        selected_subjects = [item["id"] for item in subject_page["items"]]
        query = [
            ("scope", "corpus"),
            ("subjectCategory", "E01"),
            ("relationCategory", "R01"),
            ("objectCategory", "E04"),
            *(("subjectTag", value) for value in selected_subjects),
        ]
        response = self.client.get(
            "/api/graph/categories/topology/filtered-tags",
            query_string=query,
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["focus"]["kind"], "filtered-tags")
        self.assertEqual(payload["focus"]["subjectTagIds"], selected_subjects)
        self.assertEqual(
            payload["claimCount"],
            sum(item["frequency"] for item in subject_page["items"]),
        )
        self.assertEqual(sum(pattern[4] for pattern in payload["patterns"]), payload["claimCount"])

        record = next(
            item
            for item in axial_graph.records
            if item["categories"] == {"s": "E01", "r": "R01", "o": "E04"}
        )
        exact_query = [
            ("scope", "corpus"),
            ("subjectCategory", "E01"),
            ("relationCategory", "R01"),
            ("objectCategory", "E04"),
            ("subjectTag", graph.by_label["entity"][record["subject"]]["baseKey"]),
            ("relationTag", graph.by_label["relation"][record["predicate"]]["baseKey"]),
            ("objectTag", graph.by_label["entity"][record["object"]]["baseKey"]),
        ]
        exact = self.client.get(
            "/api/graph/categories/topology/filtered-tags",
            query_string=exact_query,
        ).get_json()
        expected = [
            item
            for item in axial_graph.records
            if item["categories"] == {"s": "E01", "r": "R01", "o": "E04"}
            and item["subject"] == record["subject"]
            and item["predicate"] == record["predicate"]
            and item["object"] == record["object"]
        ]
        self.assertEqual(exact["claimCount"], len(expected))
        self.assertGreater(exact["claimCount"], 0)

        evidence_query = [
            ("source", "E01"),
            ("relation", "R01"),
            ("target", "E04"),
            *exact_query[4:],
        ]
        evidence = self.client.get(
            "/api/graph/categories/evidence",
            query_string=evidence_query,
        ).get_json()["items"]
        self.assertTrue(evidence)
        self.assertTrue(
            all(
                item["subject"] == record["subject"]
                and item["predicate"] == record["predicate"]
                and item["object"] == record["object"]
                for item in evidence
            )
        )

    def test_filtered_tags_without_categories_fall_back_to_the_scope(self):
        # An unfiltered role lists every raw tag in that position, which is what
        # lets the tag panel double as free-text semantic search.
        payload = self.client.get(
            "/api/graph/categories/tags",
            query_string={"role": "subject"},
        )
        self.assertEqual(payload.status_code, 200)
        body = payload.get_json()
        self.assertFalse(body["categoryScoped"])
        self.assertGreater(body["pagination"]["total"], 0)
        self.assertEqual(body["matchingClaimCount"], len(axial_graph.records))

        empty_topology = self.client.get(
            "/api/graph/categories/topology/filtered-tags"
        )
        self.assertEqual(empty_topology.status_code, 400)

    def test_similar_to_ranks_a_bucket_by_averaged_tag_vectors(self):
        anchors = [
            graph.by_label["entity"][label]["baseKey"]
            for label in ("King", "Alaungpaya")
        ]
        response = self.client.get(
            "/api/graph/categories/tags",
            query_string=[
                ("role", "subject"),
                ("subject", "E01"),
                ("minSimilarity", "0.5"),
                *(("similarTo", anchor) for anchor in anchors),
            ],
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["similarTo"], ["King", "Alaungpaya"])
        similarities = [item["similarity"] for item in body["items"]]
        self.assertTrue(all(value is not None for value in similarities))
        # The two anchors lead, and everything after them descends by cosine.
        self.assertEqual(
            [item["label"] for item in body["items"][:2]],
            ["King", "Alaungpaya"],
        )
        tail = similarities[2:]
        self.assertEqual(tail, sorted(tail, reverse=True))

    def test_applied_tag_filters_narrow_the_other_role_buckets(self):
        spellings = ["Alaungpaya", "Alaungmintaya", "Alaungmintayagyi"]
        applied = [
            ("subjectTag", graph.by_label["entity"][label]["baseKey"])
            for label in spellings
        ]

        def bucket(role, narrow):
            response = self.client.get(
                "/api/graph/categories/tags",
                query_string=[("role", role), *(applied if narrow else [])],
            )
            self.assertEqual(response.status_code, 200)
            return response.get_json()

        for role in ("relation", "object"):
            wide = bucket(role, False)
            narrow = bucket(role, True)
            self.assertLess(
                narrow["pagination"]["total"], wide["pagination"]["total"]
            )
            self.assertLess(
                narrow["matchingClaimCount"], wide["matchingClaimCount"]
            )
            self.assertEqual(sorted(narrow["narrowedBy"]), ["subject"])
            self.assertEqual(sorted(narrow["narrowedBy"]["subject"]), sorted(spellings))

        # A role's own tag filter never constrains its own bucket, so another
        # spelling can still be added or swapped in.
        own = bucket("subject", True)
        self.assertEqual(own["narrowedBy"], {})
        self.assertEqual(
            own["pagination"]["total"], bucket("subject", False)["pagination"]["total"]
        )

        # Every claim behind the narrowed buckets really is one of this king's.
        narrowed_relations = {
            item["label"] for item in bucket("relation", True)["items"]
        }
        subjects_for_relations = {
            record["subject"]
            for record in axial_graph.records
            if record["predicate"] in narrowed_relations
            and record["subject"] in set(spellings)
        }
        self.assertTrue(subjects_for_relations)

    def test_similar_patterns_rank_against_the_locked_selection(self):
        response = self.client.get(
            "/api/graph/categories/patterns/similar",
            query_string={
                "subject": "E01",
                "relation": "R01",
                "minSimilarity": "0.5",
                "limit": "10",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["anchor"]["subjects"], ["E01"])
        self.assertEqual(body["anchor"]["relations"], ["R01"])
        self.assertEqual(
            sorted(body["comparedRoles"]), ["object", "relation", "subject"]
        )
        scores = [item["similarity"] for item in body["items"]]
        self.assertTrue(scores)
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(value >= 0.5 for value in scores))
        # Anchor patterns are excluded from their own neighbor list.
        anchor_patterns = {
            (
                record["categories"]["s"],
                record["categories"]["r"],
                record["categories"]["o"],
            )
            for record in axial_graph.records
            if record["categories"]["s"] == "E01"
            and record["categories"]["r"] == "R01"
        }
        self.assertEqual(body["anchor"]["patternCount"], len(anchor_patterns))
        for item in body["items"]:
            self.assertNotIn(
                (item["subjectId"], item["relationId"], item["objectId"]),
                anchor_patterns,
            )

    def test_similar_patterns_require_an_applied_category(self):
        response = self.client.get("/api/graph/categories/patterns/similar")
        self.assertEqual(response.status_code, 400)
        self.assertIn("at least one", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
