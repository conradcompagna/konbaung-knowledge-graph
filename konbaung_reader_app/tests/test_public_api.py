import unittest

from app import app, public_api


class PublicApiTests(unittest.TestCase):
    def setUp(self):
        app.config.update(
            TESTING=True,
            PUBLIC_API_RATE_LIMIT=120,
            PUBLIC_API_CORS_ORIGINS="*",
            PUBLIC_API_KEYS="",
        )
        public_api.limiter.clear()
        self.client = app.test_client()

    def tearDown(self):
        app.config.update(
            PUBLIC_API_RATE_LIMIT=120,
            PUBLIC_API_CORS_ORIGINS="*",
            PUBLIC_API_KEYS="",
        )
        public_api.limiter.clear()

    def brahmin_tag_ids(self, count=20):
        anchor = self.client.get(
            "/api/v1/tags",
            query_string={"q": "Brahmin", "kind": "entity", "limit": 1},
        ).get_json()["data"][0]
        neighbors = self.client.get(
            f"/api/v1/tags/entity/{anchor['id']}/similar",
            query_string={"minimum_similarity": 0.90, "limit": 99},
        ).get_json()["data"]
        return [anchor["id"]] + [item["id"] for item in neighbors[: count - 1]]

    @staticmethod
    def tag_query(tag_ids, **extra):
        query = [("entity_tag_id", tag_id) for tag_id in tag_ids]
        query.extend((key, str(value)) for key, value in extra.items())
        return query

    def test_discovery_docs_openapi_and_cors_are_public(self):
        discovery = self.client.get("/api/v1")
        self.assertEqual(discovery.status_code, 200)
        payload = discovery.get_json()
        self.assertEqual(payload["version"], "1.0.0")
        self.assertTrue(payload["readOnly"])
        self.assertEqual(payload["limits"]["maximumPageSize"], 100)

        docs = self.client.get("/api/v1/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("Query rules", docs.get_data(as_text=True))

        schema = self.client.get("/api/v1/openapi.json")
        self.assertEqual(schema.status_code, 200)
        contract = schema.get_json()
        self.assertEqual(contract["openapi"], "3.1.0")
        self.assertIn("/triples", contract["paths"])
        entity_filter = next(
            parameter
            for parameter in contract["paths"]["/triples"]["get"]["parameters"]
            if parameter["name"] == "entity_tag_id"
        )
        self.assertEqual(entity_filter["schema"]["type"], "array")
        self.assertTrue(entity_filter["explode"])
        self.assertNotIn("style", entity_filter["schema"])
        self.assertIn("ApiKeyAuth", contract["components"]["securitySchemes"])

        cors = self.client.get(
            "/api/v1/health",
            headers={"Origin": "https://example.org"},
        )
        self.assertEqual(cors.headers["Access-Control-Allow-Origin"], "*")
        self.assertEqual(cors.headers["X-Content-Type-Options"], "nosniff")

    def test_stats_and_closed_category_catalog(self):
        stats = self.client.get("/api/v1/stats")
        self.assertEqual(stats.status_code, 200)
        counts = stats.get_json()["counts"]
        self.assertEqual(counts["triples"], 27129)
        self.assertEqual(counts["entityCategories"], 52)
        self.assertEqual(counts["relationCategories"], 81)
        self.assertEqual(counts["thematicPatterns"], 6050)
        self.assertIn("X-RateLimit-Remaining", stats.headers)
        self.assertIn("max-age=3600", stats.headers["Cache-Control"])

        categories = self.client.get(
            "/api/v1/categories", query_string={"kind": "entity"}
        ).get_json()
        self.assertEqual(len(categories["entities"]), 52)
        self.assertNotIn("relations", categories)
        self.assertEqual(
            set(categories["entities"][0]),
            {"id", "tag", "label", "definition"},
        )

    def test_tag_lookup_and_embedding_neighbors(self):
        search = self.client.get(
            "/api/v1/tags",
            query_string={"q": "Hsinbyushin", "kind": "entity", "limit": 10},
        )
        self.assertEqual(search.status_code, 200)
        payload = search.get_json()
        self.assertTrue(payload["data"])
        anchor = payload["data"][0]
        self.assertEqual(anchor["kind"], "entity")

        neighbors = self.client.get(
            f"/api/v1/tags/entity/{anchor['id']}/similar",
            query_string={"minimum_similarity": 0.9, "limit": 20},
        )
        self.assertEqual(neighbors.status_code, 200)
        similar = neighbors.get_json()
        self.assertEqual(similar["anchor"]["id"], anchor["id"])
        self.assertTrue(
            all(item["similarity"] >= 0.9 for item in similar["data"])
        )

    def test_twenty_embedding_tags_form_a_complete_paginated_union(self):
        tag_ids = self.brahmin_tag_ids(20)
        base_query = self.tag_query(tag_ids, limit=25)
        response = self.client.get("/api/v1/triples", query_string=base_query)
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["pagination"]["total"], 74)
        self.assertEqual(len(payload["query"]["entity_tag_ids"]), 20)

        ids = [item["id"] for item in payload["data"]]
        cursor = payload["pagination"]["nextCursor"]
        while cursor:
            next_query = [*base_query, ("cursor", cursor)]
            page = self.client.get(
                "/api/v1/triples", query_string=next_query
            ).get_json()
            ids.extend(item["id"] for item in page["data"])
            cursor = page["pagination"]["nextCursor"]
        self.assertEqual(len(ids), 74)
        self.assertEqual(len(set(ids)), 74)

    def test_sentence_text_claim_lookup_and_response_bounds(self):
        page = self.client.get(
            "/api/v1/triples",
            query_string={
                "volume": "vol1",
                "page": 47,
                "include_text": "true",
                "limit": 1,
            },
        )
        self.assertEqual(page.status_code, 200)
        item = page.get_json()["data"][0]
        self.assertTrue(item["sentence"]["my"])
        self.assertTrue(item["sentence"]["en"])
        self.assertIn(47, item["source"]["pages"])

        claim = self.client.get(f"/api/v1/triples/{item['id']}")
        self.assertEqual(claim.status_code, 200)
        self.assertEqual(claim.get_json()["data"]["id"], item["id"])
        self.assertIn("sentence", claim.get_json()["data"])

        oversized = self.client.get(
            "/api/v1/triples",
            query_string={"include_text": "true", "limit": 26},
        )
        self.assertEqual(oversized.status_code, 400)
        self.assertIn("25", oversized.get_json()["error"]["message"])
        self.assertEqual(
            self.client.get("/api/v1/triples", query_string={"limit": 101}).status_code,
            400,
        )

    def test_role_categories_and_thematic_aggregation(self):
        query = {
            "subject_category": "E01",
            "relation_category": "R01",
            "object_category": "E04",
            "limit": 100,
        }
        triples = self.client.get("/api/v1/triples", query_string=query)
        self.assertEqual(triples.status_code, 200)
        data = triples.get_json()["data"]
        self.assertTrue(data)
        self.assertTrue(
            all(
                item["thematic"]["subject"]["id"] == "E01"
                and item["thematic"]["relation"]["id"] == "R01"
                and item["thematic"]["object"]["id"] == "E04"
                for item in data
            )
        )

        tag_ids = self.brahmin_tag_ids(20)
        patterns = self.client.get(
            "/api/v1/thematic-patterns",
            query_string=self.tag_query(tag_ids, limit=100),
        )
        self.assertEqual(patterns.status_code, 200)
        payload = patterns.get_json()
        self.assertEqual(payload["matchingTripleCount"], 74)
        self.assertEqual(payload["pagination"]["total"], 39)
        self.assertEqual(
            sum(item["tripleCount"] for item in payload["data"]),
            74,
        )

    def test_invalid_filters_and_query_bound_cursor(self):
        self.assertEqual(
            self.client.get(
                "/api/v1/triples", query_string={"page": 47}
            ).status_code,
            400,
        )
        unknown = self.client.get(
            "/api/v1/triples", query_string={"subjet": "King"}
        )
        self.assertEqual(unknown.status_code, 400)
        self.assertEqual(unknown.get_json()["error"]["code"], "invalid_request")
        missing_tag = self.client.get(
            "/api/v1/triples",
            query_string={"entity_tag_id": "e_00000000000000000000000000000000"},
        )
        self.assertEqual(missing_tag.status_code, 404)

        first = self.client.get(
            "/api/v1/triples", query_string={"volume": "vol1", "limit": 1}
        ).get_json()
        mismatched = self.client.get(
            "/api/v1/triples",
            query_string={
                "volume": "vol2",
                "limit": 1,
                "cursor": first["pagination"]["nextCursor"],
            },
        )
        self.assertEqual(mismatched.status_code, 400)
        self.assertEqual(mismatched.get_json()["error"]["code"], "invalid_cursor")

    def test_optional_api_key_and_rate_limit(self):
        app.config["PUBLIC_API_KEYS"] = "research-secret"
        self.assertEqual(self.client.get("/api/v1/stats").status_code, 401)
        self.assertEqual(self.client.get("/api/v1/docs").status_code, 200)
        authorized = self.client.get(
            "/api/v1/stats", headers={"X-API-Key": "research-secret"}
        )
        self.assertEqual(authorized.status_code, 200)

        app.config["PUBLIC_API_KEYS"] = ""
        app.config["PUBLIC_API_RATE_LIMIT"] = 2
        public_api.limiter.clear()
        self.assertEqual(self.client.get("/api/v1/stats").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/categories").status_code, 200)
        limited = self.client.get("/api/v1/tags")
        self.assertEqual(limited.status_code, 429)
        self.assertGreaterEqual(int(limited.headers["Retry-After"]), 1)
        self.assertLessEqual(int(limited.headers["Retry-After"]), 60)
        self.assertEqual(limited.get_json()["error"]["code"], "rate_limit_exceeded")


if __name__ == "__main__":
    unittest.main()
