from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pyoxigraph import Store

try:
    from .graph_schema import (
        EMBEDDING_GRAPH,
        EMBEDDING_ROOT,
        GRAPH_DB,
        GRAPH_MANIFEST,
        GRAPH_ROOT,
        KG,
        RDF,
        RDFS,
        SOURCE_GRAPH,
        entity_uri,
        page_uri,
        read_jsonl,
        relation_uri,
    )
except ImportError:  # Direct `python app.py` execution.
    from graph_schema import (
        EMBEDDING_GRAPH,
        EMBEDDING_ROOT,
        GRAPH_DB,
        GRAPH_MANIFEST,
        GRAPH_ROOT,
        KG,
        RDF,
        RDFS,
        SOURCE_GRAPH,
        entity_uri,
        page_uri,
        read_jsonl,
        relation_uri,
    )


ID_PATTERN = re.compile(r"^e_[0-9a-f]{32}$")
SID_PATTERN = re.compile(r"^vol[123]_[A-Za-z0-9_]+$")
PREFIXES = f"""
PREFIX kg: <{KG}>
PREFIX rdf: <{RDF}>
PREFIX rdfs: <{RDFS}>
"""


def normalize_query(value: str) -> str:
    return " ".join(
        value.casefold().replace("_", " ").replace("-", " ").split()
    )


class ChronicleGraphStore:
    """Read-only RDF graph access optimized for interactive visualization.

    Topology and evidence are deliberately separate. Topology responses contain
    only indexed nodes and aggregated edges; sentence text is loaded on demand
    for one selected edge.
    """

    TOPOLOGY_SCHEMA_VERSION = 2

    def __init__(self) -> None:
        if not GRAPH_DB.exists() or not GRAPH_MANIFEST.exists():
            raise FileNotFoundError(
                "Chronicle graph database is missing. Run build_graph_database.py."
            )
        self.store = Store.read_only(str(GRAPH_DB))
        self.manifest = json.loads(GRAPH_MANIFEST.read_text(encoding="utf-8"))
        self.records = {
            "entity": list(read_jsonl(EMBEDDING_ROOT / "node_records.jsonl")),
            "relation": list(read_jsonl(EMBEDDING_ROOT / "edge_records.jsonl")),
        }
        self.vectors = {
            "entity": np.load(
                EMBEDDING_ROOT / "node_base_vectors.npy", mmap_mode="r"
            ),
            "relation": np.load(
                EMBEDDING_ROOT / "edge_base_vectors.npy", mmap_mode="r"
            ),
        }
        self.by_id: dict[str, dict[str, dict[str, Any]]] = {}
        self.by_label: dict[str, dict[str, dict[str, Any]]] = {}
        for kind, records in self.records.items():
            self.by_id[kind] = {
                record["baseKey"]: {
                    **record,
                    "id": record["baseKey"],
                    "kind": kind,
                }
                for record in records
            }
            self.by_label[kind] = {
                record["tag"]: self.by_id[kind][record["baseKey"]]
                for record in records
            }

    @staticmethod
    def _term(row: Any, name: str) -> str:
        value = row[name]
        return "" if value is None else value.value

    @staticmethod
    def _resource_id(value: str) -> str:
        return value.rsplit(":", 1)[-1]

    @staticmethod
    def _safe_id(value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError(f"Invalid graph resource ID: {value}")
        return value

    @staticmethod
    def _safe_sid(value: str) -> str:
        if not SID_PATTERN.fullmatch(value):
            raise ValueError(f"Invalid sentence ID: {value}")
        return value

    @staticmethod
    def _bounded_limit(value: int, maximum: int) -> int:
        return max(1, min(int(value), maximum))

    def stats(self) -> dict[str, Any]:
        return {
            "ok": True,
            "database": self.manifest["database"],
            "schemaVersion": self.manifest["schemaVersion"],
            "generatedAt": self.manifest["generatedAt"],
            "counts": self.manifest["counts"],
            "statements": self.manifest["statements"],
            "namedGraphs": self.manifest["namedGraphs"],
            "embeddingSort": {
                "model": self.manifest["embeddingSort"]["model"],
                "view": self.manifest["embeddingSort"]["view"],
                "dimensions": self.manifest["embeddingSort"]["dimensions"],
            },
        }

    def search(
        self,
        query: str = "",
        kind: str = "all",
        sort: str = "frequency",
        anchor_kind: str | None = None,
        anchor_id: str | None = None,
        minimum_similarity: float | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        if kind not in {"all", "entity", "relation"}:
            raise ValueError(f"Unknown graph search kind: {kind}")
        if sort not in {"frequency", "gemini"}:
            raise ValueError(f"Unknown graph search order: {sort}")
        limit = self._bounded_limit(limit, 200)
        query_normalized = normalize_query(query)
        kinds = ("entity", "relation") if kind == "all" else (kind,)

        if sort == "gemini":
            if minimum_similarity is not None:
                minimum_similarity = float(minimum_similarity)
                if not -1.0 <= minimum_similarity <= 1.0:
                    raise ValueError(
                        "Minimum Gemini similarity must be between -1 and 1"
                    )
            if anchor_kind not in {"entity", "relation"} or not anchor_id:
                raise ValueError(
                    "Gemini sorting requires a selected entity or relation"
                )
            anchor_id = self._safe_id(anchor_id)
            anchor = self.by_id[anchor_kind].get(anchor_id)
            if anchor is None:
                raise KeyError(f"Unknown {anchor_kind}: {anchor_id}")
            records = self.records[anchor_kind]
            matrix = self.vectors[anchor_kind]
            anchor_vector = np.asarray(
                matrix[int(anchor["index"])], dtype=np.float32
            )
            similarities = np.asarray(matrix @ anchor_vector, dtype=np.float32)
            candidates = [
                index
                for index, record in enumerate(records)
                if record["baseKey"] != anchor_id
                and (
                    minimum_similarity is None
                    or float(similarities[index]) >= minimum_similarity
                )
                and (
                    not query_normalized
                    or query_normalized in normalize_query(record["tag"])
                )
            ]
            candidates.sort(
                key=lambda index: (
                    -float(similarities[index]),
                    -int(records[index]["frequency"]),
                    records[index]["tag"],
                )
            )
            results = [
                {
                    "id": records[index]["baseKey"],
                    "kind": anchor_kind,
                    "label": records[index]["tag"],
                    "normalizedLabel": records[index]["normalizedTag"],
                    "frequency": int(records[index]["frequency"]),
                    "similarity": float(similarities[index]),
                }
                for index in candidates[:limit]
            ]
            return {
                "ok": True,
                "sort": sort,
                "query": query,
                "minimumSimilarity": minimum_similarity,
                "anchor": {
                    "id": anchor_id,
                    "kind": anchor_kind,
                    "label": anchor["tag"],
                    "frequency": int(anchor["frequency"]),
                },
                "results": results,
            }

        results = []
        for target_kind in kinds:
            for record in self.records[target_kind]:
                if (
                    query_normalized
                    and query_normalized not in normalize_query(record["tag"])
                ):
                    continue
                results.append(
                    {
                        "id": record["baseKey"],
                        "kind": target_kind,
                        "label": record["tag"],
                        "normalizedLabel": record["normalizedTag"],
                        "frequency": int(record["frequency"]),
                        "similarity": None,
                    }
                )
        results.sort(
            key=lambda item: (
                -item["frequency"],
                item["kind"],
                item["label"],
            )
        )
        return {
            "ok": True,
            "sort": sort,
            "query": query,
            "minimumSimilarity": None,
            "anchor": None,
            "results": results[:limit],
        }

    def tags_topology(
        self,
        kind: str,
        tag_ids: Iterable[str],
    ) -> dict[str, Any]:
        """Return the complete union graph for up to 100 selected raw tags."""

        records = self.resolve_tags(kind, tag_ids)
        ids = [record["baseKey"] for record in records]

        uri_for = entity_uri if kind == "entity" else relation_uri
        selected_uris = " ".join(f"<{uri_for(value)}>" for value in ids)
        if kind == "entity":
            filter_body = (
                f"VALUES ?selectedTag {{ {selected_uris} }} "
                "{ ?claim rdf:subject ?selectedTag . } UNION "
                "{ ?claim rdf:object ?selectedTag . }"
            )
        else:
            filter_body = (
                f"VALUES ?selectedTag {{ {selected_uris} }} "
                "?claim rdf:predicate ?selectedTag ."
            )

        rows = self._topology_rows(filter_body, None)
        labels = [record["tag"] for record in records]
        if len(labels) == 1:
            label = labels[0]
        else:
            similar_count = len(labels) - 1
            suffix = "tag" if similar_count == 1 else "tags"
            label = f"{labels[0]} + {similar_count} similar {suffix}"

        return self._compact_payload(
            rows,
            {
                "kind": "tag-filter",
                "id": f"semantic-{kind}-selection",
                "label": label,
                "tagKind": kind,
                "tagIds": ids,
                "tagLabels": labels,
            },
            {"kind": "force"},
            available_count=len(rows),
            truncated=False,
            limit=None,
            extra_nodes=(
                (
                    {
                        "id": record["baseKey"],
                        "label": record["tag"],
                        "frequency": int(record["frequency"]),
                    }
                    for record in records
                )
                if kind == "entity"
                else ()
            ),
        )

    def resolve_tags(
        self,
        kind: str,
        tag_ids: Iterable[str],
    ) -> list[dict[str, Any]]:
        """Validate raw tag IDs and return their embedding records in order."""

        if kind not in {"entity", "relation"}:
            raise ValueError(f"Unknown graph tag kind: {kind}")
        ids = list(dict.fromkeys(self._safe_id(value) for value in tag_ids))
        if not ids:
            raise ValueError("Select at least one tag")
        if len(ids) > 100:
            raise ValueError("Select no more than 100 tags")

        missing = [value for value in ids if value not in self.by_id[kind]]
        if missing:
            raise KeyError(f"Unknown {kind}: {missing[0]}")
        return [self.by_id[kind][value] for value in ids]

    def _topology_rows(
        self,
        filter_body: str,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        limit_clause = "" if limit is None else f"LIMIT {int(limit)}"
        query = (
            PREFIXES
            + f"""
SELECT DISTINCT ?claim ?subject ?subjectLabel ?predicate ?predicateLabel
                ?object ?objectLabel
WHERE {{
  GRAPH <{SOURCE_GRAPH}> {{
    ?claim a kg:Claim ;
           rdf:subject ?subject ;
           rdf:predicate ?predicate ;
           rdf:object ?object .
    {filter_body}
  }}
  GRAPH <{EMBEDDING_GRAPH}> {{
    ?subject rdfs:label ?subjectLabel .
    ?predicate rdfs:label ?predicateLabel .
    ?object rdfs:label ?objectLabel .
  }}
}}
ORDER BY ?claim
{limit_clause}
"""
        )
        rows = []
        for row in self.store.query(query):
            rows.append(
                {
                    "claimId": self._resource_id(self._term(row, "claim")),
                    "subject": {
                        "id": self._resource_id(self._term(row, "subject")),
                        "label": self._term(row, "subjectLabel"),
                    },
                    "relation": {
                        "id": self._resource_id(self._term(row, "predicate")),
                        "label": self._term(row, "predicateLabel"),
                    },
                    "object": {
                        "id": self._resource_id(self._term(row, "object")),
                        "label": self._term(row, "objectLabel"),
                    },
                }
            )
        return rows

    def _evidence_rows(
        self,
        filter_body: str,
        offset: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = (
            PREFIXES
            + f"""
SELECT ?claim ?sid ?sentenceMy ?sentenceEn ?volumeId ?ownerPage ?ordinal
WHERE {{
  GRAPH <{SOURCE_GRAPH}> {{
    ?claim a kg:Claim ;
           rdf:subject ?subject ;
           rdf:predicate ?predicate ;
           rdf:object ?object ;
           kg:sentence ?sentence ;
           kg:tripleOrdinal ?ordinal .
    ?sentence <http://purl.org/dc/terms/identifier> ?sid ;
              kg:burmeseText ?sentenceMy ;
              kg:englishTranslation ?sentenceEn ;
              kg:volumeId ?volumeId ;
              kg:ownerPage ?ownerPage .
    {filter_body}
  }}
}}
ORDER BY ?volumeId ?ownerPage ?sid ?ordinal
OFFSET {int(offset)}
LIMIT {int(limit)}
"""
        )
        rows = []
        for row in self.store.query(query):
            rows.append(
                {
                    "claimId": self._resource_id(self._term(row, "claim")),
                    "sentenceId": self._term(row, "sid"),
                    "sentenceMy": self._term(row, "sentenceMy"),
                    "sentenceEn": self._term(row, "sentenceEn"),
                    "volumeId": self._term(row, "volumeId"),
                    "ownerPage": int(
                        self._resource_id(self._term(row, "ownerPage"))
                    ),
                    "ordinal": int(self._term(row, "ordinal")),
                }
            )
        return rows

    def _compact_payload(
        self,
        claims: Iterable[dict[str, Any]],
        focus: dict[str, Any],
        layout: dict[str, Any],
        *,
        available_count: int | None,
        truncated: bool,
        limit: int | None,
        extra_nodes: Iterable[dict[str, Any]] = (),
    ) -> dict[str, Any]:
        node_map: dict[str, tuple[str, int]] = {}
        edge_map: dict[tuple[str, str, str], list[Any]] = {}
        claim_count = 0

        for claim in claims:
            claim_count += 1
            for endpoint_key in ("subject", "object"):
                endpoint = claim[endpoint_key]
                record = self.by_id["entity"][endpoint["id"]]
                node_map[endpoint["id"]] = (
                    endpoint["label"],
                    int(record["frequency"]),
                )
            key = (
                claim["subject"]["id"],
                claim["relation"]["id"],
                claim["object"]["id"],
            )
            edge = edge_map.setdefault(
                key,
                [claim["relation"]["label"], 0],
            )
            edge[1] += 1

        for node in extra_nodes:
            node_map[node["id"]] = (node["label"], int(node["frequency"]))

        ordered_nodes = sorted(
            node_map.items(),
            key=lambda item: (-item[1][1], item[1][0], item[0]),
        )
        node_indexes = {
            node_id: index
            for index, (node_id, _) in enumerate(ordered_nodes)
        }
        nodes = [
            [node_id, label, frequency, None, None]
            for node_id, (label, frequency) in ordered_nodes
        ]
        edges = [
            [
                node_indexes[source],
                node_indexes[target],
                relation_id,
                label_and_count[0],
                label_and_count[1],
            ]
            for (source, relation_id, target), label_and_count in sorted(
                edge_map.items(),
                key=lambda item: (
                    -item[1][1],
                    item[1][0],
                    item[0][0],
                    item[0][2],
                ),
            )
        ]
        return {
            "ok": True,
            "schemaVersion": self.TOPOLOGY_SCHEMA_VERSION,
            "focus": focus,
            "layout": layout,
            "nodes": nodes,
            "edges": edges,
            "claimCount": claim_count,
            "availableCount": available_count,
            "truncated": bool(truncated),
            "limit": limit,
        }

    def overview_files(self, scope: str) -> tuple[Path, Path]:
        if scope not in {"corpus", "vol1", "vol2", "vol3"}:
            raise ValueError(f"Invalid graph scope: {scope}")
        overview = self.manifest.get("overview", {})
        if int(overview.get("schemaVersion", 0)) != 3:
            raise RuntimeError(
                "Atlas artifacts are stale. Run build_graph_overview.py."
            )
        record = overview["scopes"][scope]
        return (
            GRAPH_ROOT / record["path"],
            GRAPH_ROOT / record["gzipPath"],
        )

    def page_topology(
        self,
        volume_id: str,
        page_number: int,
        limit: int = 500,
    ) -> dict[str, Any]:
        if volume_id not in {"vol1", "vol2", "vol3"}:
            raise ValueError(f"Invalid volume: {volume_id}")
        page_number = int(page_number)
        limit = self._bounded_limit(limit, 1000)
        rows = self._topology_rows(
            f"?claim kg:appearsOnPage <{page_uri(volume_id, page_number)}> .",
            limit + 1,
        )
        truncated = len(rows) > limit
        claims = rows[:limit]
        return self._compact_payload(
            claims,
            {
                "kind": "page",
                "id": f"{volume_id}:{page_number}",
                "label": f"{volume_id.upper()}, source page {page_number}",
            },
            {"kind": "triples"},
            available_count=None if truncated else len(claims),
            truncated=truncated,
            limit=limit,
        )

    def page_range_topology(
        self,
        volume_id: str,
        start_page: int,
        end_page: int,
    ) -> dict[str, Any]:
        if volume_id not in {"vol1", "vol2", "vol3"}:
            raise ValueError(f"Invalid volume: {volume_id}")
        start_page = int(start_page)
        end_page = int(end_page)
        if start_page < 1 or end_page < 1:
            raise ValueError("Page numbers must be positive")
        if end_page < start_page:
            raise ValueError("Range end page must be greater than or equal to start page")

        range_pages = " ".join(
            f"<{page_uri(volume_id, page_number)}>"
            for page_number in range(start_page, end_page + 1)
        )
        rows = self._topology_rows(
            f"""
VALUES ?rangePage {{ {range_pages} }}
?claim kg:appearsOnPage ?rangePage .
""",
            None,
        )
        payload = self._compact_payload(
            rows,
            {
                "kind": "range",
                "id": f"{volume_id}:{start_page}-{end_page}",
                "label": (
                    f"{volume_id.upper()}, source pages "
                    f"{start_page}\u2013{end_page}"
                ),
                "volumeId": volume_id,
                "startPage": start_page,
                "endPage": end_page,
            },
            {"kind": "triples"},
            available_count=len(rows),
            truncated=False,
            limit=None,
        )
        return payload

    def entity_topology(
        self,
        entity_id: str,
        depth: int = 1,
        limit: int = 300,
    ) -> dict[str, Any]:
        entity_id = self._safe_id(entity_id)
        record = self.by_id["entity"].get(entity_id)
        if record is None:
            raise KeyError(f"Unknown entity: {entity_id}")
        depth = max(1, min(int(depth), 2))
        limit = self._bounded_limit(limit, 600)
        uri = entity_uri(entity_id)
        first_budget = limit if depth == 1 else max(1, limit // 2)
        first_rows = self._topology_rows(
            "{ "
            f"?claim rdf:subject <{uri}> . "
            "} UNION { "
            f"?claim rdf:object <{uri}> . "
            "}",
            first_budget + 1,
        )
        first_truncated = len(first_rows) > first_budget
        claims = first_rows[:first_budget]
        second_truncated = False

        if depth == 2 and len(claims) < limit:
            neighbor_ids = sorted(
                {
                    endpoint["id"]
                    for claim in claims
                    for endpoint in (claim["subject"], claim["object"])
                    if endpoint["id"] != entity_id
                }
            )
            if neighbor_ids:
                remaining = limit - len(claims)
                neighbor_uris = " ".join(
                    f"<{entity_uri(item)}>" for item in neighbor_ids[:160]
                )
                candidates = self._topology_rows(
                    f"VALUES ?neighbor {{ {neighbor_uris} }} "
                    "{ ?claim rdf:subject ?neighbor . } UNION "
                    "{ ?claim rdf:object ?neighbor . }",
                    remaining + len(claims) + 1,
                )
                seen = {claim["claimId"] for claim in claims}
                unique = [
                    claim
                    for claim in candidates
                    if claim["claimId"] not in seen
                ]
                second_truncated = len(unique) > remaining
                claims.extend(unique[:remaining])

        available_count = int(record["frequency"])
        return self._compact_payload(
            claims,
            {
                "kind": "entity",
                "id": entity_id,
                "label": record["tag"],
                "frequency": available_count,
                "depth": depth,
            },
            {"kind": "radial" if depth == 1 else "force"},
            available_count=available_count,
            truncated=(
                first_truncated
                or second_truncated
                or (depth == 1 and available_count > len(claims))
            ),
            limit=limit,
            extra_nodes=(
                {
                    "id": entity_id,
                    "label": record["tag"],
                    "frequency": available_count,
                },
            ),
        )

    def relation_topology(
        self,
        relation_id: str,
        limit: int = 300,
    ) -> dict[str, Any]:
        relation_id = self._safe_id(relation_id)
        record = self.by_id["relation"].get(relation_id)
        if record is None:
            raise KeyError(f"Unknown relation: {relation_id}")
        limit = self._bounded_limit(limit, 600)
        rows = self._topology_rows(
            f"?claim rdf:predicate <{relation_uri(relation_id)}> .",
            limit + 1,
        )
        available_count = int(record["frequency"])
        truncated = len(rows) > limit or available_count > limit
        return self._compact_payload(
            rows[:limit],
            {
                "kind": "relation",
                "id": relation_id,
                "label": record["tag"],
                "frequency": available_count,
            },
            {"kind": "bipartite"},
            available_count=available_count,
            truncated=truncated,
            limit=limit,
        )

    def claim_topology(
        self,
        sentence_id: str,
        ordinal: int,
    ) -> dict[str, Any]:
        sentence_id = self._safe_sid(sentence_id)
        ordinal = int(ordinal)
        if ordinal < 1:
            raise ValueError("Triple ordinal must be positive")
        rows = self._topology_rows(
            f"?claim kg:sentence <urn:konbaung:sentence:{sentence_id}> ; "
            f"kg:tripleOrdinal {ordinal} .",
            1,
        )
        if not rows:
            raise KeyError(f"Unknown claim: {sentence_id}/{ordinal}")
        claim = rows[0]
        return self._compact_payload(
            rows,
            {
                "kind": "claim",
                "id": claim["claimId"],
                "label": (
                    f"{claim['subject']['label']} — "
                    f"{claim['relation']['label']} → "
                    f"{claim['object']['label']}"
                ),
                "sentenceId": sentence_id,
                "ordinal": ordinal,
            },
            {"kind": "claim"},
            available_count=1,
            truncated=False,
            limit=1,
        )

    def evidence(
        self,
        source_id: str,
        relation_id: str,
        target_id: str,
        *,
        offset: int = 0,
        limit: int = 20,
        sentence_id: str | None = None,
        ordinal: int | None = None,
    ) -> dict[str, Any]:
        source_id = self._safe_id(source_id)
        relation_id = self._safe_id(relation_id)
        target_id = self._safe_id(target_id)
        if source_id not in self.by_id["entity"]:
            raise KeyError(f"Unknown entity: {source_id}")
        if target_id not in self.by_id["entity"]:
            raise KeyError(f"Unknown entity: {target_id}")
        if relation_id not in self.by_id["relation"]:
            raise KeyError(f"Unknown relation: {relation_id}")
        offset = max(0, int(offset))
        limit = self._bounded_limit(limit, 100)
        filters = (
            f"?claim rdf:subject <{entity_uri(source_id)}> ; "
            f"rdf:predicate <{relation_uri(relation_id)}> ; "
            f"rdf:object <{entity_uri(target_id)}> ."
        )
        if sentence_id is not None:
            sentence_id = self._safe_sid(sentence_id)
            if ordinal is None or int(ordinal) < 1:
                raise ValueError(
                    "A positive triple ordinal is required with sentenceId"
                )
            filters += (
                f" ?claim kg:sentence <urn:konbaung:sentence:{sentence_id}> ; "
                f"kg:tripleOrdinal {int(ordinal)} ."
            )
        rows = self._evidence_rows(filters, offset, limit + 1)
        has_more = len(rows) > limit
        items = rows[:limit]
        return {
            "ok": True,
            "source": source_id,
            "relation": relation_id,
            "target": target_id,
            "items": items,
            "offset": offset,
            "limit": limit,
            "hasMore": has_more,
            "nextOffset": offset + len(items) if has_more else None,
        }
