from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_AXIAL_ROOT = APP_ROOT / "data" / "konbaung_axial_categories_v2"


class AxialCategoryStore:
    TOPOLOGY_SCHEMA_VERSION = 5

    def __init__(self, root: Path = DEFAULT_AXIAL_ROOT) -> None:
        self.root = root
        self.manifest = self._read_json(root / "manifest.json")
        self._catalog = self._read_json(root / "catalog.json")
        self.entities = {item["id"]: item for item in self._catalog["entities"]}
        self.relations = {item["id"]: item for item in self._catalog["relations"]}
        self.records = [
            json.loads(line)
            for line in (root / "triples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self._embeddings: Any | None = None
        self._pattern_centroids: dict[str, np.ndarray] | None = None
        self._role_means: dict[str, np.ndarray] = {}
        self._build_topology_index()

    # ------------------------------------------------------------------
    # Raw tag embeddings
    #
    # The axial store owns category patterns; the graph store owns the raw
    # Gemini tag vectors. Attaching the latter lets one pass over the claim
    # table produce per-pattern role centroids, which both the theme-level
    # "similar patterns" ranking and the tag-level similarity ranking reuse.
    # ------------------------------------------------------------------

    def attach_embeddings(self, embeddings: Any) -> None:
        self._embeddings = embeddings
        self._pattern_centroids = None

    def _require_embeddings(self) -> Any:
        if self._embeddings is None:
            raise ValueError("Raw tag embeddings are not available")
        return self._embeddings

    ROLE_FIELDS = {"subject": "subject", "relation": "predicate", "object": "object"}
    ROLE_KINDS = {"subject": "entity", "relation": "relation", "object": "entity"}

    def _tag_vector(self, kind: str, label: str) -> np.ndarray | None:
        record = self._require_embeddings().by_label[kind].get(label)
        if record is None:
            return None
        matrix = self._embeddings.vectors[kind]
        return np.asarray(matrix[int(record["index"])], dtype=np.float32)

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms > 0, norms, 1.0)

    def _mean_tag_vector(self, kind: str, labels: Iterable[str]) -> np.ndarray | None:
        vectors = [
            vector
            for vector in (self._tag_vector(kind, label) for label in labels)
            if vector is not None
        ]
        if not vectors:
            return None
        mean = np.mean(np.stack(vectors), axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else None

    def _role_centroids(self, records: Iterable[dict[str, Any]]) -> dict[str, np.ndarray | None]:
        selected = list(records)
        return {
            role: self._mean_tag_vector(
                self.ROLE_KINDS[role],
                (record[self.ROLE_FIELDS[role]] for record in selected),
            )
            for role in self.ROLE_FIELDS
        }

    def _pattern_centroid_matrices(self) -> dict[str, np.ndarray]:
        """Per-pattern mean subject/predicate/object vectors, corpus-wide.

        Centroids are centred on the corpus mean before renormalizing. Raw
        Gemini tag vectors for one historical corpus all sit within a narrow
        cone, so uncentred pattern cosines bunch above 0.94 and a similarity
        slider cannot separate them. Removing the shared component restores a
        usable spread while preserving the ranking's meaning.
        """

        if self._pattern_centroids is not None:
            return self._pattern_centroids
        embeddings = self._require_embeddings()
        dimensions = int(embeddings.vectors["entity"].shape[1])
        sums = {
            role: np.zeros((len(self.pattern_keys), dimensions), dtype=np.float32)
            for role in self.ROLE_FIELDS
        }
        for record in self.records:
            tags = record["categories"]
            pattern_id = self.pattern_index[(tags["s"], tags["r"], tags["o"])]
            for role, field in self.ROLE_FIELDS.items():
                vector = self._tag_vector(self.ROLE_KINDS[role], record[field])
                if vector is not None:
                    sums[role][pattern_id] += vector
        self._role_means = {
            role: np.mean(self._normalize_rows(matrix), axis=0) for role, matrix in sums.items()
        }
        self._pattern_centroids = {
            role: self._normalize_rows(self._normalize_rows(matrix) - self._role_means[role])
            for role, matrix in sums.items()
        }
        return self._pattern_centroids

    def _centred_anchor(self, role: str, vector: np.ndarray) -> np.ndarray:
        """Project an anchor centroid into the same centred pattern space."""

        self._pattern_centroid_matrices()
        centred = vector - self._role_means[role]
        norm = float(np.linalg.norm(centred))
        return centred / norm if norm > 0 else centred

    def similar_patterns(
        self,
        *,
        scope: str = "corpus",
        subject_ids: Iterable[str] = (),
        relation_ids: Iterable[str] = (),
        object_ids: Iterable[str] = (),
        volume_id: str | None = None,
        page_number: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        minimum_similarity: float = 0.9,
        limit: int = 50,
        direction: str = "ab",
        subject_tag_labels: Iterable[str] = (),
        relation_tag_labels: Iterable[str] = (),
        object_tag_labels: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Rank in-scope category patterns against the locked selection.

        The anchor is every claim currently under the applied subject /
        relation / object categories. Its three role centroids are compared
        with the corresponding centroids of every other in-scope pattern, and
        the pattern score is the mean of the available role cosines.
        """

        anchor_records, _, subjects, relations, objects = self._matching_records(
            scope=scope,
            subject_ids=subject_ids,
            relation_ids=relation_ids,
            object_ids=object_ids,
            # The anchor is the claims actually on screen, so a low-level tag
            # filter narrows what the centroid is built from: after pinning
            # "white elephant" the anchor is those claims, not the whole
            # court-minister-to-regalia pairing.
            raw_subject_labels=subject_tag_labels,
            raw_relation_labels=relation_tag_labels,
            raw_object_labels=object_tag_labels,
            direction=direction,
            volume_id=volume_id,
            page_number=page_number,
            start_page=start_page,
            end_page=end_page,
        )
        if not subjects and not relations and not objects:
            raise ValueError("Apply at least one subject, relation, or object category first")
        if not anchor_records:
            raise ValueError("The applied categories match no claims in this scope")
        minimum_similarity = float(minimum_similarity)
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("Minimum similarity must be between -1 and 1")
        limit = self._bounded_limit(limit, 200)

        anchor_centroids = self._role_centroids(anchor_records)
        available_roles = [role for role, vector in anchor_centroids.items() if vector is not None]
        if not available_roles:
            raise ValueError("The applied categories have no embedded raw tags")

        scoped_records, *_ = self._matching_records(
            scope=scope,
            volume_id=volume_id,
            page_number=page_number,
            start_page=start_page,
            end_page=end_page,
        )
        scoped_counts: Counter[int] = Counter()
        for record in scoped_records:
            tags = record["categories"]
            scoped_counts[self.pattern_index[(tags["s"], tags["r"], tags["o"])]] += 1
        anchor_patterns = {
            self.pattern_index[
                (
                    record["categories"]["s"],
                    record["categories"]["r"],
                    record["categories"]["o"],
                )
            ]
            for record in anchor_records
        }

        centroids = self._pattern_centroid_matrices()
        scores = np.mean(
            np.stack(
                [
                    centroids[role] @ self._centred_anchor(role, anchor_centroids[role])
                    for role in available_roles
                ]
            ),
            axis=0,
        )
        candidates = [
            pattern_id
            for pattern_id in scoped_counts
            if pattern_id not in anchor_patterns and float(scores[pattern_id]) >= minimum_similarity
        ]
        candidates.sort(
            key=lambda pattern_id: (
                -float(scores[pattern_id]),
                -scoped_counts[pattern_id],
            )
        )
        items = []
        for pattern_id in candidates[:limit]:
            source_id, relation_id, target_id = self.pattern_keys[pattern_id]
            items.append(
                {
                    "subjectId": source_id,
                    "relationId": relation_id,
                    "objectId": target_id,
                    "subjectLabel": self._category_label(self.entities[source_id]),
                    "relationLabel": self._category_label(self.relations[relation_id]),
                    "objectLabel": self._category_label(self.entities[target_id]),
                    "tripleCount": int(scoped_counts[pattern_id]),
                    "similarity": float(scores[pattern_id]),
                }
            )
        narrowing = {
            "subject": [str(value) for value in subject_tag_labels if str(value)],
            "relation": [str(value) for value in relation_tag_labels if str(value)],
            "object": [str(value) for value in object_tag_labels if str(value)],
        }
        return {
            "ok": True,
            "minimumSimilarity": minimum_similarity,
            "comparedRoles": available_roles,
            "narrowedBy": {role: sorted(labels) for role, labels in narrowing.items() if labels},
            "anchor": {
                "claimCount": len(anchor_records),
                "patternCount": len(anchor_patterns),
                "subjects": sorted(subjects),
                "relations": sorted(relations),
                "objects": sorted(objects),
            },
            "totalCandidates": len(candidates),
            "items": items,
        }

    def _build_topology_index(self) -> None:
        self.entity_order = [item["id"] for item in self._catalog["entities"]]
        self.relation_order = [item["id"] for item in self._catalog["relations"]]
        self.entity_index = {
            category_id: index for index, category_id in enumerate(self.entity_order)
        }
        self.relation_index = {
            category_id: index for index, category_id in enumerate(self.relation_order)
        }

        pattern_counts: Counter[tuple[str, str, str]] = Counter()
        for record in self.records:
            tags = record["categories"]
            pattern_counts[(tags["s"], tags["r"], tags["o"])] += 1
        self.pattern_keys = sorted(
            pattern_counts,
            key=lambda key: (
                self.entity_index[key[0]],
                self.relation_index[key[1]],
                self.entity_index[key[2]],
            ),
        )
        self.pattern_index = {key: index for index, key in enumerate(self.pattern_keys)}
        self.pattern_counts = pattern_counts

        entity_counterparts: dict[str, set[str]] = defaultdict(set)
        entity_relations: dict[str, set[str]] = defaultdict(set)
        entity_patterns: Counter[str] = Counter()
        relation_entities: dict[str, set[str]] = defaultdict(set)
        relation_patterns: Counter[str] = Counter()
        bundle_counts: Counter[tuple[str, str, str]] = Counter()
        bundle_patterns: dict[tuple[str, str, str], list[int]] = defaultdict(list)

        for pattern_id, (source_id, relation_id, target_id) in enumerate(self.pattern_keys):
            count = pattern_counts[(source_id, relation_id, target_id)]
            entity_counterparts[source_id].add(target_id)
            entity_counterparts[target_id].add(source_id)
            entity_relations[source_id].add(relation_id)
            entity_relations[target_id].add(relation_id)
            entity_patterns[source_id] += 1
            if target_id != source_id:
                entity_patterns[target_id] += 1
            relation_entities[relation_id].update((source_id, target_id))
            relation_patterns[relation_id] += 1

            subject_key = ("subject", source_id, relation_id)
            object_key = ("object", target_id, relation_id)
            bundle_counts[subject_key] += count
            bundle_counts[object_key] += count
            bundle_patterns[subject_key].append(pattern_id)
            bundle_patterns[object_key].append(pattern_id)

        self.entity_counterparts = entity_counterparts
        self.entity_relations = entity_relations
        self.entity_patterns = entity_patterns
        self.relation_entities = relation_entities
        self.relation_patterns = relation_patterns
        self.bundle_keys = sorted(
            bundle_counts,
            key=lambda key: (
                0 if key[0] == "subject" else 1,
                self.entity_index[key[1]],
                self.relation_index[key[2]],
            ),
        )
        self.bundle_counts = bundle_counts
        self.bundle_patterns = bundle_patterns

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(
                f"Axial category data is missing. Run import_axial_categories.py: {path}"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _bounded_limit(value: int, maximum: int = 100) -> int:
        return max(1, min(int(value), maximum))

    @staticmethod
    def _category_label(item: dict[str, Any]) -> str:
        return f"{item['tagId']} · {item['label']}"

    def catalog(self) -> dict[str, Any]:
        return {
            "ok": True,
            "schemaVersion": self._catalog["schemaVersion"],
            "manifest": self.manifest,
            "entities": self._catalog["entities"],
            "relations": self._catalog["relations"],
        }

    def _validate_filters(
        self,
        entity_ids: Iterable[str],
        relation_ids: Iterable[str],
    ) -> tuple[set[str], set[str]]:
        entities = {str(value) for value in entity_ids if str(value)}
        relations = {str(value) for value in relation_ids if str(value)}
        unknown_entities = entities - self.entities.keys()
        unknown_relations = relations - self.relations.keys()
        if unknown_entities:
            raise ValueError(f"Unknown entity categories: {', '.join(sorted(unknown_entities))}")
        if unknown_relations:
            raise ValueError(f"Unknown relation categories: {', '.join(sorted(unknown_relations))}")
        return entities, relations

    @staticmethod
    def _in_scope(
        record: dict[str, Any],
        scope: str,
        volume_id: str | None,
        page_number: int | None,
        start_page: int | None,
        end_page: int | None,
    ) -> bool:
        if scope == "corpus":
            return True
        if scope in {"vol1", "vol2", "vol3"}:
            return record["volumeId"] == scope
        if scope == "page":
            return record["volumeId"] == volume_id and int(page_number or -1) in record["pages"]
        if scope == "range":
            return record["volumeId"] == volume_id and any(
                int(start_page or -1) <= int(page) <= int(end_page or -1)
                for page in record["pages"]
            )
        raise ValueError(f"Unknown category graph scope: {scope}")

    def _matching_records(
        self,
        *,
        scope: str,
        entity_ids: Iterable[str] = (),
        subject_ids: Iterable[str] = (),
        relation_ids: Iterable[str] = (),
        object_ids: Iterable[str] = (),
        raw_kind: str | None = None,
        raw_labels: Iterable[str] = (),
        raw_subject_labels: Iterable[str] = (),
        raw_relation_labels: Iterable[str] = (),
        raw_object_labels: Iterable[str] = (),
        direction: str = "ab",
        volume_id: str | None = None,
        page_number: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        set[str],
        set[str],
        set[str],
        set[str],
    ]:
        entities, relations = self._validate_filters(entity_ids, relation_ids)
        subjects = {str(value) for value in subject_ids if str(value)}
        objects = {str(value) for value in object_ids if str(value)}
        unknown_subjects = subjects - self.entities.keys()
        unknown_objects = objects - self.entities.keys()
        if unknown_subjects:
            raise ValueError(f"Unknown subject categories: {', '.join(sorted(unknown_subjects))}")
        if unknown_objects:
            raise ValueError(f"Unknown object categories: {', '.join(sorted(unknown_objects))}")
        raw_values = {str(value) for value in raw_labels if str(value)}
        raw_subjects = {str(value) for value in raw_subject_labels if str(value)}
        raw_relations = {str(value) for value in raw_relation_labels if str(value)}
        raw_objects = {str(value) for value in raw_object_labels if str(value)}
        if raw_kind not in {None, "entity", "relation"}:
            raise ValueError(f"Unknown raw graph tag kind: {raw_kind}")
        if raw_values and raw_kind is None:
            raise ValueError("Raw graph tag labels require a tag kind")
        records = []
        for record in self.records:
            if not self._in_scope(
                record,
                scope,
                volume_id,
                page_number,
                start_page,
                end_page,
            ):
                continue
            tags = record["categories"]
            if entities and tags["s"] not in entities and tags["o"] not in entities:
                continue
            if relations and tags["r"] not in relations:
                continue
            if raw_values:
                if raw_kind == "entity" and (
                    record["subject"] not in raw_values and record["object"] not in raw_values
                ):
                    continue
                if raw_kind == "relation" and record["predicate"] not in raw_values:
                    continue
            if raw_relations and record["predicate"] not in raw_relations:
                continue
            orientations = self._orientations(
                record,
                subjects,
                objects,
                raw_subjects,
                raw_objects,
                direction,
            )
            if not orientations:
                continue
            records.append(record)
        return records, entities, subjects, relations, objects

    # ------------------------------------------------------------------
    # Entity / relation / entity matching
    #
    # The two entity slots are groups, not grammatical roles. "forward" means
    # group A sits on the claim's subject side, "reverse" means it sits on the
    # object side. `direction` picks which orientations are allowed, so the
    # same controls express both an undirected pairing and a strict one.
    # ------------------------------------------------------------------

    DIRECTIONS = ("either", "ab", "ba")

    def _orientations(
        self,
        record: dict[str, Any],
        group_a: set[str],
        group_b: set[str],
        raw_a: set[str],
        raw_b: set[str],
        direction: str,
    ) -> tuple[str, ...]:
        """Which orientations of this claim satisfy the two entity groups."""

        if direction not in self.DIRECTIONS:
            raise ValueError(f"Unknown entity pairing direction: {direction}")
        tags = record["categories"]

        def fits(side_category, side_raw, categories, raw_labels):
            if categories and side_category not in categories:
                return False
            if raw_labels and side_raw not in raw_labels:
                return False
            return True

        found = []
        if (
            direction in {"either", "ab"}
            and fits(tags["s"], record["subject"], group_a, raw_a)
            and fits(tags["o"], record["object"], group_b, raw_b)
        ):
            found.append("forward")
        if (
            direction in {"either", "ba"}
            and fits(tags["o"], record["object"], group_a, raw_a)
            and fits(tags["s"], record["subject"], group_b, raw_b)
        ):
            found.append("reverse")
        return tuple(found)

    @staticmethod
    def _normalize_tag_query(value: str) -> str:
        return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())

    def _bucket_labels(
        self,
        record: dict[str, Any],
        role: str,
        group_a: set[str],
        group_b: set[str],
        raw_a: set[str],
        raw_b: set[str],
        direction: str,
    ) -> tuple[str, ...]:
        """Raw tags this claim contributes to the given bucket.

        For the relation bucket that is simply the predicate. For an entity
        bucket it depends on orientation: a claim matched in reverse puts its
        object on the A side and its subject on the B side.
        """

        if role == "relation":
            return (record["predicate"],)
        orientations = self._orientations(record, group_a, group_b, raw_a, raw_b, direction)
        sides = {
            ("subject", "forward"): "subject",
            ("subject", "reverse"): "object",
            ("object", "forward"): "object",
            ("object", "reverse"): "subject",
        }
        labels = {record[sides[(role, orientation)]] for orientation in orientations}
        return tuple(labels)

    def filtered_tags(
        self,
        *,
        role: str,
        subject_ids: Iterable[str] = (),
        relation_ids: Iterable[str] = (),
        object_ids: Iterable[str] = (),
        query: str = "",
        offset: int = 0,
        limit: int = 100,
        scope: str = "corpus",
        volume_id: str | None = None,
        page_number: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        similar_to: Iterable[str] = (),
        minimum_similarity: float = 0.0,
        subject_tag_labels: Iterable[str] = (),
        relation_tag_labels: Iterable[str] = (),
        object_tag_labels: Iterable[str] = (),
        direction: str = "ab",
    ) -> dict[str, Any]:
        """List raw tags inside an AND-combined set of thematic role buckets.

        With no categories applied the bucket falls back to every in-scope tag
        in that role, which is what makes free-text tag lookup possible without
        first committing to a theme. Passing ``similar_to`` ranks the bucket by
        cosine against the mean vector of those anchor tags instead of by
        frequency.

        Applied low-level tag filters narrow the bucket the same way they
        narrow the graph, so a reader who pins several spellings of one king as
        the subject then sees only the relations and objects that king actually
        participates in. A role's own tag filter is deliberately not applied to
        its own bucket: that facet would otherwise collapse to the tags already
        pinned, leaving no way to add another spelling or swap one out.
        """

        if role not in {"subject", "relation", "object"}:
            raise ValueError(f"Unknown filtered tag role: {role}")
        applied_tags = {
            "subject": [str(value) for value in subject_tag_labels if str(value)],
            "relation": [str(value) for value in relation_tag_labels if str(value)],
            "object": [str(value) for value in object_tag_labels if str(value)],
        }
        narrowing = {
            other: labels for other, labels in applied_tags.items() if other != role and labels
        }
        records, _, subjects, relations, objects = self._matching_records(
            scope=scope,
            subject_ids=subject_ids,
            relation_ids=relation_ids,
            object_ids=object_ids,
            raw_subject_labels=narrowing.get("subject", ()),
            raw_relation_labels=narrowing.get("relation", ()),
            raw_object_labels=narrowing.get("object", ()),
            direction=direction,
            volume_id=volume_id,
            page_number=page_number,
            start_page=start_page,
            end_page=end_page,
        )
        selected_for_role = {
            "subject": subjects,
            "relation": relations,
            "object": objects,
        }[role]

        # Under "either" pairing a claim can match with group A on the object
        # side, so the tag that belongs in bucket A is whichever side matched.
        frequencies: Counter[str] = Counter()
        raw_a = set(narrowing.get("subject", ()))
        raw_b = set(narrowing.get("object", ()))
        for record in records:
            for label in self._bucket_labels(
                record, role, subjects, objects, raw_a, raw_b, direction
            ):
                frequencies[label] += 1
        normalized_query = self._normalize_tag_query(query)
        ranked = [
            (label, frequency)
            for label, frequency in frequencies.items()
            if not normalized_query or normalized_query in self._normalize_tag_query(label)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].casefold(), item[0]))

        anchors = [str(value) for value in similar_to if str(value)]
        similarities: dict[str, float] = {}
        if anchors:
            kind = self.ROLE_KINDS[role]
            anchor_vector = self._mean_tag_vector(kind, anchors)
            if anchor_vector is None:
                raise ValueError("None of the checked tags have an embedding")
            minimum_similarity = float(minimum_similarity)
            if not -1.0 <= minimum_similarity <= 1.0:
                raise ValueError("Minimum similarity must be between -1 and 1")
            anchor_set = set(anchors)
            scored = []
            for label, frequency in ranked:
                vector = self._tag_vector(kind, label)
                if vector is None:
                    continue
                similarity = float(vector @ anchor_vector)
                # Checked anchors stay visible so their boxes keep their state.
                if label not in anchor_set and similarity < minimum_similarity:
                    continue
                similarities[label] = similarity
                scored.append((label, frequency))
            scored.sort(
                key=lambda item: (
                    -(1.0 if item[0] in anchor_set else similarities[item[0]]),
                    -item[1],
                    item[0].casefold(),
                )
            )
            ranked = scored

        safe_offset = max(0, int(offset))
        safe_limit = self._bounded_limit(limit, 100)
        selected = ranked[safe_offset : safe_offset + safe_limit]
        next_offset = safe_offset + len(selected)
        return {
            "ok": True,
            "role": role,
            "query": query,
            "categoryScoped": bool(selected_for_role),
            "direction": direction,
            "narrowedBy": {other: sorted(labels) for other, labels in narrowing.items()},
            "similarTo": anchors,
            "minimumSimilarity": float(minimum_similarity) if anchors else None,
            "items": [
                {
                    "label": label,
                    "frequency": int(frequency),
                    "similarity": similarities.get(label),
                }
                for label, frequency in selected
            ],
            "matchingClaimCount": len(records),
            "pagination": {
                "offset": safe_offset,
                "limit": safe_limit,
                "returned": len(selected),
                "total": len(ranked),
                "hasMore": next_offset < len(ranked),
                "nextOffset": next_offset if next_offset < len(ranked) else None,
            },
            "filters": {
                "subjects": sorted(subjects),
                "relations": sorted(relations),
                "objects": sorted(objects),
            },
        }

    def topology(
        self,
        *,
        scope: str,
        entity_ids: Iterable[str] = (),
        subject_ids: Iterable[str] = (),
        relation_ids: Iterable[str] = (),
        object_ids: Iterable[str] = (),
        raw_kind: str | None = None,
        raw_tag_ids: Iterable[str] = (),
        raw_tag_labels: Iterable[str] = (),
        raw_subject_tag_ids: Iterable[str] = (),
        raw_subject_tag_labels: Iterable[str] = (),
        raw_relation_tag_ids: Iterable[str] = (),
        raw_relation_tag_labels: Iterable[str] = (),
        raw_object_tag_ids: Iterable[str] = (),
        raw_object_tag_labels: Iterable[str] = (),
        direction: str = "ab",
        volume_id: str | None = None,
        page_number: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> dict[str, Any]:
        selected_raw_ids = list(dict.fromkeys(str(value) for value in raw_tag_ids))
        selected_raw_labels = list(dict.fromkeys(str(value) for value in raw_tag_labels))
        selected_role_tag_ids = {
            "subject": list(dict.fromkeys(str(value) for value in raw_subject_tag_ids)),
            "relation": list(dict.fromkeys(str(value) for value in raw_relation_tag_ids)),
            "object": list(dict.fromkeys(str(value) for value in raw_object_tag_ids)),
        }
        selected_role_tag_labels = {
            "subject": list(dict.fromkeys(str(value) for value in raw_subject_tag_labels)),
            "relation": list(dict.fromkeys(str(value) for value in raw_relation_tag_labels)),
            "object": list(dict.fromkeys(str(value) for value in raw_object_tag_labels)),
        }
        for role in ("subject", "relation", "object"):
            if len(selected_role_tag_ids[role]) != len(selected_role_tag_labels[role]):
                raise ValueError(f"Filtered {role} tag IDs and labels must correspond")
        role_tag_count = sum(len(values) for values in selected_role_tag_ids.values())
        if role_tag_count:
            focus = {
                "kind": "filtered-tags",
                "id": "thematic-filtered-tags",
                "label": f"Thematic graph · {role_tag_count} filtered tags",
                "tagIds": [
                    value
                    for role in ("subject", "relation", "object")
                    for value in selected_role_tag_ids[role]
                ],
                "subjectTagIds": selected_role_tag_ids["subject"],
                "relationTagIds": selected_role_tag_ids["relation"],
                "objectTagIds": selected_role_tag_ids["object"],
            }
        elif selected_raw_labels:
            if raw_kind not in {"entity", "relation"}:
                raise ValueError("Semantic graph slices require a raw tag kind")
            if len(selected_raw_ids) != len(selected_raw_labels):
                raise ValueError("Semantic graph tag IDs and labels must correspond")
            label = selected_raw_labels[0]
            if len(selected_raw_labels) > 1:
                similar_count = len(selected_raw_labels) - 1
                suffix = "tag" if similar_count == 1 else "tags"
                label += f" + {similar_count} similar {suffix}"
            focus = {
                "kind": "tag-filter",
                "id": f"semantic-{raw_kind}-selection",
                "label": f"Thematic graph · {label}",
                "tagKind": raw_kind,
                "tagIds": selected_raw_ids,
                "tagLabels": selected_raw_labels,
            }
        elif scope == "page":
            focus = {
                "kind": "page",
                "id": f"{volume_id}-p{int(page_number or 0):04d}",
                "label": (
                    f"Thematic graph · {str(volume_id).upper()}, page {int(page_number or 0)}"
                ),
                "volumeId": volume_id,
                "pageNumber": page_number,
            }
        elif scope == "range":
            focus = {
                "kind": "range",
                "id": (f"{volume_id}-p{int(start_page or 0):04d}-p{int(end_page or 0):04d}"),
                "label": (
                    f"Thematic graph · {str(volume_id).upper()} "
                    f"{int(start_page or 0)}–{int(end_page or 0)}"
                ),
                "volumeId": volume_id,
                "startPage": start_page,
                "endPage": end_page,
            }
        else:
            label = "Full corpus" if scope == "corpus" else scope.upper()
            focus = {
                "kind": scope,
                "id": scope,
                "label": f"Thematic graph · {label}",
            }

        if scope in {"page", "range"}:
            focus["volumeId"] = volume_id
        if scope == "page":
            focus["pageNumber"] = page_number
        elif scope == "range":
            focus["startPage"] = start_page
            focus["endPage"] = end_page

        records, entities, subjects, relations, objects = self._matching_records(
            scope=scope,
            entity_ids=entity_ids,
            subject_ids=subject_ids,
            relation_ids=relation_ids,
            object_ids=object_ids,
            raw_kind=raw_kind,
            raw_labels=selected_raw_labels,
            raw_subject_labels=selected_role_tag_labels["subject"],
            raw_relation_labels=selected_role_tag_labels["relation"],
            raw_object_labels=selected_role_tag_labels["object"],
            direction=direction,
            volume_id=volume_id,
            page_number=page_number,
            start_page=start_page,
            end_page=end_page,
        )
        active_pattern_counts: Counter[tuple[str, str, str]] = Counter()
        active_entity_counts: Counter[str] = Counter()
        active_relation_counts: Counter[str] = Counter()
        for record in records:
            tags = record["categories"]
            pattern_key = (tags["s"], tags["r"], tags["o"])
            active_pattern_counts[pattern_key] += 1
            active_entity_counts[tags["s"]] += 1
            active_entity_counts[tags["o"]] += 1
            active_relation_counts[tags["r"]] += 1

        active_entity_counterparts: dict[str, set[str]] = defaultdict(set)
        active_entity_relations: dict[str, set[str]] = defaultdict(set)
        active_entity_patterns: Counter[str] = Counter()
        active_relation_entities: dict[str, set[str]] = defaultdict(set)
        active_relation_patterns: Counter[str] = Counter()
        for source_id, relation_id, target_id in active_pattern_counts:
            active_entity_counterparts[source_id].add(target_id)
            active_entity_counterparts[target_id].add(source_id)
            active_entity_relations[source_id].add(relation_id)
            active_entity_relations[target_id].add(relation_id)
            active_entity_patterns[source_id] += 1
            if target_id != source_id:
                active_entity_patterns[target_id] += 1
            active_relation_entities[relation_id].update((source_id, target_id))
            active_relation_patterns[relation_id] += 1

        entity_categories = []
        for category_id in self.entity_order:
            descriptor = self.entities[category_id]
            entity_categories.append(
                {
                    **descriptor,
                    "mentionCount": int(descriptor["count"]),
                    "activeMentionCount": active_entity_counts[category_id],
                    "counterpartCount": len(self.entity_counterparts[category_id]),
                    "activeCounterpartCount": len(active_entity_counterparts[category_id]),
                    "relationCount": len(self.entity_relations[category_id]),
                    "activeRelationCount": len(active_entity_relations[category_id]),
                    "patternCount": self.entity_patterns[category_id],
                    "activePatternCount": active_entity_patterns[category_id],
                }
            )

        relation_categories = []
        for category_id in self.relation_order:
            descriptor = self.relations[category_id]
            relation_categories.append(
                {
                    **descriptor,
                    "tripleCount": int(descriptor["count"]),
                    "activeTripleCount": active_relation_counts[category_id],
                    "entityCount": len(self.relation_entities[category_id]),
                    "activeEntityCount": len(active_relation_entities[category_id]),
                    "patternCount": self.relation_patterns[category_id],
                    "activePatternCount": active_relation_patterns[category_id],
                }
            )

        patterns = [
            [
                self.entity_index[source_id],
                self.relation_index[relation_id],
                self.entity_index[target_id],
                self.pattern_counts[(source_id, relation_id, target_id)],
                active_pattern_counts[(source_id, relation_id, target_id)],
            ]
            for source_id, relation_id, target_id in self.pattern_keys
        ]

        incidence_bundles = []
        for role, entity_id, relation_id in self.bundle_keys:
            active_count = 0
            for pattern_id in self.bundle_patterns[(role, entity_id, relation_id)]:
                active_count += active_pattern_counts[self.pattern_keys[pattern_id]]
            incidence_bundles.append(
                [
                    self.entity_index[entity_id],
                    self.relation_index[relation_id],
                    0 if role == "subject" else 1,
                    self.bundle_counts[(role, entity_id, relation_id)],
                    active_count,
                    self.bundle_patterns[(role, entity_id, relation_id)],
                ]
            )

        return {
            "ok": True,
            "schemaVersion": self.TOPOLOGY_SCHEMA_VERSION,
            "categoryMode": True,
            "focus": focus,
            "layout": {"kind": "thematic", "scope": scope},
            "entityCategories": entity_categories,
            "relationCategories": relation_categories,
            "patterns": patterns,
            "incidenceBundles": incidence_bundles,
            "filters": {
                "entities": sorted(entities),
                "subjects": sorted(subjects),
                "relations": sorted(relations),
                "objects": sorted(objects),
                "rawTagKind": raw_kind if selected_raw_labels else None,
                "rawTagIds": selected_raw_ids,
                "rawSubjectTagIds": selected_role_tag_ids["subject"],
                "rawRelationTagIds": selected_role_tag_ids["relation"],
                "rawObjectTagIds": selected_role_tag_ids["object"],
            },
            "claimCount": len(records),
            "entityMentionCount": sum(active_entity_counts.values()),
            "availableCount": len(self.records),
            "truncated": False,
            "limit": None,
        }

    def evidence(
        self,
        source_id: str,
        target_id: str,
        *,
        relation_ids: Iterable[str] = (),
        raw_kind: str | None = None,
        raw_labels: Iterable[str] = (),
        raw_subject_labels: Iterable[str] = (),
        raw_relation_labels: Iterable[str] = (),
        raw_object_labels: Iterable[str] = (),
        scope: str = "corpus",
        volume_id: str | None = None,
        page_number: int | None = None,
        start_page: int | None = None,
        end_page: int | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        if source_id not in self.entities or target_id not in self.entities:
            raise ValueError("Unknown entity category pair")
        records, _, _, _, _ = self._matching_records(
            scope=scope,
            relation_ids=relation_ids,
            raw_kind=raw_kind,
            raw_labels=raw_labels,
            raw_subject_labels=raw_subject_labels,
            raw_relation_labels=raw_relation_labels,
            raw_object_labels=raw_object_labels,
            volume_id=volume_id,
            page_number=page_number,
            start_page=start_page,
            end_page=end_page,
        )
        matching = [
            record
            for record in records
            if record["categories"]["s"] == source_id and record["categories"]["o"] == target_id
        ]
        safe_offset = max(0, int(offset))
        safe_limit = self._bounded_limit(limit)
        selected = matching[safe_offset : safe_offset + safe_limit]
        items = [
            {
                "claimId": record["id"],
                "sentenceId": record["sid"],
                "sentenceMy": record["sentenceMy"],
                "sentenceEn": record["sentenceEn"],
                "volumeId": record["volumeId"],
                "ownerPage": record["ownerPage"],
                "ordinal": record["ordinal"],
                "subject": record["subject"],
                "predicate": record["predicate"],
                "object": record["object"],
                "relationCategory": self.relations[record["categories"]["r"]],
            }
            for record in selected
        ]
        next_offset = safe_offset + len(items)
        has_more = next_offset < len(matching)
        return {
            "ok": True,
            "items": items,
            "hasMore": has_more,
            "nextOffset": next_offset if has_more else None,
            "availableCount": len(matching),
        }
