from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = APP_ROOT / "static" / "data" / "konbaung"
DEFAULT_V1_DATA_ROOT = (
    APP_ROOT / "data_archives" / "konbaung_reader_pre_sentence_annotations_20260713"
)
DEFAULT_V3_DATA_ROOT = APP_ROOT / "data" / "konbaung_historiography_v3_canonical_20260724"
DEFAULT_AXIAL_DATA_ROOT = APP_ROOT / "data" / "konbaung_axial_categories_v2"
DEFAULT_TRIPLE_VARIANT = "v3"


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def slice_utf16(text: str, start: int, end: int) -> str:
    encoded = text.encode("utf-16-le")
    return encoded[start * 2 : end * 2].decode("utf-16-le")


def compact(value: str) -> str:
    return "".join(str(value).split())


def exact_utf16_ranges(text: str, literal: str) -> list[tuple[int, int]]:
    compact_text: list[str] = []
    codepoint_offsets: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        compact_text.append(char)
        codepoint_offsets.append(index)
    compact_literal = compact(literal)
    if not compact_literal:
        return []
    joined = "".join(compact_text)
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        found = joined.find(compact_literal, cursor)
        if found < 0:
            break
        start_codepoint = codepoint_offsets[found]
        end_codepoint = codepoint_offsets[found + len(compact_literal) - 1] + 1
        ranges.append(
            (
                utf16_length(text[:start_codepoint]),
                utf16_length(text[:end_codepoint]),
            )
        )
        cursor = found + 1
    return ranges


def valid_utf16_range(text: str, start: Any, end: Any) -> bool:
    return (
        isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= utf16_length(text)
    )


def resolved_range(
    record: dict[str, Any],
    canonical_text: str,
    literal_keys: tuple[str, ...],
) -> tuple[int, int] | None:
    literal = next(
        (str(record.get(key, "")) for key in literal_keys if str(record.get(key, "")).strip()),
        "",
    )
    start = record.get("startUtf16")
    end = record.get("endUtf16")
    if valid_utf16_range(canonical_text, start, end):
        actual = slice_utf16(canonical_text, start, end)
        if not literal or compact(actual) == compact(literal):
            return int(start), int(end)
    candidates = exact_utf16_ranges(canonical_text, literal)
    return candidates[0] if len(candidates) == 1 else None


def range_overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def range_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0


class CorpusStore:
    def __init__(
        self,
        data_root: Path = DEFAULT_DATA_ROOT,
        v1_data_root: Path = DEFAULT_V1_DATA_ROOT,
        v3_data_root: Path = DEFAULT_V3_DATA_ROOT,
        axial_data_root: Path = DEFAULT_AXIAL_DATA_ROOT,
    ) -> None:
        self.data_root = data_root
        self.v1_data_root = v1_data_root
        self.v3_data_root = v3_data_root
        self.axial_data_root = axial_data_root

    @lru_cache(maxsize=1)
    def _base_index(self) -> dict[str, Any]:
        path = self.data_root / "index.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Chronicle data has not been built. Run build_konbaung_reader_data.py. Missing: {path}"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    @lru_cache(maxsize=1)
    def _v1_index(self) -> dict[str, Any]:
        path = self.v1_data_root / "index.json"
        if not path.exists():
            raise FileNotFoundError(f"V1 triple corpus is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @lru_cache(maxsize=1)
    def _v3_index(self) -> dict[str, Any]:
        path = self.v3_data_root / "index.json"
        if not path.exists():
            raise FileNotFoundError(f"V3 triple corpus is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @lru_cache(maxsize=1)
    def _axial_manifest(self) -> dict[str, Any]:
        path = self.axial_data_root / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Axial category manifest is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @lru_cache(maxsize=1)
    def _axial_catalog(self) -> dict[str, dict[str, Any]]:
        path = self.axial_data_root / "catalog.json"
        if not path.exists():
            raise FileNotFoundError(f"Axial category catalog is missing: {path}")
        catalog = json.loads(path.read_text(encoding="utf-8"))
        return {item["id"]: item for kind in ("entities", "relations") for item in catalog[kind]}

    @lru_cache(maxsize=3)
    def _axial_assignments(self, volume_id: str) -> dict[str, dict[str, Any]]:
        path = self.axial_data_root / "assignments" / f"{volume_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Axial assignments are missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))["sentences"]

    @lru_cache(maxsize=1)
    def index(self) -> dict[str, Any]:
        index = copy.deepcopy(self._base_index())
        v1_index = self._v1_index()
        v3_index = self._v3_index()
        axial_manifest = self._axial_manifest()
        current_pages = {
            (volume["id"], int(page))
            for volume in index["volumes"]
            for page in volume["availablePages"]
        }
        v1_pages = {
            (volume["id"], int(page))
            for volume in v1_index["volumes"]
            for page in volume["availablePages"]
        }
        excluded_v1_annotations = 0
        for volume_id, page_number in sorted(v1_pages - current_pages):
            excluded_v1_annotations += len(self._v1_page(volume_id, page_number)["annotations"])
        index["defaultTripleVariant"] = DEFAULT_TRIPLE_VARIANT
        index["tripleVariants"] = [
            {
                "id": "v1",
                "shortLabel": "V1",
                "label": "V1 · Page triples",
                "description": "Archived page-level triple annotation pass",
                "sourceTripleCount": int(v1_index["totals"]["annotations"]),
                "pageAppearanceCount": int(v1_index["totals"]["annotations"])
                - excluded_v1_annotations,
                "excludedSourcePages": len(v1_pages - current_pages),
                "groundingMode": "spans",
            },
            {
                "id": "v2",
                "shortLabel": "V2",
                "label": "V2 · Sentence triples",
                "description": "Current translated-sentence triple annotation pass",
                "sourceTripleCount": int(index["totals"]["triples"]),
                "pageAppearanceCount": int(index["totals"]["annotationPageAppearances"]),
                "excludedSourcePages": 0,
                "groundingMode": "spans",
            },
            {
                "id": "v3",
                "shortLabel": "V3",
                "label": "V3 · Historiography triples",
                "description": ("Canonical analytical triples with final axial power categories"),
                "sourceTripleCount": int(v3_index["totals"]["canonicalTriples"]),
                "pageAppearanceCount": int(v3_index["totals"]["pageAppearanceTriples"]),
                "excludedSourcePages": sum(
                    len(pages) for pages in v3_index["pagesWithUnavailableSentences"].values()
                ),
                "groundingMode": "none",
                "taggedTripleCount": int(axial_manifest["totals"]["taggedTriples"]),
                "unresolvedTripleCount": int(axial_manifest["totals"]["unresolvedTriples"]),
            },
        ]
        return index

    def triple_variant(self, triple_variant: str | None) -> dict[str, Any]:
        selected = str(triple_variant or DEFAULT_TRIPLE_VARIANT).strip().lower()
        for variant in self.index()["tripleVariants"]:
            if variant["id"] == selected:
                return variant
        raise KeyError(f"Unknown triple variant: {triple_variant}")

    @lru_cache(maxsize=64)
    def _base_page(self, volume_id: str, page_number: int) -> dict[str, Any]:
        path = self.data_root / "pages" / volume_id / f"{page_number:04d}.json"
        if not path.exists():
            raise KeyError(f"Unknown page: {volume_id}/{page_number}")
        return json.loads(path.read_text(encoding="utf-8"))

    @lru_cache(maxsize=64)
    def _v1_page(self, volume_id: str, page_number: int) -> dict[str, Any]:
        path = self.v1_data_root / "pages" / volume_id / f"{page_number:04d}.json"
        if not path.exists():
            raise KeyError(f"Unknown V1 page: {volume_id}/{page_number}")
        return json.loads(path.read_text(encoding="utf-8"))

    @lru_cache(maxsize=3)
    def _v3_sentences(self, volume_id: str) -> dict[str, dict[str, Any]]:
        path = self.v3_data_root / "sentences" / f"{volume_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"V3 sentence data is missing: {path}")
        return json.loads(path.read_text(encoding="utf-8"))["sentences"]

    @staticmethod
    def _sentence_ranges(sentence: dict[str, Any]) -> list[tuple[int, int]]:
        fragments = sentence.get("fragments") or [sentence]
        return [
            (int(fragment["startUtf16"]), int(fragment["endUtf16"]))
            for fragment in fragments
            if isinstance(fragment.get("startUtf16"), int)
            and isinstance(fragment.get("endUtf16"), int)
        ]

    @staticmethod
    def _choose_sentence(
        sentences: list[dict[str, Any]],
        anchors: list[tuple[int, int]],
    ) -> tuple[dict[str, Any], str]:
        if not sentences:
            raise ValueError("Current page has no translated sentences")
        scored: list[tuple[int, int, int, dict[str, Any]]] = []
        for order, sentence in enumerate(sentences):
            ranges = CorpusStore._sentence_ranges(sentence)
            overlap = sum(
                range_overlap(anchor, sentence_range)
                for anchor in anchors
                for sentence_range in ranges
            )
            distance = min(
                (
                    range_distance(anchor, sentence_range)
                    for anchor in anchors
                    for sentence_range in ranges
                ),
                default=10**9,
            )
            scored.append((-overlap, distance, order, sentence))
        scored.sort(key=lambda item: item[:3])
        method = "maximum_evidence_overlap" if scored[0][0] < 0 else "nearest_source_span"
        if not anchors:
            method = "first_translated_sentence_fallback"
        return scored[0][3], method

    @staticmethod
    def _adapt_endpoint(
        endpoint: dict[str, Any],
        canonical_text: str,
    ) -> tuple[dict[str, Any], tuple[int, int] | None]:
        adapted = copy.deepcopy(endpoint)
        resolved = resolved_range(adapted, canonical_text, ("text",))
        inferred = bool(adapted.get("inferred"))
        if resolved:
            start, end = resolved
            presence = "present"
            fragments = [{"startUtf16": start, "endUtf16": end}]
        else:
            start = None
            end = None
            presence = "inferred" if inferred else "unattached"
            fragments = []
        adapted.update(
            {
                "source": "v1_page_triple_pass",
                "startUtf16": start,
                "endUtf16": end,
                "fragments": fragments,
                "pagePresence": presence,
                "alignmentMethod": "v1_source_offsets" if resolved else None,
                "characterOverlap": 1.0 if resolved else 0.0,
            }
        )
        return adapted, resolved

    @staticmethod
    def _adapt_v1_annotation(
        annotation: dict[str, Any],
        canonical_text: str,
        sentences: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        source_id = str(annotation["id"])
        subject, subject_range = CorpusStore._adapt_endpoint(annotation["subject"], canonical_text)
        object_, object_range = CorpusStore._adapt_endpoint(annotation["object"], canonical_text)
        source_evidence = annotation["evidence"]
        evidence_range = resolved_range(
            source_evidence,
            canonical_text,
            ("canonicalText", "text"),
        )
        anchors = [
            span for span in (evidence_range, subject_range, object_range) if span is not None
        ]
        sentence, sentence_mapping_method = CorpusStore._choose_sentence(sentences, anchors)
        evidence_fragments: list[dict[str, Any]] = []
        if evidence_range:
            evidence_start, evidence_end = evidence_range
            evidence_fragments.append(
                {
                    "canonicalText": slice_utf16(canonical_text, evidence_start, evidence_end),
                    "startUtf16": evidence_start,
                    "endUtf16": evidence_end,
                }
            )
            evidence_status = "resolved"
        else:
            evidence_start = None
            evidence_end = None
            evidence_status = "unattached"

        relation = copy.deepcopy(annotation["relation"])
        relation.update(
            {
                "predicateText": None,
                "predicateGloss": None,
                "predicateSource": "not_available_in_v1",
                "startUtf16": None,
                "endUtf16": None,
                "fragments": [],
                "candidateCount": 0,
                "inferred": False,
                "inference": None,
                "pagePresence": "not_available",
                "alignmentMethod": None,
                "characterOverlap": 0.0,
            }
        )
        evidence = copy.deepcopy(source_evidence)
        evidence.update(
            {
                "translation": sentence["translation"],
                "startUtf16": evidence_start,
                "endUtf16": evidence_end,
                "fragments": evidence_fragments,
                "candidateCount": len(evidence_fragments),
                "status": evidence_status,
                "alignmentMethod": ("v1_source_offsets" if evidence_range else None),
                "characterOverlap": 1.0 if evidence_range else 0.0,
                "crossPage": bool(sentence["crossPage"]),
                "ownerPage": int(sentence["ownerPage"]),
            }
        )
        validation = copy.deepcopy(annotation.get("validation") or {})
        validation["sourceStatus"] = validation.get("status")
        validation["status"] = (
            validation.get("status") if evidence_status == "resolved" else "unattached"
        )
        validation["evidenceStatus"] = evidence_status
        validation["sentenceMappingMethod"] = sentence_mapping_method
        provenance = copy.deepcopy(annotation.get("provenance") or {})
        provenance.update(
            {
                "tripleVariant": "v1",
                "sourceAnnotationId": source_id,
                "sentenceId": sentence["id"],
            }
        )
        return {
            "id": f"v1-{source_id}",
            "baseTripleId": source_id,
            "sentenceId": sentence["id"],
            "subject": subject,
            "relation": relation,
            "object": object_,
            "pageAttachment": evidence_status,
            "evidence": evidence,
            "metadata": copy.deepcopy(annotation.get("metadata") or {}),
            "validation": validation,
            "provenance": provenance,
            "rawSourceRecord": copy.deepcopy(annotation.get("rawSourceRecord")),
        }, sentence_mapping_method

    @lru_cache(maxsize=64)
    def _v1_comparison_page(self, volume_id: str, page_number: int) -> dict[str, Any]:
        current = copy.deepcopy(self._base_page(volume_id, page_number))
        source = self._v1_page(volume_id, page_number)
        if source["canonicalText"] != current["canonicalText"]:
            raise ValueError(f"V1/V2 canonical text mismatch: {volume_id}/{page_number}")
        sentences = current["sentences"]
        for sentence in sentences:
            sentence["tripleIds"] = []
        sentence_map = {sentence["id"]: sentence for sentence in sentences}
        annotations: list[dict[str, Any]] = []
        mapping_counts: dict[str, int] = {}
        for source_annotation in source["annotations"]:
            annotation, mapping_method = self._adapt_v1_annotation(
                source_annotation,
                current["canonicalText"],
                sentences,
            )
            annotations.append(annotation)
            sentence_map[annotation["sentenceId"]]["tripleIds"].append(annotation["id"])
            mapping_counts[mapping_method] = mapping_counts.get(mapping_method, 0) + 1
        current["annotations"] = annotations
        current["tripleVariant"] = self.triple_variant("v1")
        current["diagnostics"] = {
            **current["diagnostics"],
            "annotationCount": len(annotations),
            "sourceAnnotationCount": len(source["annotations"]),
            "sentenceMappingCounts": mapping_counts,
        }
        return current

    @staticmethod
    def _v3_endpoint(text: str) -> dict[str, Any]:
        return {
            "text": text,
            "gloss": None,
            "type": None,
            "language": "en",
            "source": "v3_analytical_triple",
            "inferred": False,
            "inference": None,
            "startUtf16": None,
            "endUtf16": None,
            "fragments": [],
            "pagePresence": "ungrounded",
            "alignmentMethod": None,
            "characterOverlap": 0.0,
        }

    @staticmethod
    def _v3_relation(label: str) -> dict[str, Any]:
        return {
            "rawLabel": label,
            "predicateText": None,
            "predicateGloss": None,
            "predicateSource": "v3_analytical_triple",
            "startUtf16": None,
            "endUtf16": None,
            "fragments": [],
            "candidateCount": 0,
            "inferred": False,
            "inference": None,
            "pagePresence": "ungrounded",
            "alignmentMethod": None,
            "characterOverlap": 0.0,
        }

    @lru_cache(maxsize=64)
    def _v3_comparison_page(self, volume_id: str, page_number: int) -> dict[str, Any]:
        current = copy.deepcopy(self._base_page(volume_id, page_number))
        source_sentences = self._v3_sentences(volume_id)
        axial_assignments = self._axial_assignments(volume_id)
        axial_catalog = self._axial_catalog()
        unavailable_reason = "No accepted V3 annotation is available for this page."
        sentence_map = {sentence["id"]: sentence for sentence in current["sentences"]}
        for sentence in current["sentences"]:
            sentence["tripleIds"] = []
            sentence["tripleDecision"] = "unavailable"
            sentence["tripleJustification"] = unavailable_reason

        annotations: list[dict[str, Any]] = []
        annotated_sentences = 0
        skipped_sentences = 0
        unavailable_sentences = 0
        thinking_levels: dict[str, int] = {}
        for sentence_id, sentence in sentence_map.items():
            source_sentence = source_sentences.get(sentence_id)
            if source_sentence is None:
                unavailable_sentences += 1
                continue
            if (
                source_sentence["my"] != sentence["text"]
                or source_sentence["en"] != sentence["translation"]
            ):
                raise ValueError(
                    f"V3/current translation mismatch: {volume_id}/{page_number}/{sentence_id}"
                )
            decision = source_sentence["decision"]
            sentence["tripleDecision"] = decision
            sentence["tripleJustification"] = source_sentence["justification"]
            if decision == "annotate":
                annotated_sentences += 1
            else:
                skipped_sentences += 1
            selected_from = source_sentence["selectedFrom"]
            axial_assignment = axial_assignments.get(sentence_id)
            thinking_level = selected_from["thinkingLevel"]
            thinking_levels[thinking_level] = thinking_levels.get(thinking_level, 0) + 1
            for ordinal, triple in enumerate(source_sentence["triples"], start=1):
                base_triple_id = f"v3-{sentence_id}-t{ordinal:03d}"
                annotation_id = f"{base_triple_id}-p{page_number:04d}"
                annotation = {
                    "id": annotation_id,
                    "baseTripleId": base_triple_id,
                    "sentenceId": sentence_id,
                    "subject": self._v3_endpoint(triple["subject"]),
                    "relation": self._v3_relation(triple["predicate"]),
                    "object": self._v3_endpoint(triple["object"]),
                    "pageAttachment": "ungrounded",
                    "evidence": {
                        "canonicalText": sentence["canonicalText"],
                        "translation": sentence["translation"],
                        "startUtf16": None,
                        "endUtf16": None,
                        "fragments": [],
                        "candidateCount": 0,
                        "status": "unattached",
                        "alignmentMethod": None,
                        "characterOverlap": 0.0,
                        "crossPage": bool(sentence["crossPage"]),
                        "ownerPage": int(sentence["ownerPage"]),
                    },
                    "metadata": {},
                    "validation": {
                        "status": "ungrounded",
                        "evidenceStatus": "unattached",
                    },
                    "provenance": {
                        "tripleVariant": "v3",
                        "pass": "historiography_ungrounded_canonical",
                        "sourcePage": selected_from["key"],
                        "sourcePageNumber": selected_from["page"],
                        "sentenceId": sentence_id,
                        "tripleOrdinal": ordinal,
                        "thinkingLevel": thinking_level,
                        "selectionRule": source_sentence["selectionRule"],
                        "sourceAppearanceCount": len(source_sentence["sourceAppearances"]),
                    },
                    "rawSourceRecord": copy.deepcopy(triple),
                }
                axial_tags = (
                    axial_assignment.get("triples", {}).get(str(ordinal))
                    if axial_assignment
                    else None
                )
                if axial_tags:
                    annotation["axial"] = {
                        "status": "accepted",
                        "sourcePage": axial_assignment["sourcePage"],
                        "reason": None,
                        "subject": copy.deepcopy(axial_catalog[axial_tags["s"]]),
                        "relation": copy.deepcopy(axial_catalog[axial_tags["r"]]),
                        "object": copy.deepcopy(axial_catalog[axial_tags["o"]]),
                    }
                else:
                    annotation["axial"] = {
                        "status": (
                            axial_assignment["status"] if axial_assignment else "unresolved"
                        ),
                        "sourcePage": (
                            axial_assignment["sourcePage"]
                            if axial_assignment
                            else selected_from["key"]
                        ),
                        "reason": (
                            axial_assignment["reason"]
                            if axial_assignment
                            else "No final axial assignment is available."
                        ),
                        "subject": None,
                        "relation": None,
                        "object": None,
                    }
                annotations.append(annotation)
                sentence["tripleIds"].append(annotation_id)

        current["annotations"] = annotations
        current["tripleVariant"] = self.triple_variant("v3")
        available_sentences = annotated_sentences + skipped_sentences
        if unavailable_sentences == 0:
            v3_status = "accepted"
        elif available_sentences:
            v3_status = "partial"
        else:
            v3_status = "unavailable"
        current["diagnostics"] = {
            **current["diagnostics"],
            "annotationCount": len(annotations),
            "sourceAnnotationCount": len(annotations),
            "ungroundedAnnotationCount": len(annotations),
            "annotatedSentenceCount": annotated_sentences,
            "skippedSentenceCount": skipped_sentences,
            "unavailableSentenceCount": unavailable_sentences,
            "v3Status": v3_status,
            "v3ThinkingLevels": thinking_levels,
            "axialTaggedAnnotationCount": sum(
                annotation["axial"]["status"] == "accepted" for annotation in annotations
            ),
            "axialUnresolvedAnnotationCount": sum(
                annotation["axial"]["status"] != "accepted" for annotation in annotations
            ),
            "v3UnavailableReason": (None if unavailable_sentences == 0 else unavailable_reason),
        }
        return current

    @lru_cache(maxsize=128)
    def page(
        self,
        volume_id: str,
        page_number: int,
        triple_variant: str = DEFAULT_TRIPLE_VARIANT,
    ) -> dict[str, Any]:
        safe_volume = str(volume_id).strip().lower()
        if safe_volume not in {item["id"] for item in self.index()["volumes"]}:
            raise KeyError(f"Unknown volume: {volume_id}")
        variant = self.triple_variant(triple_variant)
        if variant["id"] == "v1":
            return self._v1_comparison_page(safe_volume, page_number)
        if variant["id"] == "v3":
            return self._v3_comparison_page(safe_volume, page_number)
        page = self._base_page(safe_volume, page_number)
        return {**page, "tripleVariant": variant}

    def has_page(self, volume_id: str, page_number: int) -> bool:
        try:
            self._base_page(str(volume_id).strip().lower(), page_number)
            return True
        except KeyError:
            return False
