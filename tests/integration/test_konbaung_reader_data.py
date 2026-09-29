from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

from pipeline.corpus.build_reader_data import span_candidates, utf16_offset
from pipeline.corpus.build_sentence_reader_data import (
    cross_page_endpoint_record,
    endpoint_is_present,
    resolve_cross_page_triple,
)
from konbaung_reader_app.app import (
    align_token_glosses,
    exact_segmented_gloss_schema,
    migrate_cached_gloss_pairs,
    validate_raw_gloss_fields,
    validate_token_glosses,
)
from konbaung_reader_app.corpus import CorpusStore
from konbaung_reader_app.reader_bridge import _dictionary_display_fields


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.getenv(
        "KONBAUNG_READER_DATA_ROOT",
        str(ROOT / "konbaung_reader_app" / "static" / "data" / "konbaung"),
    )
)
SOURCE_ROOT = ROOT / "konbaung_viable_pages_manual_ranges"


def compact(value: str) -> str:
    return "".join(value.split())


def slice_utf16(text: str, start: int, end: int) -> str:
    encoded = text.encode("utf-16-le")
    return encoded[start * 2 : end * 2].decode("utf-16-le")


class AlignmentUnitTests(unittest.TestCase):
    def test_structured_token_glosses_require_exact_sequence(self) -> None:
        tokens = [
            {"text": "က", "startUtf16": 0, "endUtf16": 1},
            {"text": "ခ", "startUtf16": 1, "endUtf16": 2},
        ]
        schema = exact_segmented_gloss_schema(len(tokens))
        glosses = schema.model_validate(
            {"tokens": [{"my": "က", "en": "from"}, {"my": "ခ", "en": "letter kha"}]}
        )
        validated = validate_token_glosses(glosses, tokens)
        self.assertEqual([item["en"] for item in validated], ["from", "letter kha"])

    def test_structured_token_glosses_reject_missing_positions(self) -> None:
        tokens = [{"text": "က", "startUtf16": 0, "endUtf16": 1}]
        schema = exact_segmented_gloss_schema(len(tokens))
        glosses = schema.model_validate({"tokens": []})
        with self.assertRaisesRegex(ValueError, "1 of 1 tokens unglossed"):
            validate_token_glosses(glosses, tokens)

    def test_structured_token_glosses_reject_changed_burmese(self) -> None:
        tokens = [{"text": "က", "startUtf16": 0, "endUtf16": 1}]
        schema = exact_segmented_gloss_schema(len(tokens))
        glosses = schema.model_validate({"tokens": [{"my": "ခ", "en": "changed"}]})
        with self.assertRaisesRegex(ValueError, "1 of 1 tokens unglossed"):
            validate_token_glosses(glosses, tokens)

    def test_structured_token_glosses_drop_inserted_extras(self) -> None:
        tokens = [
            {"text": "က", "startUtf16": 0, "endUtf16": 1},
            {"text": "ခ", "startUtf16": 1, "endUtf16": 2},
        ]
        schema = exact_segmented_gloss_schema(len(tokens))
        glosses = schema.model_validate(
            {
                "tokens": [
                    {"my": "က", "en": "ka"},
                    {"my": "extra", "en": "discarded"},
                    {"my": "ခ", "en": "kha"},
                ]
            }
        )
        validated = validate_token_glosses(glosses, tokens)
        self.assertEqual(
            [(item["my"], item["en"]) for item in validated], [("က", "ka"), ("ခ", "kha")]
        )

    def test_sequence_alignment_retains_matches_after_changed_item(self) -> None:
        tokens = [
            {"text": "a", "startUtf16": 0, "endUtf16": 1},
            {"text": "b", "startUtf16": 1, "endUtf16": 2},
            {"text": "c", "startUtf16": 2, "endUtf16": 3},
        ]
        schema = exact_segmented_gloss_schema(len(tokens))
        glosses = schema.model_validate(
            {"tokens": [{"my": "a", "en": "A"}, {"my": "wrong", "en": "X"}, {"my": "c", "en": "C"}]}
        )
        aligned = align_token_glosses(glosses, tokens)
        self.assertEqual(set(aligned), {0, 2})

    def test_structured_token_schema_forbids_extra_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "only my and en"):
            validate_raw_gloss_fields({"tokens": [{"my": "က", "en": "ka", "id": 1}]})

    def test_gemini_schema_has_only_two_item_fields(self) -> None:
        schema = exact_segmented_gloss_schema(2).model_json_schema()
        self.assertIn("Exactly 2", schema["properties"]["tokens"]["description"])
        self.assertEqual(set(schema["$defs"]["TokenGloss"]["properties"]), {"my", "en"})

    def test_cached_glosses_migrate_across_whitespace_boundary_split(self) -> None:
        canonical = "ဆယ့် နှစ်"
        migrated = migrate_cached_gloss_pairs(
            [{"my": canonical, "en": "twelve"}],
            [{"text": "ဆယ့်"}, {"text": " "}, {"text": "နှစ်"}],
            canonical,
        )
        self.assertEqual(
            migrated,
            [
                {"my": "ဆယ့်", "en": "twelve"},
                {"my": " ", "en": ""},
                {"my": "နှစ်", "en": "twelve"},
            ],
        )


class BuiltCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = CorpusStore(DATA_ROOT)
        cls.index = cls.store.index()

    def test_expected_page_and_annotation_counts(self) -> None:
        totals = self.index["totals"]
        self.assertEqual(totals["pages"], 1215)
        self.assertEqual(totals["sentences"], 11282)
        self.assertEqual(totals["translations"], 11282)
        self.assertEqual(totals["triples"], 23440)
        self.assertEqual(totals["annotationPageAppearances"], 24342)
        self.assertEqual(sum(volume["pageCount"] for volume in self.index["volumes"]), 1215)

    def test_only_translated_sentence_triples_are_live(self) -> None:
        base_ids = set()
        annotation_appearances = 0
        sentence_ids = set()
        for volume in self.index["volumes"]:
            for page_number in volume["availablePages"]:
                page = self.store.page(volume["id"], page_number)
                sentence_ids.update(item["id"] for item in page["sentences"])
                annotation_appearances += len(page["annotations"])
                for annotation in page["annotations"]:
                    self.assertEqual(
                        annotation["provenance"]["pass"], "translated_sentence_triples"
                    )
                    base_ids.add(annotation["baseTripleId"])
        self.assertEqual(len(sentence_ids), 11282)
        self.assertEqual(len(base_ids), 23440)
        self.assertEqual(annotation_appearances, 24342)

    def test_non_contiguous_page_ranges_are_explicit(self) -> None:
        vol1 = next(volume for volume in self.index["volumes"] if volume["id"] == "vol1")
        self.assertIn(62, vol1["availablePages"])
        self.assertNotIn(63, vol1["availablePages"])
        self.assertIn(67, vol1["availablePages"])
        for duplicate_scan in (225, 242, 295, 301):
            self.assertNotIn(duplicate_scan, vol1["availablePages"])

    def test_canonical_source_checksum_and_text(self) -> None:
        page = self.store.page("vol1", 47)
        source = SOURCE_ROOT / "konbaung_vol1" / "pages" / "page_0047.txt"
        raw = source.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        exact = raw.decode("utf-8")
        self.assertEqual(page["canonicalText"], exact)
        self.assertEqual(
            page["source"]["checksum"], hashlib.sha1(exact.encode("utf-8")).hexdigest()
        )

    def test_every_exact_endpoint_round_trips(self) -> None:
        checked = 0
        for volume in self.index["volumes"]:
            for page_number in volume["availablePages"]:
                page = self.store.page(volume["id"], page_number)
                for annotation in page["annotations"]:
                    for role in ("subject", "predicate", "object"):
                        endpoint = (
                            annotation["relation"] if role == "predicate" else annotation[role]
                        )
                        if not str(endpoint.get("alignmentMethod", "")).startswith("exact_"):
                            continue
                        if not (
                            isinstance(endpoint.get("startUtf16"), int)
                            and isinstance(endpoint.get("endUtf16"), int)
                        ):
                            self.assertEqual(endpoint.get("pagePresence"), "absent")
                            self.assertEqual(endpoint.get("fragments"), [])
                            continue
                        ranges = endpoint.get("fragments") or [endpoint]
                        actual = "".join(
                            slice_utf16(
                                page["canonicalText"],
                                fragment["startUtf16"],
                                fragment["endUtf16"],
                            )
                            for fragment in ranges
                        )
                        expected = compact(
                            endpoint["predicateText"] if role == "predicate" else endpoint["text"]
                        )
                        if endpoint.get("pagePresence") == "partial":
                            self.assertIn(compact(actual), expected)
                        else:
                            self.assertEqual(
                                compact(actual),
                                expected,
                                f"{page['id']} {annotation['id']} {role}",
                            )
                        checked += 1
        self.assertGreater(checked, 35000)

    def test_every_sentence_fragment_and_translation_is_page_local(self) -> None:
        checked = 0
        for volume in self.index["volumes"]:
            for page_number in volume["availablePages"]:
                page = self.store.page(volume["id"], page_number)
                sentence_map = {item["id"]: item for item in page["sentences"]}
                self.assertEqual(len(sentence_map), page["diagnostics"]["sentenceCount"])
                for sentence in page["sentences"]:
                    for fragment in sentence["fragments"]:
                        actual = slice_utf16(
                            page["canonicalText"], fragment["startUtf16"], fragment["endUtf16"]
                        )
                        self.assertEqual(actual, fragment["canonicalText"])
                    self.assertTrue(sentence["translation"])
                    checked += 1
                for annotation in page["annotations"]:
                    sentence = sentence_map[annotation["sentenceId"]]
                    evidence = annotation["evidence"]
                    if evidence["status"] == "unattached":
                        self.assertEqual(page["pageNumber"], sentence["ownerPage"])
                        self.assertIsNone(evidence["startUtf16"])
                        self.assertIsNone(evidence["endUtf16"])
                        self.assertEqual(evidence["fragments"], [])
                        continue
                    self.assertEqual(evidence["startUtf16"], sentence["startUtf16"])
                    self.assertEqual(evidence["endUtf16"], sentence["endUtf16"])
                    self.assertEqual(evidence["fragments"], sentence["fragments"])
                    self.assertEqual(evidence["translation"], sentence["translation"])
        self.assertEqual(checked, 12097)

    def test_cross_page_sentence_is_shared_without_merging_pages(self) -> None:
        left = self.store.page("vol1", 47)
        right = self.store.page("vol1", 48)
        sentence_id = "vol1_s000003"
        left_sentence = next(item for item in left["sentences"] if item["id"] == sentence_id)
        right_sentence = next(item for item in right["sentences"] if item["id"] == sentence_id)
        self.assertTrue(left_sentence["crossPage"])
        self.assertEqual(left_sentence["translation"], right_sentence["translation"])
        self.assertNotEqual(left_sentence["canonicalText"], right_sentence["canonicalText"])
        self.assertTrue(all(item["pageNumber"] == 47 for item in [left]))
        self.assertTrue(all(item["pageNumber"] == 48 for item in [right]))
        left_ids = {
            item["baseTripleId"]
            for item in left["annotations"]
            if item["sentenceId"] == sentence_id
        }
        right_ids = {
            item["baseTripleId"]
            for item in right["annotations"]
            if item["sentenceId"] == sentence_id
        }
        self.assertEqual(
            left_ids,
            {"vol1_s000003-t001", "vol1_s000003-t002"},
        )
        self.assertEqual(
            right_ids,
            {"vol1_s000003-t003", "vol1_s000003-t004"},
        )
        self.assertTrue(left_ids.isdisjoint(right_ids))

    def test_cross_page_annotations_have_only_exact_page_local_attachments(self) -> None:
        attached_count = 0
        unattached_count = 0
        for volume in self.index["volumes"]:
            for page_number in volume["availablePages"]:
                page = self.store.page(volume["id"], page_number)
                for annotation in page["annotations"]:
                    if not annotation["provenance"]["crossPage"]:
                        continue
                    endpoints = {
                        "subject": annotation["subject"],
                        "predicate": annotation["relation"],
                        "object": annotation["object"],
                    }
                    present_roles = []
                    for role, endpoint in endpoints.items():
                        presence = endpoint.get("pagePresence")
                        fragments = endpoint.get("fragments", [])
                        literal = (
                            endpoint.get("predicateText")
                            if role == "predicate"
                            else endpoint.get("text")
                        ) or ""
                        method = endpoint.get("alignmentMethod")
                        self.assertNotIn("maximum_character_overlap", str(method))
                        self.assertNotIn("page_context", str(method))
                        if presence in {"present", "partial"}:
                            present_roles.append(role)
                            self.assertTrue(fragments)
                            self.assertIn(
                                method,
                                {
                                    "exact_sentence_source",
                                    "exact_sentence_triple_context",
                                },
                            )
                            actual = "".join(
                                slice_utf16(
                                    page["canonicalText"],
                                    fragment["startUtf16"],
                                    fragment["endUtf16"],
                                )
                                for fragment in fragments
                            )
                            if presence == "present":
                                self.assertEqual(compact(actual), compact(literal))
                            else:
                                self.assertIn(compact(actual), compact(literal))
                        else:
                            self.assertEqual(fragments, [])
                            self.assertIsNone(endpoint.get("startUtf16"))
                            self.assertIsNone(endpoint.get("endUtf16"))

                    if annotation["pageAttachment"] == "unattached":
                        unattached_count += 1
                        self.assertEqual(page_number, annotation["provenance"]["ownerPage"])
                        self.assertEqual(present_roles, [])
                        self.assertEqual(annotation["evidence"]["status"], "unattached")
                        self.assertEqual(annotation["evidence"]["fragments"], [])
                    else:
                        attached_count += 1
                        self.assertTrue(present_roles)
                        self.assertEqual(annotation["evidence"]["status"], "resolved")
        self.assertGreater(attached_count, 800)
        self.assertEqual(unattached_count, 10)


class TripleVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = CorpusStore(DATA_ROOT)
        cls.index = cls.store.index()

    def test_index_exposes_read_only_v1_v2_and_v3_triple_passes(self) -> None:
        self.assertEqual(self.index["defaultTripleVariant"], "v2")
        variants = {variant["id"]: variant for variant in self.index["tripleVariants"]}
        self.assertEqual(set(variants), {"v1", "v2", "v3"})
        self.assertEqual(variants["v1"]["sourceTripleCount"], 19047)
        self.assertEqual(variants["v1"]["pageAppearanceCount"], 18992)
        self.assertEqual(variants["v1"]["excludedSourcePages"], 4)
        self.assertEqual(variants["v2"]["sourceTripleCount"], 23440)
        self.assertEqual(variants["v2"]["pageAppearanceCount"], 24342)
        self.assertEqual(variants["v3"]["sourceTripleCount"], 27129)
        self.assertEqual(variants["v3"]["pageAppearanceCount"], 31582)
        self.assertEqual(variants["v3"]["excludedSourcePages"], 2)
        self.assertEqual(variants["v3"]["groundingMode"], "none")

    def test_switching_triples_keeps_current_sentences_and_translations(self) -> None:
        v1 = self.store.page("vol1", 47, "v1")
        v2 = self.store.page("vol1", 47, "v2")
        v3 = self.store.page("vol1", 47, "v3")
        sentence_fields = lambda page: [
            (
                sentence["id"],
                sentence["text"],
                sentence["translation"],
                sentence["fragments"],
            )
            for sentence in page["sentences"]
        ]
        self.assertEqual(v1["canonicalText"], v2["canonicalText"])
        self.assertEqual(sentence_fields(v1), sentence_fields(v2))
        self.assertEqual(v2["canonicalText"], v3["canonicalText"])
        self.assertEqual(sentence_fields(v2), sentence_fields(v3))
        self.assertEqual(len(v1["annotations"]), 15)
        self.assertEqual(len(v2["annotations"]), 9)
        self.assertEqual(len(v3["annotations"]), 16)
        self.assertTrue(all(sentence["translation"] for sentence in v1["sentences"]))
        self.assertTrue(all(sentence["translation"] for sentence in v3["sentences"]))
        self.assertTrue(
            all(
                annotation["provenance"]["tripleVariant"] == "v1"
                for annotation in v1["annotations"]
            )
        )
        self.assertTrue(
            all(
                annotation["provenance"]["tripleVariant"] == "v3"
                and annotation["pageAttachment"] == "ungrounded"
                and annotation["evidence"]["fragments"] == []
                and annotation["evidence"]["status"] == "unattached"
                for annotation in v3["annotations"]
            )
        )

    def test_v3_is_sentence_linked_but_has_no_page_spans(self) -> None:
        annotation_count = 0
        page_count = 0
        canonical_triple_ids: set[str] = set()
        unavailable_sentence_count = 0
        partial_pages: list[str] = []
        unavailable_pages: list[str] = []
        for volume in self.index["volumes"]:
            for page_number in volume["availablePages"]:
                page = self.store.page(volume["id"], page_number, "v3")
                page_count += 1
                sentence_map = {sentence["id"]: sentence for sentence in page["sentences"]}
                if page["diagnostics"]["v3Status"] == "unavailable":
                    unavailable_pages.append(f"{volume['id']}-p{page_number:04d}")
                    self.assertEqual(page["annotations"], [])
                    self.assertTrue(
                        all(
                            sentence["tripleDecision"] == "unavailable"
                            and sentence["tripleJustification"]
                            for sentence in page["sentences"]
                        )
                    )
                    continue
                if page["diagnostics"]["v3Status"] == "partial":
                    partial_pages.append(f"{volume['id']}-p{page_number:04d}")

                annotations = {annotation["id"]: annotation for annotation in page["annotations"]}
                assigned = [
                    triple_id
                    for sentence in page["sentences"]
                    for triple_id in sentence["tripleIds"]
                ]
                self.assertEqual(len(assigned), len(set(assigned)))
                self.assertEqual(set(assigned), set(annotations))
                for annotation in annotations.values():
                    self.assertIn(annotation["sentenceId"], sentence_map)
                    self.assertEqual(annotation["pageAttachment"], "ungrounded")
                    self.assertEqual(annotation["evidence"]["status"], "unattached")
                    for endpoint in (
                        annotation["subject"],
                        annotation["relation"],
                        annotation["object"],
                        annotation["evidence"],
                    ):
                        self.assertIsNone(endpoint["startUtf16"])
                        self.assertIsNone(endpoint["endUtf16"])
                        self.assertEqual(endpoint["fragments"], [])
                    canonical_triple_ids.add(annotation["baseTripleId"])
                annotation_count += len(annotations)
                unavailable_sentence_count += page["diagnostics"]["unavailableSentenceCount"]

        self.assertEqual(page_count, 1215)
        self.assertEqual(annotation_count, 31582)
        self.assertEqual(len(canonical_triple_ids), 27129)
        self.assertEqual(unavailable_sentence_count, 60)
        self.assertEqual(partial_pages, ["vol2-p0132", "vol3-p0511"])
        self.assertEqual(unavailable_pages, [])

    def test_v3_duplicate_selection_prefers_most_triples(self) -> None:
        duplicate_sentence_count = 0
        removed_appearances = 0
        for volume in self.index["volumes"]:
            for sentence in self.store._v3_sentences(volume["id"]).values():
                appearances = sentence["sourceAppearances"]
                if len(appearances) == 1:
                    continue
                duplicate_sentence_count += 1
                removed_appearances += len(appearances) - 1
                maximum = max(item["tripleCount"] for item in appearances)
                self.assertEqual(sentence["selectedFrom"]["tripleCount"], maximum)
                top = [item for item in appearances if item["tripleCount"] == maximum]
                owner_page = sentence["ownerPage"]
                expected_page = min((item["page"] != owner_page, item["page"]) for item in top)[1]
                self.assertEqual(sentence["selectedFrom"]["page"], expected_page)

        self.assertEqual(duplicate_sentence_count, 806)
        self.assertEqual(removed_appearances, 813)

    def test_every_v1_triple_maps_once_to_a_current_translated_sentence(self) -> None:
        annotation_count = 0
        mapping_counts: dict[str, int] = {}
        for volume in self.index["volumes"]:
            for page_number in volume["availablePages"]:
                v1 = self.store.page(volume["id"], page_number, "v1")
                v2 = self.store.page(volume["id"], page_number, "v2")
                v1_sentences = [
                    (
                        sentence["id"],
                        sentence["text"],
                        sentence["translation"],
                        sentence["fragments"],
                    )
                    for sentence in v1["sentences"]
                ]
                v2_sentences = [
                    (
                        sentence["id"],
                        sentence["text"],
                        sentence["translation"],
                        sentence["fragments"],
                    )
                    for sentence in v2["sentences"]
                ]
                self.assertEqual(v1_sentences, v2_sentences)
                annotations = {annotation["id"]: annotation for annotation in v1["annotations"]}
                assigned = [
                    triple_id for sentence in v1["sentences"] for triple_id in sentence["tripleIds"]
                ]
                self.assertEqual(len(assigned), len(set(assigned)))
                self.assertEqual(set(assigned), set(annotations))
                self.assertTrue(
                    all(
                        annotation["evidence"]["status"] == "resolved"
                        for annotation in annotations.values()
                    )
                )
                annotation_count += len(annotations)
                for method, count in v1["diagnostics"]["sentenceMappingCounts"].items():
                    mapping_counts[method] = mapping_counts.get(method, 0) + count
        self.assertEqual(annotation_count, 18992)
        self.assertEqual(mapping_counts["maximum_evidence_overlap"], 18933)
        self.assertEqual(mapping_counts["nearest_source_span"], 59)
        self.assertNotIn("first_translated_sentence_fallback", mapping_counts)


if __name__ == "__main__":
    unittest.main()
