"""Source-span and cross-page regressions using synthetic text only."""

import unittest

from pipeline.corpus.build_reader_data import span_candidates, utf16_offset
from pipeline.corpus.build_sentence_reader_data import (
    cross_page_endpoint_record,
    endpoint_is_present,
    resolve_cross_page_triple,
)
from konbaung_reader_app.reader_bridge import _dictionary_display_fields


def compact(value):
    return "".join(value.split())


def slice_utf16(text, start, end):
    return text.encode("utf-16-le")[start * 2 : end * 2].decode("utf-16-le")


class AlignmentUnitTests(unittest.TestCase):
    def test_whitespace_insensitive_candidates_map_to_source(self) -> None:
        text = "alpha beta\ngamma beta"
        candidates = span_candidates(text, "beta gamma")
        self.assertEqual(candidates, [{"start": 6, "end": 16}])
        self.assertEqual(compact(text[6:16]), compact("beta gamma"))

    def test_repeated_literals_remain_multiple_candidates(self) -> None:
        candidates = span_candidates("king and king", "king")
        self.assertEqual(candidates, [{"start": 0, "end": 4}, {"start": 9, "end": 13}])

    def test_utf16_offsets_are_not_codepoint_offsets(self) -> None:
        text = "က😀ခ"
        self.assertEqual(utf16_offset(text, 1), 1)
        self.assertEqual(utf16_offset(text, 2), 3)

    def test_dictionary_rows_are_deduplicated_and_preserve_multiple_pos(self) -> None:
        romanizations, parts_of_speech, definitions = _dictionary_display_fields(
            {
                "roman": "min:",
                "pos": "n",
                "senses": [
                    "မင်း\tmin:\tn\t1 monarch; king",
                    "\t\tadj\thonorific term pron 1 you",
                    "\t\tpart\tparticle",
                ],
            }
        )
        self.assertEqual(romanizations, ["min:"])
        self.assertEqual(parts_of_speech, ["n", "adj", "pron", "part"])
        self.assertEqual(definitions, ["monarch; king", "honorific term", "you", "particle"])


class CrossPageProjectionUnitTests(unittest.TestCase):
    def test_projects_only_exact_page_source(self) -> None:
        sentence = {
            "id": "synthetic_cross_page",
            "source": [
                {
                    "page": 1,
                    "start_codepoint": 7,
                    "text": "Alpha ruled",
                    "source_text": "Alpha\r\nruled",
                },
                {
                    "page": 2,
                    "start_codepoint": 6,
                    "text": "Omega.",
                    "source_text": "Omega.",
                },
            ],
        }
        triple = {
            "s": {"my": "Alpha", "en": "Alpha", "source": "sentence"},
            "p": "RULES",
            "o": {"my": "Omega", "en": "Omega", "source": "sentence"},
            "predicate_grounding": {"my": "ruled", "en": "ruled", "source": "sentence"},
        }
        resolutions = resolve_cross_page_triple(sentence, triple)
        page_one = "decoy  Alpha\r\nruled"
        page_two = "Alpha Omega."

        subject_one, _ = cross_page_endpoint_record(resolutions["subject"], page_one, 1)
        subject_two, _ = cross_page_endpoint_record(resolutions["subject"], page_two, 2)
        object_one, _ = cross_page_endpoint_record(resolutions["object"], page_one, 1)
        object_two, _ = cross_page_endpoint_record(resolutions["object"], page_two, 2)

        self.assertEqual(subject_one["pagePresence"], "present")
        self.assertEqual(
            slice_utf16(page_one, subject_one["startUtf16"], subject_one["endUtf16"]),
            "Alpha",
        )
        self.assertEqual(subject_two["pagePresence"], "absent")
        self.assertIsNone(subject_two["startUtf16"])
        self.assertEqual(object_one["pagePresence"], "absent")
        self.assertEqual(object_two["pagePresence"], "present")
        self.assertEqual(
            slice_utf16(page_two, object_two["startUtf16"], object_two["endUtf16"]),
            "Omega",
        )
        self.assertNotEqual(subject_one["startUtf16"], 0)

    def test_endpoint_can_have_exact_fragments_on_both_pages(self) -> None:
        sentence = {
            "id": "synthetic_split_endpoint",
            "source": [
                {
                    "page": 1,
                    "start_codepoint": 3,
                    "text": "The King",
                    "source_text": "The King",
                },
                {
                    "page": 2,
                    "start_codepoint": 4,
                    "text": "dom endured.",
                    "source_text": "dom endured.",
                },
            ],
        }
        triple = {
            "s": {"my": "Kingdom", "source": "sentence"},
            "p": "ENDURES",
            "o": {"my": "Kingdom", "source": "sentence"},
            "predicate_grounding": {"my": "endured", "source": "sentence"},
        }
        resolutions = resolve_cross_page_triple(sentence, triple)
        page_one = "---The King"
        page_two = "----dom endured."
        subject_one, _ = cross_page_endpoint_record(resolutions["subject"], page_one, 1)
        subject_two, _ = cross_page_endpoint_record(resolutions["subject"], page_two, 2)

        self.assertEqual(subject_one["pagePresence"], "partial")
        self.assertEqual(subject_two["pagePresence"], "partial")
        self.assertEqual(
            slice_utf16(page_one, subject_one["startUtf16"], subject_one["endUtf16"]),
            "King",
        )
        self.assertEqual(
            slice_utf16(page_two, subject_two["startUtf16"], subject_two["endUtf16"]),
            "dom",
        )

    def test_repeated_exact_predicate_is_resolved_with_its_local_triple(self) -> None:
        sentence = {
            "id": "synthetic_repeated_predicate",
            "source": [
                {
                    "page": 1,
                    "start_codepoint": 0,
                    "text": "Alpha ruled Omega.",
                    "source_text": "Alpha ruled Omega.",
                },
                {
                    "page": 2,
                    "start_codepoint": 0,
                    "text": "Beta ruled Gamma.",
                    "source_text": "Beta ruled Gamma.",
                },
            ],
        }
        triple = {
            "s": {"my": "Beta", "source": "sentence"},
            "p": "RULES",
            "o": {"my": "Gamma", "source": "sentence"},
            "predicate_grounding": {"my": "ruled", "source": "sentence"},
        }
        resolutions = resolve_cross_page_triple(sentence, triple)
        predicate_one, _ = cross_page_endpoint_record(
            resolutions["predicate"], "Alpha ruled Omega.", 1
        )
        predicate_two, _ = cross_page_endpoint_record(
            resolutions["predicate"], "Beta ruled Gamma.", 2
        )

        self.assertEqual(predicate_one["pagePresence"], "absent")
        self.assertEqual(predicate_two["pagePresence"], "present")
        self.assertEqual(
            slice_utf16(
                "Beta ruled Gamma.",
                predicate_two["startUtf16"],
                predicate_two["endUtf16"],
            ),
            "ruled",
        )

    def test_triple_without_a_page_endpoint_is_not_page_content(self) -> None:
        sentence = {
            "id": "synthetic_irrelevant_triple",
            "source": [
                {
                    "page": 1,
                    "start_codepoint": 0,
                    "text": "Alpha ruled Omega.",
                    "source_text": "Alpha ruled Omega.",
                },
                {
                    "page": 2,
                    "start_codepoint": 0,
                    "text": "Beta observed.",
                    "source_text": "Beta observed.",
                },
            ],
        }
        triple = {
            "s": {"my": "Alpha", "source": "sentence"},
            "p": "RULES",
            "o": {"my": "Omega", "source": "sentence"},
            "predicate_grounding": {"my": "ruled", "source": "sentence"},
        }
        resolutions = resolve_cross_page_triple(sentence, triple)
        page_two_records = [
            cross_page_endpoint_record(resolutions[role], "Beta observed.", 2)[0]
            for role in ("subject", "predicate", "object")
        ]
        self.assertFalse(any(endpoint_is_present(endpoint) for endpoint in page_two_records))
