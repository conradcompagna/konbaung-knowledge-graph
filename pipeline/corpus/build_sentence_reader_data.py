#!/usr/bin/env python3
"""Compile translated sentences and sentence triples for the Konbaung reader."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline.corpus.build_reader_data import (
    best_character_overlap_span,
    compact_with_offsets,
    span_candidates,
    utf16_offset,
)


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_LIVE_ROOT = WORKSPACE / "konbaung_reader_app" / "static" / "data" / "konbaung"
DEFAULT_V3_ROOT = WORKSPACE / "konbaung_dataset_v3_20260718"
DEFAULT_SENTENCE_ROOT = DEFAULT_V3_ROOT / "sentence_corpus"
DEFAULT_TRANSLATION_ROOT = DEFAULT_V3_ROOT / "sentence_translations"
DEFAULT_TRIPLE_ROOT = DEFAULT_V3_ROOT / "triples"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE / "konbaung_reader_app" / "static" / "data" / "konbaung_sentence_staging"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def shift_span(span: dict[str, int], offset: int) -> dict[str, int]:
    return {"start": span["start"] + offset, "end": span["end"] + offset}


def inside(span: dict[str, int], evidence: dict[str, int]) -> bool:
    return evidence["start"] <= span["start"] and span["end"] <= evidence["end"]


def endpoint_record(
    endpoint: dict[str, Any],
    canonical_text: str,
    evidence_spans: list[dict[str, int]],
) -> tuple[dict[str, Any], list[str]]:
    literal = str(endpoint.get("my", "")).strip()
    source = str(endpoint.get("source", "inferred"))
    exact = span_candidates(canonical_text, literal)
    exact_inside = [
        candidate
        for candidate in exact
        if any(inside(candidate, evidence_span) for evidence_span in evidence_spans)
    ]
    selected: dict[str, int] | None = None
    method: str | None = None
    overlap = 0.0

    if exact_inside:
        selected = exact_inside[0]
        method = "exact_sentence_evidence"
        overlap = 1.0
    elif exact:
        selected = exact[0]
        method = "exact_page_context"
        overlap = 1.0
    elif literal and source != "inferred":
        fuzzy_candidates: list[tuple[float, dict[str, int]]] = []
        for evidence_span in evidence_spans:
            evidence_text = canonical_text[evidence_span["start"] : evidence_span["end"]]
            fuzzy, score = best_character_overlap_span(evidence_text, literal)
            if fuzzy:
                fuzzy_candidates.append((score, shift_span(fuzzy, evidence_span["start"])))
        if fuzzy_candidates:
            overlap, selected = max(fuzzy_candidates, key=lambda item: item[0])
            method = "maximum_character_overlap_evidence"
        else:
            selected, overlap = best_character_overlap_span(canonical_text, literal)
            if selected:
                method = "maximum_character_overlap_page"

    in_evidence = bool(selected) and any(inside(selected, span) for span in evidence_spans)
    is_inferred = source == "inferred" or not selected or not in_evidence
    warnings: list[str] = []
    if not selected:
        warnings.append("endpoint_not_in_page")
    elif method and method.startswith("maximum_character_overlap"):
        warnings.append("endpoint_fuzzy_alignment")
    elif not in_evidence:
        warnings.append("endpoint_anchored_in_page_context")

    return {
        "text": literal,
        "gloss": str(endpoint.get("en", "")).strip() or None,
        "type": str(endpoint.get("tag", "")).strip() or None,
        "source": source,
        "startUtf16": utf16_offset(canonical_text, selected["start"]) if selected else None,
        "endUtf16": utf16_offset(canonical_text, selected["end"]) if selected else None,
        "candidateCount": len(exact),
        "inferred": is_inferred,
        "inference": (None if not is_inferred else "page_context" if selected else "not_in_page"),
        "alignmentMethod": method,
        "characterOverlap": overlap,
    }, warnings


def exact_compact_candidates(
    compact_source: str,
    literal: str,
    source_positions: list[dict[str, int]],
) -> list[dict[str, Any]]:
    compact_literal = "".join(str(literal).split())
    if not compact_literal:
        return []
    candidates: list[dict[str, Any]] = []
    cursor = 0
    while True:
        found = compact_source.find(compact_literal, cursor)
        if found < 0:
            break
        end = found + len(compact_literal)
        mapped = source_positions[found:end]
        fragments: list[dict[str, int]] = []
        group_start = 0
        for index in range(1, len(mapped) + 1):
            if (
                index < len(mapped)
                and mapped[index]["pieceIndex"] == mapped[group_start]["pieceIndex"]
            ):
                continue
            first = mapped[group_start]
            last = mapped[index - 1]
            fragments.append(
                {
                    "page": first["page"],
                    "pieceIndex": first["pieceIndex"],
                    "start": first["canonicalOffset"],
                    "end": last["canonicalOffset"] + 1,
                }
            )
            group_start = index
        candidates.append(
            {
                "compactStart": found,
                "compactEnd": end,
                "fragments": fragments,
            }
        )
        cursor = found + 1
    return candidates


def stitched_sentence_source(sentence: dict[str, Any]) -> tuple[str, list[dict[str, int]]]:
    chars: list[str] = []
    positions: list[dict[str, int]] = []
    for piece_index, source_piece in enumerate(sentence["source"]):
        compact_piece, source_offsets = compact_with_offsets(source_piece["source_text"])
        if compact_piece != "".join(str(source_piece["text"]).split()):
            raise ValueError(
                f"Sentence source normalization mismatch: {sentence['id']} page {source_piece['page']}"
            )
        chars.append(compact_piece)
        positions.extend(
            {
                "page": int(source_piece["page"]),
                "pieceIndex": piece_index,
                "canonicalOffset": int(source_piece["start_codepoint"]) + source_offset,
            }
            for source_offset in source_offsets
        )
    return "".join(chars), positions


def candidate_group_score(candidates: list[dict[str, Any]]) -> tuple[int, int, float]:
    starts = [int(candidate["compactStart"]) for candidate in candidates]
    ends = [int(candidate["compactEnd"]) for candidate in candidates]
    centers = [(start + end) / 2 for start, end in zip(starts, ends)]
    envelope = max(ends) - min(starts)
    gaps = 0
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            left_start = int(left["compactStart"])
            left_end = int(left["compactEnd"])
            right_start = int(right["compactStart"])
            right_end = int(right["compactEnd"])
            if left_end <= right_start:
                gaps += right_start - left_end
            elif right_end <= left_start:
                gaps += left_start - right_end
    center_spread = sum(
        abs(left - right)
        for left_index, left in enumerate(centers)
        for right in centers[left_index + 1 :]
    )
    return envelope, gaps, center_spread


def resolve_cross_page_triple(
    sentence: dict[str, Any],
    triple: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    compact_source, source_positions = stitched_sentence_source(sentence)
    predicate_grounding = triple.get("predicate_grounding") or {}
    endpoints = {
        "subject": triple["s"],
        "predicate": {
            "my": predicate_grounding.get("my", ""),
            "en": predicate_grounding.get("en", ""),
            "source": predicate_grounding.get("source", "inferred"),
        },
        "object": triple["o"],
    }
    resolutions: dict[str, dict[str, Any]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for role, endpoint in endpoints.items():
        source = str(endpoint.get("source", "inferred"))
        candidates = (
            []
            if source == "inferred"
            else exact_compact_candidates(compact_source, endpoint.get("my", ""), source_positions)
        )
        resolutions[role] = {
            "endpoint": endpoint,
            "candidates": candidates,
            "selected": None,
            "selectionMethod": None,
        }
        if len(candidates) == 1:
            selected[role] = candidates[0]
            resolutions[role]["selected"] = candidates[0]
            resolutions[role]["selectionMethod"] = "exact_sentence_source"

    unresolved_roles = sorted(
        role
        for role, resolution in resolutions.items()
        if len(resolution["candidates"]) > 1 and resolution["selected"] is None
    )
    combination_count = 1
    for role in unresolved_roles:
        combination_count *= len(resolutions[role]["candidates"])
    if unresolved_roles and combination_count <= 4096:
        scored_combinations: list[tuple[tuple[int, int, float], tuple[dict[str, Any], ...]]] = []
        for combination in itertools.product(
            *(resolutions[role]["candidates"] for role in unresolved_roles)
        ):
            scored_combinations.append(
                (candidate_group_score([*selected.values(), *combination]), combination)
            )
        scored_combinations.sort(key=lambda item: item[0])
        if len(scored_combinations) == 1 or scored_combinations[0][0] != scored_combinations[1][0]:
            for role, candidate in zip(unresolved_roles, scored_combinations[0][1]):
                resolutions[role]["selected"] = candidate
                resolutions[role]["selectionMethod"] = "exact_sentence_triple_context"
    return resolutions


def cross_page_endpoint_record(
    resolution: dict[str, Any],
    canonical_text: str,
    page_number: int,
) -> tuple[dict[str, Any], list[str]]:
    endpoint = resolution["endpoint"]
    source = str(endpoint.get("source", "inferred"))
    candidates = resolution["candidates"]
    selected = resolution["selected"]
    selected_fragments = selected["fragments"] if selected else []
    page_fragments = [
        {
            "startUtf16": utf16_offset(canonical_text, int(fragment["start"])),
            "endUtf16": utf16_offset(canonical_text, int(fragment["end"])),
        }
        for fragment in selected_fragments
        if int(fragment["page"]) == page_number
    ]
    selected_pages = {int(fragment["page"]) for fragment in selected_fragments}
    if source == "inferred":
        page_presence = "inferred"
    elif selected is None:
        page_presence = "ambiguous" if len(candidates) > 1 else "not_in_sentence"
    elif not page_fragments:
        page_presence = "absent"
    elif len(selected_pages) > 1:
        page_presence = "partial"
    else:
        page_presence = "present"

    warnings: list[str] = []
    if page_presence == "ambiguous":
        warnings.append("endpoint_ambiguous_in_sentence")
    elif page_presence == "not_in_sentence":
        warnings.append("endpoint_not_in_sentence")

    start_utf16 = page_fragments[0]["startUtf16"] if page_fragments else None
    end_utf16 = page_fragments[-1]["endUtf16"] if page_fragments else None
    return {
        "text": str(endpoint.get("my", "")).strip(),
        "gloss": str(endpoint.get("en", "")).strip() or None,
        "type": str(endpoint.get("tag", "")).strip() or None,
        "source": source,
        "startUtf16": start_utf16,
        "endUtf16": end_utf16,
        "fragments": page_fragments,
        "candidateCount": len(candidates),
        "inferred": source == "inferred",
        "inference": "source_record" if source == "inferred" else None,
        "pagePresence": page_presence,
        "alignmentMethod": resolution["selectionMethod"],
        "characterOverlap": 1.0 if selected else 0.0,
    }, warnings


def endpoint_is_present(endpoint: dict[str, Any]) -> bool:
    return endpoint.get("pagePresence") in {"present", "partial"}


def display_relation(label: str) -> str:
    return " ".join(part.capitalize() for part in label.split("_") if part)


def load_sentences(sentence_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for volume in (1, 2, 3):
        for record in read_jsonl(sentence_root / "sentences" / f"vol{volume}.jsonl"):
            sentence_id = record["id"]
            if sentence_id in records:
                raise ValueError(f"Duplicate sentence id: {sentence_id}")
            records[sentence_id] = record
    return records


def load_translations(translation_root: Path) -> dict[str, str]:
    path = translation_root / "translations" / "all_volumes.jsonl"
    translations: dict[str, str] = {}
    for record in read_jsonl(path):
        sentence_id = record["id"]
        if sentence_id in translations:
            raise ValueError(f"Duplicate translation id: {sentence_id}")
        translations[sentence_id] = str(record["en"]).strip()
    return translations


def load_triples(
    triple_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[int, int], str]]:
    triples: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[tuple[int, int], str] = {}
    for page in read_jsonl(triple_root / "annotations" / "all_pages.jsonl"):
        summaries[(int(page["volume"]), int(page["page"]))] = str(page.get("summary", "")).strip()
        for group in page.get("S", []):
            sentence_id = group["sid"]
            if sentence_id in triples:
                raise ValueError(f"Duplicate triple group: {sentence_id}")
            triples[sentence_id] = list(group.get("T", []))
    return triples, summaries


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    old_index = read_json(args.live_root / "index.json")
    old_volumes = {item["id"]: item for item in old_index["volumes"]}
    sentences = load_sentences(args.sentence_root)
    translations = load_translations(args.translation_root)
    triples, generated_summaries = load_triples(args.triple_root)

    if set(translations) != set(sentences):
        missing = sorted(set(sentences) - set(translations))[:10]
        extra = sorted(set(translations) - set(sentences))[:10]
        raise ValueError(f"Sentence/translation mismatch; missing={missing}, extra={extra}")

    page_sources: dict[tuple[int, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    for sentence in sentences.values():
        for source in sentence["source"]:
            page_sources[(int(sentence["volume"]), int(source["page"]))].append((sentence, source))

    page_record_paths = sorted(args.sentence_root.glob("page_records/vol*/page_*.json"))
    expected_pages = {
        (int(read_json(path)["volume"]), int(read_json(path)["page"])) for path in page_record_paths
    }
    if set(page_sources) != expected_pages:
        raise ValueError("Sentence source pages do not match repaired page records")

    totals = Counter()
    volume_pages: dict[int, list[int]] = defaultdict(list)
    diagnostics: list[dict[str, Any]] = []
    base_triple_ids = {
        f"{sentence_id}-t{triple_index:03d}"
        for sentence_id, sentence_triples in triples.items()
        for triple_index in range(1, len(sentence_triples) + 1)
    }
    cross_page_resolution_cache: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}

    for volume, page_number in sorted(expected_pages):
        volume_id = f"vol{volume}"
        old_path = args.live_root / "pages" / volume_id / f"{page_number:04d}.json"
        old_page = read_json(old_path)
        canonical_text = old_page["canonicalText"]
        page_sentences: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        page_warning_counts = Counter()

        grouped_sources: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for sentence, source_piece in page_sources[(volume, page_number)]:
            grouped_sources.setdefault(sentence["id"], (sentence, []))[1].append(source_piece)
        ordered_sources = sorted(
            grouped_sources.values(),
            key=lambda item: min(int(source["start_utf16"]) for source in item[1]),
        )
        for sentence, source_pieces in ordered_sources:
            sentence_id = sentence["id"]
            source_pieces.sort(key=lambda item: int(item["start_utf16"]))
            fragments: list[dict[str, Any]] = []
            evidence_spans: list[dict[str, int]] = []
            for source_piece in source_pieces:
                start_cp = int(source_piece["start_codepoint"])
                end_cp = int(source_piece["end_codepoint"])
                source_text = source_piece["source_text"]
                if canonical_text[start_cp:end_cp] != source_text:
                    raise ValueError(
                        f"Source offset mismatch: {volume_id} page {page_number}, {sentence_id}"
                    )
                if utf16_offset(canonical_text, start_cp) != int(source_piece["start_utf16"]):
                    raise ValueError(
                        f"UTF-16 start mismatch: {volume_id} page {page_number}, {sentence_id}"
                    )
                if utf16_offset(canonical_text, end_cp) != int(source_piece["end_utf16"]):
                    raise ValueError(
                        f"UTF-16 end mismatch: {volume_id} page {page_number}, {sentence_id}"
                    )
                evidence_spans.append({"start": start_cp, "end": end_cp})
                fragments.append(
                    {
                        "canonicalText": source_text,
                        "startUtf16": int(source_piece["start_utf16"]),
                        "endUtf16": int(source_piece["end_utf16"]),
                    }
                )

            sentence_triples = triples.get(sentence_id, [])
            annotation_ids: list[str] = []
            for triple_index, triple in enumerate(sentence_triples, start=1):
                base_id = f"{sentence_id}-t{triple_index:03d}"
                annotation_id = f"{base_id}-p{page_number:04d}"
                predicate_grounding = triple.get("predicate_grounding") or {}
                unattached_cross_page = False
                if sentence["cross_page"]:
                    cache_key = (sentence_id, triple_index)
                    if cache_key not in cross_page_resolution_cache:
                        cross_page_resolution_cache[cache_key] = resolve_cross_page_triple(
                            sentence,
                            triple,
                        )
                    resolutions = cross_page_resolution_cache[cache_key]
                    subject, subject_warnings = cross_page_endpoint_record(
                        resolutions["subject"],
                        canonical_text,
                        page_number,
                    )
                    predicate, predicate_warnings = cross_page_endpoint_record(
                        resolutions["predicate"],
                        canonical_text,
                        page_number,
                    )
                    object_, object_warnings = cross_page_endpoint_record(
                        resolutions["object"],
                        canonical_text,
                        page_number,
                    )
                    if not any(
                        endpoint_is_present(endpoint) for endpoint in (subject, predicate, object_)
                    ):
                        has_exact_endpoint = any(
                            resolution["selected"] is not None
                            for resolution in resolutions.values()
                        )
                        if has_exact_endpoint or page_number != int(sentence["owner_page"]):
                            continue
                        unattached_cross_page = True
                else:
                    subject, subject_warnings = endpoint_record(
                        triple["s"],
                        canonical_text,
                        evidence_spans,
                    )
                    object_, object_warnings = endpoint_record(
                        triple["o"],
                        canonical_text,
                        evidence_spans,
                    )
                    predicate, predicate_warnings = endpoint_record(
                        {
                            "my": predicate_grounding.get("my", ""),
                            "en": predicate_grounding.get("en", ""),
                            "source": predicate_grounding.get("source", "inferred"),
                        },
                        canonical_text,
                        evidence_spans,
                    )
                annotation_ids.append(annotation_id)
                warnings = (
                    [f"subject_{item}" for item in subject_warnings]
                    + [f"object_{item}" for item in object_warnings]
                    + [f"predicate_{item}" for item in predicate_warnings]
                )
                page_warning_counts.update(warnings)
                label = str(triple["p"]).strip()
                annotation_evidence_fragments = [] if unattached_cross_page else fragments
                evidence_status = "unattached" if unattached_cross_page else "resolved"
                annotations.append(
                    {
                        "id": annotation_id,
                        "baseTripleId": base_id,
                        "sentenceId": sentence_id,
                        "subject": subject,
                        "relation": {
                            "rawLabel": label,
                            "displayLabel": display_relation(label),
                            "predicateText": predicate["text"] or None,
                            "predicateGloss": predicate["gloss"],
                            "predicateSource": predicate["source"],
                            "startUtf16": predicate["startUtf16"],
                            "endUtf16": predicate["endUtf16"],
                            "fragments": predicate.get("fragments", []),
                            "candidateCount": predicate["candidateCount"],
                            "inferred": predicate["inferred"],
                            "inference": predicate["inference"],
                            "pagePresence": predicate.get(
                                "pagePresence",
                                "present" if predicate["startUtf16"] is not None else "inferred",
                            ),
                            "alignmentMethod": predicate["alignmentMethod"],
                            "characterOverlap": predicate["characterOverlap"],
                        },
                        "object": object_,
                        "pageAttachment": evidence_status,
                        "evidence": {
                            "text": sentence["text"],
                            "canonicalText": "\n[… ]\n".join(
                                item["canonicalText"] for item in fragments
                            ),
                            "translation": translations[sentence_id],
                            "startUtf16": (
                                None if unattached_cross_page else fragments[0]["startUtf16"]
                            ),
                            "endUtf16": (
                                None if unattached_cross_page else fragments[-1]["endUtf16"]
                            ),
                            "fragments": annotation_evidence_fragments,
                            "candidateCount": len(annotation_evidence_fragments),
                            "status": evidence_status,
                            "alignmentMethod": (
                                None if unattached_cross_page else "sentence_source_offsets"
                            ),
                            "characterOverlap": 0.0 if unattached_cross_page else 1.0,
                            "crossPage": bool(sentence["cross_page"]),
                            "ownerPage": int(sentence["owner_page"]),
                        },
                        "metadata": {},
                        "validation": {
                            "status": (
                                "unattached"
                                if unattached_cross_page
                                else "resolved"
                                if not warnings
                                else "resolved_with_warnings"
                            ),
                            "warnings": warnings,
                            "evidenceStatus": evidence_status,
                        },
                        "provenance": {
                            "pass": "translated_sentence_triples",
                            "sentenceId": sentence_id,
                            "ownerPage": int(sentence["owner_page"]),
                            "crossPage": bool(sentence["cross_page"]),
                            "tripleIndex": triple_index,
                        },
                        "rawSourceRecord": triple,
                    }
                )

            page_sentences.append(
                {
                    "id": sentence_id,
                    "ownerPage": int(sentence["owner_page"]),
                    "pages": [int(page) for page in sentence["pages"]],
                    "crossPage": bool(sentence["cross_page"]),
                    "text": sentence["text"],
                    "translation": translations[sentence_id],
                    "startUtf16": fragments[0]["startUtf16"],
                    "endUtf16": fragments[-1]["endUtf16"],
                    "canonicalText": "\n[… ]\n".join(item["canonicalText"] for item in fragments),
                    "fragments": fragments,
                    "tripleIds": annotation_ids,
                }
            )

        summary = generated_summaries.get((volume, page_number)) or old_page.get("summary")
        page_diagnostics = {
            "canonicalCodePointLength": len(canonical_text),
            "canonicalUtf16Length": utf16_offset(canonical_text, len(canonical_text)),
            "sentenceCount": len(page_sentences),
            "translatedSentenceCount": len(page_sentences),
            "crossPageSentenceCount": sum(item["crossPage"] for item in page_sentences),
            "annotationCount": len(annotations),
            "endpointWarningCount": sum(page_warning_counts.values()),
            "endpointWarningCounts": dict(sorted(page_warning_counts.items())),
        }
        page = {
            "schemaVersion": 6,
            "id": old_page["id"],
            "bookId": old_page["bookId"],
            "volumeId": volume_id,
            "sourceVolumeId": old_page["sourceVolumeId"],
            "pageNumber": page_number,
            "printedPageNumber": old_page.get("printedPageNumber"),
            "canonicalText": canonical_text,
            "summary": summary,
            "sentences": page_sentences,
            "annotations": annotations,
            "source": {
                **old_page["source"],
                "annotationCorpus": "translated_sentence_triples_20260713_high_thinking",
                "sentenceCorpus": "sentence_corpus_20260713_repaired",
            },
            "diagnostics": page_diagnostics,
        }
        write_json(args.output_root / "pages" / volume_id / f"{page_number:04d}.json", page)

        totals["pages"] += 1
        totals["sentencePageAppearances"] += len(page_sentences)
        totals["annotationPageAppearances"] += len(annotations)
        totals["crossPageSentenceAppearances"] += page_diagnostics["crossPageSentenceCount"]
        totals["endpointWarnings"] += page_diagnostics["endpointWarningCount"]
        volume_pages[volume].append(page_number)
        diagnostics.append({"pageId": page["id"], **page_diagnostics})

    totals["sentences"] = len(sentences)
    totals["translations"] = len(translations)
    totals["triples"] = len(base_triple_ids)
    totals["tripleBearingSentences"] = len(triples)
    totals["nonPropositionalSentences"] = len(sentences) - len(triples)
    expected_triple_count = sum(len(group) for group in triples.values())
    if totals["triples"] != expected_triple_count:
        raise ValueError(f"Triple count mismatch: {totals['triples']} != {expected_triple_count}")

    volumes: list[dict[str, Any]] = []
    for volume in (1, 2, 3):
        volume_id = f"vol{volume}"
        pages = sorted(volume_pages[volume])
        volumes.append(
            {
                "id": volume_id,
                "label": old_volumes[volume_id]["label"],
                "availablePages": pages,
                "pageCount": len(pages),
                "firstPage": pages[0],
                "lastPage": pages[-1],
            }
        )

    index = {
        "schemaVersion": 6,
        "bookId": old_index["bookId"],
        "title": old_index["title"],
        "volumes": volumes,
        "totals": dict(totals),
        "annotationSource": "Konbaung triples v3 with predicate grounding and restored chronicle material",
        "sentenceSource": "Konbaung sentence corpus v3 with conservative restoration integration",
    }
    write_json(args.output_root / "index.json", index)
    write_json(
        args.output_root / "diagnostics" / "sentence-integration.json",
        {
            "status": "passed",
            "sourceCorpusHash": hashlib.sha1(
                (args.sentence_root / "manifest.json").read_bytes()
            ).hexdigest(),
            "totals": dict(totals),
            "pages": diagnostics,
        },
    )
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, default=DEFAULT_LIVE_ROOT)
    parser.add_argument("--sentence-root", type=Path, default=DEFAULT_SENTENCE_ROOT)
    parser.add_argument("--translation-root", type=Path, default=DEFAULT_TRANSLATION_ROOT)
    parser.add_argument("--triple-root", type=Path, default=DEFAULT_TRIPLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result["totals"], indent=2))
