#!/usr/bin/env python3
"""Build lazy, canonical page records for the standalone Konbaung reader."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = WORKSPACE / "konbaung_viable_pages_manual_ranges"
DEFAULT_ANNOTATION_ROOT = WORKSPACE / "konbaung_open_coding_structured_full_batch_20260709"
DEFAULT_EVIDENCE_ROOT = WORKSPACE / "konbaung_cite_sources_full_batch_20260710"
DEFAULT_THIRD_PASS_ROOT = WORKSPACE / "konbaung_summary_gap_full_batch_20260711"
DEFAULT_THIRD_PASS_PILOT_ROOT = WORKSPACE / "konbaung_remainder_completion_test"
DEFAULT_FOURTH_PASS_ROOT = WORKSPACE / "konbaung_quantitative_fourth_pass_full_batch_20260712"
DEFAULT_FOURTH_PASS_PILOT_ROOT = WORKSPACE / "konbaung_quantitative_fourth_pass_test"
DEFAULT_OUTPUT_ROOT = WORKSPACE / "konbaung_reader_app" / "static" / "data" / "konbaung"
WARNING_INDEX_RE = re.compile(r"^T\[(\d+)]\.(\w+):\s*(.*)$")


def read_exact_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.decode("utf-8", errors="strict")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utf16_offset(text: str, codepoint_offset: int) -> int:
    return len(text[:codepoint_offset].encode("utf-16-le")) // 2


def compact_with_offsets(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        chars.append(char)
        offsets.append(index)
    return "".join(chars), offsets


def span_candidates(canonical_text: str, literal: str) -> list[dict[str, int]]:
    compact_text, source_offsets = compact_with_offsets(canonical_text)
    compact_literal = "".join(str(literal).split())
    if not compact_literal:
        return []
    candidates: list[dict[str, int]] = []
    cursor = 0
    while True:
        found = compact_text.find(compact_literal, cursor)
        if found < 0:
            break
        start = source_offsets[found]
        end = source_offsets[found + len(compact_literal) - 1] + 1
        candidates.append({"start": start, "end": end})
        cursor = found + 1
    return candidates


def best_character_overlap_span(canonical_text: str, literal: str) -> tuple[dict[str, int] | None, float]:
    compact_text, source_offsets = compact_with_offsets(canonical_text)
    compact_literal = "".join(str(literal).split())
    if not compact_text or not compact_literal:
        return None, 0.0

    ngram_size = min(5, len(compact_literal))
    votes: Counter[int] = Counter()
    for literal_index in range(len(compact_literal) - ngram_size + 1):
        ngram = compact_literal[literal_index : literal_index + ngram_size]
        cursor = 0
        occurrences = 0
        while occurrences < 24:
            found = compact_text.find(ngram, cursor)
            if found < 0:
                break
            votes[found - literal_index] += 1
            cursor = found + 1
            occurrences += 1

    estimated_starts = [start for start, _ in votes.most_common(8)]
    if not estimated_starts:
        match = difflib.SequenceMatcher(None, compact_literal, compact_text, autojunk=False).find_longest_match()
        estimated_starts = [match.b - match.a]

    best: tuple[float, int, int] | None = None
    for estimated in estimated_starts:
        start = max(0, min(estimated, len(compact_text) - 1))
        end = min(len(compact_text), start + len(compact_literal))
        if end <= start:
            continue
        ratio = difflib.SequenceMatcher(
            None, compact_literal, compact_text[start:end], autojunk=False
        ).ratio()
        candidate = (ratio, start, end)
        if best is None or candidate > best:
            best = candidate
    if not best:
        return None, 0.0
    return {
        "start": source_offsets[best[1]],
        "end": source_offsets[best[2] - 1] + 1,
    }, best[0]


def endpoint_record(
    literal: str,
    gloss: str,
    open_code: str,
    selected: dict[str, int] | None,
    candidates: list[dict[str, int]],
    canonical_text: str,
    inferred: bool,
) -> dict[str, Any]:
    return {
        "text": literal,
        "gloss": gloss or None,
        "type": open_code or None,
        "startUtf16": utf16_offset(canonical_text, selected["start"]) if selected else None,
        "endUtf16": utf16_offset(canonical_text, selected["end"]) if selected else None,
        "candidateCount": len(candidates),
        "inferred": inferred,
        "inference": ("outside_evidence" if selected else "not_in_page") if inferred else None,
    }


def span_record(
    literal: str,
    canonical_text: str,
    subject_candidates: list[dict[str, int]],
    object_candidates: list[dict[str, int]],
) -> tuple[dict[str, Any], dict[str, int] | None]:
    def context_envelope(start: int, end: int) -> dict[str, int]:
        left = max(canonical_text.rfind("။", 0, start), canonical_text.rfind("\n", 0, start))
        start = 0 if left < 0 else left + 1
        right_candidates = [
            position
            for position in (canonical_text.find("။", end), canonical_text.find("\n", end))
            if position >= 0
        ]
        end = min(right_candidates) + 1 if right_candidates else len(canonical_text)
        while start < end and canonical_text[start].isspace():
            start += 1
        return {"start": start, "end": end}

    candidates = span_candidates(canonical_text, literal)
    selected = candidates[0] if len(candidates) == 1 else None
    method = "exact" if selected else None
    if not selected and candidates:
        scored = []
        for candidate in candidates:
            subject_hits = sum(
                candidate["start"] <= item["start"] and item["end"] <= candidate["end"]
                for item in subject_candidates
            )
            object_hits = sum(
                candidate["start"] <= item["start"] and item["end"] <= candidate["end"]
                for item in object_candidates
            )
            scored.append((int(subject_hits > 0) + int(object_hits > 0), candidate))
        scored.sort(key=lambda item: (-item[0], item[1]["start"]))
        if scored:
            selected = scored[0][1]
            method = "exact_evidence_first"

    if not selected and literal.strip():
        selected, overlap_score = best_character_overlap_span(canonical_text, literal)
        if selected:
            method = "maximum_character_overlap"

    fallback_anchors = [
        candidates[0]
        for candidates in (subject_candidates, object_candidates)
        if candidates
    ]
    if not selected and fallback_anchors:
        selected = context_envelope(
            min(anchor["start"] for anchor in fallback_anchors),
            max(anchor["end"] for anchor in fallback_anchors),
        )
        method = "endpoint_context_envelope"

    record = {
        "text": literal,
        "canonicalText": canonical_text[selected["start"] : selected["end"]] if selected else "",
        "startUtf16": utf16_offset(canonical_text, selected["start"]) if selected else None,
        "endUtf16": utf16_offset(canonical_text, selected["end"]) if selected else None,
        "candidateCount": len(candidates),
        "status": "resolved" if selected else ("ambiguous" if candidates else "unresolved"),
        "alignmentMethod": method,
        "characterOverlap": overlap_score if method == "maximum_character_overlap" else 1.0 if selected else 0.0,
    }
    return record, selected


def select_endpoints_in_evidence(
    subject_candidates: list[dict[str, int]],
    object_candidates: list[dict[str, int]],
    evidence_span: dict[str, int] | None,
) -> tuple[dict[str, int] | None, bool, dict[str, int] | None, bool]:
    def inside(candidates: list[dict[str, int]]) -> list[dict[str, int]]:
        if not evidence_span:
            return []
        return [
            item
            for item in candidates
            if evidence_span["start"] <= item["start"] and item["end"] <= evidence_span["end"]
        ]

    subject_inside = inside(subject_candidates)
    object_inside = inside(object_candidates)
    selected_subject = subject_inside[0] if subject_inside else (subject_candidates[0] if subject_candidates else None)
    selected_object = object_inside[0] if object_inside else (object_candidates[0] if object_candidates else None)
    return selected_subject, not bool(subject_inside), selected_object, not bool(object_inside)


def metadata_record(
    literal: Any,
    gloss: Any,
    canonical_text: str,
    evidence_span: dict[str, int] | None,
) -> dict[str, Any]:
    text = str(literal or "")
    candidates = span_candidates(canonical_text, text)
    inside = [
        item
        for item in candidates
        if evidence_span
        and evidence_span["start"] <= item["start"]
        and item["end"] <= evidence_span["end"]
    ]
    selected = inside[0] if inside else (candidates[0] if candidates else None)
    return {
        "originalText": text or None,
        "gloss": str(gloss or "") or None,
        "startUtf16": utf16_offset(canonical_text, selected["start"]) if selected else None,
        "endUtf16": utf16_offset(canonical_text, selected["end"]) if selected else None,
        "inEvidence": bool(inside),
        "inferred": bool(text and not inside),
    }


def locate_record(root: Path, phase: str, volume_id: str, page_number: int) -> tuple[Path, str]:
    filename = f"page_{page_number:04d}.json"
    matches: list[tuple[Path, str]] = []
    for bucket in ("valid", "invalid"):
        path = root / phase / bucket / volume_id / filename
        if path.exists():
            matches.append((path, bucket))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {phase} record for {volume_id}/{filename}; found {matches}")
    return matches[0]


def locate_third_pass_record(
    root: Path,
    pilot_root: Path,
    volume_id: str,
    page_number: int,
) -> tuple[Path, str] | None:
    filename = f"page_{page_number:04d}.json"
    production_matches = [
        (root / "postprocessed" / bucket / volume_id / filename, bucket)
        for bucket in ("valid", "invalid")
        if (root / "postprocessed" / bucket / volume_id / filename).exists()
    ]
    if len(production_matches) > 1:
        raise RuntimeError(f"Multiple third-pass records for {volume_id}/{filename}")
    if production_matches:
        return production_matches[0]

    short_volume_id = volume_id.replace("konbaung_", "")
    pilot_matches = sorted(
        pilot_root.glob(
            f"{short_volume_id}/page_{page_number:04d}/anchor_gap_analysis_*/postprocessed.json"
        )
    )
    if len(pilot_matches) > 1:
        raise RuntimeError(f"Multiple third-pass pilot records for {volume_id}/{filename}")
    return (pilot_matches[0], "pilot") if pilot_matches else None


def locate_fourth_pass_record(
    root: Path,
    pilot_root: Path,
    volume_id: str,
    page_number: int,
) -> tuple[Path, str] | None:
    filename = f"page_{page_number:04d}.json"
    production_matches = [
        (root / "postprocessed" / bucket / volume_id / filename, bucket)
        for bucket in ("valid", "invalid")
        if (root / "postprocessed" / bucket / volume_id / filename).exists()
    ]
    if len(production_matches) > 1:
        raise RuntimeError(f"Multiple fourth-pass records for {volume_id}/{filename}")
    if production_matches:
        return production_matches[0]

    short_volume_id = volume_id.replace("konbaung_", "")
    pilot_matches = sorted(
        pilot_root.glob(
            f"{short_volume_id}/page_{page_number:04d}/fourth_pass_*nonredundant/postprocessed.json"
        )
    )
    if len(pilot_matches) > 1:
        raise RuntimeError(f"Multiple fourth-pass pilot records for {volume_id}/{filename}")
    return (pilot_matches[0], "pilot") if pilot_matches else None


def warnings_by_index(validation: dict[str, Any]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for warning in validation.get("validation_warnings", []):
        match = WARNING_INDEX_RE.match(str(warning))
        if match:
            grouped.setdefault(int(match.group(1)), []).append(str(warning))
    return grouped


def resolution_by_index(validation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(item["n"]): item
        for item in validation.get("span_resolutions", [])
        if isinstance(item, dict) and item.get("n") is not None
    }


def build_page(
    source_path: Path,
    annotation_root: Path,
    evidence_root: Path,
    third_pass_root: Path,
    third_pass_pilot_root: Path,
    fourth_pass_root: Path,
    fourth_pass_pilot_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_volume_id = source_path.parent.parent.name
    volume_id = source_volume_id.replace("konbaung_", "")
    page_number = int(source_path.stem.replace("page_", ""))
    canonical_text = read_exact_text(source_path)
    annotation_path, source_bucket = locate_record(
        annotation_root, "postprocessed", source_volume_id, page_number
    )
    validation_path, _ = locate_record(annotation_root, "validation", source_volume_id, page_number)
    annotation_data = json.loads(annotation_path.read_text(encoding="utf-8"))
    validation_data = json.loads(validation_path.read_text(encoding="utf-8"))
    evidence_path, evidence_bucket = locate_record(
        evidence_root, "postprocessed", source_volume_id, page_number
    )
    evidence_data = json.loads(evidence_path.read_text(encoding="utf-8"))
    third_pass_record = locate_third_pass_record(
        third_pass_root,
        third_pass_pilot_root,
        source_volume_id,
        page_number,
    )
    third_pass_data: dict[str, Any] = {}
    third_pass_bucket: str | None = None
    if third_pass_record:
        third_pass_path, third_pass_bucket = third_pass_record
        third_pass_data = json.loads(third_pass_path.read_text(encoding="utf-8-sig"))
    fourth_pass_record = locate_fourth_pass_record(
        fourth_pass_root,
        fourth_pass_pilot_root,
        source_volume_id,
        page_number,
    )
    fourth_pass_data: dict[str, Any] = {}
    fourth_pass_bucket: str | None = None
    if fourth_pass_record:
        fourth_pass_path, fourth_pass_bucket = fourth_pass_record
        fourth_pass_data = json.loads(fourth_pass_path.read_text(encoding="utf-8-sig"))
    indexed_warnings = warnings_by_index(validation_data)
    indexed_resolutions = resolution_by_index(validation_data)

    existing_evidence = {
        int(item.get("n", 0)): str(item.get("e", ""))
        for item in evidence_data.get("existing", [])
        if item.get("n") is not None
    }
    triple_inputs: list[dict[str, Any]] = []
    for index, triple in enumerate(annotation_data.get("T", []), start=1):
        triple_inputs.append(
            {
                "triple": triple,
                "evidence": existing_evidence.get(index, ""),
                "source": "first_pass",
                "sourceIndex": index,
                "idPrefix": "a",
            }
        )
    addition_index = 0
    for block_index, addition in enumerate(evidence_data.get("additions", []), start=1):
        evidence_text = str(addition.get("e", ""))
        for triple_index, triple in enumerate(addition.get("t", []), start=1):
            addition_index += 1
            triple_inputs.append(
                {
                    "triple": triple,
                    "evidence": evidence_text,
                    "source": "second_pass",
                    "sourceIndex": addition_index,
                    "blockIndex": block_index,
                    "blockTripleIndex": triple_index,
                    "idPrefix": "x",
                }
            )
    for index, triple in enumerate(third_pass_data.get("T", []), start=1):
        triple_inputs.append(
            {
                "triple": triple,
                "evidence": str(triple.get("e", "")),
                "source": "third_pass",
                "sourceIndex": index,
                "idPrefix": "g",
                "sourceBucket": third_pass_bucket,
            }
        )
    for index, triple in enumerate(fourth_pass_data.get("T", []), start=1):
        triple_inputs.append(
            {
                "triple": triple,
                "evidence": str(triple.get("e", "")),
                "source": "fourth_pass",
                "sourceIndex": index,
                "idPrefix": "h",
                "sourceBucket": fourth_pass_bucket,
            }
        )

    annotations: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for input_record in triple_inputs:
        triple = input_record["triple"]
        index = int(input_record["sourceIndex"])
        is_first_pass = input_record["source"] == "first_pass"
        subject_text = str(triple.get("s", ""))
        object_text = str(triple.get("o", ""))
        subject_candidates = span_candidates(canonical_text, subject_text)
        object_candidates = span_candidates(canonical_text, object_text)
        evidence, evidence_span = span_record(
            str(input_record["evidence"]),
            canonical_text,
            subject_candidates,
            object_candidates,
        )
        selected_subject, subject_inferred, selected_object, object_inferred = select_endpoints_in_evidence(
            subject_candidates, object_candidates, evidence_span
        )
        method = "evidence_first"
        if subject_inferred or object_inferred:
            method = "evidence_first_with_inference"
        warnings = indexed_warnings.get(index, []) if is_first_pass else []
        if selected_subject is None or selected_object is None:
            status = "unresolved"
        elif warnings:
            status = "invalid"
        else:
            status = "resolved"

        annotation_id = f"{volume_id}-p{page_number:04d}-{input_record['idPrefix']}{index:03d}"
        record = {
            "id": annotation_id,
            "subject": endpoint_record(
                subject_text,
                str(triple.get("sg", "")),
                str(triple.get("st", "")),
                selected_subject,
                subject_candidates,
                canonical_text,
                subject_inferred,
            ),
            "relation": {
                "rawLabel": str(triple.get("p", "")),
                "displayLabel": str(triple.get("p", "")).replace("_", " ").title(),
            },
            "object": endpoint_record(
                object_text,
                str(triple.get("og", "")),
                str(triple.get("ot", "")),
                selected_object,
                object_candidates,
                canonical_text,
                object_inferred,
            ),
            "evidence": evidence,
            "metadata": {
                "date": metadata_record(
                    triple.get("d"), triple.get("dg"), canonical_text, evidence_span
                ),
                "location": metadata_record(
                    triple.get("l"), triple.get("lg"), canonical_text, evidence_span
                ),
                "quantity": metadata_record(
                    triple.get("q"), triple.get("qg"), canonical_text, evidence_span
                ),
            },
            "rawOpenCodes": {
                "subjectType": triple.get("st") or None,
                "objectType": triple.get("ot") or None,
                "relationCode": triple.get("p") or None,
            },
            "supportedClaimIds": triple.get("c", []) if input_record["source"] == "third_pass" else [],
            "validation": {
                "status": status,
                "warnings": warnings,
                "alignmentMethod": method,
                "sourceBucket": (
                    source_bucket
                    if is_first_pass
                    else input_record.get("sourceBucket", evidence_bucket)
                ),
                "evidenceStatus": evidence["status"],
            },
            "provenance": {
                "pass": input_record["source"],
                "sourceIndex": index,
                "evidenceBlockIndex": input_record.get("blockIndex"),
                "blockTripleIndex": input_record.get("blockTripleIndex"),
            },
            "rawSourceRecord": triple,
        }
        annotations.append(record)
        diagnostics.append(
            {
                "pageId": f"{volume_id}-p{page_number:04d}",
                "annotationId": annotation_id,
                "status": status,
                "alignmentMethod": method,
                "subjectLiteral": subject_text,
                "subjectCandidates": subject_candidates,
                "objectLiteral": object_text,
                "objectCandidates": object_candidates,
                "warnings": warnings,
                "evidenceLiteral": evidence["text"],
                "evidenceStatus": evidence["status"],
                "evidenceCandidateCount": evidence["candidateCount"],
                "subjectInferred": subject_inferred,
                "objectInferred": object_inferred,
            }
        )

    status_counts = Counter(item["validation"]["status"] for item in annotations)
    evidence_counts = Counter(item["evidence"]["status"] for item in annotations)
    checksum = hashlib.sha1(canonical_text.encode("utf-8")).hexdigest()
    page_record = {
        "schemaVersion": 4,
        "id": f"{volume_id}-p{page_number:04d}",
        "bookId": "konbaung_chronicles",
        "volumeId": volume_id,
        "sourceVolumeId": source_volume_id,
        "pageNumber": page_number,
        "printedPageNumber": None,
        "canonicalText": canonical_text,
        "summary": annotation_data.get("summary") or None,
        "annotations": annotations,
        "source": {
            "filename": source_path.name,
            "relativePath": source_path.relative_to(DEFAULT_SOURCE_ROOT).as_posix()
            if source_path.is_relative_to(DEFAULT_SOURCE_ROOT)
            else str(source_path),
            "checksum": checksum,
            "annotationBucket": source_bucket,
        },
        "diagnostics": {
            "canonicalCodePointLength": len(canonical_text),
            "canonicalUtf16Length": utf16_offset(canonical_text, len(canonical_text)),
            "annotationCount": len(annotations),
            "resolvedAnnotationCount": status_counts["resolved"],
            "ambiguousAnnotationCount": status_counts["ambiguous"],
            "unresolvedAnnotationCount": status_counts["unresolved"],
            "invalidAnnotationCount": status_counts["invalid"],
            "resolvedEvidenceCount": evidence_counts["resolved"],
            "ambiguousEvidenceCount": evidence_counts["ambiguous"],
            "unresolvedEvidenceCount": evidence_counts["unresolved"],
            "firstPassAnnotationCount": sum(
                item["provenance"]["pass"] == "first_pass" for item in annotations
            ),
            "secondPassAnnotationCount": sum(
                item["provenance"]["pass"] == "second_pass" for item in annotations
            ),
            "thirdPassAnnotationCount": sum(
                item["provenance"]["pass"] == "third_pass" for item in annotations
            ),
            "fourthPassAnnotationCount": sum(
                item["provenance"]["pass"] == "fourth_pass" for item in annotations
            ),
        },
    }
    write_json(output_root / "pages" / volume_id / f"{page_number:04d}.json", page_record)
    return page_record, diagnostics


def build(
    source_root: Path,
    annotation_root: Path,
    evidence_root: Path,
    third_pass_root: Path,
    third_pass_pilot_root: Path,
    fourth_pass_root: Path,
    fourth_pass_pilot_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_paths = sorted(source_root.glob("konbaung_vol*/pages/page_*.txt"))
    if not source_paths:
        raise FileNotFoundError(f"No source pages found under {source_root}")

    pages_by_volume: dict[str, list[int]] = {}
    all_diagnostics: list[dict[str, Any]] = []
    aggregate = Counter()
    for source_path in source_paths:
        page, diagnostics = build_page(
            source_path,
            annotation_root,
            evidence_root,
            third_pass_root,
            third_pass_pilot_root,
            fourth_pass_root,
            fourth_pass_pilot_root,
            output_root,
        )
        pages_by_volume.setdefault(page["volumeId"], []).append(page["pageNumber"])
        all_diagnostics.extend(diagnostics)
        aggregate["pages"] += 1
        aggregate["annotations"] += len(page["annotations"])
        for annotation in page["annotations"]:
            aggregate[annotation["validation"]["status"]] += 1

    labels = {"vol1": "Volume I", "vol2": "Volume II", "vol3": "Volume III"}
    volumes = []
    for volume_id in sorted(pages_by_volume):
        pages = sorted(pages_by_volume[volume_id])
        volumes.append(
            {
                "id": volume_id,
                "label": labels.get(volume_id, volume_id),
                "availablePages": pages,
                "firstPage": pages[0],
                "lastPage": pages[-1],
                "pageCount": len(pages),
            }
        )
    index = {
        "schemaVersion": 4,
        "bookId": "konbaung_chronicles",
        "title": "Konbaung Chronicles",
        "volumes": volumes,
        "totals": dict(aggregate),
    }
    write_json(output_root / "index.json", index)
    write_json(output_root / "diagnostics" / "annotation-alignment.json", all_diagnostics)
    write_json(output_root / "diagnostics" / "summary.json", {"totals": dict(aggregate)})
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--third-pass-root", type=Path, default=DEFAULT_THIRD_PASS_ROOT)
    parser.add_argument(
        "--third-pass-pilot-root", type=Path, default=DEFAULT_THIRD_PASS_PILOT_ROOT
    )
    parser.add_argument("--fourth-pass-root", type=Path, default=DEFAULT_FOURTH_PASS_ROOT)
    parser.add_argument(
        "--fourth-pass-pilot-root", type=Path, default=DEFAULT_FOURTH_PASS_PILOT_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build(
        args.source_root.resolve(),
        args.annotation_root.resolve(),
        args.evidence_root.resolve(),
        args.third_pass_root.resolve(),
        args.third_pass_pilot_root.resolve(),
        args.fourth_pass_root.resolve(),
        args.fourth_pass_pilot_root.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps(result["totals"], indent=2))
