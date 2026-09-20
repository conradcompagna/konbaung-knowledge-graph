#!/usr/bin/env python3
"""Retry unresolved axial pages as ordered chunks of at most 30 triples."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

import konbaung_gemini_flashlite_axial_full_corpus_batch as full


FULL_ROOT = full.DEFAULT_OUT_DIR
RETRY_ROOT = FULL_ROOT / "retry_chunked_minimal_20260729"
PAGE_KEYS = (
    "vol1-p0206",
    "vol1-p0208",
    "vol1-p0209",
    "vol2-p0024",
    "vol2-p0032",
    "vol2-p0086",
    "vol2-p0165",
    "vol2-p0178",
)
CHUNK_SIZE = 30
MAX_OUTPUT_TOKENS = 15000
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
}


@dataclass
class Chunk:
    original: full.PageJob
    job: full.PageJob
    index: int
    total: int
    start: int
    stop: int


def original_jobs() -> list[full.PageJob]:
    by_key = {job.key: job for job in full.load_jobs()}
    jobs = [by_key[key] for key in PAGE_KEYS]
    if sum(len(job.triples) for job in jobs) != 744:
        raise ValueError("Chunk retry triple count differs from the audited 744")
    return jobs


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for original in original_jobs():
        total = (len(original.triples) + CHUNK_SIZE - 1) // CHUNK_SIZE
        sentence_by_sid = {
            sentence["sid"]: sentence for sentence in original.sentences
        }
        selection_by_sid = {
            selection["sid"]: selection for selection in original.selections
        }
        for index, start in enumerate(
            range(0, len(original.triples), CHUNK_SIZE),
            start=1,
        ):
            stop = min(start + CHUNK_SIZE, len(original.triples))
            source_triples = original.triples[start:stop]
            sids = list(dict.fromkeys(triple["sid"] for triple in source_triples))
            triples = []
            for local_index, triple in enumerate(source_triples):
                value = copy.deepcopy(triple)
                value["i"] = local_index
                triples.append(value)
            chunk_key = f"{original.key}-c{index:02d}of{total:02d}"
            job = full.PageJob(
                key=chunk_key,
                volume=original.volume,
                page=original.page,
                summary=original.summary,
                sentences=[copy.deepcopy(sentence_by_sid[sid]) for sid in sids],
                triples=triples,
                selections=[copy.deepcopy(selection_by_sid[sid]) for sid in sids],
            )
            chunks.append(
                Chunk(
                    original=original,
                    job=job,
                    index=index,
                    total=total,
                    start=start,
                    stop=stop,
                )
            )
    if sum(len(chunk.job.triples) for chunk in chunks) != 744:
        raise ValueError("Chunk plan changed the total triple count")
    return chunks


def client_and_cache() -> tuple[genai.Client, str, dict[str, Any]]:
    key = full.api_key()
    if not key:
        raise RuntimeError("No Gemini API key found")
    client = genai.Client(api_key=key)
    cache_record = full.read_json(FULL_ROOT / "cache_record.json")
    cache = client.caches.get(name=cache_record["name"])
    return client, cache.name, cache_record


def build_request(
    chunk: Chunk,
    cache_name: str,
    template: str,
    schema: dict[str, Any],
) -> types.InlinedRequest:
    job = chunk.job
    return types.InlinedRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=full.dynamic_text(template, job))],
            )
        ],
        metadata={
            "key": job.key,
            "original_key": chunk.original.key,
            "chunk": str(chunk.index),
            "triple_start": str(chunk.start),
            "triple_stop": str(chunk.stop),
        },
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            response_json_schema=full.response_schema(
                schema,
                len(job.triples),
            ),
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
            ),
        ),
    )


def write_plan(chunks: list[Chunk], cache_record: dict[str, Any]) -> None:
    full.write_json(
        RETRY_ROOT / "chunk_plan.json",
        {
            "model": full.MODEL,
            "thinking_level": "minimal",
            "chunk_size": CHUNK_SIZE,
            "page_count": len(PAGE_KEYS),
            "chunk_count": len(chunks),
            "triple_count": sum(len(chunk.job.triples) for chunk in chunks),
            "cached_prefix": cache_record,
            "chunks": [
                {
                    "key": chunk.job.key,
                    "original_key": chunk.original.key,
                    "index": chunk.index,
                    "total": chunk.total,
                    "original_triple_start": chunk.start,
                    "original_triple_stop": chunk.stop,
                    "triple_count": len(chunk.job.triples),
                    "sentence_ids": [
                        sentence["sid"] for sentence in chunk.job.sentences
                    ],
                }
                for chunk in chunks
            ],
        },
    )


def submit() -> dict[str, Any]:
    RETRY_ROOT.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    _, template, schema = full.load_prompt_package()
    client, cache_name, cache_record = client_and_cache()
    write_plan(chunks, cache_record)
    submitted_path = RETRY_ROOT / "batch_submitted.json"
    if submitted_path.exists():
        batch = client.batches.get(name=full.read_json(submitted_path)["name"])
    else:
        batch = client.batches.create(
            model=full.MODEL,
            src=[
                build_request(chunk, cache_name, template, schema)
                for chunk in chunks
            ],
            config=types.CreateBatchJobConfig(
                display_name=f"konbaung_axial_chunk_retry_{full.utc_stamp()}"
            ),
        )
        full.write_json(submitted_path, full.json_safe(batch))
    record = {
        "status": "submitted",
        "submitted_at": full.utc_now(),
        "batch_name": batch.name,
        "state": full.enum_name(batch.state),
        "page_count": len(PAGE_KEYS),
        "chunk_count": len(chunks),
        "triple_count": sum(len(chunk.job.triples) for chunk in chunks),
        "chunk_size": CHUNK_SIZE,
        "thinking_level": "minimal",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "cache_name": cache_name,
    }
    full.write_json(RETRY_ROOT / "submission_summary.json", record)
    return record


def status() -> dict[str, Any]:
    client, _, _ = client_and_cache()
    submitted = full.read_json(RETRY_ROOT / "batch_submitted.json")
    batch = client.batches.get(name=submitted["name"])
    record = {
        "checked_at": full.utc_now(),
        "name": batch.name,
        "state": full.enum_name(batch.state),
        "completion_stats": full.json_safe(
            getattr(batch, "completion_stats", None)
        ),
    }
    full.write_json(RETRY_ROOT / "batch_latest.json", full.json_safe(batch))
    full.write_json(RETRY_ROOT / "status.json", record)
    return record


def process_batch(
    batch: Any,
    chunks: list[Chunk],
    artifact_root: Path = RETRY_ROOT,
) -> dict[str, Any]:
    if full.enum_name(batch.state) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"{batch.name} ended in {full.enum_name(batch.state)}"
        )
    by_key = {chunk.job.key: chunk for chunk in chunks}
    responses = getattr(getattr(batch, "dest", None), "inlined_responses", None) or []
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for inlined in responses:
        metadata = getattr(inlined, "metadata", None) or {}
        chunk = by_key.get(metadata.get("key", ""))
        if chunk is None:
            rows.append(
                {"key": metadata.get("key"), "category": "unknown_response"}
            )
            totals["unknown_response"] += 1
            continue
        job = chunk.job
        seen.add(job.key)
        if getattr(inlined, "error", None) is not None:
            attempt = {
                "key": job.key,
                "category": "transport_error",
                "error": full.json_safe(inlined.error),
            }
        elif getattr(inlined, "response", None) is None:
            attempt = {
                "key": job.key,
                "category": "transport_error",
                "error": "missing response",
            }
        else:
            response = inlined.response
            full.write_json(
                artifact_root / "raw_responses" / f"{job.key}.json",
                full.json_safe(response),
            )
            attempt = full.classify_response(
                job,
                response,
                MAX_OUTPUT_TOKENS,
            )
        totals[attempt["category"]] += 1
        for usage_key in full.USAGE_KEYS:
            value = attempt.get("usage", {}).get(usage_key)
            if isinstance(value, int):
                totals[usage_key] += value
        full.write_json(artifact_root / "attempts" / f"{job.key}.json", attempt)
        rows.append(
            {
                "key": job.key,
                "original_key": chunk.original.key,
                "chunk": chunk.index,
                "start": chunk.start,
                "stop": chunk.stop,
                "category": attempt["category"],
                "finish_reasons": attempt.get("finish_reasons", []),
                "usage": attempt.get("usage", {}),
                "errors": attempt.get("errors", attempt.get("error")),
            }
        )
    for chunk in chunks:
        if chunk.job.key in seen:
            continue
        attempt = {
            "key": chunk.job.key,
            "category": "transport_error",
            "error": "missing inlined batch response",
        }
        full.write_json(
            artifact_root / "attempts" / f"{chunk.job.key}.json",
            attempt,
        )
        totals["transport_error"] += 1
        rows.append(
            {
                "key": chunk.job.key,
                "original_key": chunk.original.key,
                "chunk": chunk.index,
                "start": chunk.start,
                "stop": chunk.stop,
                "category": "transport_error",
                "errors": "missing inlined batch response",
            }
        )
    return {
        "batch_name": batch.name,
        "state": full.enum_name(batch.state),
        "thinking_level": "minimal",
        "chunk_size": CHUNK_SIZE,
        "totals": dict(totals),
        "results": rows,
    }


def normalized_label(value: str) -> str:
    return " ".join(value.casefold().split())


def reassemble_page(
    original: full.PageJob,
    page_chunks: list[Chunk],
    artifact_root: Path = RETRY_ROOT,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    results: list[dict[str, Any]] = []
    for chunk in page_chunks:
        attempt = full.read_json(
            artifact_root / "attempts" / f"{chunk.job.key}.json"
        )
        if attempt["category"] != "accepted":
            errors.append(
                f"{chunk.job.key}: {attempt['category']}: "
                f"{attempt.get('errors', attempt.get('error'))}"
            )
        else:
            results.append(attempt["result"])
    if errors:
        return None, errors

    final_entities: list[dict[str, str]] = []
    final_relations: list[dict[str, str]] = []
    entity_by_label: dict[str, str] = {}
    relation_by_label: dict[str, str] = {}
    annotations: list[dict[str, str]] = []
    for result in results:
        entity_map: dict[str, str] = {}
        relation_map: dict[str, str] = {}
        for source_key, proposals, final_values, by_label, prefix in (
            (
                "new_entity_categories",
                result["new_entity_categories"],
                final_entities,
                entity_by_label,
                "NE",
            ),
            (
                "new_relation_categories",
                result["new_relation_categories"],
                final_relations,
                relation_by_label,
                "NR",
            ),
        ):
            local_map = entity_map if source_key.startswith("new_entity") else relation_map
            for proposal in proposals:
                label_key = normalized_label(proposal["label"])
                final_id = by_label.get(label_key)
                if final_id is None:
                    final_id = f"{prefix}{len(final_values) + 1:02d}"
                    if len(final_values) >= 99:
                        return None, [f"{original.key}: too many provisional categories"]
                    value = copy.deepcopy(proposal)
                    value["id"] = final_id
                    final_values.append(value)
                    by_label[label_key] = final_id
                local_map[proposal["id"]] = final_id
        for annotation in result["annotations"]:
            value = copy.deepcopy(annotation)
            if value["s"].startswith("NE"):
                value["s"] = entity_map[value["s"]]
            if value["o"].startswith("NE"):
                value["o"] = entity_map[value["o"]]
            if value["r"].startswith("NR"):
                value["r"] = relation_map[value["r"]]
            annotations.append(value)

    reassembled = {
        "new_entity_categories": final_entities,
        "new_relation_categories": final_relations,
        "annotations": annotations,
    }
    validation_errors = full.validate_result(original, reassembled)
    if validation_errors:
        return None, validation_errors
    return reassembled, []


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    relative = path.relative_to(FULL_ROOT)
    target = RETRY_ROOT / "premerge_backup" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)


def reassemble_and_merge(
    chunk_summary: dict[str, Any],
    chunks: list[Chunk],
) -> dict[str, Any]:
    previous_report = full.read_json(FULL_ROOT / "final_report.json")
    by_original: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_original.setdefault(chunk.original.key, []).append(chunk)

    page_rows: list[dict[str, Any]] = []
    accepted_pages: set[str] = set()
    accepted_triples = 0
    for original in original_jobs():
        page_chunks = sorted(
            by_original[original.key],
            key=lambda value: value.index,
        )
        result, errors = reassemble_page(original, page_chunks)
        reassembly_record = {
            "key": original.key,
            "chunk_keys": [chunk.job.key for chunk in page_chunks],
            "chunk_ranges": [
                [chunk.start, chunk.stop] for chunk in page_chunks
            ],
            "triple_count": len(original.triples),
            "accepted": result is not None,
            "errors": errors,
        }
        full.write_json(
            RETRY_ROOT / "reassembled" / f"{original.key}.json",
            {**reassembly_record, "result": result},
        )
        if result is None:
            page_rows.append(
                {
                    "key": original.key,
                    "category": "chunk_validation_invalid",
                    "errors": errors,
                }
            )
            continue

        accepted_pages.add(original.key)
        accepted_triples += len(original.triples)
        for path in (
            FULL_ROOT
            / "attempts"
            / f"vol{original.volume}"
            / f"{original.key}.json",
            full.page_output_dir(FULL_ROOT, original) / "review.md",
        ):
            backup_file(path)
        target = full.page_output_dir(FULL_ROOT, original)
        full.write_json(target / "page_data.json", original.payload)
        full.write_json(target / "result.json", result)
        chunk_usage: Counter[str] = Counter()
        for chunk in page_chunks:
            attempt = full.read_json(
                RETRY_ROOT / "attempts" / f"{chunk.job.key}.json"
            )
            for usage_key in full.USAGE_KEYS:
                chunk_usage[usage_key] += int(
                    attempt.get("usage", {}).get(usage_key, 0)
                )
        full.write_json(
            target / "run_metadata.json",
            {
                "status": "completed_valid",
                "source": "gemini_batch_chunked_minimal_retry",
                "model": full.MODEL,
                "thinking_level": "minimal",
                "temperature": 0.0,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "batch_name": chunk_summary["batch_name"],
                "chunk_size": CHUNK_SIZE,
                "chunk_keys": [chunk.job.key for chunk in page_chunks],
                "usage": dict(chunk_usage),
                "validation": {"accepted": True, "errors": []},
            },
        )
        full.write_json(
            FULL_ROOT
            / "attempts"
            / f"vol{original.volume}"
            / f"{original.key}.json",
            {
                "key": original.key,
                "category": "accepted",
                "source": "chunked_minimal_retry",
                "chunk_keys": [chunk.job.key for chunk in page_chunks],
                "result": result,
                "errors": [],
            },
        )
        page_rows.append(
            {
                "key": original.key,
                "category": "accepted",
                "errors": [],
            }
        )

    for aggregate in (
        FULL_ROOT / "final_report.json",
        FULL_ROOT / "run_summary.json",
        FULL_ROOT / "final" / "all_pages.jsonl",
        FULL_ROOT / "final" / "all_tagged_triples.jsonl",
        FULL_ROOT / "final" / "all_provisional_categories.jsonl",
        FULL_ROOT / "final" / "truncated_pages.json",
        FULL_ROOT / "final" / "unresolved_pages.json",
    ):
        backup_file(aggregate)

    page_totals: Counter[str] = Counter(row["category"] for row in page_rows)
    for usage_key in full.USAGE_KEYS:
        page_totals[usage_key] = int(
            chunk_summary["totals"].get(usage_key, 0)
        )
    page_summary = {
        "batch_name": chunk_summary["batch_name"],
        "state": chunk_summary["state"],
        "thinking_level": "minimal",
        "totals": dict(page_totals),
        "results": page_rows,
    }
    full.write_json(RETRY_ROOT / "page_result_summary.json", page_summary)

    all_jobs = full.load_jobs()
    prefix, _, _ = full.load_prompt_package()
    original_summaries = [
        full.read_json(FULL_ROOT / "batch_results" / f"axial_vol{volume}.json")
        for volume in (1, 2, 3)
    ]
    first_retry = full.read_json(
        FULL_ROOT / "retry_minimal_20260729" / "batch_result.json"
    )
    report = full.finalize(
        FULL_ROOT,
        all_jobs,
        [*original_summaries, first_retry, page_summary],
        prefix,
    )
    if "minimal_retry" in previous_report:
        report["minimal_retry"] = previous_report["minimal_retry"]
    remaining = [
        row for row in page_rows if row["category"] != "accepted"
    ]
    report["retry_policy_applied"] = (
        "one 16-page minimal-thinking retry followed by one ordered "
        "30-triple chunk retry for the eight remaining pages"
    )
    report["chunked_minimal_retry"] = {
        "batch_name": chunk_summary["batch_name"],
        "submitted_pages": len(PAGE_KEYS),
        "submitted_chunks": len(chunks),
        "submitted_triples": sum(len(chunk.job.triples) for chunk in chunks),
        "chunk_size": CHUNK_SIZE,
        "accepted_pages": len(accepted_pages),
        "accepted_triples": accepted_triples,
        "remaining_pages": remaining,
        "usage": {
            key: chunk_summary["totals"].get(key, 0)
            for key in full.USAGE_KEYS
        },
    }
    full.write_json(FULL_ROOT / "final_report.json", report)
    full.write_json(
        FULL_ROOT / "run_summary.json",
        {
            "batches": [
                *original_summaries,
                first_retry,
                page_summary,
            ],
            "final": report,
        },
    )

    combined_review = RETRY_ROOT / "combined_review.md"
    combined_parts = [
        "# Konbaung axial-coding chunked minimal-thinking retry review",
        "",
        f"- Submitted pages: **{len(PAGE_KEYS)}**",
        f"- Submitted chunks: **{len(chunks)}**",
        f"- Submitted triples: **{sum(len(chunk.job.triples) for chunk in chunks)}**",
        f"- Accepted pages: **{len(accepted_pages)}**",
        f"- Accepted triples: **{accepted_triples}**",
        f"- Remaining unresolved pages: **{len(remaining)}**",
        "",
    ]
    by_key = {job.key: job for job in original_jobs()}
    for key in PAGE_KEYS:
        combined_parts.extend(
            [
                "---",
                "",
                (full.page_output_dir(FULL_ROOT, by_key[key]) / "review.md")
                .read_text(encoding="utf-8")
                .rstrip(),
                "",
            ]
        )
    full.write_text(combined_review, "\n".join(combined_parts).rstrip() + "\n")
    merge_record = {
        "merged_at": full.utc_now(),
        "accepted_pages": sorted(accepted_pages),
        "accepted_triples": accepted_triples,
        "remaining_pages": remaining,
        "final_status": report["status"],
        "tagged_triples": report["tagged_triples"],
        "combined_review": str(combined_review),
    }
    full.write_json(RETRY_ROOT / "merge_record.json", merge_record)
    return merge_record


def collect() -> dict[str, Any]:
    client, _, _ = client_and_cache()
    submitted = full.read_json(RETRY_ROOT / "batch_submitted.json")
    batch = client.batches.get(name=submitted["name"])
    state = full.enum_name(batch.state)
    if state not in TERMINAL_STATES:
        return {"status": "not_ready", "batch": status()}
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Chunk retry batch ended in {state}")
    chunks = build_chunks()
    summary_path = RETRY_ROOT / "batch_result.json"
    if summary_path.exists():
        summary = full.read_json(summary_path)
    else:
        summary = process_batch(batch, chunks)
        full.write_json(summary_path, summary)
    return {
        "status": "collected",
        "summary": summary,
        "merge": reassemble_and_merge(summary, chunks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "submit", "status", "collect"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        chunks = build_chunks()
        _, _, cache_record = client_and_cache()
        RETRY_ROOT.mkdir(parents=True, exist_ok=True)
        write_plan(chunks, cache_record)
        result = {
            "pages": len(PAGE_KEYS),
            "chunks": len(chunks),
            "triples": sum(len(chunk.job.triples) for chunk in chunks),
            "max_chunk_triples": max(len(chunk.job.triples) for chunk in chunks),
        }
    elif args.command == "submit":
        result = submit()
    elif args.command == "status":
        result = status()
    else:
        result = collect()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
