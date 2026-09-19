#!/usr/bin/env python3
"""Repair the two failed 30-triple chunks as six 10-triple requests."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from google.genai import types

import konbaung_gemini_flashlite_axial_full_corpus_batch as full
import konbaung_gemini_flashlite_axial_retry_chunked as chunked


REPAIR_ROOT = chunked.RETRY_ROOT / "repair_10"
PARENT_KEYS = (
    "vol1-p0208-c03of04",
    "vol2-p0086-c01of02",
)
REPAIR_CHUNK_SIZE = 10


def parent_chunks() -> list[chunked.Chunk]:
    by_key = {chunk.job.key: chunk for chunk in chunked.build_chunks()}
    return [by_key[key] for key in PARENT_KEYS]


def repair_chunks() -> list[chunked.Chunk]:
    values: list[chunked.Chunk] = []
    for parent in parent_chunks():
        parent_job = parent.job
        total = (
            len(parent_job.triples) + REPAIR_CHUNK_SIZE - 1
        ) // REPAIR_CHUNK_SIZE
        sentence_by_sid = {
            sentence["sid"]: sentence for sentence in parent_job.sentences
        }
        selection_by_sid = {
            selection["sid"]: selection for selection in parent_job.selections
        }
        for index, start in enumerate(
            range(0, len(parent_job.triples), REPAIR_CHUNK_SIZE),
            start=1,
        ):
            stop = min(start + REPAIR_CHUNK_SIZE, len(parent_job.triples))
            source_triples = parent_job.triples[start:stop]
            sids = list(dict.fromkeys(triple["sid"] for triple in source_triples))
            triples = []
            for local_index, triple in enumerate(source_triples):
                value = copy.deepcopy(triple)
                value["i"] = local_index
                triples.append(value)
            job = full.PageJob(
                key=f"{parent_job.key}-r{index:02d}of{total:02d}",
                volume=parent_job.volume,
                page=parent_job.page,
                summary=parent_job.summary,
                sentences=[copy.deepcopy(sentence_by_sid[sid]) for sid in sids],
                triples=triples,
                selections=[
                    copy.deepcopy(selection_by_sid[sid]) for sid in sids
                ],
            )
            values.append(
                chunked.Chunk(
                    original=parent_job,
                    job=job,
                    index=index,
                    total=total,
                    start=start,
                    stop=stop,
                )
            )
    if len(values) != 6 or sum(len(value.job.triples) for value in values) != 60:
        raise ValueError("Repair plan must contain six chunks and 60 triples")
    return values


def write_plan(values: list[chunked.Chunk]) -> None:
    full.write_json(
        REPAIR_ROOT / "repair_plan.json",
        {
            "thinking_level": "minimal",
            "chunk_size": REPAIR_CHUNK_SIZE,
            "parent_chunks": list(PARENT_KEYS),
            "repair_chunks": [
                {
                    "key": value.job.key,
                    "parent": value.original.key,
                    "start": value.start,
                    "stop": value.stop,
                    "triple_count": len(value.job.triples),
                }
                for value in values
            ],
        },
    )


def submit() -> dict[str, Any]:
    REPAIR_ROOT.mkdir(parents=True, exist_ok=True)
    values = repair_chunks()
    write_plan(values)
    _, template, schema = full.load_prompt_package()
    client, cache_name, _ = chunked.client_and_cache()
    submitted_path = REPAIR_ROOT / "batch_submitted.json"
    if submitted_path.exists():
        batch = client.batches.get(name=full.read_json(submitted_path)["name"])
    else:
        batch = client.batches.create(
            model=full.MODEL,
            src=[
                chunked.build_request(value, cache_name, template, schema)
                for value in values
            ],
            config=types.CreateBatchJobConfig(
                display_name=f"konbaung_axial_chunk_repair_{full.utc_stamp()}"
            ),
        )
        full.write_json(submitted_path, full.json_safe(batch))
    record = {
        "status": "submitted",
        "submitted_at": full.utc_now(),
        "batch_name": batch.name,
        "state": full.enum_name(batch.state),
        "parent_chunk_count": len(PARENT_KEYS),
        "repair_chunk_count": len(values),
        "triple_count": sum(len(value.job.triples) for value in values),
        "chunk_size": REPAIR_CHUNK_SIZE,
        "thinking_level": "minimal",
        "cache_name": cache_name,
    }
    full.write_json(REPAIR_ROOT / "submission_summary.json", record)
    return record


def status() -> dict[str, Any]:
    client, _, _ = chunked.client_and_cache()
    submitted = full.read_json(REPAIR_ROOT / "batch_submitted.json")
    batch = client.batches.get(name=submitted["name"])
    record = {
        "checked_at": full.utc_now(),
        "name": batch.name,
        "state": full.enum_name(batch.state),
        "completion_stats": full.json_safe(
            getattr(batch, "completion_stats", None)
        ),
    }
    full.write_json(REPAIR_ROOT / "batch_latest.json", full.json_safe(batch))
    full.write_json(REPAIR_ROOT / "status.json", record)
    return record


def replace_parent_attempts(
    repair_summary: dict[str, Any],
    values: list[chunked.Chunk],
) -> dict[str, Any]:
    by_parent: dict[str, list[chunked.Chunk]] = {}
    for value in values:
        by_parent.setdefault(value.original.key, []).append(value)
    replacements: list[dict[str, Any]] = []
    for parent in parent_chunks():
        children = sorted(by_parent[parent.job.key], key=lambda value: value.index)
        result, errors = chunked.reassemble_page(
            parent.job,
            children,
            REPAIR_ROOT,
        )
        if result is None:
            replacements.append(
                {
                    "key": parent.job.key,
                    "category": "repair_validation_invalid",
                    "errors": errors,
                }
            )
            continue
        target = chunked.RETRY_ROOT / "attempts" / f"{parent.job.key}.json"
        backup = REPAIR_ROOT / "premerge_backup" / f"{parent.job.key}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not backup.exists():
            shutil.copy2(target, backup)
        usage: Counter[str] = Counter()
        for child in children:
            attempt = full.read_json(
                REPAIR_ROOT / "attempts" / f"{child.job.key}.json"
            )
            for usage_key in full.USAGE_KEYS:
                usage[usage_key] += int(
                    attempt.get("usage", {}).get(usage_key, 0)
                )
        full.write_json(
            target,
            {
                "key": parent.job.key,
                "category": "accepted",
                "source": "10_triple_repair_chunks",
                "repair_batch_name": repair_summary["batch_name"],
                "repair_chunk_keys": [child.job.key for child in children],
                "usage": dict(usage),
                "result": result,
                "errors": [],
            },
        )
        replacements.append(
            {
                "key": parent.job.key,
                "category": "accepted",
                "errors": [],
                "usage": dict(usage),
            }
        )
    return {
        "repair_batch_name": repair_summary["batch_name"],
        "replacements": replacements,
    }


def combined_chunk_summary(
    repair_summary: dict[str, Any],
    replacement_record: dict[str, Any],
) -> dict[str, Any]:
    original = full.read_json(chunked.RETRY_ROOT / "batch_result.json")
    combined = copy.deepcopy(original)
    replacement_by_key = {
        row["key"]: row for row in replacement_record["replacements"]
    }
    for row in combined["results"]:
        replacement = replacement_by_key.get(row["key"])
        if replacement is None:
            continue
        row["category"] = replacement["category"]
        row["errors"] = replacement["errors"]
        if replacement.get("usage"):
            row["usage"] = replacement["usage"]
        row["repair_batch_name"] = repair_summary["batch_name"]
    combined["batch_name"] = (
        f"{original['batch_name']} + {repair_summary['batch_name']}"
    )
    combined["totals"]["accepted"] = sum(
        row["category"] == "accepted" for row in combined["results"]
    )
    combined["totals"].pop("validation_invalid", None)
    for usage_key in full.USAGE_KEYS:
        combined["totals"][usage_key] = int(
            original["totals"].get(usage_key, 0)
        ) + int(repair_summary["totals"].get(usage_key, 0))
    full.write_json(REPAIR_ROOT / "combined_chunk_result.json", combined)
    return combined


def collect() -> dict[str, Any]:
    client, _, _ = chunked.client_and_cache()
    submitted = full.read_json(REPAIR_ROOT / "batch_submitted.json")
    batch = client.batches.get(name=submitted["name"])
    state = full.enum_name(batch.state)
    if state not in chunked.TERMINAL_STATES:
        return {"status": "not_ready", "batch": status()}
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Repair batch ended in {state}")
    values = repair_chunks()
    summary_path = REPAIR_ROOT / "batch_result.json"
    if summary_path.exists():
        repair_summary = full.read_json(summary_path)
    else:
        repair_summary = chunked.process_batch(
            batch,
            values,
            REPAIR_ROOT,
        )
        full.write_json(summary_path, repair_summary)
    replacement_record = replace_parent_attempts(repair_summary, values)
    full.write_json(REPAIR_ROOT / "replacement_record.json", replacement_record)
    if any(
        row["category"] != "accepted"
        for row in replacement_record["replacements"]
    ):
        return {
            "status": "repair_incomplete",
            "summary": repair_summary,
            "replacements": replacement_record,
        }
    combined = combined_chunk_summary(repair_summary, replacement_record)
    merge_record = chunked.reassemble_and_merge(
        combined,
        chunked.build_chunks(),
    )
    report = full.read_json(chunked.FULL_ROOT / "final_report.json")
    report["chunked_minimal_retry"]["repair"] = {
        "batch_name": repair_summary["batch_name"],
        "parent_chunks": list(PARENT_KEYS),
        "repair_chunk_size": REPAIR_CHUNK_SIZE,
        "repair_chunks": len(values),
        "repair_triples": sum(len(value.job.triples) for value in values),
        "usage": {
            key: repair_summary["totals"].get(key, 0)
            for key in full.USAGE_KEYS
        },
    }
    full.write_json(chunked.FULL_ROOT / "final_report.json", report)
    run_summary = full.read_json(chunked.FULL_ROOT / "run_summary.json")
    run_summary["final"] = report
    full.write_json(chunked.FULL_ROOT / "run_summary.json", run_summary)
    merge_record["repair"] = report["chunked_minimal_retry"]["repair"]
    full.write_json(REPAIR_ROOT / "merge_record.json", merge_record)
    return {
        "status": "collected",
        "summary": repair_summary,
        "replacements": replacement_record,
        "merge": merge_record,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "submit", "status", "collect"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        values = repair_chunks()
        REPAIR_ROOT.mkdir(parents=True, exist_ok=True)
        write_plan(values)
        result = {
            "parents": len(PARENT_KEYS),
            "chunks": len(values),
            "triples": sum(len(value.job.triples) for value in values),
            "max_chunk_triples": max(len(value.job.triples) for value in values),
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
