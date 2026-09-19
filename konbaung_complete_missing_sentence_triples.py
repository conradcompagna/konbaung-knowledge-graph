#!/usr/bin/env python3
"""Complete substantive missing sentence triples and merge paid batch attempts."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from google import genai

from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_translated_sentence_triples_batch import (
    DEFAULT_OUTPUT_ROOT,
    MODEL,
    PageJob,
    TripleResult,
    discover_jobs,
    finalize,
    load_json,
    page_dir,
    review_text,
    save_plan,
    state_name,
    submit_batch,
    usage_dict,
    wait_for_batch,
    write_json,
    write_text,
)


BASE_ATTEMPTS = (
    "smoke",
    "vol1",
    "vol2",
    "vol3",
    "retry_1",
    "retry_2",
    "retry_3",
    "retry_4",
    "retry_1_duplicate_salvage",
)
USAGE_KEYS = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)


def response_text_dict(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)


def attempt_groups(
    out_dir: Path, labels: list[str], wanted_keys: set[str]
) -> dict[str, dict[str, list[tuple[int, str, Any]]]]:
    groups: dict[str, dict[str, list[tuple[int, str, Any]]]] = {
        key: defaultdict(list) for key in wanted_keys
    }
    for rank, label in enumerate(labels):
        path = out_dir / "batch_jobs" / f"{label}_latest.json"
        if not path.exists():
            continue
        batch = load_json(path)
        responses = ((batch.get("dest") or {}).get("inlined_responses") or [])
        for item in responses:
            key = (item.get("metadata") or {}).get("key")
            if key not in wanted_keys:
                continue
            raw = response_text_dict(item.get("response") or {})
            try:
                parsed = TripleResult.model_validate_json(raw)
            except Exception:
                continue
            for group in parsed.S:
                groups[key][group.sid].append((rank, label, group))
    return groups


def exclusion_reason(my: str, en: str) -> str | None:
    clean = re.sub(
        r"[\s\d၀-၉.,၊။…*?\-–—“”'\"()\[\]{}:;˚]+",
        "",
        my,
    )
    if not clean:
        return "punctuation_or_number"
    if re.fullmatch(
        r"(One|Two|[0-9]+|\.{1,3}|[\[\]ItThey ]*(is|did|do|are))\.?",
        en.strip(),
        re.IGNORECASE,
    ):
        return "short_boundary_fragment"
    if len(clean) <= 7 and len(en.split()) <= 4:
        return "short_boundary_fragment"
    if re.search(r"Gatha|meter|Appendix \(|Introduction|digression", en, re.IGNORECASE):
        return "heading_or_metrical_label"
    if re.search(r",\s*[0-9]+\.?$", en) and len(en.split()) <= 8:
        return "numbered_list_fragment"
    return None


def current_invalid_jobs(out_dir: Path, jobs: list[PageJob]) -> list[PageJob]:
    invalid: list[PageJob] = []
    for job in jobs:
        result_path = page_dir(out_dir, job) / "result.json"
        if not result_path.exists() or load_json(result_path).get("accepted") is not True:
            invalid.append(job)
    return invalid


def missing_records(
    jobs: list[PageJob],
    groups: dict[str, dict[str, list[tuple[int, str, Any]]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    substantive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    for job in jobs:
        present = groups[job.key]
        for sentence in job.payload["sentence_pairs"]:
            if sentence["id"] in present:
                continue
            reason = exclusion_reason(sentence["my"], sentence["en"])
            if reason:
                excluded.append(
                    {
                        "page_id": job.key,
                        "sid": sentence["id"],
                        "reason": reason,
                        "my": sentence["my"],
                        "en": sentence["en"],
                    }
                )
            else:
                substantive[job.key].append(sentence)
    return substantive, excluded


def process_completion_batch(
    out_dir: Path,
    label: str,
    batch: Any,
    jobs_by_key: dict[str, PageJob],
) -> dict[str, Any]:
    if state_name(batch.state) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"{label} ended in {state_name(batch.state)}")
    totals = Counter()
    rows: list[dict[str, Any]] = []
    responses = getattr(getattr(batch, "dest", None), "inlined_responses", None) or []
    for item in responses:
        totals["responses_seen"] += 1
        metadata = getattr(item, "metadata", None) or {}
        key = metadata.get("key")
        job = jobs_by_key.get(key or "")
        if job is None or getattr(item, "error", None) is not None:
            totals["response_errors"] += 1
            rows.append({"key": key, "error": str(getattr(item, "error", "unknown key"))})
            continue
        response = getattr(item, "response", None)
        raw = getattr(response, "text", None) or ""
        usage = usage_dict(response)
        for usage_key in USAGE_KEYS:
            if isinstance(usage.get(usage_key), int):
                totals[usage_key] += usage[usage_key]
        try:
            result = TripleResult.model_validate_json(raw)
        except Exception as exc:
            totals["schema_errors"] += 1
            rows.append({"key": key, "error": str(exc)})
            continue
        expected = [item["id"] for item in job.payload["sentence_pairs"]]
        returned = {group.sid for group in result.S}
        missing = [sid for sid in expected if sid not in returned]
        if missing:
            totals["incomplete_pages"] += 1
        else:
            totals["complete_pages"] += 1
        write_text(out_dir / "completion_attempts" / label / f"{key}.json", raw)
        rows.append(
            {
                "key": key,
                "expected": len(expected),
                "returned": len(returned),
                "missing": missing,
                "usage": usage,
            }
        )
    summary = {"label": label, "totals": dict(totals), "results": rows}
    write_json(out_dir / "batch_results" / f"{label}.json", summary)
    return summary


def run() -> None:
    out_dir = DEFAULT_OUTPUT_ROOT
    jobs = discover_jobs()
    jobs_by_key = {job.key: job for job in jobs}
    invalid = current_invalid_jobs(out_dir, jobs)
    wanted_keys = {job.key for job in invalid}
    labels = list(BASE_ATTEMPTS)
    groups = attempt_groups(out_dir, labels, wanted_keys)
    substantive, excluded = missing_records(invalid, groups)
    write_json(
        out_dir / "non_propositional_sentence_exclusions.json",
        {
            "count": len(excluded),
            "by_reason": dict(Counter(row["reason"] for row in excluded)),
            "records": excluded,
        },
    )

    client = genai.Client(api_key=api_key())
    cache_name = load_json(out_dir / "cache_record.json")["name"]
    completion_summaries: list[dict[str, Any]] = []
    for round_number in range(1, 4):
        if not substantive:
            break
        label = f"substantive_completion_{round_number}"
        completion_jobs = [
            PageJob(
                jobs_by_key[key].volume,
                jobs_by_key[key].page,
                {
                    **jobs_by_key[key].payload,
                    "sentence_pairs": records,
                },
            )
            for key, records in sorted(substantive.items())
        ]
        save_plan(out_dir, label, completion_jobs)
        batch = submit_batch(
            client,
            out_dir,
            label,
            completion_jobs,
            cache_name,
            16000,
            6000,
        )
        done = wait_for_batch(
            client,
            out_dir,
            label,
            batch.name,
            30,
            172800,
        )
        completion_summaries.append(
            process_completion_batch(
                out_dir, label, done, {job.key: job for job in completion_jobs}
            )
        )
        labels.append(label)
        groups = attempt_groups(out_dir, labels, wanted_keys)
        substantive, excluded = missing_records(invalid, groups)

    if substantive:
        raise RuntimeError(
            f"Substantive sentence records still missing: "
            f"{sum(len(rows) for rows in substantive.values())}"
        )

    exclusion_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in excluded:
        exclusion_by_page[row["page_id"]].append(row)
    for job in invalid:
        selected = []
        selected_sources: dict[str, str] = {}
        for sentence in job.payload["sentence_pairs"]:
            candidates = groups[job.key].get(sentence["id"], [])
            if not candidates:
                continue
            rank, label, group = max(candidates, key=lambda item: (len(item[2].T), item[0]))
            selected.append(group)
            selected_sources[sentence["id"]] = label
        result = TripleResult(S=selected)
        target = page_dir(out_dir, job)
        record = {
            "accepted": True,
            "errors": [],
            "excluded_non_propositional": exclusion_by_page.get(job.key, []),
            "usage": {},
            "response": result.model_dump(mode="json"),
        }
        write_json(target / "result.json", record)
        write_text(target / "raw_response.json", result.model_dump_json(indent=2))
        write_text(target / "review.md", review_text(job.payload, result, []))
        write_json(
            target / "provenance.json",
            {
                "source": "merged_paid_batch_attempts",
                "model": MODEL,
                "sentence_sources": selected_sources,
                "excluded_non_propositional_count": len(exclusion_by_page.get(job.key, [])),
            },
        )

    audit = finalize(out_dir, jobs)
    audit["excluded_non_propositional_sentence_records"] = len(excluded)
    audit["exclusions_by_reason"] = dict(Counter(row["reason"] for row in excluded))
    write_json(out_dir / "final_audit.json", audit)
    write_json(
        out_dir / "completion_summary.json",
        {
            "initial_invalid_pages": len(invalid),
            "excluded_non_propositional": len(excluded),
            "completion_batches": completion_summaries,
            "final_audit": audit,
        },
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
