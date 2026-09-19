#!/usr/bin/env python3
"""Translate every sentence in the repaired Konbaung corpus with Gemini Batch API."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from konbaung_gemini_cite_sources_annotator import MODEL
from konbaung_gemini_evidence_translation_batch import (
    TERMINAL_STATES,
    TranslationJob,
    TranslationResult,
    json_safe,
    normalize_batch_result,
    response_text,
    state_name,
    utc_stamp,
    wait_for_batch,
    write_json,
    write_text,
)
from konbaung_gemini_sentence_translation_test import (
    TRANSLATION_PROMPT,
    normalize,
    response_schema,
    validate,
    write_review,
)
from konbaung_gemini_summary_claim_completion_annotator import api_key
from konbaung_gemini_structured_open_coding_annotator import usage_dict


ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = ROOT / "konbaung_sentence_corpus_20260713_repaired"
DEFAULT_OUT_DIR = ROOT / "konbaung_sentence_translation_full_batch_20260713_repaired"
DEFAULT_SMOKE_MANIFEST = ROOT / "konbaung_sentence_translation_smoke_test_manifest.json"


@cache
def sentence_rows(volume: int) -> list[dict[str, Any]]:
    path = CORPUS_ROOT / "sentences" / f"vol{volume}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@cache
def rows_by_page(volume: int) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sentence_rows(volume):
        grouped[int(row["owner_page"])].append(row)
    return dict(grouped)


def smoke_pages(path: Path) -> set[tuple[int, int]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(row["volume"]), int(row["page"]))
        for row in manifest.get("pages", [])
        if row.get("schema_valid") is True
    }


def build_payload(volume: int, page_number: int) -> dict[str, Any]:
    grouped = rows_by_page(volume)
    rows = grouped.get(page_number, [])
    if not rows:
        raise ValueError(f"No owned sentences for volume {volume}, page {page_number}")
    page_path = CORPUS_ROOT / "cleaned_pages" / f"vol{volume}" / f"page_{page_number:04d}.txt"
    return {
        "page_id": f"vol{volume}-p{page_number:04d}",
        "page_context": normalize(page_path.read_text(encoding="utf-8")),
        "sentences": [{"id": row["id"], "my": normalize(row["text"])} for row in rows],
    }


def prompt_and_payload(job: TranslationJob) -> tuple[str, dict[str, Any]]:
    payload = build_payload(job.volume, job.page)
    prompt = (
        TRANSLATION_PROMPT
        + "\n\n<INPUT>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</INPUT>\n"
    )
    return prompt, payload


def jobs_for_volume(volume: int, excluded: set[tuple[int, int]]) -> list[TranslationJob]:
    return [
        TranslationJob(volume, page)
        for page in sorted(rows_by_page(volume))
        if (volume, page) not in excluded
    ]


def build_request(job: TranslationJob, max_output_tokens: int) -> types.InlinedRequest:
    prompt, payload = prompt_and_payload(job)
    ids = [item["id"] for item in payload["sentences"]]
    return types.InlinedRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        metadata={"key": job.key, "volume": str(job.volume), "page": f"{job.page:04d}"},
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=response_schema(ids),
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )


def validate_batch_result(payload: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors = validate(payload, result)
    for index, item in enumerate(result.get("translations", [])):
        translation = item.get("translation") if isinstance(item, dict) else None
        if not isinstance(translation, str):
            errors.append(f"translations[{index}] is not text")
        elif re.search(r"[\u1000-\u109f\uaa60-\uaa7f\ua9e0-\ua9ff]", translation):
            errors.append(f"translations[{index}] still contains Burmese script")
    return errors


def save_plan(out_dir: Path, label: str, jobs: list[TranslationJob]) -> None:
    write_json(
        out_dir / "batch_plans" / f"{label}.json",
        {
            "label": label,
            "page_count": len(jobs),
            "sentence_count": sum(len(build_payload(job.volume, job.page)["sentences"]) for job in jobs),
            "pages": [{"key": job.key, "volume": job.volume, "page": job.page} for job in jobs],
        },
    )


def load_plan(out_dir: Path, label: str) -> list[TranslationJob]:
    data = json.loads((out_dir / "batch_plans" / f"{label}.json").read_text(encoding="utf-8"))
    return [TranslationJob(int(row["volume"]), int(row["page"])) for row in data["pages"]]


def materialize_smoke_pages(out_dir: Path, manifest_path: Path) -> dict[str, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    totals = {"pages": 0, "sentences": 0}
    for row in manifest.get("pages", []):
        if row.get("schema_valid") is not True:
            continue
        volume = int(row["volume"])
        page = int(row["page"])
        old_dir = ROOT / row["output"]
        old_payload = json.loads((old_dir / "payload.json").read_text(encoding="utf-8"))
        old_result = json.loads((old_dir / "result.json").read_text(encoding="utf-8"))
        new_payload = build_payload(volume, page)
        old_sources = [normalize(item["my"]) for item in old_payload["sentences"]]
        new_sources = [normalize(item["my"]) for item in new_payload["sentences"]]
        if old_sources != new_sources:
            raise RuntimeError(f"Smoke-page sentence text changed for vol{volume} page {page}")
        old_translations = old_result["response"]["translations"]
        if len(old_translations) != len(new_payload["sentences"]):
            raise RuntimeError(f"Smoke-page translation count changed for vol{volume} page {page}")
        result = {
            "translations": [
                {"id": source["id"], "translation": translated["translation"]}
                for source, translated in zip(new_payload["sentences"], old_translations)
            ]
        }
        errors = validate_batch_result(new_payload, result)
        if errors:
            raise RuntimeError(f"Remapped smoke page vol{volume} page {page} failed: {errors}")
        page_dir = out_dir / "pages" / f"vol{volume}" / f"page_{page:04d}"
        write_json(page_dir / "payload.json", new_payload)
        write_json(
            page_dir / "result.json",
            {
                "accepted": True,
                "errors": [],
                "usage": old_result.get("usage", {}),
                "reused_smoke_test": True,
                "response": result,
            },
        )
        write_review(page_dir / "review.md", new_payload, result, [])
        totals["pages"] += 1
        totals["sentences"] += len(result["translations"])
    return totals


def submit_batch(
    client: genai.Client,
    model: str,
    out_dir: Path,
    label: str,
    jobs: list[TranslationJob],
    max_output_tokens: int,
) -> Any:
    requests = [build_request(job, max_output_tokens) for job in jobs]
    batch = client.batches.create(
        model=model,
        src=requests,
        config=types.CreateBatchJobConfig(
            display_name=f"konbaung_sentence_translation_{label}_{utc_stamp()}"
        ),
    )
    write_json(out_dir / "batch_jobs" / f"{label}_submitted.json", json_safe(batch))
    return batch


def process_batch(
    out_dir: Path,
    label: str,
    batch: Any,
    jobs: list[TranslationJob],
) -> dict[str, Any]:
    if state_name(batch.state) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"{label} ended in {state_name(batch.state)}")
    by_key = {job.key: job for job in jobs}
    totals = {
        "expected_pages": len(jobs),
        "responses_seen": 0,
        "valid_pages": 0,
        "flagged_pages": 0,
        "response_errors": 0,
        "schema_errors": 0,
        "sentences": 0,
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "total_token_count": 0,
    }
    rows = []
    responses = getattr(getattr(batch, "dest", None), "inlined_responses", None) or []
    for inlined in responses:
        totals["responses_seen"] += 1
        metadata = getattr(inlined, "metadata", None) or {}
        key = metadata.get("key", "")
        job = by_key.get(key)
        if job is None:
            totals["response_errors"] += 1
            rows.append({"key": key, "error": "unknown response key"})
            continue
        if getattr(inlined, "error", None) is not None or getattr(inlined, "response", None) is None:
            totals["response_errors"] += 1
            error = json_safe(getattr(inlined, "error", None)) or "missing response"
            write_json(out_dir / "errors" / f"vol{job.volume}" / f"page_{job.page:04d}.json", error)
            rows.append({"key": key, "error": error})
            continue

        response = inlined.response
        usage = usage_dict(response)
        for name in (
            "prompt_token_count",
            "cached_content_token_count",
            "candidates_token_count",
            "total_token_count",
        ):
            if isinstance(usage.get(name), int):
                totals[name] += usage[name]

        prompt, payload = prompt_and_payload(job)
        page_dir = out_dir / "pages" / f"vol{job.volume}" / f"page_{job.page:04d}"
        raw = response_text(response)
        write_text(page_dir / "prompt_sent.txt", prompt)
        write_json(page_dir / "payload.json", payload)
        write_text(page_dir / "raw_response.json", raw)
        try:
            result = normalize_batch_result(json.loads(raw))
        except (json.JSONDecodeError, TypeError) as exc:
            totals["schema_errors"] += 1
            write_json(page_dir / "result.json", {"accepted": False, "errors": [str(exc)], "usage": usage})
            rows.append({"key": key, "error": f"schema parse failed: {exc}"})
            continue

        errors = validate_batch_result(payload, result)
        totals["sentences"] += len(payload["sentences"])
        totals["flagged_pages" if errors else "valid_pages"] += 1
        write_review(page_dir / "review.md", payload, result, errors)
        write_json(
            page_dir / "result.json",
            {"accepted": not errors, "errors": errors, "usage": usage, "response": result},
        )
        rows.append(
            {
                "key": key,
                "volume": job.volume,
                "page": job.page,
                "accepted": not errors,
                "sentences": len(payload["sentences"]),
                "usage": usage,
            }
        )

    summary = {
        "label": label,
        "batch_name": batch.name,
        "state": state_name(batch.state),
        "completion_stats": json_safe(getattr(batch, "completion_stats", None)),
        "totals": totals,
        "results": rows,
    }
    write_json(out_dir / "batch_results" / f"{label}_summary.json", summary)
    write_text(
        out_dir / "batch_results" / f"{label}_results.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
    )
    return summary


def finalize(out_dir: Path) -> dict[str, Any]:
    all_output_rows = []
    totals = {"pages": 0, "sentences": 0, "valid_pages": 0, "flagged_pages": 0, "missing_pages": 0}
    for volume in (1, 2, 3):
        volume_rows = []
        for page, rows in sorted(rows_by_page(volume).items()):
            page_dir = out_dir / "pages" / f"vol{volume}" / f"page_{page:04d}"
            result_path = page_dir / "result.json"
            totals["pages"] += 1
            if not result_path.exists():
                totals["missing_pages"] += 1
                continue
            stored = json.loads(result_path.read_text(encoding="utf-8"))
            totals["valid_pages" if stored.get("accepted") else "flagged_pages"] += 1
            translations = stored.get("response", {}).get("translations", [])
            if len(translations) != len(rows):
                continue
            for source, translated in zip(rows, translations):
                if translated.get("id") != source["id"]:
                    continue
                output = {
                    "id": source["id"],
                    "volume": volume,
                    "owner_page": source["owner_page"],
                    "pages": source["pages"],
                    "cross_page": source["cross_page"],
                    "my": normalize(source["text"]),
                    "en": translated["translation"],
                }
                volume_rows.append(output)
                all_output_rows.append(output)
        totals["sentences"] += len(volume_rows)
        write_text(
            out_dir / "translations" / f"vol{volume}.jsonl",
            "\n".join(json.dumps(row, ensure_ascii=False) for row in volume_rows) + "\n",
        )
    write_text(
        out_dir / "translations" / "all_volumes.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False) for row in all_output_rows) + "\n",
    )
    expected_sentences = sum(len(sentence_rows(volume)) for volume in (1, 2, 3))
    totals["expected_sentences"] = expected_sentences
    totals["complete"] = (
        totals["missing_pages"] == 0
        and totals["flagged_pages"] == 0
        and totals["sentences"] == expected_sentences
    )
    write_json(out_dir / "final_audit.json", totals)
    return totals


def load_submitted_name(out_dir: Path, label: str) -> str:
    data = json.loads((out_dir / "batch_jobs" / f"{label}_submitted.json").read_text(encoding="utf-8"))
    return data["name"]


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.smoke_manifest)
    excluded = smoke_pages(manifest_path)
    labels = [f"vol{volume}" for volume in (1, 2, 3)]

    if args.prepare_only:
        prompt_tokens = None
        if args.count_tokens:
            client = genai.Client(api_key=api_key())
            prompt_tokens = client.models.count_tokens(model=args.model, contents=TRANSLATION_PROMPT).total_tokens
        reused = materialize_smoke_pages(out_dir, manifest_path)
        counts = {}
        sentence_counts = {}
        for volume, label in zip((1, 2, 3), labels):
            jobs = jobs_for_volume(volume, excluded)
            save_plan(out_dir, label, jobs)
            counts[label] = len(jobs)
            sentence_counts[label] = sum(len(build_payload(job.volume, job.page)["sentences"]) for job in jobs)
        write_text(out_dir / "fixed_prompt.txt", TRANSLATION_PROMPT)
        readiness = {
            "model": args.model,
            "fixed_prompt_tokens": prompt_tokens,
            "explicit_cache_used": False,
            "cache_reason": "The fixed translation instruction is below the useful explicit-cache threshold.",
            "reused_smoke_test": reused,
            "batch_page_counts": counts,
            "batch_sentence_counts": sentence_counts,
            "total_batch_pages": sum(counts.values()),
            "total_batch_sentences": sum(sentence_counts.values()),
        }
        write_json(out_dir / "readiness.json", readiness)
        print(json.dumps(readiness, indent=2))
        return

    if not args.execute:
        raise RuntimeError("Refusing API work without --execute or --prepare-only")
    client = genai.Client(api_key=api_key())
    batches = []
    for label in labels:
        jobs = load_plan(out_dir, label)
        if args.process_existing:
            batch = client.batches.get(name=load_submitted_name(out_dir, label))
        else:
            batch = submit_batch(client, args.model, out_dir, label, jobs, args.max_output_tokens)
        batches.append((label, batch, jobs))
    if args.submit_only:
        print(json.dumps({label: batch.name for label, batch, _ in batches}, indent=2))
        return

    summaries = []
    for label, batch, jobs in batches:
        done = wait_for_batch(
            client,
            out_dir,
            label,
            batch.name,
            args.poll_seconds,
            args.max_wait_seconds,
        )
        summaries.append(process_batch(out_dir, label, done, jobs))
    write_json(
        out_dir / "run_summary.json",
        {
            "model": args.model,
            "explicit_cache_used": False,
            "reused_smoke_pages": len(excluded),
            "volumes": summaries,
        },
    )
    audit = finalize(out_dir)
    print(json.dumps({"complete": True, "audit": audit}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--smoke-manifest", default=str(DEFAULT_SMOKE_MANIFEST))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=86400)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--count-tokens", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--process-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
