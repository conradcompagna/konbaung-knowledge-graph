#!/usr/bin/env python3
"""Run the accepted evidence-block translation setup over all valid pages."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from pipeline.extraction.cite_sources_annotator import MODEL
from pipeline.translation.evidence_translation import (
    TRANSLATION_PROMPT,
    build_payload,
    validate,
    write_review,
)
from pipeline.extraction.summary_claim_completion_annotator import READER_DATA, api_key
from pipeline.extraction.structured_open_coding_annotator import usage_dict


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "konbaung_evidence_translation_full_batch_20260713"
DEFAULT_SMOKE_MANIFEST = ROOT / "konbaung_evidence_translation_smoke_test_manifest.json"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
}


class TranslationItem(BaseModel):
    id: str
    translation: str


class TranslationResult(BaseModel):
    translations: list[TranslationItem] = Field(
        description="One ordered translation per evidence item."
    )


@dataclass(frozen=True)
class TranslationJob:
    volume: int
    page: int

    @property
    def key(self) -> str:
        return f"vol{self.volume}-p{self.page:04d}"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def state_name(state: Any) -> str:
    return getattr(state, "value", str(state or ""))


def completed_smoke_pages(path: Path) -> set[tuple[int, int]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(row["volume"]), int(row["page"]))
        for row in manifest.get("pages", [])
        if row.get("status") == "complete"
    }


def jobs_for_volume(volume: int, smoke_manifest: Path) -> list[TranslationJob]:
    excluded = completed_smoke_pages(smoke_manifest)
    jobs: list[TranslationJob] = []
    for path in sorted((READER_DATA / f"vol{volume}").glob("*.json")):
        page = int(path.stem)
        if (volume, page) in excluded:
            continue
        payload, _ = build_payload(volume, page)
        if payload["evidence"]:
            jobs.append(TranslationJob(volume, page))
    return jobs


def prompt_and_payload(job: TranslationJob) -> tuple[str, dict[str, Any]]:
    payload, _ = build_payload(job.volume, job.page)
    prompt = (
        TRANSLATION_PROMPT
        + "\n\n<INPUT>\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n</INPUT>\n"
    )
    return prompt, payload


def build_request(job: TranslationJob, max_output_tokens: int) -> types.InlinedRequest:
    prompt, payload = prompt_and_payload(job)
    return types.InlinedRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        metadata={"key": job.key, "volume": str(job.volume), "page": f"{job.page:04d}"},
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TranslationResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )


def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "\n".join(parts)


def normalize_batch_result(value: Any) -> dict[str, Any]:
    """Normalize the Batch API's flattened array/renamed value field."""
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {
            "translations": [
                {
                    "id": item.get("id"),
                    "translation": item.get("translation", item.get("my", "")),
                }
                for item in value
                if isinstance(item, dict)
            ]
        }
    raise TypeError(f"Unexpected batch response type: {type(value).__name__}")


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
            "pages": [{"key": job.key, "volume": job.volume, "page": job.page} for job in jobs],
        },
    )


def load_plan(out_dir: Path, label: str) -> list[TranslationJob]:
    data = json.loads((out_dir / "batch_plans" / f"{label}.json").read_text(encoding="utf-8"))
    return [TranslationJob(int(row["volume"]), int(row["page"])) for row in data["pages"]]


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
            display_name=f"konbaung_translation_{label}_{utc_stamp()}"
        ),
    )
    write_json(out_dir / "batch_jobs" / f"{label}_submitted.json", json_safe(batch))
    return batch


def wait_for_batch(
    client: genai.Client,
    out_dir: Path,
    label: str,
    batch_name: str,
    poll_seconds: int,
    max_wait_seconds: int,
) -> Any:
    started = time.time()
    while True:
        batch = client.batches.get(name=batch_name)
        write_json(out_dir / "batch_jobs" / f"{label}_latest.json", json_safe(batch))
        state = state_name(batch.state)
        elapsed = int(time.time() - started)
        print(json.dumps({"label": label, "state": state, "elapsed_seconds": elapsed}), flush=True)
        if state in TERMINAL_STATES:
            return batch
        if elapsed >= max_wait_seconds:
            raise TimeoutError(f"{label} did not finish within {max_wait_seconds} seconds")
        time.sleep(poll_seconds)


def process_batch(
    out_dir: Path, label: str, batch: Any, jobs: list[TranslationJob]
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
        "evidence_blocks": 0,
        "prompt_token_count": 0,
        "cached_content_token_count": 0,
        "candidates_token_count": 0,
        "total_token_count": 0,
    }
    rows: list[dict[str, Any]] = []
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
        if (
            getattr(inlined, "error", None) is not None
            or getattr(inlined, "response", None) is None
        ):
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
            write_json(
                page_dir / "result.json", {"accepted": False, "errors": [str(exc)], "usage": usage}
            )
            rows.append({"key": key, "error": f"schema parse failed: {exc}"})
            continue

        errors = validate_batch_result(payload, result)
        totals["evidence_blocks"] += len(payload["evidence"])
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
                "evidence_blocks": len(payload["evidence"]),
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


def load_submitted_name(out_dir: Path, label: str) -> str:
    data = json.loads(
        (out_dir / "batch_jobs" / f"{label}_submitted.json").read_text(encoding="utf-8")
    )
    return data["name"]


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_manifest = Path(args.smoke_manifest)
    labels = [f"vol{volume}" for volume in (1, 2, 3)]

    if args.prepare_only:
        prompt_tokens = None
        if args.count_tokens:
            client = genai.Client(api_key=api_key())
            prompt_tokens = client.models.count_tokens(
                model=args.model, contents=TRANSLATION_PROMPT
            ).total_tokens
        counts = {}
        for volume, label in zip((1, 2, 3), labels):
            jobs = jobs_for_volume(volume, smoke_manifest)
            save_plan(out_dir, label, jobs)
            counts[label] = len(jobs)
        write_text(out_dir / "fixed_prompt.txt", TRANSLATION_PROMPT)
        readiness = {
            "model": args.model,
            "prompt_tokens": prompt_tokens,
            "explicit_cache_used": False,
            "reason": "The accepted fixed prompt is too short to benefit from explicit caching.",
            "excluded_smoke_pages": len(completed_smoke_pages(smoke_manifest)),
            "page_counts": counts,
            "total_pages": sum(counts.values()),
        }
        write_json(out_dir / "readiness.json", readiness)
        print(json.dumps(readiness, indent=2))
        return

    if not args.execute:
        raise RuntimeError("Refusing API work without --execute or --prepare-only")

    client = genai.Client(api_key=api_key())
    batches: list[tuple[str, Any, list[TranslationJob]]] = []
    for label in labels:
        jobs = load_plan(out_dir, label)
        if args.process_existing:
            batch_name = load_submitted_name(out_dir, label)
            batch = client.batches.get(name=batch_name)
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
            "excluded_smoke_pages": len(completed_smoke_pages(smoke_manifest)),
            "volumes": summaries,
        },
    )
    print(
        json.dumps({"complete": True, "volumes": [item["totals"] for item in summaries]}, indent=2)
    )


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
