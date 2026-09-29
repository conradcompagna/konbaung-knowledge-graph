#!/usr/bin/env python3
"""Submit and collect lean predicate-grounding jobs through the Gemini Batch API."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from konbaung_gemini_relation_predicate_grounding_trial import (
    INSTRUCTION,
    MODEL,
    Result,
    prepare_records,
    render_review,
    validate,
)
from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = (
    ROOT / "konbaung_translated_sentence_triples_full_batch_20260713_high_thinking" / "pages"
)
OUTPUT_ROOT = ROOT / "konbaung_relation_predicate_grounding_full_batch_20260717"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
SEED_TRIALS = {
    (1, 193): ROOT
    / "konbaung_relation_predicate_grounding_trials/vol1/page_0193/user_prompt_minimal_trial_02",
    (1, 392): ROOT
    / "konbaung_relation_predicate_grounding_trials/vol1/page_0392/user_prompt_minimal_trial_03",
    (2, 245): ROOT
    / "konbaung_relation_predicate_grounding_trials/vol2/page_0245/user_prompt_minimal_trial_04",
    (2, 336): ROOT
    / "konbaung_relation_predicate_grounding_trials/vol2/page_0336/user_prompt_minimal_trial_05",
    (2, 342): ROOT
    / "konbaung_relation_predicate_grounding_trials/vol2/page_0342/user_prompt_minimal_trial_06",
}


@dataclass(frozen=True)
class PageJob:
    volume: int
    page: int
    payload: dict[str, Any]
    records: list[dict]
    expected: list[str]

    @property
    def key(self) -> str:
        return f"vol{self.volume}-p{self.page:04d}"


# Gemini Batch accepts the API's reduced schema dialect, which does not include
# JSON Schema's additionalProperties keyword. Keep strict Result validation for
# collected responses, but use this equivalent lean model for request shaping.
class BatchGrounding(BaseModel):
    id: str
    my: str
    en: str
    src: Literal["sentence", "page", "inferred"]


class BatchResult(BaseModel):
    R: list[BatchGrounding]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="json"))
    return str(value)


def state_name(state: Any) -> str:
    return getattr(state, "value", str(state or ""))


def discover_jobs() -> list[PageJob]:
    jobs: list[PageJob] = []
    for result_path in sorted(SOURCE_ROOT.glob("vol*/page_*/result.json")):
        stored = json.loads(result_path.read_text(encoding="utf-8"))
        if stored.get("accepted") is not True:
            continue
        volume = int(result_path.parent.parent.name.removeprefix("vol"))
        page = int(result_path.parent.name.removeprefix("page_"))
        payload, records, expected = prepare_records(volume, page)
        jobs.append(PageJob(volume, page, payload, records, expected))
    return jobs


def prompt_for(job: PageJob) -> str:
    input_data = {"page_id": job.payload["page_id"], "records": job.records}
    return (
        INSTRUCTION
        + "\n\n<INPUT>\n"
        + json.dumps(input_data, ensure_ascii=False, separators=(",", ":"))
        + "\n</INPUT>"
    )


def build_request(job: PageJob) -> types.InlinedRequest:
    max_output_tokens = max(4096, min(12288, len(job.expected) * 112))
    return types.InlinedRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt_for(job))])],
        metadata={"key": job.key},
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
        ),
    )


def page_dir(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "pages" / f"vol{job.volume}" / f"page_{job.page:04d}"


def save_page(
    out_dir: Path,
    job: PageJob,
    result: Result,
    errors: list[str],
    raw: str,
    usage: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    target = page_dir(out_dir, job)
    target.mkdir(parents=True, exist_ok=True)
    input_data = {"page_id": job.payload["page_id"], "records": job.records}
    write_json(target / "payload.json", input_data)
    (target / "prompt_sent.txt").write_text(prompt_for(job), encoding="utf-8")
    (target / "raw_response.json").write_text(raw, encoding="utf-8")
    write_json(
        target / "result.json",
        {
            "accepted": not errors,
            "errors": errors,
            "usage": usage,
            "response": result.model_dump(mode="json"),
        },
    )
    (target / "review.md").write_text(
        render_review(job.payload, job.records, result, errors), encoding="utf-8"
    )
    write_json(target / "provenance.json", provenance)


def seed_trials(out_dir: Path, jobs_by_key: dict[str, PageJob]) -> set[str]:
    seeded: set[str] = set()
    manifest: list[dict[str, Any]] = []
    for (volume, page), source in SEED_TRIALS.items():
        key = f"vol{volume}-p{page:04d}"
        job = jobs_by_key[key]
        stored = json.loads((source / "result.json").read_text(encoding="utf-8"))
        converted = {
            "R": [
                {
                    "id": item["tid"],
                    "my": item["predicate_my"],
                    "en": item["predicate_en"],
                    "src": item["source"],
                }
                for item in stored["response"]["R"]
            ]
        }
        result = Result.model_validate(converted)
        errors = validate(job.records, job.expected, result)
        if errors:
            raise RuntimeError(f"Seed trial failed current validation: {key}: {errors}")
        raw = json.dumps(converted, ensure_ascii=False, separators=(",", ":"))
        save_page(
            out_dir,
            job,
            result,
            errors,
            raw,
            stored.get("usage", {}),
            {
                "source": "converted_final_prompt_trial",
                "source_dir": str(source),
                "original_thinking": stored.get("request", {}).get("thinking_level", "MINIMAL"),
                "production_schema": "lean_v1",
            },
        )
        seeded.add(key)
        manifest.append({"key": key, "source_dir": str(source)})
    write_json(out_dir / "seeded_trials.json", manifest)
    return seeded


def submit(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = discover_jobs()
    jobs_by_key = {job.key: job for job in jobs}
    seeded = seed_trials(out_dir, jobs_by_key)
    client = genai.Client(api_key=api_key())
    submitted: list[dict[str, Any]] = []
    for volume in (1, 2, 3):
        volume_jobs = [job for job in jobs if job.volume == volume and job.key not in seeded]
        label = f"vol{volume}"
        requests = [build_request(job) for job in volume_jobs]
        batch = client.batches.create(
            model=MODEL,
            src=requests,
            config=types.CreateBatchJobConfig(
                display_name=f"konbaung_predicate_grounding_{label}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            ),
        )
        plan = {
            "label": label,
            "batch_name": batch.name,
            "request_count": len(volume_jobs),
            "triple_count": sum(len(job.expected) for job in volume_jobs),
            "keys": [job.key for job in volume_jobs],
            "submitted": json_safe(batch),
        }
        write_json(out_dir / "batch_jobs" / f"{label}.json", plan)
        submitted.append(plan)
    manifest = {
        "model": MODEL,
        "thinking_budget": 0,
        "include_thoughts": False,
        "schema": {"R": [{"id": "...", "my": "...", "en": "...", "src": "sentence|page|inferred"}]},
        "source_pages": len(jobs),
        "source_triples": sum(len(job.expected) for job in jobs),
        "seeded_pages": len(seeded),
        "submitted_pages": sum(item["request_count"] for item in submitted),
        "submitted_batches": [
            {key: item[key] for key in ("label", "batch_name", "request_count", "triple_count")}
            for item in submitted
        ],
    }
    write_json(out_dir / "submission_manifest.json", manifest)
    return manifest


def response_text(response: Any) -> str:
    if getattr(response, "text", None):
        return response.text
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts)


def collect(out_dir: Path) -> dict[str, Any]:
    jobs = discover_jobs()
    jobs_by_key = {job.key: job for job in jobs}
    client = genai.Client(api_key=api_key())
    states: list[dict[str, Any]] = []
    for volume in (1, 2, 3):
        label = f"vol{volume}"
        plan_path = out_dir / "batch_jobs" / f"{label}.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        batch = client.batches.get(name=plan["batch_name"])
        state = state_name(batch.state)
        states.append({"label": label, "batch_name": batch.name, "state": state})
        write_json(out_dir / "batch_jobs" / f"{label}_latest.json", json_safe(batch))
        if state != "JOB_STATE_SUCCEEDED":
            continue
        responses = getattr(getattr(batch, "dest", None), "inlined_responses", None) or []
        rows: list[dict[str, Any]] = []
        for inlined in responses:
            metadata = getattr(inlined, "metadata", None) or {}
            key = metadata.get("key", "")
            job = jobs_by_key.get(key)
            if job is None or getattr(inlined, "error", None) is not None:
                rows.append(
                    {"key": key, "error": json_safe(getattr(inlined, "error", "unknown key"))}
                )
                continue
            response = getattr(inlined, "response", None)
            raw = response_text(response)
            try:
                result = Result.model_validate_json(raw)
                errors = validate(job.records, job.expected, result)
            except Exception as exc:
                rows.append({"key": key, "error": str(exc), "raw": raw})
                continue
            usage = json_safe(getattr(response, "usage_metadata", None))
            save_page(
                out_dir,
                job,
                result,
                errors,
                raw,
                usage,
                {
                    "source": "gemini_batch",
                    "batch_name": batch.name,
                    "model": MODEL,
                    "thinking_budget": 0,
                },
            )
            rows.append({"key": key, "accepted": not errors, "errors": errors, "usage": usage})
        write_json(out_dir / "batch_results" / f"{label}.json", {"state": state, "results": rows})
    summary = finalize(out_dir, jobs)
    summary["batch_states"] = states
    write_json(out_dir / "status.json", summary)
    return summary


def finalize(out_dir: Path, jobs: list[PageJob]) -> dict[str, Any]:
    accepted_pages = 0
    accepted_groundings = 0
    rows: list[dict[str, Any]] = []
    for job in jobs:
        result_path = page_dir(out_dir, job) / "result.json"
        if not result_path.exists():
            continue
        stored = json.loads(result_path.read_text(encoding="utf-8"))
        if stored.get("accepted") is not True:
            continue
        accepted_pages += 1
        original = {triple["id"]: triple for record in job.records for triple in record["T"]}
        for item in stored["response"]["R"]:
            triple = original[item["id"]]
            rows.append(
                {
                    "page_id": job.payload["page_id"],
                    "id": item["id"],
                    "s": triple["s"],
                    "p": triple["p"],
                    "o": triple["o"],
                    "my": item["my"],
                    "en": item["en"],
                    "src": item["src"],
                }
            )
            accepted_groundings += 1
    aggregate = out_dir / "predicate_groundings.jsonl"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    aggregate.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "expected_pages": len(jobs),
        "expected_groundings": sum(len(job.expected) for job in jobs),
        "accepted_pages": accepted_pages,
        "accepted_groundings": accepted_groundings,
        "complete": accepted_pages == len(jobs)
        and accepted_groundings == sum(len(job.expected) for job in jobs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "collect"))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    result = submit(args.out_dir) if args.command == "submit" else collect(args.out_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
