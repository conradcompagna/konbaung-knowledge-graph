#!/usr/bin/env python3
"""Repair only tag-like English predicate glosses using sentence-only Gemini batches."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "konbaung_relation_predicate_grounding_full_batch_20260717" / "pages"
TRIAL = ROOT / "konbaung_predicate_gloss_repair_trials" / "five_sentence_trial_01"
OUTPUT = ROOT / "konbaung_predicate_gloss_repair_full_batch_20260718"
RETRY = OUTPUT / "retry_01"
MODEL = "gemini-3.1-flash-lite"
SENTENCES_PER_REQUEST = 20
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

INSTRUCTION = """Translate each exact Burmese predicate span (`my`) as it functions in its sentence. Return a short, natural English verb or relational phrase. Translate the span itself, not the broader historical relation, subject, object, motive, consequence, or analytical interpretation. Never output a database tag, an ALL_CAPS label, or underscores. Use the sentence and translation only for context. Return every ID exactly once in order. Return only schema-valid JSON."""


class Gloss(BaseModel):
    id: str
    en: str


class Result(BaseModel):
    R: list[Gloss]


class BatchGloss(BaseModel):
    id: str
    en: str


class BatchResult(BaseModel):
    R: list[BatchGloss]


@dataclass(frozen=True)
class Job:
    key: str
    volume: int
    sentences: list[dict[str, Any]]
    expected: list[str]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def tag_like(value: str) -> bool:
    stripped = value.strip()
    return "_" in stripped or bool(re.fullmatch(r"[A-Z0-9 ]+", stripped))


def discover() -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    sentences: dict[str, dict[str, Any]] = {}
    source: dict[str, dict[str, str]] = {}
    for result_path in sorted(SOURCE.glob("vol*/page_*/result.json")):
        page_dir = result_path.parent
        stored = read_json(result_path)
        payload = read_json(page_dir / "payload.json")
        by_sid = {record["sid"]: record for record in payload["records"]}
        for grounding in stored.get("response", {}).get("R", []):
            if not tag_like(str(grounding.get("en", ""))):
                continue
            triple_id = grounding["id"]
            sid = re.sub(r"_t\d+$", "", triple_id)
            sentence = by_sid.get(sid)
            if sentence is None:
                raise ValueError(f"Missing sentence payload for {triple_id}")
            record = sentences.setdefault(
                sid,
                {
                    "sid": sid,
                    "volume": int(payload["page_id"][3]),
                    "my": sentence["my"],
                    "en": sentence["en"],
                    "I": [],
                },
            )
            if any(item["id"] == triple_id for item in record["I"]):
                raise ValueError(f"Duplicate grounding ID: {triple_id}")
            record["I"].append({"id": triple_id, "my": grounding["my"]})
            source[triple_id] = {"sid": sid, "my": grounding["my"], "old_en": grounding["en"]}
    ordered = sorted(sentences.values(), key=lambda item: item["sid"])
    for sentence in ordered:
        sentence["I"].sort(key=lambda item: item["id"])
    return ordered, source


def seeded() -> dict[str, str]:
    if not (TRIAL / "result.json").exists():
        return {}
    stored = read_json(TRIAL / "result.json")
    if stored.get("accepted") is not True:
        return {}
    return {item["id"]: item["en"] for item in stored["response"]["R"]}


def build_jobs() -> tuple[list[Job], dict[str, dict[str, str]], dict[str, str]]:
    sentences, source = discover()
    seed = seeded()
    remaining = []
    for sentence in sentences:
        items = [item for item in sentence["I"] if item["id"] not in seed]
        if items:
            remaining.append({**sentence, "I": items})
    jobs = []
    for volume in (1, 2, 3):
        subset = [sentence for sentence in remaining if sentence["volume"] == volume]
        for offset in range(0, len(subset), SENTENCES_PER_REQUEST):
            chunk = subset[offset : offset + SENTENCES_PER_REQUEST]
            expected = [item["id"] for sentence in chunk for item in sentence["I"]]
            jobs.append(
                Job(
                    f"vol{volume}-g{offset // SENTENCES_PER_REQUEST + 1:04d}",
                    volume,
                    chunk,
                    expected,
                )
            )
    return jobs, source, seed


def existing_corrections() -> dict[str, str]:
    path = OUTPUT / "predicate_gloss_corrections.jsonl"
    if not path.exists():
        return {}
    return {
        item["id"]: item["en"]
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def build_retry_jobs() -> tuple[list[Job], dict[str, dict[str, str]], dict[str, str]]:
    sentences, source = discover()
    corrections = existing_corrections()
    remaining = []
    for sentence in sentences:
        items = [item for item in sentence["I"] if item["id"] not in corrections]
        if items:
            remaining.append({**sentence, "I": items})
    jobs = []
    for volume in (1, 2, 3):
        subset = [sentence for sentence in remaining if sentence["volume"] == volume]
        for offset in range(0, len(subset), 5):
            chunk = subset[offset : offset + 5]
            expected = [item["id"] for sentence in chunk for item in sentence["I"]]
            jobs.append(Job(f"r1-vol{volume}-g{offset // 5 + 1:04d}", volume, chunk, expected))
    return jobs, source, corrections


def prompt(job: Job) -> str:
    lean = [
        {"sid": item["sid"], "my": item["my"], "en": item["en"], "I": item["I"]}
        for item in job.sentences
    ]
    return (
        INSTRUCTION
        + "\n<INPUT>"
        + json.dumps({"S": lean}, ensure_ascii=False, separators=(",", ":"))
        + "</INPUT>"
    )


def request(job: Job) -> types.InlinedRequest:
    return types.InlinedRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt(job))])],
        metadata={"key": job.key},
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max(1024, min(8192, len(job.expected) * 32)),
            thinking_config=types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
        ),
    )


def retry_request(job: Job) -> types.InlinedRequest:
    item_count = len(job.expected)
    return types.InlinedRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt(job))])],
        metadata={"key": job.key},
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchResult,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max(2048, min(8192, item_count * 96)),
            thinking_config=types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
        ),
    )


def validate(job: Job, result: Result) -> list[str]:
    errors = []
    ids = [item.id for item in result.R]
    if ids != job.expected:
        errors.append("ID/order mismatch")
    for item in result.R:
        if not item.en.strip() or "_" in item.en or re.fullmatch(r"[A-Z0-9 ]+", item.en.strip()):
            errors.append(f"{item.id}: still tag-like")
    return errors


def salvage_complete_items(raw: str, expected: set[str]) -> dict[str, str]:
    salvaged: dict[str, str] = {}
    pattern = re.compile(
        r'\{\s*"id"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"en"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    )
    for match in pattern.finditer(raw):
        try:
            item_id = json.loads('"' + match.group(1) + '"')
            gloss = json.loads('"' + match.group(2) + '"')
        except json.JSONDecodeError:
            continue
        if (
            item_id not in expected
            or not gloss.strip()
            or "_" in gloss
            or re.fullmatch(r"[A-Z0-9 ]+", gloss.strip())
        ):
            continue
        salvaged[item_id] = gloss
    return salvaged


def state_name(value: Any) -> str:
    return getattr(value, "value", str(value or ""))


def response_text(response: Any) -> str:
    if getattr(response, "text", None):
        return response.text
    return "".join(
        part.text
        for candidate in (getattr(response, "candidates", None) or [])
        for part in (getattr(getattr(candidate, "content", None), "parts", None) or [])
        if getattr(part, "text", None)
    )


def submit() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    jobs, source, seed = build_jobs()
    client = genai.Client(api_key=api_key())
    submitted = []
    for volume in (1, 2, 3):
        volume_jobs = [job for job in jobs if job.volume == volume]
        batch = client.batches.create(
            model=MODEL,
            src=[request(job) for job in volume_jobs],
            config=types.CreateBatchJobConfig(
                display_name=f"konbaung_gloss_repair_vol{volume}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            ),
        )
        plan = {
            "volume": volume,
            "batch_name": batch.name,
            "request_count": len(volume_jobs),
            "sentence_count": sum(len(job.sentences) for job in volume_jobs),
            "item_count": sum(len(job.expected) for job in volume_jobs),
            "keys": [job.key for job in volume_jobs],
        }
        write_json(OUTPUT / "batch_jobs" / f"vol{volume}.json", plan)
        submitted.append(plan)
    manifest = {
        "model": MODEL,
        "thinking_budget": 0,
        "sentences_per_request": SENTENCES_PER_REQUEST,
        "candidate_items": len(source),
        "candidate_sentences": len({item["sid"] for item in source.values()}),
        "seeded_items": len(seed),
        "submitted": submitted,
    }
    write_json(OUTPUT / "submission_manifest.json", manifest)
    return manifest


def collect() -> dict[str, Any]:
    jobs, source, seed = build_jobs()
    by_key = {job.key: job for job in jobs}
    client = genai.Client(api_key=api_key())
    corrections = dict(seed)
    states = []
    usage_totals = {"prompt_token_count": 0, "candidates_token_count": 0, "thoughts_token_count": 0}
    for volume in (1, 2, 3):
        plan = read_json(OUTPUT / "batch_jobs" / f"vol{volume}.json")
        batch = client.batches.get(name=plan["batch_name"])
        state = state_name(batch.state)
        states.append({"volume": volume, "batch_name": batch.name, "state": state})
        write_json(OUTPUT / "batch_jobs" / f"vol{volume}_latest.json", json_safe(batch))
        if state != "JOB_STATE_SUCCEEDED":
            continue
        rows = []
        for inlined in getattr(getattr(batch, "dest", None), "inlined_responses", None) or []:
            key = (getattr(inlined, "metadata", None) or {}).get("key", "")
            job = by_key.get(key)
            response = getattr(inlined, "response", None)
            raw = response_text(response)
            if job is None or getattr(inlined, "error", None) is not None:
                rows.append(
                    {"key": key, "error": json_safe(getattr(inlined, "error", "unknown key"))}
                )
                continue
            usage = json_safe(getattr(response, "usage_metadata", None)) or {}
            for field in usage_totals:
                usage_totals[field] += int(usage.get(field) or 0)
            try:
                parsed = Result.model_validate_json(raw)
                errors = validate(job, parsed)
            except Exception as error:
                salvaged = salvage_complete_items(raw, set(job.expected))
                corrections.update(salvaged)
                rows.append(
                    {
                        "key": key,
                        "error": str(error),
                        "raw": raw,
                        "salvaged_items": len(salvaged),
                        "usage": usage,
                    }
                )
                continue
            if not errors:
                corrections.update({item.id: item.en for item in parsed.R})
            rows.append(
                {
                    "key": key,
                    "accepted": not errors,
                    "errors": errors,
                    "usage": usage,
                    "response": parsed.model_dump(mode="json"),
                }
            )
        write_json(
            OUTPUT / "batch_results" / f"vol{volume}.json", {"state": state, "results": rows}
        )
    ordered_ids = sorted(source)
    (OUTPUT / "predicate_gloss_corrections.jsonl").write_text(
        "".join(
            json.dumps(
                {"id": item_id, "en": corrections[item_id]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for item_id in ordered_ids
            if item_id in corrections
        ),
        encoding="utf-8",
    )
    (OUTPUT / "audit_changes.jsonl").write_text(
        "".join(
            json.dumps(
                {"id": item_id, **source[item_id], "new_en": corrections[item_id]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for item_id in ordered_ids
            if item_id in corrections
        ),
        encoding="utf-8",
    )
    status = {
        "expected_items": len(source),
        "corrected_items": sum(item_id in corrections for item_id in source),
        "complete": all(item_id in corrections for item_id in source),
        "usage": usage_totals,
        "batch_states": states,
    }
    if status["complete"]:
        status["calculated_batch_cost_usd"] = round(
            usage_totals["prompt_token_count"] / 1_000_000 * 0.125
            + usage_totals["candidates_token_count"] / 1_000_000 * 0.75,
            8,
        )
    write_json(OUTPUT / "status.json", status)
    return status


def submit_retry() -> dict[str, Any]:
    if RETRY.exists():
        raise FileExistsError(RETRY)
    RETRY.mkdir(parents=True)
    jobs, source, corrections = build_retry_jobs()
    client = genai.Client(api_key=api_key())
    submitted = []
    for volume in (1, 2, 3):
        volume_jobs = [job for job in jobs if job.volume == volume]
        if not volume_jobs:
            continue
        batch = client.batches.create(
            model=MODEL,
            src=[retry_request(job) for job in volume_jobs],
            config=types.CreateBatchJobConfig(
                display_name=f"konbaung_gloss_retry_vol{volume}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            ),
        )
        plan = {
            "volume": volume,
            "batch_name": batch.name,
            "request_count": len(volume_jobs),
            "sentence_count": sum(len(job.sentences) for job in volume_jobs),
            "item_count": sum(len(job.expected) for job in volume_jobs),
            "keys": [job.key for job in volume_jobs],
        }
        write_json(RETRY / "batch_jobs" / f"vol{volume}.json", plan)
        submitted.append(plan)
    manifest = {
        "model": MODEL,
        "thinking_budget": 0,
        "already_corrected_items": len(corrections),
        "retry_items": sum(len(job.expected) for job in jobs),
        "retry_sentences": sum(len(job.sentences) for job in jobs),
        "submitted": submitted,
    }
    write_json(RETRY / "submission_manifest.json", manifest)
    return manifest


def collect_retry() -> dict[str, Any]:
    jobs, source, corrections = build_retry_jobs()
    by_key = {job.key: job for job in jobs}
    client = genai.Client(api_key=api_key())
    states = []
    retry_usage = {"prompt_token_count": 0, "candidates_token_count": 0, "thoughts_token_count": 0}
    for plan_path in sorted((RETRY / "batch_jobs").glob("vol?.json")):
        plan = read_json(plan_path)
        batch = client.batches.get(name=plan["batch_name"])
        state = state_name(batch.state)
        states.append({"volume": plan["volume"], "batch_name": batch.name, "state": state})
        write_json(plan_path.with_name(plan_path.stem + "_latest.json"), json_safe(batch))
        if state != "JOB_STATE_SUCCEEDED":
            continue
        rows = []
        for inlined in getattr(getattr(batch, "dest", None), "inlined_responses", None) or []:
            key = (getattr(inlined, "metadata", None) or {}).get("key", "")
            job = by_key.get(key)
            response = getattr(inlined, "response", None)
            raw = response_text(response)
            usage = json_safe(getattr(response, "usage_metadata", None)) or {}
            for field in retry_usage:
                retry_usage[field] += int(usage.get(field) or 0)
            if job is None or getattr(inlined, "error", None) is not None:
                rows.append(
                    {
                        "key": key,
                        "error": json_safe(getattr(inlined, "error", "unknown key")),
                        "usage": usage,
                    }
                )
                continue
            try:
                parsed = Result.model_validate_json(raw)
                errors = validate(job, parsed)
            except Exception as error:
                salvaged = salvage_complete_items(raw, set(job.expected))
                corrections.update(salvaged)
                rows.append(
                    {
                        "key": key,
                        "error": str(error),
                        "salvaged_items": len(salvaged),
                        "usage": usage,
                        "raw": raw,
                    }
                )
                continue
            if not errors:
                corrections.update({item.id: item.en for item in parsed.R})
            rows.append(
                {
                    "key": key,
                    "accepted": not errors,
                    "errors": errors,
                    "usage": usage,
                    "response": parsed.model_dump(mode="json"),
                }
            )
        write_json(
            RETRY / "batch_results" / f"vol{plan['volume']}.json", {"state": state, "results": rows}
        )
    ordered_ids = sorted(source)
    (OUTPUT / "predicate_gloss_corrections.jsonl").write_text(
        "".join(
            json.dumps(
                {"id": item_id, "en": corrections[item_id]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for item_id in ordered_ids
            if item_id in corrections
        ),
        encoding="utf-8",
    )
    (OUTPUT / "audit_changes.jsonl").write_text(
        "".join(
            json.dumps(
                {"id": item_id, **source[item_id], "new_en": corrections[item_id]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for item_id in ordered_ids
            if item_id in corrections
        ),
        encoding="utf-8",
    )
    first_usage = read_json(OUTPUT / "status.json").get("usage", {})
    total_usage = {
        field: int(first_usage.get(field) or 0) + retry_usage[field] for field in retry_usage
    }
    status = {
        "expected_items": len(source),
        "corrected_items": sum(item_id in corrections for item_id in source),
        "complete": all(item_id in corrections for item_id in source),
        "usage": total_usage,
        "retry_usage": retry_usage,
        "batch_states": states,
    }
    status["calculated_batch_cost_usd"] = round(
        total_usage["prompt_token_count"] / 1_000_000 * 0.125
        + total_usage["candidates_token_count"] / 1_000_000 * 0.75,
        8,
    )
    write_json(OUTPUT / "status.json", status)
    write_json(RETRY / "status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "collect", "retry-submit", "retry-collect"))
    args = parser.parse_args()
    actions = {
        "submit": submit,
        "collect": collect,
        "retry-submit": submit_retry,
        "retry-collect": collect_retry,
    }
    print(json.dumps(actions[args.command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
