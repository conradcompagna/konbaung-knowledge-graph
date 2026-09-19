#!/usr/bin/env python3
"""Recover all predicate groundings for the exceptional 104-triple vol1 page 208."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "konbaung_relation_predicate_grounding_full_batch_20260717" / "pages" / "vol1" / "page_0208"
OUTPUT = ROOT / "konbaung_relation_predicate_grounding_page0208_repair_20260718"
MODEL = "gemini-3.1-flash-lite"
ITEMS_PER_REQUEST = 15

INSTRUCTION = """For every supplied existing triple, copy the shortest exact contiguous Burmese span in its assigned sentence that expresses or textually anchors the relation, then give a short natural English gloss of that Burmese span itself. Copy Burmese verbatim. Do not invent or rewrite it. The English must translate the Burmese span, not repeat the analytical relation label: never output ALL_CAPS tags or underscores. Return every ID once in order. Return only schema-valid JSON."""


class Grounding(BaseModel):
    id: str
    my: str
    en: str


class Result(BaseModel):
    R: list[Grounding]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jobs() -> list[dict[str, Any]]:
    payload = read_json(SOURCE / "payload.json")
    output = []
    sequence = 0
    for sentence in payload["records"]:
        for offset in range(0, len(sentence["T"]), ITEMS_PER_REQUEST):
            sequence += 1
            triples = sentence["T"][offset : offset + ITEMS_PER_REQUEST]
            output.append(
                {
                    "key": f"p0208-g{sequence:03d}",
                    "sid": sentence["sid"],
                    "my": sentence["my"],
                    "en": sentence["en"],
                    "T": triples,
                    "expected": [triple["id"] for triple in triples],
                }
            )
    return output


def prompt(job: dict[str, Any]) -> str:
    triples = [{"id": item["id"], "s": item["s"], "p": item["p"], "o": item["o"]} for item in job["T"]]
    payload = {"sid": job["sid"], "my": job["my"], "en": job["en"], "T": triples}
    return INSTRUCTION + "\n<INPUT>" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "</INPUT>"


def request(job: dict[str, Any]) -> types.InlinedRequest:
    return types.InlinedRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt(job))])],
        metadata={"key": job["key"]},
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Result,
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
        ),
    )


def response_text(response: Any) -> str:
    if getattr(response, "text", None):
        return response.text
    return "".join(
        part.text
        for candidate in (getattr(response, "candidates", None) or [])
        for part in (getattr(getattr(candidate, "content", None), "parts", None) or [])
        if getattr(part, "text", None)
    )


def state_name(value: Any) -> str:
    return getattr(value, "value", str(value or ""))


def submit() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    work = jobs()
    client = genai.Client(api_key=api_key())
    batch = client.batches.create(
        model=MODEL,
        src=[request(job) for job in work],
        config=types.CreateBatchJobConfig(display_name=f"konbaung_p0208_repair_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"),
    )
    manifest = {
        "batch_name": batch.name,
        "model": MODEL,
        "thinking_budget": 0,
        "request_count": len(work),
        "item_count": sum(len(job["expected"]) for job in work),
        "keys": [job["key"] for job in work],
    }
    write_json(OUTPUT / "submission_manifest.json", manifest)
    return manifest


def collect() -> dict[str, Any]:
    manifest = read_json(OUTPUT / "submission_manifest.json")
    work = {job["key"]: job for job in jobs()}
    client = genai.Client(api_key=api_key())
    batch = client.batches.get(name=manifest["batch_name"])
    state = state_name(batch.state)
    if state != "JOB_STATE_SUCCEEDED":
        status = {"state": state, "complete": False}
        write_json(OUTPUT / "status.json", status)
        return status
    returned: dict[str, Grounding] = {}
    errors = []
    usage = {"prompt_token_count": 0, "candidates_token_count": 0, "thoughts_token_count": 0}
    for inlined in getattr(getattr(batch, "dest", None), "inlined_responses", None) or []:
        key = (getattr(inlined, "metadata", None) or {}).get("key", "")
        job = work.get(key)
        response = getattr(inlined, "response", None)
        response_usage = getattr(response, "usage_metadata", None)
        if response_usage:
            dumped = response_usage.model_dump(mode="json")
            for field in usage:
                usage[field] += int(dumped.get(field) or 0)
        if job is None or getattr(inlined, "error", None) is not None:
            errors.append(f"{key}: batch error")
            continue
        try:
            parsed = Result.model_validate_json(response_text(response))
        except Exception as error:
            errors.append(f"{key}: {error}")
            continue
        if [item.id for item in parsed.R] != job["expected"]:
            errors.append(f"{key}: ID/order mismatch")
            continue
        for item in parsed.R:
            if item.my not in job["my"]:
                errors.append(f"{item.id}: span not exact in assigned sentence")
            if "_" in item.en or re.fullmatch(r"[A-Z0-9 ]+", item.en.strip()):
                errors.append(f"{item.id}: tag-like gloss")
            returned[item.id] = item
    expected = [item for job in work.values() for item in job["expected"]]
    rows = [returned[item].model_dump(mode="json") for item in expected if item in returned]
    (OUTPUT / "predicate_groundings.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    status = {
        "state": state,
        "expected_items": len(expected),
        "returned_items": len(rows),
        "errors": errors,
        "complete": len(rows) == len(expected) and not errors,
        "usage": usage,
        "calculated_batch_cost_usd": round(usage["prompt_token_count"] / 1_000_000 * 0.125 + usage["candidates_token_count"] / 1_000_000 * 0.75, 8),
    }
    write_json(OUTPUT / "status.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("submit", "collect"))
    args = parser.parse_args()
    print(json.dumps(submit() if args.command == "submit" else collect(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
