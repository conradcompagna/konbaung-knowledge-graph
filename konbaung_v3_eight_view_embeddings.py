from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = (
    ROOT
    / "konbaung_reader_app"
    / "data"
    / "konbaung_historiography_v3_canonical_20260724"
)
SOURCE_INDEX = SOURCE_ROOT / "index.json"
SOURCE_SENTENCES = SOURCE_ROOT / "sentences"
OUTPUT_ROOT = ROOT / "konbaung_v3_eight_view_embeddings_20260724"

MODEL = "gemini-embedding-2"
DIMENSIONS = 768
BATCH_PRICE_PER_MILLION_TOKENS_USD = 0.10
BUDGET_CEILING_USD = 7.00
MODEL_TOKEN_LIMIT = 8192
SAFE_TOKEN_LIMIT = 7800
TOKEN_PREFLIGHT_CHAR_THRESHOLD = 9000
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

VIEW_ORDER = ("S", "P", "O", "SPO", "SBE", "PBE", "OBE", "SPOBE")
SHARD_ORDER = (
    "argument",
    "predicate",
    "triple",
    "argument_context",
    "predicate_context",
    "triple_context",
)
VIEW_TO_SHARD = {
    "S": "argument",
    "O": "argument",
    "P": "predicate",
    "SPO": "triple",
    "SBE": "argument_context",
    "OBE": "argument_context",
    "PBE": "predicate_context",
    "SPOBE": "triple_context",
}

# Conservative, tokenizer-sampled estimates from the frozen V3 corpus.
ESTIMATED_CHARS_PER_TOKEN = {
    "argument": 3.75,
    "predicate": 3.75,
    "triple": 3.25,
    "argument_context": 2.00,
    "predicate_context": 2.00,
    "triple_context": 2.00,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def embedding_key(text: str) -> str:
    return f"e_{sha256_text(text)[:32]}"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


def make_client() -> genai.Client:
    return genai.Client(api_key=api_key())


def source_files() -> list[Path]:
    return [SOURCE_INDEX, *(SOURCE_SENTENCES / f"vol{i}.json" for i in (1, 2, 3))]


def source_manifest() -> dict[str, Any]:
    files = []
    for path in source_files():
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": str(path),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "createdAt": now_iso(),
        "sourceId": "konbaung_historiography_v3_canonical_20260724",
        "sourceRoot": str(SOURCE_ROOT),
        "files": files,
        "sourceIndex": read_json(SOURCE_INDEX),
    }


def iter_sentences() -> Iterable[dict[str, Any]]:
    for volume_number in (1, 2, 3):
        path = SOURCE_SENTENCES / f"vol{volume_number}.json"
        payload = read_json(path)
        records = payload["sentences"]
        if isinstance(records, dict):
            values = records.values()
        else:
            values = records
        yield from values


def triple_record_hash(
    sid: str,
    ordinal: int,
    subject: str,
    predicate: str,
    object_: str,
    sentence_my: str,
    sentence_en: str,
) -> str:
    payload = {
        "sid": sid,
        "ordinal": ordinal,
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "sentenceMy": sentence_my,
        "sentenceEn": sentence_en,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def view_inputs(
    subject: str,
    predicate: str,
    object_: str,
    sentence_my: str,
    sentence_en: str,
) -> dict[str, tuple[str, str | None, str | None, str]]:
    argument_s = f"Historical argument: {subject}"
    argument_o = f"Historical argument: {object_}"
    predicate_text = f"Directed historical relation: {predicate}"
    triple_text = f"Subject: {subject}\nPredicate: {predicate}\nObject: {object_}"

    def contextual(header: str) -> str:
        return (
            f"{header}\n"
            f"Burmese context: {sentence_my}\n"
            f"English translation: {sentence_en}"
        )

    return {
        "S": (argument_s, None, None, argument_s),
        "P": (predicate_text, None, None, predicate_text),
        "O": (argument_o, None, None, argument_o),
        "SPO": (triple_text, None, None, triple_text),
        "SBE": (contextual(argument_s), sentence_my, sentence_en, argument_s),
        "PBE": (contextual(predicate_text), sentence_my, sentence_en, predicate_text),
        "OBE": (contextual(argument_o), sentence_my, sentence_en, argument_o),
        "SPOBE": (contextual(triple_text), sentence_my, sentence_en, triple_text),
    }


@dataclass
class RequestSpec:
    key: str
    text: str
    shard: str
    views: set[str] = field(default_factory=set)
    header: str | None = None
    sentence_my: str | None = None
    sentence_en: str | None = None
    token_count: int | None = None
    derived_from_key: str | None = None
    chunk_ordinal: int | None = None

    @property
    def is_contextual(self) -> bool:
        return self.sentence_my is not None and self.sentence_en is not None


def split_near_middle(text: str) -> tuple[str, str]:
    if len(text) < 2:
        return text, ""
    midpoint = len(text) // 2
    punctuation = {"\n", " ", ".", ";", ",", "။", "၊", ":", "—"}
    radius_limit = min(max(100, len(text) // 4), midpoint)
    for radius in range(radius_limit + 1):
        for position in (midpoint + radius, midpoint - radius):
            if 0 < position < len(text) and text[position] in punctuation:
                left = text[: position + 1].strip()
                right = text[position + 1 :].strip()
                if left and right:
                    return left, right
    return text[:midpoint].strip(), text[midpoint:].strip()


def contextual_text(header: str, sentence_my: str, sentence_en: str) -> str:
    return (
        f"{header}\n"
        f"Burmese context: {sentence_my}\n"
        f"English translation: {sentence_en}"
    )


def count_tokens_with_retry(client: genai.Client, text: str, attempts: int = 5) -> int:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = client.models.count_tokens(model=MODEL, contents=text)
            return int(result.total_tokens)
        except Exception as exc:  # API errors are retried with bounded backoff.
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"Token counting failed after {attempts} attempts: {last_error}")


def count_token_candidates(
    specs: dict[str, RequestSpec],
    max_workers: int = 8,
) -> dict[str, int]:
    candidates = {
        key: spec
        for key, spec in specs.items()
        if len(spec.text) > TOKEN_PREFLIGHT_CHAR_THRESHOLD
    }
    if not candidates:
        return {}
    client = make_client()
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(count_tokens_with_retry, client, spec.text): key
            for key, spec in candidates.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            counts[key] = future.result()
    return counts


def chunk_oversized_spec(
    client: genai.Client,
    spec: RequestSpec,
) -> list[RequestSpec]:
    if not spec.is_contextual or spec.header is None:
        raise ValueError(f"Cannot safely chunk non-contextual request {spec.key}")

    pending: list[tuple[str, str]] = [(spec.sentence_my or "", spec.sentence_en or "")]
    accepted: list[tuple[str, str, str, int]] = []

    while pending:
        sentence_my, sentence_en = pending.pop(0)
        text = contextual_text(spec.header, sentence_my, sentence_en)
        token_count = count_tokens_with_retry(client, text)
        if token_count <= SAFE_TOKEN_LIMIT:
            accepted.append((text, sentence_my, sentence_en, token_count))
            continue

        my_left, my_right = split_near_middle(sentence_my)
        en_left, en_right = split_near_middle(sentence_en)
        if not my_right and not en_right:
            raise RuntimeError(
                f"Unable to split oversized request {spec.key} with {token_count} tokens"
            )
        pending.insert(0, (my_right, en_right))
        pending.insert(0, (my_left, en_left))

    chunks = []
    for ordinal, (text, sentence_my, sentence_en, token_count) in enumerate(
        accepted, start=1
    ):
        key = embedding_key(text)
        chunks.append(
            RequestSpec(
                key=key,
                text=text,
                shard=spec.shard,
                views=set(spec.views),
                header=spec.header,
                sentence_my=sentence_my,
                sentence_en=sentence_en,
                token_count=token_count,
                derived_from_key=spec.key,
                chunk_ordinal=ordinal,
            )
        )
    return chunks


def request_row(spec: RequestSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "request": {
            "output_dimensionality": DIMENSIONS,
            "content": {"parts": [{"text": spec.text}]},
        },
    }


def prepare() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "inputs").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "batch_jobs").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "raw_results").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "vectors").mkdir(exist_ok=True)

    manifest_path = OUTPUT_ROOT / "source_manifest.json"
    current_manifest = source_manifest()
    if manifest_path.exists():
        existing = read_json(manifest_path)
        existing_hashes = [entry["sha256"] for entry in existing["files"]]
        current_hashes = [entry["sha256"] for entry in current_manifest["files"]]
        if existing_hashes != current_hashes:
            raise RuntimeError("Existing output directory belongs to different source hashes")
    atomic_write_json(manifest_path, current_manifest)

    specs: dict[str, RequestSpec] = {}
    occurrence_rows: list[dict[str, Any]] = []
    view_reference_counts: Counter[str] = Counter()
    source_triple_count = 0

    for sentence in iter_sentences():
        sid = str(sentence["sid"])
        sentence_my = str(sentence["my"]).strip()
        sentence_en = str(sentence["en"]).strip()
        for ordinal, triple in enumerate(sentence.get("triples", []), start=1):
            source_triple_count += 1
            subject = str(triple["subject"]).strip()
            predicate = str(triple["predicate"]).strip()
            object_ = str(triple["object"]).strip()
            record_hash = triple_record_hash(
                sid,
                ordinal,
                subject,
                predicate,
                object_,
                sentence_my,
                sentence_en,
            )
            occurrence_id = f"{sid}_t{ordinal:02d}_{record_hash[:12]}"
            key_map: dict[str, list[str]] = {}

            for view, (text, contextual_my, contextual_en, header) in view_inputs(
                subject,
                predicate,
                object_,
                sentence_my,
                sentence_en,
            ).items():
                key = embedding_key(text)
                shard = VIEW_TO_SHARD[view]
                if key not in specs:
                    specs[key] = RequestSpec(
                        key=key,
                        text=text,
                        shard=shard,
                        views={view},
                        header=header if contextual_my is not None else None,
                        sentence_my=contextual_my,
                        sentence_en=contextual_en,
                    )
                else:
                    existing = specs[key]
                    if existing.text != text or existing.shard != shard:
                        raise RuntimeError(f"Embedding-key collision for {key}")
                    existing.views.add(view)
                key_map[view] = [key]
                view_reference_counts[view] += 1

            occurrence_rows.append(
                {
                    "occurrenceId": occurrence_id,
                    "sid": sid,
                    "volumeId": sentence["volumeId"],
                    "ownerPage": sentence.get("ownerPage"),
                    "pageOccurrences": sentence.get("pages", []),
                    "tripleOrdinal": ordinal,
                    "subject": subject,
                    "predicate": predicate,
                    "object": object_,
                    "sentenceMy": sentence_my,
                    "sentenceEn": sentence_en,
                    "sourceRecordHash": record_hash,
                    "embeddingKeys": key_map,
                }
            )

    if source_triple_count != 27129:
        raise RuntimeError(f"Expected 27,129 triples, found {source_triple_count:,}")

    print(
        f"Built {len(occurrence_rows):,} occurrences and "
        f"{len(specs):,} unique embedding inputs"
    )
    token_counts = count_token_candidates(specs)
    for key, token_count in token_counts.items():
        specs[key].token_count = token_count

    oversized_keys = [
        key
        for key, token_count in token_counts.items()
        if token_count > SAFE_TOKEN_LIMIT
    ]
    replacements: dict[str, list[str]] = {}
    chunk_audit: list[dict[str, Any]] = []
    if oversized_keys:
        print(f"Chunking {len(oversized_keys):,} oversized embedding inputs")
        client = make_client()
        for original_key in sorted(oversized_keys):
            original = specs.pop(original_key)
            chunks = chunk_oversized_spec(client, original)
            chunk_keys = []
            for chunk in chunks:
                if chunk.key in specs:
                    existing = specs[chunk.key]
                    if existing.text != chunk.text or existing.shard != chunk.shard:
                        raise RuntimeError(f"Chunk-key collision for {chunk.key}")
                    existing.views.update(chunk.views)
                else:
                    specs[chunk.key] = chunk
                chunk_keys.append(chunk.key)
            replacements[original_key] = chunk_keys
            chunk_audit.append(
                {
                    "originalKey": original_key,
                    "originalChars": len(original.text),
                    "originalTokens": original.token_count,
                    "chunkKeys": chunk_keys,
                    "chunkTokenCounts": [chunk.token_count for chunk in chunks],
                }
            )

        for occurrence in occurrence_rows:
            for view, keys in occurrence["embeddingKeys"].items():
                if len(keys) == 1 and keys[0] in replacements:
                    occurrence["embeddingKeys"][view] = replacements[keys[0]]

    requests_by_shard: dict[str, list[RequestSpec]] = defaultdict(list)
    for spec in specs.values():
        requests_by_shard[spec.shard].append(spec)
    for shard in requests_by_shard:
        requests_by_shard[shard].sort(key=lambda item: item.key)

    input_files = []
    estimated_tokens = 0.0
    for shard in SHARD_ORDER:
        shard_specs = requests_by_shard.get(shard, [])
        path = OUTPUT_ROOT / "inputs" / f"{shard}.jsonl"
        write_jsonl(path, (request_row(spec) for spec in shard_specs))
        char_count = sum(len(spec.text) for spec in shard_specs)
        token_estimate = char_count / ESTIMATED_CHARS_PER_TOKEN[shard]
        estimated_tokens += token_estimate
        input_files.append(
            {
                "shard": shard,
                "path": str(path),
                "requests": len(shard_specs),
                "inputChars": char_count,
                "estimatedTokens": round(token_estimate),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    write_jsonl(
        OUTPUT_ROOT / "occurrences.jsonl",
        sorted(occurrence_rows, key=lambda row: (row["sid"], row["tripleOrdinal"])),
    )
    write_jsonl(
        OUTPUT_ROOT / "request_index.jsonl",
        (
            {
                "key": spec.key,
                "shard": spec.shard,
                "views": sorted(spec.views),
                "textSha256": sha256_text(spec.text),
                "chars": len(spec.text),
                "tokenCountIfPreflighted": spec.token_count,
                "derivedFromKey": spec.derived_from_key,
                "chunkOrdinal": spec.chunk_ordinal,
            }
            for spec in sorted(specs.values(), key=lambda item: item.key)
        ),
    )
    atomic_write_json(OUTPUT_ROOT / "chunk_audit.json", chunk_audit)

    estimated_cost = (
        estimated_tokens / 1_000_000 * BATCH_PRICE_PER_MILLION_TOKENS_USD
    )
    preflight = {
        "createdAt": now_iso(),
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "views": list(VIEW_ORDER),
        "shards": input_files,
        "sourceTripleOccurrences": source_triple_count,
        "viewReferenceCounts": dict(view_reference_counts),
        "uniqueEmbeddingRequests": len(specs),
        "oversizedOriginalRequests": len(chunk_audit),
        "estimatedTokens": round(estimated_tokens),
        "batchPricePerMillionTokensUsd": BATCH_PRICE_PER_MILLION_TOKENS_USD,
        "estimatedBatchCostUsd": round(estimated_cost, 4),
        "budgetCeilingUsd": BUDGET_CEILING_USD,
        "autoTruncate": "unsupported_by_file_batch_schema",
        "longInputSafeguard": (
            f"Exact token preflight above {TOKEN_PREFLIGHT_CHAR_THRESHOLD:,} "
            f"characters; split above {SAFE_TOKEN_LIMIT:,} tokens"
        ),
        "modelTokenLimit": MODEL_TOKEN_LIMIT,
        "safeTokenLimit": SAFE_TOKEN_LIMIT,
    }
    if estimated_cost > BUDGET_CEILING_USD:
        raise RuntimeError(
            f"Projected batch cost ${estimated_cost:.2f} exceeds "
            f"${BUDGET_CEILING_USD:.2f} ceiling"
        )
    atomic_write_json(OUTPUT_ROOT / "preflight_report.json", preflight)
    print(json.dumps(preflight, indent=2))


def smoke_job_record_path() -> Path:
    return OUTPUT_ROOT / "batch_jobs" / "smoke.json"


def smoke() -> None:
    preflight_path = OUTPUT_ROOT / "preflight_report.json"
    if not preflight_path.exists():
        raise RuntimeError("Run prepare first")
    record_path = smoke_job_record_path()
    client = make_client()

    if record_path.exists():
        record = read_json(record_path)
        job_name = record.get("jobName")
        if job_name:
            job = client.batches.get(name=job_name)
            state = job.state.name
            print(f"Existing smoke job {job_name}: {state}")
            if state == "JOB_STATE_SUCCEEDED":
                validate_smoke_result(client, job, record_path)
                return
            if state not in TERMINAL_STATES:
                job = wait_for_job(client, job_name, record_path, poll_seconds=5)
                validate_smoke_result(client, job, record_path)
                return
            raise RuntimeError(f"Existing smoke job ended in {state}: {job.error}")

    smoke_rows: list[dict[str, Any]] = []
    for shard in SHARD_ORDER:
        input_path = OUTPUT_ROOT / "inputs" / f"{shard}.jsonl"
        smoke_rows.extend(list(iter_jsonl(input_path))[:2])
    smoke_input = OUTPUT_ROOT / "inputs" / "smoke.jsonl"
    write_jsonl(smoke_input, smoke_rows)
    uploaded = client.files.upload(
        file=str(smoke_input),
        config=types.UploadFileConfig(
            display_name=f"konbaung-v3-8view-smoke-{sha256_file(smoke_input)[:10]}",
            mime_type="jsonl",
        ),
    )
    atomic_write_json(
        record_path,
        {
            "createdAt": now_iso(),
            "stage": "uploaded",
            "uploadedFileName": uploaded.name,
            "inputPath": str(smoke_input),
            "inputSha256": sha256_file(smoke_input),
            "requests": len(smoke_rows),
        },
    )
    job = client.batches.create_embeddings(
        model=MODEL,
        src=types.EmbeddingsBatchJobSource(file_name=uploaded.name),
        config={"display_name": f"konbaung-v3-8view-smoke-{int(time.time())}"},
    )
    record = read_json(record_path)
    record.update(
        {
            "stage": "submitted",
            "jobName": job.name,
            "state": job.state.name,
            "submittedAt": now_iso(),
        }
    )
    atomic_write_json(record_path, record)
    job = wait_for_job(client, job.name, record_path, poll_seconds=5)
    validate_smoke_result(client, job, record_path)


def validate_smoke_result(
    client: genai.Client,
    job: Any,
    record_path: Path,
) -> None:
    if not job.dest or not job.dest.file_name:
        raise RuntimeError("Succeeded smoke batch has no result file")
    content = client.files.download(file=job.dest.file_name)
    result_path = OUTPUT_ROOT / "raw_results" / "smoke.jsonl"
    temp = result_path.with_suffix(".jsonl.tmp")
    temp.write_bytes(content)
    os.replace(temp, result_path)

    successes = 0
    errors = []
    dimensions = set()
    keys = set()
    for line_number, row in enumerate(iter_jsonl(result_path), start=1):
        if row.get("error"):
            errors.append({"line": line_number, "error": row["error"]})
            continue
        values = extract_embedding_values(row)
        dimensions.add(len(values))
        key = row.get("key")
        if key in keys:
            errors.append({"line": line_number, "error": f"Duplicate key {key}"})
            continue
        keys.add(key)
        successes += 1

    record = read_json(record_path)
    expected = int(record["requests"])
    validated = (
        successes == expected and not errors and dimensions == {DIMENSIONS}
    )
    record.update(
        {
            "state": "JOB_STATE_SUCCEEDED",
            "checkedAt": now_iso(),
            "resultFileName": job.dest.file_name,
            "resultPath": str(result_path),
            "resultSha256": sha256_file(result_path),
            "successfulResponses": successes,
            "errors": errors,
            "dimensionsSeen": sorted(dimensions),
            "smokeValidated": validated,
        }
    )
    atomic_write_json(record_path, record)
    if not validated:
        raise RuntimeError(f"Smoke result validation failed: {record}")
    print(
        f"Smoke validated: {successes} responses, "
        f"{DIMENSIONS} dimensions, no errors"
    )


def wait_for_job(
    client: genai.Client,
    job_name: str,
    record_path: Path,
    poll_seconds: int = 30,
) -> Any:
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name
        record = read_json(record_path) if record_path.exists() else {}
        record.update({"jobName": job_name, "state": state, "checkedAt": now_iso()})
        atomic_write_json(record_path, record)
        print(f"{job_name}: {state}", flush=True)
        if state in TERMINAL_STATES:
            if state != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"Batch {job_name} ended in {state}: {job.error}")
            return job
        time.sleep(poll_seconds)


def ensure_smoke_accepted() -> None:
    path = smoke_job_record_path()
    if not path.exists():
        raise RuntimeError("Smoke batch has not been submitted")
    record = read_json(path)
    if record.get("state") == "JOB_STATE_SUCCEEDED" and record.get("smokeValidated"):
        return
    if (
        record.get("stage") == "submitted"
        and record.get("jobName")
        and record.get("state")
        in {"JOB_STATE_PENDING", "JOB_STATE_RUNNING"}
    ):
        # The service has accepted and begun processing the exact file-batch schema.
        # Result dimensions are still validated before any shard can be certified.
        return
    raise RuntimeError(f"Smoke batch has not been accepted: {record}")


def submit(shard: str) -> None:
    if shard not in SHARD_ORDER:
        raise ValueError(f"Unknown shard {shard}")
    ensure_smoke_accepted()
    preflight = read_json(OUTPUT_ROOT / "preflight_report.json")
    if preflight["estimatedBatchCostUsd"] > preflight["budgetCeilingUsd"]:
        raise RuntimeError("Preflight budget gate failed")

    record_path = OUTPUT_ROOT / "batch_jobs" / f"{shard}.json"
    client = make_client()
    existing_record: dict[str, Any] | None = None
    if record_path.exists():
        existing_record = read_json(record_path)
        if existing_record.get("jobName"):
            job = client.batches.get(name=existing_record["jobName"])
            existing_record.update({"state": job.state.name, "checkedAt": now_iso()})
            atomic_write_json(record_path, existing_record)
            print(f"{shard} already submitted as {job.name}: {job.state.name}")
            return

    input_path = OUTPUT_ROOT / "inputs" / f"{shard}.jsonl"
    input_sha = sha256_file(input_path)
    shard_info = next(item for item in preflight["shards"] if item["shard"] == shard)
    display = f"konbaung-v3-8view-{shard}-{input_sha[:10]}"
    if (
        existing_record
        and existing_record.get("stage") == "uploaded"
        and existing_record.get("uploadedFileName")
        and existing_record.get("inputSha256") == input_sha
    ):
        record = existing_record
        uploaded_file_name = record["uploadedFileName"]
        print(f"Reusing uploaded input {uploaded_file_name} for {shard}")
    else:
        uploaded = client.files.upload(
            file=str(input_path),
            config=types.UploadFileConfig(display_name=display, mime_type="jsonl"),
        )
        uploaded_file_name = uploaded.name
        record = {
            "createdAt": now_iso(),
            "shard": shard,
            "stage": "uploaded",
            "uploadedFileName": uploaded_file_name,
            "inputPath": str(input_path),
            "inputSha256": input_sha,
            "requests": shard_info["requests"],
            "estimatedTokens": shard_info["estimatedTokens"],
            "estimatedCostUsd": round(
                shard_info["estimatedTokens"]
                / 1_000_000
                * BATCH_PRICE_PER_MILLION_TOKENS_USD,
                4,
            ),
        }
        atomic_write_json(record_path, record)
    job = client.batches.create_embeddings(
        model=MODEL,
        src=types.EmbeddingsBatchJobSource(file_name=uploaded_file_name),
        config={"display_name": display},
    )
    record.update(
        {
            "stage": "submitted",
            "jobName": job.name,
            "state": job.state.name,
            "submittedAt": now_iso(),
        }
    )
    atomic_write_json(record_path, record)
    print(json.dumps(record, indent=2))


def update_status(shard: str | None = None) -> dict[str, dict[str, Any]]:
    client = make_client()
    shards = [shard] if shard else list(SHARD_ORDER)
    statuses = {}
    for item in shards:
        record_path = OUTPUT_ROOT / "batch_jobs" / f"{item}.json"
        if not record_path.exists():
            statuses[item] = {"state": "NOT_SUBMITTED"}
            continue
        record = read_json(record_path)
        if not record.get("jobName"):
            statuses[item] = record
            continue
        job = client.batches.get(name=record["jobName"])
        record.update({"state": job.state.name, "checkedAt": now_iso()})
        if job.dest and job.dest.file_name:
            record["resultFileName"] = job.dest.file_name
        if job.error:
            record["error"] = str(job.error)
        atomic_write_json(record_path, record)
        statuses[item] = record
    print(
        json.dumps(
            {
                key: {
                    "jobName": value.get("jobName"),
                    "state": value.get("state"),
                    "requests": value.get("requests"),
                    "resultFileName": value.get("resultFileName"),
                }
                for key, value in statuses.items()
            },
            indent=2,
        )
    )
    return statuses


def extract_embedding_values(response_row: dict[str, Any]) -> list[float]:
    response = response_row.get("response") or {}
    if response.get("embedding") and response["embedding"].get("values") is not None:
        return response["embedding"]["values"]
    embeddings = response.get("embeddings")
    if embeddings and embeddings[0].get("values") is not None:
        return embeddings[0]["values"]
    raise ValueError("Successful response did not contain embedding values")


def usage_tokens(response_row: dict[str, Any]) -> int:
    usage = (response_row.get("response") or {}).get("usageMetadata") or {}
    for key in ("totalTokenCount", "promptTokenCount", "inputTokenCount"):
        if usage.get(key) is not None:
            return int(usage[key])
    return 0


def collect(shard: str) -> None:
    if shard not in SHARD_ORDER:
        raise ValueError(f"Unknown shard {shard}")
    statuses = update_status(shard)
    record = statuses[shard]
    if record.get("state") != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"{shard} is not complete: {record.get('state')}")

    raw_path = OUTPUT_ROOT / "raw_results" / f"{shard}.jsonl"
    if not raw_path.exists():
        client = make_client()
        result_file_name = record.get("resultFileName")
        if not result_file_name:
            raise RuntimeError(f"No result file for {shard}")
        content = client.files.download(file=result_file_name)
        temp = raw_path.with_suffix(".jsonl.tmp")
        temp.write_bytes(content)
        os.replace(temp, raw_path)

    expected = int(record["requests"])
    vector_path = OUTPUT_ROOT / "vectors" / f"{shard}.npy"
    key_path = OUTPUT_ROOT / "vectors" / f"{shard}.keys.jsonl"
    error_path = OUTPUT_ROOT / "vectors" / f"{shard}.errors.jsonl"
    vectors = np.lib.format.open_memmap(
        vector_path, mode="w+", dtype=np.float32, shape=(expected, DIMENSIONS)
    )
    keys = []
    errors = []
    token_total = 0
    success_count = 0

    for line_number, row in enumerate(iter_jsonl(raw_path), start=1):
        key = row.get("key")
        if row.get("error"):
            errors.append(
                {"line": line_number, "key": key, "error": row["error"]}
            )
            continue
        try:
            values = extract_embedding_values(row)
        except Exception as exc:
            errors.append(
                {"line": line_number, "key": key, "error": str(exc)}
            )
            continue
        if len(values) != DIMENSIONS:
            errors.append(
                {
                    "line": line_number,
                    "key": key,
                    "error": f"Expected {DIMENSIONS} dimensions, got {len(values)}",
                }
            )
            continue
        vectors[success_count] = np.asarray(values, dtype=np.float32)
        keys.append({"row": success_count, "key": key})
        success_count += 1
        token_total += usage_tokens(row)

    vectors.flush()
    del vectors
    if success_count != expected:
        # Preserve the complete memmap shape for forensic inspection, but do not certify it.
        print(
            f"WARNING: {shard} expected {expected:,}, parsed {success_count:,} successes "
            f"and {len(errors):,} errors",
            file=sys.stderr,
        )
    write_jsonl(key_path, keys)
    write_jsonl(error_path, errors)

    if success_count:
        loaded = np.load(vector_path, mmap_mode="r")
        norms = np.linalg.norm(loaded[:success_count], axis=1)
        norm_min = float(norms.min())
        norm_max = float(norms.max())
        norm_mean = float(norms.mean())
    else:
        norm_min = norm_max = norm_mean = math.nan

    collection = {
        "collectedAt": now_iso(),
        "shard": shard,
        "expectedResponses": expected,
        "successfulResponses": success_count,
        "failedResponses": len(errors),
        "reportedInputTokens": token_total,
        "actualBatchCostUsd": round(
            token_total / 1_000_000 * BATCH_PRICE_PER_MILLION_TOKENS_USD, 6
        ),
        "rawResultPath": str(raw_path),
        "rawResultSizeBytes": raw_path.stat().st_size,
        "rawResultSha256": sha256_file(raw_path),
        "vectorPath": str(vector_path),
        "vectorSizeBytes": vector_path.stat().st_size,
        "keysPath": str(key_path),
        "errorsPath": str(error_path),
        "normMin": norm_min,
        "normMax": norm_max,
        "normMean": norm_mean,
        "certified": success_count == expected and not errors,
    }
    atomic_write_json(OUTPUT_ROOT / "vectors" / f"{shard}.manifest.json", collection)
    print(json.dumps(collection, indent=2))


def finalize() -> None:
    manifests = []
    for shard in SHARD_ORDER:
        path = OUTPUT_ROOT / "vectors" / f"{shard}.manifest.json"
        if not path.exists():
            raise RuntimeError(f"Missing collected manifest for {shard}")
        manifest = read_json(path)
        if not manifest.get("certified"):
            raise RuntimeError(f"Shard {shard} is not certified")
        manifests.append(manifest)

    preflight = read_json(OUTPUT_ROOT / "preflight_report.json")
    request_keys = {
        row["key"] for row in iter_jsonl(OUTPUT_ROOT / "request_index.jsonl")
    }
    collected_keys: set[str] = set()
    for shard in SHARD_ORDER:
        for row in iter_jsonl(OUTPUT_ROOT / "vectors" / f"{shard}.keys.jsonl"):
            key = row["key"]
            if key in collected_keys:
                raise RuntimeError(f"Duplicate collected embedding key {key}")
            collected_keys.add(key)
    missing = sorted(request_keys - collected_keys)
    unexpected = sorted(collected_keys - request_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"Embedding-key mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )

    actual_tokens = sum(item["reportedInputTokens"] for item in manifests)
    actual_cost = (
        actual_tokens / 1_000_000 * BATCH_PRICE_PER_MILLION_TOKENS_USD
    )
    final = {
        "completedAt": now_iso(),
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "views": list(VIEW_ORDER),
        "sourceTripleOccurrences": preflight["sourceTripleOccurrences"],
        "uniqueEmbeddingRequests": len(request_keys),
        "collectedEmbeddingKeys": len(collected_keys),
        "actualInputTokens": actual_tokens,
        "batchPricePerMillionTokensUsd": BATCH_PRICE_PER_MILLION_TOKENS_USD,
        "actualBatchCostUsd": round(actual_cost, 6),
        "allShardsCertified": True,
        "shards": manifests,
    }
    atomic_write_json(OUTPUT_ROOT / "final_report.json", final)
    report_lines = [
        "# Konbaung V3 Eight-View Embedding Run",
        "",
        f"- Completed: {final['completedAt']}",
        f"- Model: `{MODEL}`",
        f"- Dimensions: {DIMENSIONS}",
        f"- Canonical triple occurrences: {final['sourceTripleOccurrences']:,}",
        f"- Unique embedding requests: {final['uniqueEmbeddingRequests']:,}",
        f"- Actual input tokens: {actual_tokens:,}",
        f"- Actual batch cost: **${actual_cost:.4f}**",
        f"- Certified shards: {len(manifests)}/{len(SHARD_ORDER)}",
        "",
        "All source fields remain unchanged. Vectors are keyed through "
        "`occurrences.jsonl` and `request_index.jsonl`.",
        "",
    ]
    report_path = OUTPUT_ROOT / "RUN_REPORT.md"
    temp = report_path.with_suffix(".md.tmp")
    temp.write_text("\n".join(report_lines), encoding="utf-8")
    os.replace(temp, report_path)
    print(json.dumps(final, indent=2))


def orchestrate(poll_seconds: int, max_hours: float) -> None:
    started = time.monotonic()
    state_path = OUTPUT_ROOT / "orchestrator_state.json"
    client = make_client()

    while True:
        if (time.monotonic() - started) > max_hours * 3600:
            raise TimeoutError(f"Orchestrator exceeded {max_hours} hours")

        smoke_record = read_json(smoke_job_record_path())
        smoke_job = client.batches.get(name=smoke_record["jobName"])
        smoke_state = smoke_job.state.name
        smoke_record.update({"state": smoke_state, "checkedAt": now_iso()})
        atomic_write_json(smoke_job_record_path(), smoke_record)
        if smoke_state == "JOB_STATE_SUCCEEDED" and not smoke_record.get(
            "smokeValidated"
        ):
            validate_smoke_result(client, smoke_job, smoke_job_record_path())
            smoke_record = read_json(smoke_job_record_path())
        elif smoke_state in TERMINAL_STATES and smoke_state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Smoke batch ended in {smoke_state}: {smoke_job.error}")

        shard_states: dict[str, str] = {}
        active_jobs = 0 if smoke_state in TERMINAL_STATES else 1
        for shard in SHARD_ORDER:
            record_path = OUTPUT_ROOT / "batch_jobs" / f"{shard}.json"
            if not record_path.exists():
                shard_states[shard] = "NOT_SUBMITTED"
                continue
            record = read_json(record_path)
            if not record.get("jobName"):
                shard_states[shard] = "UPLOADED"
                continue
            job = client.batches.get(name=record["jobName"])
            state = job.state.name
            record.update({"state": state, "checkedAt": now_iso()})
            if job.dest and job.dest.file_name:
                record["resultFileName"] = job.dest.file_name
            if job.error:
                record["error"] = str(job.error)
            atomic_write_json(record_path, record)
            shard_states[shard] = state
            if state not in TERMINAL_STATES:
                active_jobs += 1
            elif state != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"{shard} batch ended in {state}: {job.error}")

        # Download and certify completed shards before moving on.
        for shard, state in list(shard_states.items()):
            manifest_path = OUTPUT_ROOT / "vectors" / f"{shard}.manifest.json"
            if state == "JOB_STATE_SUCCEEDED" and not manifest_path.exists():
                collect(shard)
                manifest = read_json(manifest_path)
                if not manifest.get("certified"):
                    raise RuntimeError(f"{shard} collection was not certified")

        # Keep at most three live jobs, matching the account's observed quota.
        for shard in SHARD_ORDER:
            if active_jobs >= 3:
                break
            record_path = OUTPUT_ROOT / "batch_jobs" / f"{shard}.json"
            record = read_json(record_path) if record_path.exists() else {}
            if record.get("jobName"):
                continue
            try:
                submit(shard)
            except genai_errors.ClientError as exc:
                if getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(
                    exc
                ):
                    print(
                        "Batch concurrency quota is full; retaining uploaded input "
                        f"for {shard}",
                        flush=True,
                    )
                    break
                raise
            active_jobs += 1
            shard_states[shard] = "SUBMITTED"

        certified = {
            shard: (
                (OUTPUT_ROOT / "vectors" / f"{shard}.manifest.json").exists()
                and read_json(
                    OUTPUT_ROOT / "vectors" / f"{shard}.manifest.json"
                ).get("certified")
            )
            for shard in SHARD_ORDER
        }
        snapshot = {
            "updatedAt": now_iso(),
            "smokeState": smoke_state,
            "smokeValidated": bool(smoke_record.get("smokeValidated")),
            "activeJobs": active_jobs,
            "shardStates": shard_states,
            "certified": certified,
        }
        atomic_write_json(state_path, snapshot)
        print(json.dumps(snapshot, sort_keys=True), flush=True)

        if smoke_record.get("smokeValidated") and all(certified.values()):
            if not (OUTPUT_ROOT / "final_report.json").exists():
                finalize()
            return
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, submit, and collect Konbaung V3 eight-view embeddings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("smoke")
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--shard", required=True, choices=SHARD_ORDER)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--shard", choices=SHARD_ORDER)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--shard", required=True, choices=SHARD_ORDER)
    orchestrate_parser = subparsers.add_parser("orchestrate")
    orchestrate_parser.add_argument("--poll-seconds", type=int, default=30)
    orchestrate_parser.add_argument("--max-hours", type=float, default=26)
    subparsers.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "smoke":
        smoke()
    elif args.command == "submit":
        submit(args.shard)
    elif args.command == "status":
        update_status(args.shard)
    elif args.command == "collect":
        collect(args.shard)
    elif args.command == "orchestrate":
        orchestrate(args.poll_seconds, args.max_hours)
    elif args.command == "finalize":
        finalize()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
