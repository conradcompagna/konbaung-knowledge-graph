from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from google.genai import errors as genai_errors
from google.genai import types

from pipeline.embeddings import v3_eight_view_embeddings as batch_run


OUTPUT_ROOT = batch_run.OUTPUT_ROOT
MODEL = batch_run.MODEL
DIMENSIONS = batch_run.DIMENSIONS
STANDARD_PRICE_PER_MILLION_TOKENS_USD = 0.20
TOTAL_RUN_COST_CEILING_USD = 13.00

UNFINISHED_SHARDS = (
    "triple",
    "argument_context",
    "predicate_context",
    "triple_context",
)

MAX_GROUP_ITEMS = 64
TARGET_GROUP_TOKENS = 45_000
TARGET_TOKENS_PER_MINUTE = 800_000
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRY_DELAY_SECONDS = 120


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def input_path(shard: str) -> Path:
    return OUTPUT_ROOT / "inputs" / f"{shard}.jsonl"


def state_path(shard: str) -> Path:
    return OUTPUT_ROOT / "standard_progress" / f"{shard}.state.json"


def vector_path(shard: str) -> Path:
    return OUTPUT_ROOT / "vectors" / f"{shard}.npy"


def manifest_path(shard: str) -> Path:
    return OUTPUT_ROOT / "vectors" / f"{shard}.manifest.json"


def extract_text(row: dict[str, Any]) -> str:
    return row["request"]["content"]["parts"][0]["text"]


def load_rows(shard: str) -> list[dict[str, Any]]:
    rows = list(batch_run.iter_jsonl(input_path(shard)))
    if not rows:
        raise RuntimeError(f"No requests found for {shard}")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        key = row.get("key")
        if not key:
            raise RuntimeError(f"Missing key in {shard} row {index}")
        if key in seen:
            raise RuntimeError(f"Duplicate key in {shard}: {key}")
        seen.add(key)
        text = extract_text(row)
        if not text:
            raise RuntimeError(f"Empty text in {shard} row {index}: {key}")
    return rows


def existing_run_cost() -> tuple[int, float]:
    tokens = 0
    cost = 0.0
    vectors_dir = OUTPUT_ROOT / "vectors"
    for shard in batch_run.SHARD_ORDER:
        path = vectors_dir / f"{shard}.manifest.json"
        if path.exists():
            manifest = batch_run.read_json(path)
            if manifest.get("certified"):
                tokens += int(manifest.get("reportedInputTokens", 0))
                cost += float(
                    manifest.get(
                        "actualCostUsd",
                        manifest.get(
                            "actualStandardCostUsd",
                            manifest.get("actualBatchCostUsd", 0.0),
                        ),
                    )
                )
                continue
        progress = state_path(shard)
        if progress.exists():
            state = batch_run.read_json(progress)
            tokens += int(state.get("reportedInputTokens", 0))
            cost += float(
                state.get(
                    "standardCostUsd",
                    int(state.get("reportedInputTokens", 0))
                    / 1_000_000
                    * STANDARD_PRICE_PER_MILLION_TOKENS_USD,
                )
            )
    return tokens, cost


def validate_projected_cost() -> None:
    preflight = batch_run.read_json(OUTPUT_ROOT / "preflight_report.json")
    completed_shards = {
        shard
        for shard in batch_run.SHARD_ORDER
        if manifest_path(shard).exists()
        and batch_run.read_json(manifest_path(shard)).get("certified")
    }
    unfinished_estimated_tokens = sum(
        int(item["estimatedTokens"])
        for item in preflight["shards"]
        if item["shard"] not in completed_shards
    )
    _, completed_cost = existing_run_cost()
    projected = (
        completed_cost
        + unfinished_estimated_tokens / 1_000_000 * STANDARD_PRICE_PER_MILLION_TOKENS_USD
    )
    if projected > TOTAL_RUN_COST_CEILING_USD:
        raise RuntimeError(
            f"Projected mixed run cost ${projected:.4f} exceeds "
            f"${TOTAL_RUN_COST_CEILING_USD:.2f} ceiling"
        )


def estimated_tokens(shard: str, text: str) -> int:
    chars_per_token = batch_run.ESTIMATED_CHARS_PER_TOKEN[shard]
    return max(1, math.ceil(len(text) / chars_per_token * 1.10))


def choose_group(
    shard: str, rows: list[dict[str, Any]], start: int
) -> tuple[int, list[dict[str, Any]]]:
    total = 0
    end = start
    while end < len(rows) and (end - start) < MAX_GROUP_ITEMS:
        row_tokens = estimated_tokens(shard, extract_text(rows[end]))
        if end > start and total + row_tokens > TARGET_GROUP_TOKENS:
            break
        total += row_tokens
        end += 1
    return end, rows[start:end]


def contents_for(rows: list[dict[str, Any]]) -> list[types.Content]:
    return [types.Content(parts=[types.Part(text=extract_text(row))]) for row in rows]


def call_with_retry(label: str, fn):
    attempt = 0
    while True:
        try:
            return fn(), attempt
        except genai_errors.APIError as exc:
            code = int(getattr(exc, "code", 0) or 0)
            if code not in RETRYABLE_STATUS_CODES:
                raise
            attempt += 1
            delay = min(MAX_RETRY_DELAY_SECONDS, 5 * (2 ** min(attempt - 1, 5)))
            if code == 429:
                delay = max(delay, 60)
            print(
                json.dumps(
                    {
                        "event": "retry",
                        "label": label,
                        "statusCode": code,
                        "attempt": attempt,
                        "delaySeconds": delay,
                        "at": now_iso(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay)


class TokenWindow:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.events: deque[tuple[float, int]] = deque()
        self.total = 0

    def _purge(self, now: float) -> None:
        while self.events and now - self.events[0][0] >= 60:
            _, tokens = self.events.popleft()
            self.total -= tokens

    def wait_for_capacity(self, tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._purge(now)
            if not self.events or self.total + tokens <= self.limit:
                return
            wait_seconds = max(0.25, 60 - (now - self.events[0][0]) + 0.1)
            time.sleep(min(wait_seconds, 60))

    def record(self, tokens: int) -> None:
        now = time.monotonic()
        self._purge(now)
        self.events.append((now, tokens))
        self.total += tokens


def create_or_open_state(
    shard: str, rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], np.memmap]:
    progress_dir = OUTPUT_ROOT / "standard_progress"
    progress_dir.mkdir(exist_ok=True)
    path = state_path(shard)
    source_hash = batch_run.sha256_file(input_path(shard))
    vec_path = vector_path(shard)

    if path.exists():
        state = batch_run.read_json(path)
        if state["inputSha256"] != source_hash:
            raise RuntimeError(f"Input changed beneath standard run for {shard}")
        if int(state["expectedResponses"]) != len(rows):
            raise RuntimeError(f"Request count changed beneath standard run for {shard}")
        if not vec_path.exists():
            raise RuntimeError(f"Missing checkpoint vector file for {shard}")
        vectors = np.load(vec_path, mmap_mode="r+")
        if vectors.shape != (len(rows), DIMENSIONS):
            raise RuntimeError(f"Unexpected checkpoint shape for {shard}: {vectors.shape}")
        return state, vectors

    if vec_path.exists() and not manifest_path(shard).exists():
        raise RuntimeError(f"Unclaimed vector file already exists for {shard}: {vec_path}")
    vectors = np.lib.format.open_memmap(
        vec_path, mode="w+", dtype=np.float32, shape=(len(rows), DIMENSIONS)
    )
    state = {
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "mode": "standard",
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "shard": shard,
        "inputPath": str(input_path(shard)),
        "inputSha256": source_hash,
        "expectedResponses": len(rows),
        "nextRow": 0,
        "reportedInputTokens": 0,
        "standardCostUsd": 0.0,
        "apiCalls": 0,
        "retryCount": 0,
        "groups": [],
    }
    batch_run.atomic_write_json(path, state)
    return state, vectors


def group_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["key"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(extract_text(row).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def certify_shard(
    shard: str,
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    vectors: np.memmap,
) -> None:
    vectors.flush()
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all():
        raise RuntimeError(f"Non-finite vector norm in {shard}")
    if float(norms.min()) < 0.99 or float(norms.max()) > 1.01:
        raise RuntimeError(
            f"Unexpected vector norms in {shard}: {float(norms.min())}..{float(norms.max())}"
        )

    keys_path = OUTPUT_ROOT / "vectors" / f"{shard}.keys.jsonl"
    errors_path = OUTPUT_ROOT / "vectors" / f"{shard}.errors.jsonl"
    batch_run.write_jsonl(
        keys_path, ({"row": index, "key": row["key"]} for index, row in enumerate(rows))
    )
    batch_run.write_jsonl(errors_path, [])
    tokens = int(state["reportedInputTokens"])
    standard_cost = tokens / 1_000_000 * STANDARD_PRICE_PER_MILLION_TOKENS_USD
    manifest = {
        "collectedAt": now_iso(),
        "shard": shard,
        "billingMode": "standard",
        "expectedResponses": len(rows),
        "successfulResponses": len(rows),
        "failedResponses": 0,
        "reportedInputTokens": tokens,
        "actualStandardCostUsd": round(standard_cost, 6),
        "actualCostUsd": round(standard_cost, 6),
        "vectorPath": str(vector_path(shard)),
        "vectorSizeBytes": vector_path(shard).stat().st_size,
        "vectorSha256": batch_run.sha256_file(vector_path(shard)),
        "keysPath": str(keys_path),
        "keysSha256": batch_run.sha256_file(keys_path),
        "errorsPath": str(errors_path),
        "normMin": float(norms.min()),
        "normMax": float(norms.max()),
        "normMean": float(norms.mean()),
        "apiCalls": int(state["apiCalls"]),
        "retryCount": int(state["retryCount"]),
        "certified": True,
    }
    batch_run.atomic_write_json(manifest_path(shard), manifest)
    state.update({"updatedAt": now_iso(), "certified": True})
    batch_run.atomic_write_json(state_path(shard), state)
    print(json.dumps({"event": "shard_certified", **manifest}, sort_keys=True), flush=True)


def run_shard(shard: str, token_window: TokenWindow) -> None:
    existing_manifest = manifest_path(shard)
    if existing_manifest.exists():
        manifest = batch_run.read_json(existing_manifest)
        if manifest.get("certified"):
            print(
                json.dumps(
                    {"event": "shard_already_certified", "shard": shard},
                    sort_keys=True,
                ),
                flush=True,
            )
            return

    rows = load_rows(shard)
    state, vectors = create_or_open_state(shard, rows)
    client = batch_run.make_client()

    while int(state["nextRow"]) < len(rows):
        start = int(state["nextRow"])
        end, group_rows = choose_group(shard, rows, start)
        contents = contents_for(group_rows)
        count_response, count_retries = call_with_retry(
            f"{shard}:{start}-{end}:count_tokens",
            lambda: client.models.count_tokens(model=MODEL, contents=contents),
        )
        token_count = int(count_response.total_tokens)
        if token_count <= 0:
            raise RuntimeError(f"Non-positive token count for {shard}:{start}-{end}")

        _, current_cost = existing_run_cost()
        pending_group_cost = token_count / 1_000_000 * STANDARD_PRICE_PER_MILLION_TOKENS_USD
        if current_cost + pending_group_cost > TOTAL_RUN_COST_CEILING_USD:
            raise RuntimeError(f"Hard cost ceiling would be exceeded before {shard}:{start}-{end}")

        token_window.wait_for_capacity(token_count)
        response, embed_retries = call_with_retry(
            f"{shard}:{start}-{end}:embed_content",
            lambda: client.models.embed_content(
                model=MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS),
            ),
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(group_rows):
            raise RuntimeError(
                f"{shard}:{start}-{end} returned {len(embeddings)} embeddings "
                f"for {len(group_rows)} inputs"
            )
        for offset, embedding in enumerate(embeddings):
            values = embedding.values or []
            if len(values) != DIMENSIONS:
                raise RuntimeError(f"{shard}:{start + offset} returned {len(values)} dimensions")
            vector = np.asarray(values, dtype=np.float32)
            if not np.isfinite(vector).all():
                raise RuntimeError(f"Non-finite vector at {shard}:{start + offset}")
            vectors[start + offset] = vector
        vectors.flush()
        token_window.record(token_count)

        group_record = {
            "startRow": start,
            "endRowExclusive": end,
            "responses": len(embeddings),
            "inputTokens": token_count,
            "inputDigest": group_digest(group_rows),
            "countTokenRetries": count_retries,
            "embedRetries": embed_retries,
            "completedAt": now_iso(),
        }
        state["nextRow"] = end
        state["reportedInputTokens"] = int(state["reportedInputTokens"]) + token_count
        state["standardCostUsd"] = round(
            int(state["reportedInputTokens"]) / 1_000_000 * STANDARD_PRICE_PER_MILLION_TOKENS_USD,
            6,
        )
        state["apiCalls"] = int(state["apiCalls"]) + 1
        state["retryCount"] = int(state["retryCount"]) + count_retries + embed_retries
        state["groups"].append(group_record)
        state["updatedAt"] = now_iso()
        batch_run.atomic_write_json(state_path(shard), state)
        print(
            json.dumps(
                {
                    "event": "group_completed",
                    "shard": shard,
                    "nextRow": end,
                    "expectedResponses": len(rows),
                    "inputTokens": token_count,
                    "shardTokens": state["reportedInputTokens"],
                    "shardCostUsd": state["standardCostUsd"],
                    "at": state["updatedAt"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    certify_shard(shard, rows, state, vectors)


def finalize_mixed_run() -> None:
    manifests: list[dict[str, Any]] = []
    for shard in batch_run.SHARD_ORDER:
        path = manifest_path(shard)
        if not path.exists():
            raise RuntimeError(f"Missing manifest for {shard}")
        manifest = batch_run.read_json(path)
        if not manifest.get("certified"):
            raise RuntimeError(f"Uncertified shard {shard}")
        manifests.append(manifest)

    request_keys = {row["key"] for row in batch_run.iter_jsonl(OUTPUT_ROOT / "request_index.jsonl")}
    collected_keys: set[str] = set()
    for shard in batch_run.SHARD_ORDER:
        for row in batch_run.iter_jsonl(OUTPUT_ROOT / "vectors" / f"{shard}.keys.jsonl"):
            key = row["key"]
            if key in collected_keys:
                raise RuntimeError(f"Duplicate collected key {key}")
            collected_keys.add(key)
    missing = request_keys - collected_keys
    unexpected = collected_keys - request_keys
    if missing or unexpected:
        raise RuntimeError(
            f"Embedding key mismatch: {len(missing)} missing, {len(unexpected)} unexpected"
        )

    batch_tokens = sum(
        int(item.get("reportedInputTokens", 0))
        for item in manifests
        if item.get("billingMode", "batch") == "batch"
    )
    standard_tokens = sum(
        int(item.get("reportedInputTokens", 0))
        for item in manifests
        if item.get("billingMode") == "standard"
    )
    batch_cost = sum(float(item.get("actualBatchCostUsd", 0.0)) for item in manifests)
    standard_cost = sum(float(item.get("actualStandardCostUsd", 0.0)) for item in manifests)
    total_cost = batch_cost + standard_cost
    if total_cost > TOTAL_RUN_COST_CEILING_USD:
        raise RuntimeError(
            f"Final cost ${total_cost:.4f} exceeded ${TOTAL_RUN_COST_CEILING_USD:.2f} ceiling"
        )

    preflight = batch_run.read_json(OUTPUT_ROOT / "preflight_report.json")
    final = {
        "completedAt": now_iso(),
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "views": list(batch_run.VIEW_ORDER),
        "sourceTripleOccurrences": preflight["sourceTripleOccurrences"],
        "uniqueEmbeddingRequests": len(request_keys),
        "collectedEmbeddingKeys": len(collected_keys),
        "batchInputTokens": batch_tokens,
        "standardInputTokens": standard_tokens,
        "actualInputTokens": batch_tokens + standard_tokens,
        "batchCostUsd": round(batch_cost, 6),
        "standardCostUsd": round(standard_cost, 6),
        "actualTotalCostUsd": round(total_cost, 6),
        "allShardsCertified": True,
        "shards": manifests,
    }
    batch_run.atomic_write_json(OUTPUT_ROOT / "final_report.json", final)
    report_lines = [
        "# Konbaung V3 Eight-View Embedding Run",
        "",
        f"- Completed: {final['completedAt']}",
        f"- Model: `{MODEL}`",
        f"- Dimensions: {DIMENSIONS}",
        f"- Canonical triple occurrences: {final['sourceTripleOccurrences']:,}",
        f"- Unique embedding requests: {final['uniqueEmbeddingRequests']:,}",
        f"- Collected embedding keys: {final['collectedEmbeddingKeys']:,}",
        f"- Batch input tokens: {batch_tokens:,}",
        f"- Standard input tokens: {standard_tokens:,}",
        f"- Actual total input tokens: {final['actualInputTokens']:,}",
        f"- Batch cost: ${batch_cost:.6f}",
        f"- Standard cost: ${standard_cost:.6f}",
        f"- **Actual total cost: ${total_cost:.6f}**",
        f"- Certified shards: {len(manifests)}/{len(batch_run.SHARD_ORDER)}",
        "",
        "All source fields remain unchanged. Exact keys are preserved through "
        "`occurrences.jsonl`, `request_index.jsonl`, and each shard's key file.",
        "",
    ]
    report_path = OUTPUT_ROOT / "RUN_REPORT.md"
    temp = report_path.with_suffix(".md.tmp")
    temp.write_text("\n".join(report_lines), encoding="utf-8")
    os.replace(temp, report_path)
    print(json.dumps({"event": "run_finalized", **final}, sort_keys=True), flush=True)


def status() -> None:
    result: dict[str, Any] = {}
    for shard in batch_run.SHARD_ORDER:
        manifest = manifest_path(shard)
        state = state_path(shard)
        if manifest.exists():
            data = batch_run.read_json(manifest)
            result[shard] = {
                "certified": bool(data.get("certified")),
                "billingMode": data.get("billingMode", "batch"),
                "successfulResponses": data.get("successfulResponses"),
                "reportedInputTokens": data.get("reportedInputTokens"),
                "actualCostUsd": data.get(
                    "actualCostUsd",
                    data.get(
                        "actualStandardCostUsd",
                        data.get("actualBatchCostUsd"),
                    ),
                ),
            }
        elif state.exists():
            data = batch_run.read_json(state)
            result[shard] = {
                "certified": False,
                "billingMode": "standard",
                "nextRow": data.get("nextRow"),
                "expectedResponses": data.get("expectedResponses"),
                "reportedInputTokens": data.get("reportedInputTokens"),
                "standardCostUsd": data.get("standardCostUsd"),
                "updatedAt": data.get("updatedAt"),
            }
        else:
            result[shard] = {"certified": False, "state": "not_started"}
    print(json.dumps(result, indent=2))


def run() -> None:
    validate_projected_cost()
    token_window = TokenWindow(TARGET_TOKENS_PER_MINUTE)
    for shard in UNFINISHED_SHARDS:
        run_shard(shard, token_window)
    finalize_mixed_run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the Konbaung V3 eight-view run with standard embeddings."
    )
    parser.add_argument("command", choices=("run", "status", "finalize"))
    args = parser.parse_args()
    if args.command == "run":
        run()
    elif args.command == "status":
        status()
    else:
        finalize_mixed_run()


if __name__ == "__main__":
    main()
