#!/usr/bin/env python3
"""Run the final Flash-Lite axial-coding prompt over the canonical corpus."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
MODEL = "gemini-3.1-flash-lite"
SOURCE_PAGES = (
    ROOT
    / "konbaung_historiography_ungrounded_v3_full_batch_20260723"
    / "final"
    / "all_pages.jsonl"
)
CANONICAL_ROOT = (
    ROOT
    / "konbaung_reader_app"
    / "data"
    / "konbaung_historiography_v3_canonical_20260724"
)
PROMPT_PACKAGE = ROOT / "konbaung_flashlite_axial_prompt_package_v2.zip"
DEFAULT_OUT_DIR = ROOT / "konbaung_flashlite_axial_full_corpus_20260729"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
}
EXPECTED_CANONICAL_SENTENCES = 11222
EXPECTED_CANONICAL_TRIPLES = 27129
EXPECTED_CANONICAL_PAGES = 1210

PROMPT_ENTRY = "konbaung_flashlite_axial_coding_prompt_v2.txt"
TEMPLATE_ENTRY = "konbaung_flashlite_axial_page_template_v2.txt"
SCHEMA_ENTRY = "konbaung_flashlite_axial_response_schema_v2.json"

COMPLETED_PAGE_DIRS = {
    "vol2-p0236": ROOT
    / "konbaung_flashlite_axial_provisional_random_five_page_trial_20260728"
    / "vol2-p0236",
    "vol2-p0067": ROOT
    / "konbaung_flashlite_axial_provisional_random_five_page_trial_20260728"
    / "vol2-p0067",
    "vol2-p0190": ROOT
    / "konbaung_flashlite_axial_provisional_random_five_page_trial_20260728"
    / "vol2-p0190",
    "vol3-p0375": ROOT
    / "konbaung_flashlite_axial_provisional_random_five_page_trial_20260728"
    / "vol3-p0375",
    "vol1-p0175": ROOT
    / "konbaung_flashlite_axial_provisional_random_five_page_trial_20260728"
    / "vol1-p0175",
    "vol2-p0245": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol2-p0245",
    "vol1-p0210": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol1-p0210",
    "vol3-p0509": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol3-p0509",
    "vol3-p0377": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol3-p0377",
    "vol3-p0263": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol3-p0263",
    "vol2-p0102": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol2-p0102",
    "vol3-p0276": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol3-p0276",
    "vol2-p0222": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol2-p0222",
    "vol1-p0281": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol1-p0281",
    "vol1-p0142": ROOT
    / "konbaung_flashlite_axial_provisional_random_ten_page_trial_20260729"
    / "vol1-p0142",
}
REQUEUED_V1_PAGES = {
    "vol3-p0332",
    "vol3-p0267",
    "vol2-p0095",
    "vol3-p0285",
    "vol2-p0374",
}

ENTITY_PATTERN = re.compile(r"^(?:E(?:0[1-9]|[1-4][0-9]|5[0-2])|NE(?:0[1-9]|[1-9][0-9]))$")
RELATION_PATTERN = re.compile(r"^(?:R(?:0[1-9]|[1-7][0-9]|8[01])|NR(?:0[1-9]|[1-9][0-9]))$")
PROVISIONAL_ENTITY_PATTERN = re.compile(r"^NE(?:0[1-9]|[1-9][0-9])$")
PROVISIONAL_RELATION_PATTERN = re.compile(r"^NR(?:0[1-9]|[1-9][0-9])$")
TAG_LINE_PATTERN = re.compile(r"^- \*\*((?:E|R)\d{2}) ([^*]+)\*\*:", re.MULTILINE)
USAGE_KEYS = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)


@dataclass
class PageJob:
    key: str
    volume: int
    page: int
    summary: str
    sentences: list[dict[str, str]]
    triples: list[dict[str, Any]]
    selections: list[dict[str, Any]]

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "summary": self.summary,
            "sentences": self.sentences,
            "triples": self.triples,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def enum_name(value: Any) -> str:
    if value is None:
        return ""
    return getattr(value, "value", str(value))


def load_prompt_package() -> tuple[str, str, dict[str, Any]]:
    with zipfile.ZipFile(PROMPT_PACKAGE) as archive:
        prefix = archive.read(PROMPT_ENTRY).decode("utf-8")
        template = archive.read(TEMPLATE_ENTRY).decode("utf-8")
        schema = json.loads(archive.read(SCHEMA_ENTRY).decode("utf-8"))
    return prefix, template, schema


def load_source_pages() -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    with SOURCE_PAGES.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            pages[record["key"]] = record
    return pages


def load_jobs() -> list[PageJob]:
    source_pages = load_source_pages()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for volume in (1, 2, 3):
        document = read_json(CANONICAL_ROOT / "sentences" / f"vol{volume}.json")
        for sentence in document["sentences"].values():
            grouped.setdefault(sentence["selectedFrom"]["key"], []).append(sentence)

    jobs: list[PageJob] = []
    for key, selected_sentences in grouped.items():
        source = source_pages[key]
        source_order = {
            sentence["sid"]: index
            for index, sentence in enumerate(source["sentences"])
        }
        if len(source_order) != len(source["sentences"]):
            raise ValueError(f"{key}: repeated sentence ID inside one source page")
        for sentence in selected_sentences:
            if sentence["sid"] not in source_order:
                raise ValueError(f"{key}/{sentence['sid']}: absent from selected source page")
        selected_sentences.sort(key=lambda sentence: source_order[sentence["sid"]])

        sentences = [
            {
                "sid": sentence["sid"],
                "my": sentence["my"],
                "en": sentence["en"],
            }
            for sentence in selected_sentences
        ]
        triples: list[dict[str, Any]] = []
        for sentence in selected_sentences:
            for triple in sentence["triples"]:
                triples.append(
                    {
                        "i": len(triples),
                        "sid": sentence["sid"],
                        "subject": triple["subject"],
                        "predicate": triple["predicate"],
                        "object": triple["object"],
                    }
                )
        if not triples:
            raise ValueError(f"{key}: canonical selected page has no triples")
        selections = [
            {
                "sid": sentence["sid"],
                "ownerPage": sentence["ownerPage"],
                "selectedFrom": sentence["selectedFrom"],
                "sourceAppearances": sentence["sourceAppearances"],
            }
            for sentence in selected_sentences
        ]
        jobs.append(
            PageJob(
                key=key,
                volume=int(source["volume"]),
                page=int(source["page"]),
                summary=source["summary"],
                sentences=sentences,
                triples=triples,
                selections=selections,
            )
        )

    jobs.sort(key=lambda job: (job.volume, job.page))
    sentence_count = sum(len(job.sentences) for job in jobs)
    triple_count = sum(len(job.triples) for job in jobs)
    if len(jobs) != EXPECTED_CANONICAL_PAGES:
        raise ValueError(f"Canonical page count changed: {len(jobs)}")
    if sentence_count != EXPECTED_CANONICAL_SENTENCES:
        raise ValueError(f"Canonical sentence count changed: {sentence_count}")
    if triple_count != EXPECTED_CANONICAL_TRIPLES:
        raise ValueError(f"Canonical triple count changed: {triple_count}")
    return jobs


def triple_signature(triple: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        triple["sid"],
        triple["subject"],
        triple["predicate"],
        triple["object"],
    )


def validate_result(job: PageJob, result: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["top-level response is not an object"]
    expected_top = {
        "new_entity_categories",
        "new_relation_categories",
        "annotations",
    }
    if set(result) != expected_top:
        errors.append(f"top-level keys differ: {sorted(result)}")
    entities = result.get("new_entity_categories")
    relations = result.get("new_relation_categories")
    annotations = result.get("annotations")
    if not isinstance(entities, list):
        errors.append("new_entity_categories is not an array")
        entities = []
    if not isinstance(relations, list):
        errors.append("new_relation_categories is not an array")
        relations = []
    if not isinstance(annotations, list):
        errors.append("annotations is not an array")
        annotations = []
    if len(annotations) != len(job.triples):
        errors.append(
            f"annotation count {len(annotations)} != triple count {len(job.triples)}"
        )

    proposal_keys = {"id", "label", "definition", "justification"}
    entity_ids: list[str] = []
    relation_ids: list[str] = []
    for kind, proposals, pattern, ids in (
        ("entity", entities, PROVISIONAL_ENTITY_PATTERN, entity_ids),
        ("relation", relations, PROVISIONAL_RELATION_PATTERN, relation_ids),
    ):
        for index, proposal in enumerate(proposals):
            if not isinstance(proposal, dict):
                errors.append(f"{kind} proposal {index} is not an object")
                continue
            if set(proposal) != proposal_keys:
                errors.append(
                    f"{kind} proposal {index} keys differ: {sorted(proposal)}"
                )
            proposal_id = proposal.get("id")
            if not isinstance(proposal_id, str) or not pattern.fullmatch(proposal_id):
                errors.append(f"{kind} proposal {index} has invalid ID {proposal_id!r}")
            else:
                ids.append(proposal_id)
            for field in ("label", "definition", "justification"):
                if not isinstance(proposal.get(field), str) or not proposal[field].strip():
                    errors.append(f"{kind} proposal {index} has empty {field}")

    if len(entity_ids) != len(set(entity_ids)):
        errors.append("duplicate provisional entity ID")
    if len(relation_ids) != len(set(relation_ids)):
        errors.append("duplicate provisional relation ID")

    used_entities: set[str] = set()
    used_relations: set[str] = set()
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            errors.append(f"annotation {index} is not an object")
            continue
        if set(annotation) != {"s", "r", "o"}:
            errors.append(f"annotation {index} keys differ: {sorted(annotation)}")
        for field in ("s", "o"):
            value = annotation.get(field)
            if not isinstance(value, str) or not ENTITY_PATTERN.fullmatch(value):
                errors.append(f"annotation {index}.{field} invalid: {value!r}")
            elif value.startswith("NE"):
                used_entities.add(value)
        relation = annotation.get("r")
        if not isinstance(relation, str) or not RELATION_PATTERN.fullmatch(relation):
            errors.append(f"annotation {index}.r invalid: {relation!r}")
        elif relation.startswith("NR"):
            used_relations.add(relation)

    if used_entities != set(entity_ids):
        errors.append(
            f"declared/used provisional entities differ: declared={entity_ids}, used={sorted(used_entities)}"
        )
    if used_relations != set(relation_ids):
        errors.append(
            f"declared/used provisional relations differ: declared={relation_ids}, used={sorted(used_relations)}"
        )
    return errors


def completed_subset(job: PageJob) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dir = COMPLETED_PAGE_DIRS[job.key]
    old_page = read_json(source_dir / "page_data.json")
    old_result = read_json(source_dir / "result.json")
    if old_result["new_entity_categories"] or old_result["new_relation_categories"]:
        raise ValueError(f"{job.key}: completed-page proposals cannot be subset safely")
    old_triples = old_page["triples"]
    old_annotations = old_result["annotations"]
    if len(old_triples) != len(old_annotations):
        raise ValueError(f"{job.key}: old triple/annotation count mismatch")

    selected_annotations: list[dict[str, str]] = []
    selected_old_indices: list[int] = []
    cursor = 0
    for canonical_triple in job.triples:
        signature = triple_signature(canonical_triple)
        while cursor < len(old_triples) and triple_signature(old_triples[cursor]) != signature:
            cursor += 1
        if cursor >= len(old_triples):
            raise ValueError(f"{job.key}: canonical triple absent from completed v2 page")
        selected_annotations.append(old_annotations[cursor])
        selected_old_indices.append(cursor)
        cursor += 1
    result = {
        "new_entity_categories": [],
        "new_relation_categories": [],
        "annotations": selected_annotations,
    }
    errors = validate_result(job, result)
    if errors:
        raise ValueError(f"{job.key}: invalid completed subset: {errors}")
    record = {
        "page_key": job.key,
        "source_directory": str(source_dir),
        "old_triple_count": len(old_triples),
        "canonical_triple_count": len(job.triples),
        "selected_old_indices": selected_old_indices,
        "discarded_duplicate_boundary_triples": len(old_triples) - len(job.triples),
        "prior_run_metadata": read_json(source_dir / "run_metadata.json"),
    }
    return result, record


def page_input_path(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "inputs" / f"vol{job.volume}" / f"{job.key}.json"


def page_selection_path(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "canonical_selection" / f"vol{job.volume}" / f"{job.key}.json"


def page_output_dir(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "pages" / f"vol{job.volume}" / job.key


def result_path(out_dir: Path, job: PageJob) -> Path:
    return page_output_dir(out_dir, job) / "result.json"


def prepare(out_dir: Path, max_output_tokens: int) -> tuple[list[PageJob], str, str, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs()
    prefix, template, schema = load_prompt_package()
    completed_records: list[dict[str, Any]] = []

    for job in jobs:
        write_json(page_input_path(out_dir, job), job.payload)
        write_json(
            page_selection_path(out_dir, job),
            {
                "key": job.key,
                "selectionRule": "max_triples_then_owner_page_then_lowest_page",
                "sentences": job.selections,
            },
        )
        if job.key in COMPLETED_PAGE_DIRS:
            result, reuse_record = completed_subset(job)
            target = page_output_dir(out_dir, job)
            write_json(target / "page_data.json", job.payload)
            write_json(target / "result.json", result)
            write_json(target / "reuse_record.json", reuse_record)
            completed_records.append(reuse_record)

    queued = [job for job in jobs if job.key not in COMPLETED_PAGE_DIRS]
    if not REQUEUED_V1_PAGES.issubset({job.key for job in queued}):
        raise ValueError("One or more original v1 pages were not requeued")
    if sum(len(job.triples) for job in queued) != 26795:
        raise ValueError("Queued canonical triple count differs from audited 26,795")

    plans: list[dict[str, Any]] = []
    for volume in (1, 2, 3):
        volume_jobs = [job for job in queued if job.volume == volume]
        plan = {
            "label": f"axial_vol{volume}",
            "volume": volume,
            "page_count": len(volume_jobs),
            "sentence_count": sum(len(job.sentences) for job in volume_jobs),
            "triple_count": sum(len(job.triples) for job in volume_jobs),
            "pages": [
                {
                    "key": job.key,
                    "page": job.page,
                    "sentence_count": len(job.sentences),
                    "triple_count": len(job.triples),
                }
                for job in volume_jobs
            ],
        }
        write_json(out_dir / "batch_plans" / f"vol{volume}.json", plan)
        plans.append(plan)

    canonical_index = read_json(CANONICAL_ROOT / "index.json")
    manifest = {
        "status": "prepared",
        "prepared_at": utc_now(),
        "model": MODEL,
        "prompt_version": "v2.0",
        "thinking_level": "low",
        "temperature": 0.0,
        "max_output_tokens": max_output_tokens,
        "batching": "one Gemini Batch API job per volume",
        "retry_policy": "no retries; report truncations and other unresolved pages",
        "source_pages": str(SOURCE_PAGES),
        "source_pages_sha256": sha256(SOURCE_PAGES),
        "canonical_root": str(CANONICAL_ROOT),
        "canonical_index_sha256": sha256(CANONICAL_ROOT / "index.json"),
        "selection_rule": canonical_index["selectionRule"],
        "source_triples": canonical_index["totals"]["sourceTriples"],
        "canonical_pages": len(jobs),
        "canonical_sentences": sum(len(job.sentences) for job in jobs),
        "canonical_triples": sum(len(job.triples) for job in jobs),
        "removed_duplicate_appearances": canonical_index["totals"][
            "removedDuplicateAppearances"
        ],
        "completed_v2_pages_reused": sorted(COMPLETED_PAGE_DIRS),
        "completed_v2_triples_reused": sum(
            len(job.triples) for job in jobs if job.key in COMPLETED_PAGE_DIRS
        ),
        "original_v1_pages_requeued": sorted(REQUEUED_V1_PAGES),
        "queued_pages": len(queued),
        "queued_sentences": sum(len(job.sentences) for job in queued),
        "queued_triples": sum(len(job.triples) for job in queued),
        "volume_plans": plans,
        "prompt_package": str(PROMPT_PACKAGE),
        "prompt_package_sha256": sha256(PROMPT_PACKAGE),
    }
    write_json(out_dir / "source_manifest.json", manifest)
    write_json(out_dir / "completed_reuse.json", completed_records)
    write_text(out_dir / "cached_prefix.txt", prefix)
    write_text(out_dir / "page_template.txt", template)
    write_json(out_dir / "base_response_schema.json", schema)
    return jobs, prefix, template, schema


def token_count_total(response: Any) -> int | None:
    value = getattr(response, "total_tokens", None)
    if value is not None:
        return int(value)
    dumped = json_safe(response)
    if isinstance(dumped, dict):
        for key in ("total_tokens", "totalTokens"):
            if dumped.get(key) is not None:
                return int(dumped[key])
    return None


def create_cache(
    client: genai.Client,
    out_dir: Path,
    prefix: str,
    ttl_seconds: int,
) -> tuple[str, dict[str, Any]]:
    record_path = out_dir / "cache_record.json"
    if record_path.exists():
        record = read_json(record_path)
        client.caches.get(name=record["name"])
        return record["name"], record
    count = client.models.count_tokens(model=MODEL, contents=prefix)
    cache = client.caches.create(
        model=MODEL,
        config=types.CreateCachedContentConfig(
            display_name=f"konbaung_axial_v2_full_{utc_stamp()}",
            contents=prefix,
            ttl=f"{ttl_seconds}s",
        ),
    )
    record = {
        "name": cache.name,
        "model": MODEL,
        "display_name": f"konbaung_axial_v2_full_{utc_stamp()}",
        "prefix_tokens": token_count_total(count),
        "prefix_characters": len(prefix),
        "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
        "ttl_seconds": ttl_seconds,
        "created_at": utc_now(),
        "cache": json_safe(cache),
    }
    write_json(record_path, record)
    return cache.name, record


def dynamic_text(template: str, job: PageJob) -> str:
    compact = json.dumps(job.payload, ensure_ascii=False, separators=(",", ":"))
    return template.replace("{{PAGE_DATA_JSON}}", compact)


def response_schema(base_schema: dict[str, Any], triple_count: int) -> dict[str, Any]:
    schema = copy.deepcopy(base_schema)
    annotations = schema["properties"]["annotations"]
    annotations["minItems"] = triple_count
    annotations["maxItems"] = triple_count
    return schema


def build_request(
    job: PageJob,
    cache_name: str,
    template: str,
    base_schema: dict[str, Any],
    max_output_tokens: int,
) -> types.InlinedRequest:
    return types.InlinedRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=dynamic_text(template, job))],
            )
        ],
        metadata={
            "key": job.key,
            "volume": str(job.volume),
            "page": f"{job.page:04d}",
            "triple_count": str(len(job.triples)),
        },
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            response_mime_type="application/json",
            response_json_schema=response_schema(base_schema, len(job.triples)),
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
            ),
        ),
    )


def submit(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    jobs, prefix, template, schema = prepare(out_dir, args.max_output_tokens)
    key = api_key()
    if not key:
        raise RuntimeError("No Gemini API key found")
    client = genai.Client(api_key=key)
    cache_name, cache_record = create_cache(
        client,
        out_dir,
        prefix,
        args.cache_ttl_seconds,
    )
    queued = [job for job in jobs if job.key not in COMPLETED_PAGE_DIRS]
    submitted: list[dict[str, Any]] = []
    for volume in (1, 2, 3):
        label = f"axial_vol{volume}"
        submitted_path = out_dir / "batch_jobs" / f"{label}_submitted.json"
        volume_jobs = [job for job in queued if job.volume == volume]
        if submitted_path.exists():
            batch = client.batches.get(name=read_json(submitted_path)["name"])
        else:
            requests = [
                build_request(
                    job,
                    cache_name,
                    template,
                    schema,
                    args.max_output_tokens,
                )
                for job in volume_jobs
            ]
            batch = client.batches.create(
                model=MODEL,
                src=requests,
                config=types.CreateBatchJobConfig(
                    display_name=f"konbaung_axial_v2_vol{volume}_{utc_stamp()}"
                ),
            )
            write_json(submitted_path, json_safe(batch))
        submitted.append(
            {
                "label": label,
                "name": batch.name,
                "state": enum_name(batch.state),
                "page_count": len(volume_jobs),
                "triple_count": sum(len(job.triples) for job in volume_jobs),
            }
        )
    record = {
        "status": "submitted",
        "submitted_at": utc_now(),
        "cache": cache_record,
        "batches": submitted,
    }
    write_json(out_dir / "submission_summary.json", record)
    return record


def batch_status(out_dir: Path, client: genai.Client) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for volume in (1, 2, 3):
        label = f"axial_vol{volume}"
        submitted_path = out_dir / "batch_jobs" / f"{label}_submitted.json"
        if not submitted_path.exists():
            rows.append({"label": label, "state": "NOT_SUBMITTED"})
            continue
        name = read_json(submitted_path)["name"]
        batch = client.batches.get(name=name)
        write_json(
            out_dir / "batch_jobs" / f"{label}_latest.json",
            json_safe(batch),
        )
        rows.append(
            {
                "label": label,
                "name": name,
                "state": enum_name(batch.state),
                "completion_stats": json_safe(
                    getattr(batch, "completion_stats", None)
                ),
            }
        )
    write_json(
        out_dir / "batch_jobs" / "status.json",
        {"checked_at": utc_now(), "batches": rows},
    )
    return rows


def response_text(response: Any) -> str:
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if text and not getattr(part, "thought", False):
                parts.append(text)
    return "".join(parts).strip()


def finish_reasons(response: Any) -> list[str]:
    return [
        enum_name(getattr(candidate, "finish_reason", None))
        for candidate in getattr(response, "candidates", None) or []
    ]


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    return json_safe(usage) if usage is not None else {}


def decode_nested_json_objects(result: Any) -> tuple[Any, int]:
    if not isinstance(result, dict):
        return result, 0
    decoded = copy.deepcopy(result)
    decoded_count = 0
    expected_fields = {
        "new_entity_categories": ("id", "label", "definition", "justification"),
        "new_relation_categories": ("id", "label", "definition", "justification"),
        "annotations": ("s", "r", "o"),
    }
    for key, fields in expected_fields.items():
        values = decoded.get(key)
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            if not isinstance(value, str):
                continue
            try:
                nested = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(nested, dict):
                values[index] = nested
                decoded_count += 1
        if values and all(isinstance(value, str) for value in values):
            width = len(fields) * 2
            if len(values) % width == 0:
                rebuilt: list[dict[str, str]] = []
                for start in range(0, len(values), width):
                    chunk = values[start : start + width]
                    if tuple(chunk[::2]) != fields:
                        rebuilt = []
                        break
                    rebuilt.append(dict(zip(chunk[::2], chunk[1::2])))
                if rebuilt:
                    decoded[key] = rebuilt
                    decoded_count += len(rebuilt)
                    continue
            if fields[0] == "id":
                width = 1 + (len(fields) - 1) * 2
                if len(values) % width == 0:
                    rebuilt = []
                    for start in range(0, len(values), width):
                        chunk = values[start : start + width]
                        if tuple(chunk[1::2]) != fields[1:]:
                            rebuilt = []
                            break
                        proposal = {"id": chunk[0]}
                        proposal.update(dict(zip(chunk[1::2], chunk[2::2])))
                        rebuilt.append(proposal)
                    if rebuilt:
                        decoded[key] = rebuilt
                        decoded_count += len(rebuilt)
    return decoded, decoded_count


def classify_response(
    job: PageJob,
    response: Any,
    max_output_tokens: int,
) -> dict[str, Any]:
    raw = response_text(response)
    usage = usage_dict(response)
    reasons = finish_reasons(response)
    generation_tokens = int(usage.get("candidates_token_count") or 0) + int(
        usage.get("thoughts_token_count") or 0
    )
    capped = any("MAX_TOKENS" in reason for reason in reasons)
    near_cap = generation_tokens >= max_output_tokens - 64
    attempt: dict[str, Any] = {
        "key": job.key,
        "finish_reasons": reasons,
        "usage": usage,
        "generation_tokens": generation_tokens,
        "max_output_tokens": max_output_tokens,
        "raw_response": raw,
    }
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        attempt["category"] = "truncated" if capped or near_cap else "schema_invalid"
        attempt["errors"] = [f"JSON parse failed: {exc}"]
        return attempt
    result, decoded_count = decode_nested_json_objects(result)
    attempt["nested_json_strings_decoded"] = decoded_count
    errors = validate_result(job, result)
    attempt["result"] = result
    attempt["errors"] = errors
    if not errors:
        attempt["category"] = "accepted"
    elif capped or near_cap:
        attempt["category"] = "truncated"
    else:
        attempt["category"] = "validation_invalid"
    return attempt


def load_batch_jobs(out_dir: Path, all_jobs: list[PageJob], volume: int) -> list[PageJob]:
    page_keys = {
        row["key"]
        for row in read_json(out_dir / "batch_plans" / f"vol{volume}.json")["pages"]
    }
    jobs = [job for job in all_jobs if job.key in page_keys]
    if len(jobs) != len(page_keys):
        raise ValueError(f"vol{volume}: batch plan cannot be reconstructed")
    return jobs


def process_batch(
    out_dir: Path,
    batch: Any,
    jobs: list[PageJob],
    max_output_tokens: int,
) -> dict[str, Any]:
    if enum_name(batch.state) != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"{batch.name} ended in {enum_name(batch.state)}")
    by_key = {job.key: job for job in jobs}
    responses = getattr(getattr(batch, "dest", None), "inlined_responses", None) or []
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for inlined in responses:
        metadata = getattr(inlined, "metadata", None) or {}
        key = metadata.get("key")
        job = by_key.get(key or "")
        if job is None:
            rows.append({"key": key, "category": "unknown_response"})
            totals["unknown_response"] += 1
            continue
        seen.add(job.key)
        if getattr(inlined, "error", None) is not None:
            attempt = {
                "key": job.key,
                "category": "transport_error",
                "error": json_safe(inlined.error),
            }
        elif getattr(inlined, "response", None) is None:
            attempt = {
                "key": job.key,
                "category": "transport_error",
                "error": "missing response",
            }
        else:
            response = inlined.response
            write_json(
                out_dir / "raw_responses" / f"vol{job.volume}" / f"{job.key}.json",
                json_safe(response),
            )
            attempt = classify_response(job, response, max_output_tokens)
        totals[attempt["category"]] += 1
        for usage_key in USAGE_KEYS:
            value = attempt.get("usage", {}).get(usage_key)
            if isinstance(value, int):
                totals[usage_key] += value
        write_json(
            out_dir / "attempts" / f"vol{job.volume}" / f"{job.key}.json",
            attempt,
        )
        if attempt["category"] == "accepted":
            target = page_output_dir(out_dir, job)
            write_json(target / "page_data.json", job.payload)
            write_json(target / "result.json", attempt["result"])
            write_json(
                target / "run_metadata.json",
                {
                    "status": "completed_valid",
                    "source": "new_gemini_batch",
                    "model": MODEL,
                    "thinking_level": "low",
                    "temperature": 0.0,
                    "max_output_tokens": max_output_tokens,
                    "batch_name": batch.name,
                    "finish_reasons": attempt.get("finish_reasons", []),
                    "usage": attempt.get("usage", {}),
                    "validation": {"accepted": True, "errors": []},
                },
            )
        rows.append(
            {
                "key": job.key,
                "category": attempt["category"],
                "finish_reasons": attempt.get("finish_reasons", []),
                "usage": attempt.get("usage", {}),
                "errors": attempt.get("errors", attempt.get("error")),
            }
        )
    for job in jobs:
        if job.key in seen:
            continue
        attempt = {
            "key": job.key,
            "category": "transport_error",
            "error": "missing inlined batch response",
        }
        write_json(
            out_dir / "attempts" / f"vol{job.volume}" / f"{job.key}.json",
            attempt,
        )
        totals["transport_error"] += 1
        rows.append(attempt)
    return {
        "batch_name": batch.name,
        "state": enum_name(batch.state),
        "completion_stats": json_safe(getattr(batch, "completion_stats", None)),
        "totals": dict(totals),
        "results": rows,
    }


def taxonomy_labels(prefix: str) -> dict[str, str]:
    labels = {tag: label.strip() for tag, label in TAG_LINE_PATTERN.findall(prefix)}
    if len([tag for tag in labels if tag.startswith("E")]) != 52:
        raise ValueError("Could not parse all 52 entity labels from prompt")
    if len([tag for tag in labels if tag.startswith("R")]) != 81:
        raise ValueError("Could not parse all 81 relation labels from prompt")
    return labels


def md_text(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").strip()


def render_review(
    job: PageJob,
    result: dict[str, Any] | None,
    status: str,
    labels: dict[str, str],
    errors: Any = None,
    thinking_level: str = "low",
) -> str:
    lines = [
        f"# Flash-Lite axial-coding full corpus — {job.key}",
        "",
        f"- Status: **{status}**",
        "- Prompt version: **v2.0**",
        f"- Model: `{MODEL}`",
        f"- Thinking level: **{thinking_level}**",
        f"- Sentences: **{len(job.sentences)}**",
        f"- Triples: **{len(job.triples)}**",
        "",
        "## Provisional categories",
        "",
    ]
    provisional_labels: dict[str, str] = {}
    if result is None:
        lines.extend(["- No valid result returned.", ""])
    else:
        entities = result["new_entity_categories"]
        relations = result["new_relation_categories"]
        lines.extend(["### Entity categories", ""])
        if entities:
            for proposal in entities:
                provisional_labels[proposal["id"]] = proposal["label"]
                lines.extend(
                    [
                        f"- `{proposal['id']}` — **{proposal['label']}**: {proposal['definition']}",
                        f"  - Justification: {proposal['justification']}",
                    ]
                )
        else:
            lines.append("- None proposed.")
        lines.extend(["", "### Relation categories", ""])
        if relations:
            for proposal in relations:
                provisional_labels[proposal["id"]] = proposal["label"]
                lines.extend(
                    [
                        f"- `{proposal['id']}` — **{proposal['label']}**: {proposal['definition']}",
                        f"  - Justification: {proposal['justification']}",
                    ]
                )
        else:
            lines.append("- None proposed.")
        lines.append("")
    if errors:
        lines.extend(["## Unresolved response details", "", f"`{errors}`", ""])
    lines.extend(["## Page summary", "", job.summary, "", "## Sentences and tagged triples", ""])

    annotations = result["annotations"] if result is not None else []
    triples_by_sid: dict[str, list[dict[str, Any]]] = {}
    for triple in job.triples:
        triples_by_sid.setdefault(triple["sid"], []).append(triple)
    for sentence in job.sentences:
        lines.extend(
            [
                f"### `{sentence['sid']}`",
                "",
                sentence["my"],
                "",
                sentence["en"],
                "",
                "#### Triples",
                "",
            ]
        )
        sentence_triples = triples_by_sid.get(sentence["sid"], [])
        if not sentence_triples:
            lines.extend(["- No accepted triples in this canonical sentence.", ""])
            continue
        for triple in sentence_triples:
            index = int(triple["i"])
            lines.append(
                f"- **{index}.** **{md_text(triple['subject'])}** — "
                f"`{md_text(triple['predicate'])}` → **{md_text(triple['object'])}**"
            )
            if index < len(annotations):
                annotation = annotations[index]
                tag_parts: list[str] = []
                for field in ("s", "r", "o"):
                    tag = annotation[field]
                    label = labels.get(tag, provisional_labels.get(tag, "ProvisionalCategory"))
                    tag_parts.append(f"`{tag}` — **{label}**")
                lines.append(f"  - Axial tags: {'; '.join(tag_parts)}")
            else:
                lines.append("  - Axial tags: **not returned**")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def finalize(
    out_dir: Path,
    jobs: list[PageJob],
    batch_summaries: list[dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    labels = taxonomy_labels(prefix)
    usage_totals: Counter[str] = Counter()
    category_by_key: dict[str, str] = {}
    errors_by_key: dict[str, Any] = {}
    for summary in batch_summaries:
        for key in USAGE_KEYS:
            usage_totals[key] += int(summary["totals"].get(key, 0))
        for row in summary["results"]:
            if row.get("key"):
                category_by_key[row["key"]] = row["category"]
                if row.get("errors"):
                    errors_by_key[row["key"]] = row["errors"]

    page_rows: list[dict[str, Any]] = []
    triple_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    truncated: list[str] = []
    unresolved: list[dict[str, Any]] = []
    for job in jobs:
        if job.key in COMPLETED_PAGE_DIRS:
            status = "completed_valid_reused_v2"
        else:
            status = category_by_key.get(job.key, "missing")
        page_result_path = result_path(out_dir, job)
        result = read_json(page_result_path) if page_result_path.exists() else None
        if status == "truncated":
            truncated.append(job.key)
        if result is None:
            unresolved.append(
                {
                    "key": job.key,
                    "category": status,
                    "errors": errors_by_key.get(job.key),
                }
            )
        run_metadata_path = page_output_dir(out_dir, job) / "run_metadata.json"
        thinking_level = "low"
        if run_metadata_path.exists():
            thinking_level = str(
                read_json(run_metadata_path).get("thinking_level") or thinking_level
            )
        review = render_review(
            job,
            result,
            status,
            labels,
            errors_by_key.get(job.key),
            thinking_level,
        )
        write_text(page_output_dir(out_dir, job) / "review.md", review)
        page_rows.append(
            {
                "key": job.key,
                "volume": job.volume,
                "page": job.page,
                "sentence_count": len(job.sentences),
                "triple_count": len(job.triples),
                "status": status,
                "new_entity_categories": (
                    result["new_entity_categories"] if result else None
                ),
                "new_relation_categories": (
                    result["new_relation_categories"] if result else None
                ),
            }
        )
        if result is None:
            continue
        for kind in ("new_entity_categories", "new_relation_categories"):
            for proposal in result[kind]:
                proposal_rows.append(
                    {"key": job.key, "kind": kind, **proposal}
                )
        for triple, annotation in zip(job.triples, result["annotations"]):
            triple_rows.append(
                {
                    "key": job.key,
                    "volume": job.volume,
                    "page": job.page,
                    **triple,
                    "tags": annotation,
                }
            )
        write_json(
            page_output_dir(out_dir, job) / "tagged_triples.json",
            [
                {**triple, "tags": annotation}
                for triple, annotation in zip(job.triples, result["annotations"])
            ],
        )

    final_dir = out_dir / "final"
    write_text(
        final_dir / "all_pages.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in page_rows),
    )
    write_text(
        final_dir / "all_tagged_triples.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in triple_rows),
    )
    write_text(
        final_dir / "all_provisional_categories.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in proposal_rows),
    )
    write_json(final_dir / "truncated_pages.json", truncated)
    write_json(final_dir / "unresolved_pages.json", unresolved)

    combined_path = final_dir / "combined_review.md"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as combined:
        combined.write("# Konbaung Flash-Lite axial-coding full-corpus review\n\n")
        combined.write(
            f"- Canonical triples: **{EXPECTED_CANONICAL_TRIPLES}**\n"
            f"- Tagged triples returned or reused: **{len(triple_rows)}**\n"
            f"- Truncated pages: **{len(truncated)}**\n"
            f"- Other unresolved pages: **{len(unresolved) - len(truncated)}**\n\n"
        )
        for job in jobs:
            combined.write("\n---\n\n")
            combined.write(
                (page_output_dir(out_dir, job) / "review.md").read_text(
                    encoding="utf-8"
                )
            )

    category_totals = Counter(row["status"] for row in page_rows)
    cache_record = read_json(out_dir / "cache_record.json")
    report = {
        "status": "completed" if not unresolved else "completed_with_unresolved_pages",
        "completed_at": utc_now(),
        "model": MODEL,
        "prompt_version": "v2.0",
        "thinking_level": "low",
        "max_output_tokens": read_json(out_dir / "source_manifest.json")[
            "max_output_tokens"
        ],
        "selection_rule": "max_triples_then_owner_page_then_lowest_page",
        "source_triples": 30180,
        "canonical_sentences": EXPECTED_CANONICAL_SENTENCES,
        "canonical_triples": EXPECTED_CANONICAL_TRIPLES,
        "completed_v2_pages_reused": len(COMPLETED_PAGE_DIRS),
        "completed_v2_triples_reused": 334,
        "new_batch_pages_submitted": 1195,
        "new_batch_triples_submitted": 26795,
        "page_status_totals": dict(category_totals),
        "tagged_triples": len(triple_rows),
        "provisional_categories": len(proposal_rows),
        "truncated_pages": truncated,
        "other_unresolved_pages": [
            row for row in unresolved if row["category"] != "truncated"
        ],
        "usage_new_batch": dict(usage_totals),
        "explicit_cache": cache_record,
        "cached_prefix_verified_in_usage": usage_totals[
            "cached_content_token_count"
        ]
        > 0,
        "combined_review": str(combined_path),
        "all_tagged_triples": str(final_dir / "all_tagged_triples.jsonl"),
        "existing_database_modified": False,
        "retry_policy_applied": "none",
    }
    write_json(out_dir / "final_report.json", report)
    return report


def collect(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    jobs, prefix, _, _ = prepare(out_dir, args.max_output_tokens)
    key = api_key()
    if not key:
        raise RuntimeError("No Gemini API key found")
    client = genai.Client(api_key=key)
    statuses = batch_status(out_dir, client)
    nonterminal = [
        row for row in statuses if row["state"] not in TERMINAL_STATES
    ]
    if nonterminal:
        return {"status": "not_ready", "batches": statuses}
    failed = [
        row for row in statuses if row["state"] != "JOB_STATE_SUCCEEDED"
    ]
    if failed:
        raise RuntimeError(f"One or more volume batches failed: {failed}")

    summaries: list[dict[str, Any]] = []
    for volume in (1, 2, 3):
        label = f"axial_vol{volume}"
        batch_name = read_json(
            out_dir / "batch_jobs" / f"{label}_submitted.json"
        )["name"]
        batch = client.batches.get(name=batch_name)
        volume_jobs = load_batch_jobs(out_dir, jobs, volume)
        summary_path = out_dir / "batch_results" / f"{label}.json"
        if summary_path.exists():
            summary = read_json(summary_path)
        else:
            summary = process_batch(
                out_dir,
                batch,
                volume_jobs,
                args.max_output_tokens,
            )
            write_json(summary_path, summary)
        summaries.append(summary)
    report = finalize(out_dir, jobs, summaries, prefix)
    write_json(
        out_dir / "run_summary.json",
        {"batches": summaries, "final": report},
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cached Gemini batch runner for canonical Konbaung axial coding."
    )
    parser.add_argument(
        "command",
        choices=("prepare", "submit", "status", "collect"),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-output-tokens", type=int, default=15000)
    parser.add_argument("--cache-ttl-seconds", type=int, default=172800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.command == "prepare":
        jobs, _, _, _ = prepare(out_dir, args.max_output_tokens)
        result: Any = {
            "status": "prepared",
            "pages": len(jobs),
            "sentences": sum(len(job.sentences) for job in jobs),
            "triples": sum(len(job.triples) for job in jobs),
            "queued_pages": sum(
                job.key not in COMPLETED_PAGE_DIRS for job in jobs
            ),
            "queued_triples": sum(
                len(job.triples)
                for job in jobs
                if job.key not in COMPLETED_PAGE_DIRS
            ),
            "out_dir": str(out_dir),
        }
    elif args.command == "submit":
        result = submit(args)
    elif args.command == "status":
        key = api_key()
        if not key:
            raise RuntimeError("No Gemini API key found")
        result = {
            "checked_at": utc_now(),
            "batches": batch_status(out_dir, genai.Client(api_key=key)),
        }
    else:
        result = collect(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
