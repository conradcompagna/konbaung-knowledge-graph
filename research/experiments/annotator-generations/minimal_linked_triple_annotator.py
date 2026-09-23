#!/usr/bin/env python3
"""Minimal one-pass linked triple annotator for Konbaung pages.

Model schema contains only:
- Burmese subject span
- subject entity label
- subject English gloss
- open relation phrase
- Burmese object span
- object entity label
- object English gloss

All mapping, offsets, ambiguity warnings, and review text are produced locally.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Literal, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from konbaung_gemini_page_kg_annotator import (
    DEFAULT_ENV_FILE,
    DEFAULT_SOURCE_ROOT,
    PageJob,
    resolve_api_key,
)
from konbaung_gemini_xlmr_seed_annotator import (
    DEFAULT_CONTEXT_RATIO,
    build_page_index_with_neighbors,
    context_blocks,
    page_job_from_path,
    resolve_text_ref,
    write_json,
    write_text,
)


EntityTag = Literal["PERSON", "GROUP", "PLACE", "OBJECT", "TIME", "EVENT"]
ENTITY_TAGS = {"PERSON", "GROUP", "PLACE", "OBJECT", "TIME", "EVENT"}

DEFAULT_OUT_DIR = Path("konbaung_minimal_linked_triples_ready")
DEFAULT_MAX_OUTPUT_TOKENS = 5000


class MinimalTriple(BaseModel):
    s: str = Field(description="Exact Burmese subject span.")
    sl: EntityTag = Field(description="Subject label.")
    sg: str = Field(description="English subject gloss.")
    p: str = Field(description="Short English relation phrase.")
    o: str = Field(description="Exact Burmese object span.")
    ol: EntityTag = Field(description="Object label.")
    og: str = Field(description="English object gloss.")


class MinimalAnnotation(BaseModel):
    T: list[MinimalTriple]


PROMPT_TEMPLATE = """Extract the main narrative of TARGET_PAGE_TEXT as linked triples.

You should aim for around 20 triples per page, comprehensively covering the main spine of the narrative without needless repetition or coverage of unimportant details. It should be possible to reverse-engineer a faithful summary of the events on the page from the triples themselves without looking at the page. That is the high-level target. If your triples do not allow a reader to get a sense of the main spine of events on the page, you have failed.

Entity labels are closed-class, but relations are open-coded; handle them concisely and try to reuse terminology to cut down on variance. However, clarity comes first. Ensure that all triples are accurately represented.

Do not directly annotate spans from previous/next page context. Only use it to better annotate the current page.

Skip page numbers, footnotes, headers, footers, OCR debris, and other useless fragments that will not contribute to comprehension.

Schema fields: s/o are exact Burmese spans copied from TARGET_PAGE_TEXT; sl/ol are labels; sg/og are English glosses; p is the English relation. Never put English paraphrases in s or o.

<metadata>
job_id: {job_id}
volume_id: {volume_id}
page_num: {page_num}
</metadata>

<PREVIOUS_CONTEXT annotate="no">
{previous_context}
</PREVIOUS_CONTEXT>

<TARGET_PAGE_TEXT annotate="yes">
{target_page_text}
</TARGET_PAGE_TEXT>

<NEXT_CONTEXT annotate="no">
{next_context}
</NEXT_CONTEXT>
"""


def prompt_output_path(out_dir: Path, job: PageJob) -> Path:
    return out_dir / "prompts_sent" / job.volume_id / f"page_{job.page_num:04d}.txt"


def output_path(out_dir: Path, phase: str, bucket: str, job: PageJob) -> Path:
    return out_dir / phase / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def review_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:
    return out_dir / "review" / bucket / job.volume_id / f"page_{job.page_num:04d}.txt"


def validation_output_path(out_dir: Path, bucket: str, job: PageJob) -> Path:
    return out_dir / "validation" / bucket / job.volume_id / f"page_{job.page_num:04d}.json"


def build_prompt(
    job: PageJob, page_index: dict[tuple[str, int], PageJob], context_ratio: float
) -> str:
    blocks = context_blocks(job, page_index, context_ratio)
    return PROMPT_TEMPLATE.format(
        job_id=job.job_id,
        volume_id=job.volume_id,
        page_num=f"{job.page_num:04d}",
        **blocks,
    )


def usage_dict(response: Any) -> dict[str, Any]:
    if getattr(response, "usage_metadata", None) is None:
        return {}
    raw_usage = response.usage_metadata
    if hasattr(raw_usage, "model_dump"):
        return raw_usage.model_dump()
    return dict(raw_usage)


def call_gemini(
    client: genai.Client,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> tuple[MinimalAnnotation, dict[str, Any]]:
    last_err: Optional[Exception] = None
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=MinimalAnnotation,
        temperature=0.0,
        candidate_count=1,
        max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            usage = usage_dict(response)
            if getattr(response, "parsed", None) is not None:
                return response.parsed, usage
            return MinimalAnnotation.model_validate_json(response.text), usage
        except (ValidationError, Exception) as exc:
            last_err = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (2**attempt) + random.random())
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def page_metadata(job: PageJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "volume_id": job.volume_id,
        "page_num": job.page_num,
        "source_path": job.source_path,
        "source_sha1": job.sha1,
        "char_count": job.char_count,
        "line_count": job.line_count,
    }


def exact_line_matches(lines: list[str], tx: str) -> list[tuple[int, int, int]]:
    matches: list[tuple[int, int, int]] = []
    for ln, line in enumerate(lines, start=1):
        start = 0
        while True:
            pos = line.find(tx, start)
            if pos == -1:
                break
            matches.append((ln, pos, pos + len(tx)))
            start = pos + max(len(tx), 1)
    return matches


def resolve_span(job: PageJob, tx: str) -> tuple[dict[str, Any], list[str], Optional[str]]:
    lines = job.raw_text.splitlines()
    warnings: list[str] = []
    matches = exact_line_matches(lines, tx)
    if len(matches) > 1:
        warnings.append(f"ambiguous repeated span {tx!r}; used first exact occurrence")
    if matches:
        ln, start, end = matches[0]
        resolved, error = resolve_text_ref(job.raw_text, lines, ln=ln, tx=tx, i=1)
        return resolved or {}, warnings, error
    resolved, error = resolve_text_ref(job.raw_text, lines, ln=1, tx=tx, i=1)
    if resolved and resolved.get("line_hint_corrected"):
        warnings.append(f"span {tx!r}: line inferred locally")
    return resolved or {}, warnings, error


def validate_annotation(
    job: PageJob, annotation: MinimalAnnotation
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    resolved: dict[str, Any] = {"T": [], "_review_T": []}
    seen: set[tuple[str, str, str]] = set()

    for idx, triple in enumerate(annotation.T, start=1):
        ref = f"T[{idx}]"
        s = triple.s
        sl = str(triple.sl)
        sg = triple.sg
        p = triple.p
        o = triple.o
        ol = str(triple.ol)
        og = triple.og
        if not s.strip():
            errors.append(f"{ref}: blank subject span")
        if not o.strip():
            errors.append(f"{ref}: blank object span")
        if not p.strip():
            errors.append(f"{ref}: blank relation")

        key = (s.strip(), p.strip().lower(), o.strip())
        if key in seen:
            warnings.append(f"{ref}: duplicate triple dropped")
            continue
        seen.add(key)

        s_resolved, s_warnings, s_error = resolve_span(job, s)
        o_resolved, o_warnings, o_error = resolve_span(job, o)
        warnings.extend(f"{ref}.s: {warning}" for warning in s_warnings)
        warnings.extend(f"{ref}.o: {warning}" for warning in o_warnings)
        if s_error:
            errors.append(f"{ref}.s: {s_error}")
        if o_error:
            errors.append(f"{ref}.o: {o_error}")

        item = triple.model_dump()
        resolved["T"].append(item)
        resolved["_review_T"].append(
            {
                "s": s,
                "sl": sl,
                "sg": sg,
                "p": p,
                "o": o,
                "ol": ol,
                "og": og,
                "n": idx,
                "s_resolved": s_resolved,
                "o_resolved": o_resolved,
            }
        )

    return resolved, errors, warnings


def collect_review_spans(annotation: dict[str, Any]) -> None:
    next_id = 1
    entity_ids: dict[tuple[int, int, str], str] = {}
    for triple in annotation.get("T", []):
        for prefix, resolved_key, label_key, gloss_key in (
            ("s", "s_resolved", "sl", "sg"),
            ("o", "o_resolved", "ol", "og"),
        ):
            resolved = triple.get(resolved_key, {})
            abs_s = resolved.get("abs_s")
            abs_e = resolved.get("abs_e")
            label = triple.get(label_key)
            if not isinstance(abs_s, int) or not isinstance(abs_e, int) or not label:
                continue
            key = (abs_s, abs_e, str(label))
            if key not in entity_ids:
                entity_ids[key] = f"E{next_id}"
                next_id += 1
            triple[f"{prefix}_eid"] = entity_ids[key]
            triple[f"{prefix}_gloss"] = triple.get(gloss_key, "")


def annotate_line(line: str, spans: list[dict[str, Any]]) -> str:
    opens: dict[int, list[dict[str, Any]]] = {}
    closes: dict[int, list[dict[str, Any]]] = {}
    for span in spans:
        opens.setdefault(span["s"], []).append(span)
        closes.setdefault(span["e"], []).append(span)
    parts: list[str] = []
    for pos in range(len(line) + 1):
        for _span in sorted(closes.get(pos, []), key=lambda item: item["s"], reverse=True):
            parts.append("]")
        for span in sorted(opens.get(pos, []), key=lambda item: item["e"], reverse=True):
            parts.append(f"[{span['id']}|{span['label']} ")
        if pos < len(line):
            parts.append(line[pos])
    return "".join(parts)


def build_review_text(job: PageJob, annotation: dict[str, Any]) -> str:
    collect_review_spans(annotation)
    lines = job.raw_text.splitlines()
    spans_by_line: dict[int, list[dict[str, Any]]] = {}
    seen_spans: set[tuple[int, int, int, str, str]] = set()
    relations_by_line: dict[int, list[str]] = {}

    for triple in annotation.get("T", []):
        for side, resolved_key, label_key, eid_key in (
            ("s", "s_resolved", "sl", "s_eid"),
            ("o", "o_resolved", "ol", "o_eid"),
        ):
            resolved = triple.get(resolved_key, {})
            eid = triple.get(eid_key)
            if not eid:
                continue
            for part in resolved.get("spans", []):
                ln = part.get("ln")
                start = part.get("s")
                end = part.get("e")
                if isinstance(ln, int) and isinstance(start, int) and isinstance(end, int):
                    key = (ln, start, end, str(eid), str(triple.get(label_key)))
                    if key in seen_spans:
                        continue
                    seen_spans.add(key)
                    spans_by_line.setdefault(ln, []).append(
                        {"id": eid, "label": triple.get(label_key), "s": start, "e": end}
                    )
        s_ln = triple.get("s_resolved", {}).get("ln")
        o_ln = triple.get("o_resolved", {}).get("ln")
        ln_candidates = [ln for ln in (s_ln, o_ln) if isinstance(ln, int)]
        if ln_candidates:
            ln = max(ln_candidates)
            relations_by_line.setdefault(ln, []).append(
                f"R{triple.get('n')} "
                f"{triple.get('s_eid')}|{triple.get('sl')}|{triple.get('sg')} "
                f"--{triple.get('p')}--> "
                f"{triple.get('o_eid')}|{triple.get('ol')}|{triple.get('og')}"
            )

    output: list[str] = []
    for ln, line in enumerate(lines, start=1):
        output.append(annotate_line(line, spans_by_line.get(ln, [])))
        output.extend(relations_by_line.get(ln, []))
    return "\n".join(output) + "\n"


def run_page(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    job_path = source_root / args.volume_id / "pages" / f"page_{args.page_num:04d}.txt"
    if not job_path.exists():
        raise FileNotFoundError(job_path)

    job = page_job_from_path(job_path)
    page_index = build_page_index_with_neighbors(source_root, [job])
    out_dir = Path(args.out_dir)
    prompt = build_prompt(job, page_index, args.context_ratio)
    prompt_path = prompt_output_path(out_dir, job)
    write_text(prompt_path, prompt)

    summary = {
        **page_metadata(job),
        "out_dir": str(out_dir),
        "model": args.model,
        "schema": "minimal linked triples: each T item has only s/sl/sg/p/o/ol/og",
        "prompt_path": str(prompt_path),
        "will_call_api": bool(args.run),
    }
    write_json(out_dir / "run_summary.json", summary)

    if not args.run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    api_key = resolve_api_key(args)
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY, pass --api-key, or use --env-file."
        )

    client = genai.Client(api_key=api_key)
    annotation, usage = call_gemini(
        client,
        model=args.model,
        prompt=prompt,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    resolved, errors, warnings = validate_annotation(job, annotation)
    review_annotation = {"T": resolved.pop("_review_T", [])}
    bucket = "invalid" if errors else "valid"
    validation_record = {
        **page_metadata(job),
        "stage": "minimal_linked_triples",
        "usage_metadata": usage,
        "prompt_path": str(prompt_path),
        "validation_errors": errors,
        "validation_warnings": warnings,
    }
    write_json(output_path(out_dir, "raw", bucket, job), annotation.model_dump())
    write_json(output_path(out_dir, "postprocessed", bucket, job), resolved)
    write_json(validation_output_path(out_dir, bucket, job), validation_record)
    write_text(review_output_path(out_dir, bucket, job), build_review_text(job, review_annotation))
    print(
        json.dumps(
            {
                **summary,
                "bucket": bucket,
                "usage_metadata": usage,
                "triple_count": len(resolved["T"]),
                "errors": errors,
                "warnings": warnings,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal linked triple extraction on one Konbaung OCR page."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--volume-id", default="konbaung_vol1")
    parser.add_argument("--page-num", type=int, default=177)
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--context-ratio", type=float, default=DEFAULT_CONTEXT_RATIO)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    run_page(parse_args(argv))


if __name__ == "__main__":
    main()
