#!/usr/bin/env python3
"""Run two sequential 20-page contextual entity-resolution windows with Gemini."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from konbaung_gemini_summary_claim_completion_annotator import api_key


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831_WITH_METHODOLOGY.zip"
OUTPUT = ROOT / "konbaung_contextual_entity_resolution_trial_20260902_first40"
MODEL = "gemini-3.1-flash-lite"
WINDOW_SIZE = 20
PAGE_COUNT = 40
MAX_OUTPUT_TOKENS = 30000


def read_jsonl_member(archive: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    """Read one UTF-8 JSONL table from the canonical package."""
    rows: list[dict[str, Any]] = []
    with archive.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8") as text:
            for line in text:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def clean_line(value: Any) -> str:
    """Keep each sentence and entity record on one compact prompt line."""
    return " ".join(str(value).replace("\t", " ").split())


def load_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the three normalized source tables required for contextual resolution."""
    with zipfile.ZipFile(ARCHIVE) as archive:
        pages = read_jsonl_member(archive, "data/pages.jsonl")
        sentences = read_jsonl_member(archive, "data/sentences.jsonl")
        triples = read_jsonl_member(archive, "data/triples.jsonl")
    if len(pages) < PAGE_COUNT:
        raise RuntimeError(f"Archive has only {len(pages)} pages")
    return pages, sentences, triples


def build_window(
    selected_pages: list[dict[str, Any]],
    sentences: list[dict[str, Any]],
    triples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a compact sentence corpus and an entity-to-sentence context map."""
    page_order = {
        (page["volume_id"], int(page["page_number"])): position
        for position, page in enumerate(selected_pages)
    }
    window_sentences = [
        sentence
        for sentence in sentences
        if (sentence["volume_id"], int(sentence["owner_page"])) in page_order
    ]
    window_sentences.sort(
        key=lambda sentence: (
            page_order[(sentence["volume_id"], int(sentence["owner_page"]))],
            sentence["sentence_id"],
        )
    )
    short_id_by_sentence = {
        sentence["sentence_id"]: f"s{index}"
        for index, sentence in enumerate(window_sentences, start=1)
    }

    entity_contexts: dict[str, set[str]] = defaultdict(set)
    for triple in triples:
        short_id = short_id_by_sentence.get(triple["sentence_id"])
        if short_id is None:
            continue
        entity_contexts[str(triple["subject"]["tag"])].add(short_id)
        entity_contexts[str(triple["object"]["tag"])].add(short_id)

    sentences_by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sentence in window_sentences:
        key = (sentence["volume_id"], int(sentence["owner_page"]))
        sentences_by_page[key].append(sentence)

    lines = ["SENTENCES"]
    for page in selected_pages:
        key = (page["volume_id"], int(page["page_number"]))
        lines.append(f"P\t{page['page_id']}")
        for sentence in sentences_by_page[key]:
            lines.append(
                "\t".join(
                    (
                        short_id_by_sentence[sentence["sentence_id"]],
                        clean_line(sentence["burmese"]),
                        clean_line(sentence["english_translation"]),
                    )
                )
            )

    lines.append("ENTITIES\t(tag then supporting sentence IDs)")
    for tag in sorted(entity_contexts, key=lambda value: (value.casefold(), value)):
        sentence_ids = sorted(entity_contexts[tag], key=lambda value: int(value.removeprefix("s")))
        lines.append(f"{clean_line(tag)}\t{','.join(sentence_ids)}")

    return {
        "pages": [page["page_id"] for page in selected_pages],
        "sentenceCount": len(window_sentences),
        "entityCount": len(entity_contexts),
        "entities": sorted(entity_contexts, key=lambda value: (value.casefold(), value)),
        "bundle": "\n".join(lines),
    }


def compact_json(value: Any) -> str:
    """Serialize continuity state without whitespace that would waste prompt tokens."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_prompt(window: dict[str, Any], prior: dict[str, Any] | None) -> str:
    """Ask for strict identity resolution and concise ambiguity flags using full context."""
    prior_text = "none" if prior is None else compact_json(prior)
    if prior is None:
        continuity_instructions = """Return one complete register covering every exact raw
tag in this window. Every raw tag must occur exactly once, either in one resolved alias
list or as one ambiguous tag. Include singleton resolved groups."""
    else:
        continuity_instructions = """Return only a delta for this window, not the complete
prior register. Every exact raw tag in CURRENT_WINDOW must occur exactly once in the
delta, either in one resolved alias list or as one ambiguous tag. Include a prior raw tag
only when the new context links it to a current tag or revises its status. Include
singleton groups for current tags that do not merge. The local pipeline will merge this
delta into the prior register."""
    return f"""You are resolving raw entity tags in a Burmese Konbaung historical corpus.

Use the Burmese and English sentence context. Resolve identity, not topical similarity.

Tasks:
1. Group tags only when they denote the same real referent: the same person, place,
organization, group, office, object, event, or concept. Spelling, transliteration,
capitalization, word order, and title variants may merge. Do not merge relatives,
successive title-holders, related places, a ruler with a dynasty, or a part with a whole.
2. Put a tag in manual review when its supplied occurrences may denote different
referents or the intended referent remains genuinely uncertain. Give the shortest useful
best guess. If occurrences differ, cite compact sID=guess pairs in the guess.

The prior register is continuity evidence, not unquestionable truth. Reuse its canonical
name for the same referent, but correct it if the new context proves it wrong.
{continuity_instructions}
Copy raw tags exactly.

Return JSON only, with exactly this compact form:
{{"r":[["canonical",["exact raw tag", "..."]],...],"a":[["exact raw tag","best guess","reason <= 12 words"],...]}}

`r` means resolved groups; `a` means manual-review ambiguities. Use no prose outside JSON
and no explanations for resolved groups. Prefer an observed tag as the canonical name.

PRIOR_REGISTER_JSON
{prior_text}

CURRENT_WINDOW
{window["bundle"]}
"""


def parse_result(text: str) -> dict[str, Any]:
    """Parse compact JSON and recover a trailing ambiguity row from Gemini JSON glitches."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        result = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        result, end = decoder.raw_decode(stripped)
        trailing = stripped[end:].strip()
        fragment_start = trailing.find("[")
        if fragment_start >= 0:
            try:
                fragment, _ = decoder.raw_decode(trailing[fragment_start:])
            except json.JSONDecodeError:
                fragment = None
            if (
                isinstance(fragment, list)
                and len(fragment) == 3
                and all(isinstance(value, str) for value in fragment)
                and isinstance(result, dict)
                and isinstance(result.get("a"), list)
            ):
                result["a"].append(fragment)
    if not isinstance(result, dict):
        raise RuntimeError("Gemini response is not a JSON object")
    return result


def result_tags(result: dict[str, Any]) -> list[str]:
    """Flatten every exact raw tag represented in the compact output register."""
    tags: list[str] = []
    resolved = result.get("r")
    ambiguous = result.get("a")
    if not isinstance(resolved, list) or not isinstance(ambiguous, list):
        raise RuntimeError("Result must contain list-valued r and a keys")
    for group in resolved:
        if (
            not isinstance(group, list)
            or len(group) != 2
            or not isinstance(group[0], str)
            or not isinstance(group[1], list)
            or not group[1]
            or not all(isinstance(tag, str) for tag in group[1])
        ):
            raise RuntimeError(f"Invalid resolved group: {group!r}")
        tags.extend(group[1])
    for ambiguity in ambiguous:
        if (
            not isinstance(ambiguity, list)
            or len(ambiguity) != 3
            or not all(isinstance(value, str) for value in ambiguity)
        ):
            raise RuntimeError(f"Invalid ambiguity row: {ambiguity!r}")
        tags.append(ambiguity[0])
    return tags


def validate_result(result: dict[str, Any], expected_tags: set[str]) -> dict[str, Any]:
    """Require exact, unique coverage so no entity silently disappears between windows."""
    if set(result) != {"r", "a"}:
        raise RuntimeError(f"Result keys must be exactly r and a, got {sorted(result)}")
    tags = result_tags(result)
    duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
    missing = sorted(expected_tags - set(tags))
    unknown = sorted(set(tags) - expected_tags)
    if duplicates or missing or unknown:
        raise RuntimeError(
            compact_json(
                {
                    "duplicateTags": duplicates,
                    "missingTags": missing,
                    "unknownTags": unknown,
                }
            )
        )
    return {
        "resolvedGroups": len(result["r"]),
        "multiAliasGroups": sum(len(group[1]) > 1 for group in result["r"]),
        "ambiguousTags": len(result["a"]),
        "coveredTags": len(tags),
    }


def validate_delta(
    result: dict[str, Any], required_tags: set[str], allowed_tags: set[str]
) -> dict[str, Any]:
    """Require every current tag once while allowing only relevant prior-register tags."""
    if set(result) != {"r", "a"}:
        raise RuntimeError(f"Delta keys must be exactly r and a, got {sorted(result)}")
    tags = result_tags(result)
    duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
    missing = sorted(required_tags - set(tags))
    unknown = sorted(set(tags) - allowed_tags)
    if duplicates or missing or unknown:
        raise RuntimeError(
            compact_json(
                {
                    "duplicateTags": duplicates,
                    "missingCurrentTags": missing,
                    "unknownTags": unknown,
                }
            )
        )
    return {
        "resolvedGroups": len(result["r"]),
        "multiAliasGroups": sum(len(group[1]) > 1 for group in result["r"]),
        "ambiguousTags": len(result["a"]),
        "coveredCurrentTags": len(required_tags),
        "mentionedPriorTags": len(set(tags) - required_tags),
    }


def mechanically_conform_result(
    result: dict[str, Any], allowed_tags: set[str], required_tags: set[str] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conform aliases to the exact inventory without making new identity merges."""
    if required_tags is None:
        required_tags = allowed_tags

    def normalized(value: str) -> str:
        text = unicodedata.normalize("NFKC", value).casefold()
        return "".join(character for character in text if character.isalnum())

    exact_by_normalized: dict[str, list[str]] = defaultdict(list)
    for tag in allowed_tags:
        exact_by_normalized[normalized(tag)].append(tag)

    returned_exact = {tag for tag in result_tags(result) if tag in allowed_tags}
    initially_missing = required_tags - returned_exact
    mapped_aliases: list[dict[str, str]] = []
    removed_unknown: list[str] = []

    def exact_or_safe_replacement(value: str) -> str | None:
        if value in allowed_tags:
            return value
        normalized_matches = exact_by_normalized.get(normalized(value), [])
        if len(normalized_matches) == 1:
            replacement = normalized_matches[0]
            mapped_aliases.append({"from": value, "to": replacement})
            return replacement
        fuzzy_matches = sorted(
            (
                SequenceMatcher(None, normalized(value), normalized(tag)).ratio(),
                tag,
            )
            for tag in initially_missing
        )
        if fuzzy_matches and fuzzy_matches[-1][0] >= 0.94:
            best_score, replacement = fuzzy_matches[-1]
            tied = [score for score, _ in fuzzy_matches if score == best_score]
            if len(tied) == 1:
                mapped_aliases.append({"from": value, "to": replacement})
                return replacement
        removed_unknown.append(value)
        return None

    cleaned_groups: list[list[Any]] = []
    for canonical, aliases in result["r"]:
        cleaned_aliases: list[str] = []
        for alias in aliases:
            replacement = exact_or_safe_replacement(alias)
            if replacement is not None and replacement not in cleaned_aliases:
                cleaned_aliases.append(replacement)
        if cleaned_aliases:
            cleaned_groups.append([canonical, cleaned_aliases])

    cleaned_ambiguities: list[list[str]] = []
    for tag, guess, reason in result["a"]:
        replacement = exact_or_safe_replacement(tag)
        if replacement is not None and all(row[0] != replacement for row in cleaned_ambiguities):
            cleaned_ambiguities.append([replacement, guess, reason])
    ambiguous_tags = {row[0] for row in cleaned_ambiguities}

    group_positions: dict[str, list[int]] = defaultdict(list)
    for group_index, (_, aliases) in enumerate(cleaned_groups):
        for alias in aliases:
            if alias not in ambiguous_tags:
                group_positions[alias].append(group_index)
    preferred_group = {
        tag: max(indices, key=lambda index: (len(cleaned_groups[index][1]), -index))
        for tag, indices in group_positions.items()
    }
    deduplicated: list[str] = []
    final_groups: list[list[Any]] = []
    for group_index, (canonical, aliases) in enumerate(cleaned_groups):
        kept: list[str] = []
        for alias in aliases:
            if alias in ambiguous_tags:
                deduplicated.append(alias)
            elif preferred_group.get(alias) == group_index:
                kept.append(alias)
            else:
                deduplicated.append(alias)
        if kept:
            final_groups.append([canonical, kept])

    covered = {tag for _, aliases in final_groups for tag in aliases} | ambiguous_tags
    added_singletons = sorted(required_tags - covered, key=lambda value: (value.casefold(), value))
    final_groups.extend([[tag, [tag]] for tag in added_singletons])
    conformed = {"r": final_groups, "a": cleaned_ambiguities}
    audit = {
        "method": "mechanical exact-inventory conformance; no new identity merges",
        "mappedAliases": mapped_aliases,
        "removedUnknownAliases": sorted(set(removed_unknown)),
        "deduplicatedTags": sorted(set(deduplicated)),
        "addedAsSingletons": added_singletons,
    }
    validate_delta(conformed, required_tags, allowed_tags)
    return conformed, audit


def apply_delta(
    prior: dict[str, Any], delta: dict[str, Any], cumulative_tags: set[str]
) -> dict[str, Any]:
    """Merge a validated window delta into the prior register without losing old aliases."""
    delta_resolved_tags = {tag for _, aliases in delta["r"] for tag in aliases}
    delta_ambiguous_tags = {row[0] for row in delta["a"]}
    explicitly_mentioned = delta_resolved_tags | delta_ambiguous_tags

    prior_group_by_tag: dict[str, list[str]] = {}
    for _, aliases in prior["r"]:
        for alias in aliases:
            prior_group_by_tag[alias] = aliases

    expanded_groups: list[list[Any]] = []
    expansion_claims: set[str] = set()
    for canonical, aliases in delta["r"]:
        expanded = list(aliases)
        for alias in aliases:
            for prior_alias in prior_group_by_tag.get(alias, []):
                claimed_elsewhere = (
                    prior_alias in explicitly_mentioned and prior_alias not in aliases
                )
                if (
                    not claimed_elsewhere
                    and prior_alias not in delta_ambiguous_tags
                    and prior_alias not in expansion_claims
                    and prior_alias not in expanded
                ):
                    expanded.append(prior_alias)
                    expansion_claims.add(prior_alias)
        expansion_claims.update(expanded)
        expanded_groups.append([canonical, expanded])

    consumed = {tag for _, aliases in expanded_groups for tag in aliases} | delta_ambiguous_tags
    merged_groups: list[list[Any]] = []
    for canonical, aliases in prior["r"]:
        remaining = [alias for alias in aliases if alias not in consumed]
        if remaining:
            merged_groups.append([canonical, remaining])
    merged_groups.extend(expanded_groups)

    merged_ambiguities = [row for row in prior["a"] if row[0] not in consumed]
    merged_ambiguities.extend(delta["a"])
    cumulative = {"r": merged_groups, "a": merged_ambiguities}
    validate_result(cumulative, cumulative_tags)
    return cumulative


def render_review(
    window: dict[str, Any], result: dict[str, Any], validation: dict[str, Any]
) -> str:
    """Render only merges and ambiguities for quick human review; JSON retains singletons."""
    lines = [
        f"# Contextual entity resolution: {window['pages'][0]} to {window['pages'][-1]}",
        "",
        f"- Pages in window: {len(window['pages'])}",
        f"- Sentences: {window['sentenceCount']}",
        f"- Current-window raw tags: {window['entityCount']}",
        f"- Cumulative covered tags: {validation['coveredTags']}",
        f"- Resolved groups: {validation['resolvedGroups']}",
        f"- Multi-alias groups: {validation['multiAliasGroups']}",
        f"- Manual-review tags: {validation['ambiguousTags']}",
        "",
        "## Resolved groups with multiple tags",
        "",
    ]
    multi_alias = [group for group in result["r"] if len(group[1]) > 1]
    if multi_alias:
        for canonical, aliases in multi_alias:
            lines.append(f"- {canonical} <- {' | '.join(aliases)}")
    else:
        lines.append("- None")
    lines.extend(["", "## Manual review", ""])
    if result["a"]:
        for tag, guess, reason in result["a"]:
            lines.append(f"- {tag} -> {guess} — {reason}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_json(path: Path, value: Any) -> None:
    """Write readable UTF-8 JSON for audit and downstream reuse."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def submit_window(
    client: genai.Client,
    window_number: int,
    window: dict[str, Any],
    prior: dict[str, Any] | None,
    cumulative_expected_tags: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Submit one window, preserve the raw response, and validate exact tag coverage."""
    output_dir = OUTPUT / f"window_{window_number:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(window, prior)
    (output_dir / "prompt_sent.txt").write_text(prompt, encoding="utf-8", newline="\n")
    write_json(
        output_dir / "window_manifest.json",
        {
            "pages": window["pages"],
            "sentenceCount": window["sentenceCount"],
            "currentEntityCount": window["entityCount"],
            "expectedCumulativeEntityCount": len(cumulative_expected_tags),
            "responseMode": "complete" if prior is None else "delta",
            "promptCharacters": len(prompt),
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
                include_thoughts=False,
            ),
        ),
    )
    raw_response = response.model_dump(mode="json", exclude_none=True)
    write_json(output_dir / "raw_response.json", raw_response)
    (output_dir / "response_text.json").write_text(response.text, encoding="utf-8", newline="\n")

    model_result = parse_result(response.text)
    conformance_audit: dict[str, Any] = {}
    if prior is None:
        try:
            validate_result(model_result, cumulative_expected_tags)
        except RuntimeError:
            write_json(output_dir / "result_model_raw.json", model_result)
            model_result, conformance_audit = mechanically_conform_result(
                model_result, cumulative_expected_tags
            )
            write_json(output_dir / "mechanical_conformance.json", conformance_audit)
        result = model_result
    else:
        current_tags = set(window["entities"])
        try:
            validate_delta(model_result, current_tags, cumulative_expected_tags)
        except RuntimeError:
            write_json(output_dir / "delta_result_model_raw.json", model_result)
            model_result, conformance_audit = mechanically_conform_result(
                model_result,
                cumulative_expected_tags,
                required_tags=current_tags,
            )
            write_json(output_dir / "mechanical_conformance.json", conformance_audit)
        validate_delta(model_result, current_tags, cumulative_expected_tags)
        write_json(output_dir / "delta_result.json", model_result)
        result = apply_delta(prior, model_result, cumulative_expected_tags)

    write_json(output_dir / "result.json", result)
    validation = validate_result(result, cumulative_expected_tags)
    write_json(output_dir / "validation.json", validation)
    (output_dir / "review.md").write_text(
        render_review(window, result, validation), encoding="utf-8", newline="\n"
    )
    usage = (
        response.usage_metadata.model_dump(mode="json")
        if response.usage_metadata is not None
        else {}
    )
    combined_usage = {"initial": usage}
    write_json(output_dir / "usage.json", combined_usage)
    return result, {"validation": validation, "usage": combined_usage}


def write_entity_index(windows: list[dict[str, Any]]) -> None:
    """Write a compact inventory showing which raw tags enter in each page window."""
    with (OUTPUT / "window_entity_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("window", "entity"))
        for window_number, window in enumerate(windows, start=1):
            for entity in window["entities"]:
                writer.writerow((window_number, entity))


def main() -> None:
    """Build, submit, validate, and package the first two contextual windows."""
    if not api_key():
        raise RuntimeError("GEMINI_API_KEY is not configured")

    pages, sentences, triples = load_corpus()
    selected_pages = pages[:PAGE_COUNT]
    windows = [
        build_window(selected_pages[start : start + WINDOW_SIZE], sentences, triples)
        for start in range(0, PAGE_COUNT, WINDOW_SIZE)
    ]
    OUTPUT.mkdir(exist_ok=True)
    write_entity_index(windows)

    client = genai.Client(api_key=api_key())
    prior: dict[str, Any] | None = None
    cumulative_tags: set[str] = set()
    run_rows: list[dict[str, Any]] = []
    for window_number, window in enumerate(windows, start=1):
        cumulative_tags.update(window["entities"])
        output_dir = OUTPUT / f"window_{window_number:02d}"
        existing_result_path = output_dir / "result.json"
        if existing_result_path.exists():
            prior = json.loads(existing_result_path.read_text(encoding="utf-8"))
            conformance_audit: dict[str, Any] = {}
            try:
                validation = validate_result(prior, cumulative_tags)
            except RuntimeError:
                write_json(output_dir / "result_model_raw.json", prior)
                prior, conformance_audit = mechanically_conform_result(prior, cumulative_tags)
                write_json(output_dir / "mechanical_conformance.json", conformance_audit)
                write_json(output_dir / "result.json", prior)
                validation = validate_result(prior, cumulative_tags)
            raw_response_path = output_dir / "raw_response.json"
            initial_usage: dict[str, Any] = {}
            if raw_response_path.exists():
                raw_response = json.loads(raw_response_path.read_text(encoding="utf-8"))
                initial_usage = raw_response.get("usage_metadata", {})
            combined_usage = {"initial": initial_usage}
            write_json(output_dir / "validation.json", validation)
            write_json(output_dir / "usage.json", combined_usage)
            (output_dir / "review.md").write_text(
                render_review(window, prior, validation),
                encoding="utf-8",
                newline="\n",
            )
            metadata = {"validation": validation, "usage": combined_usage}
        else:
            prior, metadata = submit_window(client, window_number, window, prior, cumulative_tags)
        run_rows.append(
            {
                "window": window_number,
                "pages": window["pages"],
                "sentenceCount": window["sentenceCount"],
                "currentEntityCount": window["entityCount"],
                **metadata,
            }
        )

    if prior is None:
        raise RuntimeError("No contextual windows were submitted")
    write_json(OUTPUT / "cumulative_result_after_40_pages.json", prior)
    write_json(
        OUTPUT / "run_manifest.json",
        {
            "status": "completed_with_research_warnings",
            "completedAt": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "thinkingLevel": "low",
            "temperature": 0.0,
            "windowSize": WINDOW_SIZE,
            "pageCount": PAGE_COUNT,
            "outputSchema": {"r": "resolved groups", "a": "manual review"},
            "windows": run_rows,
        },
    )
    print(
        json.dumps(
            {
                "status": "completed_with_research_warnings",
                "output": str(OUTPUT),
                "windows": run_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
