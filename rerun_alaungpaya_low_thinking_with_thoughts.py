#!/usr/bin/env python3
"""Rerun the mistaken Alaungpaya page at low thinking with thoughts returned."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from konbaung_gemini_summary_claim_completion_annotator import api_key
from run_konbaung_top20_resolution_trial import Decision, validate_decision


ROOT = Path(__file__).resolve().parent
ORIGINAL_PASS = (
    ROOT
    / "konbaung_top20_gemini_resolution_trial_20260902_first10"
    / "pass_02_e00001_Alaungpaya"
)
OUTPUT = ROOT / "konbaung_top20_alaungpaya_low_thinking_thoughts_20260902"
MODEL = "gemini-3.1-flash-lite"
MAX_OUTPUT_TOKENS = 2000
TARGET_ERROR = "Nga Lwan Ya"


def write_json(path: Path, value: Any) -> None:
    """Write readable JSON with Unicode entity names preserved."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def response_parts(response: Any) -> tuple[str, list[str]]:
    """Separate Gemini's JSON answer from its explicitly returned thought summaries."""
    answer_parts: list[str] = []
    thought_parts: list[str] = []
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            text = getattr(part, "text", None)
            if not text:
                continue
            if getattr(part, "thought", False):
                thought_parts.append(text)
            else:
                answer_parts.append(text)
    return "".join(answer_parts).strip(), thought_parts


def mistake_status(decision: Decision) -> str:
    """Classify whether the rerun still falsely resolves Nga Lwan Ya as Alaungpaya."""
    if TARGET_ERROR in decision.s:
        return "NOT_CAUGHT: Nga Lwan Ya was again merged with Alaungpaya."
    if TARGET_ERROR in decision.u:
        return "PARTLY_CAUGHT: Nga Lwan Ya was withheld as uncertain rather than merged."
    return "CAUGHT: Nga Lwan Ya was correctly omitted as a different entity."


def main() -> None:
    """Make one controlled API call using the original prompt and save an A/B review."""
    if not api_key():
        raise RuntimeError("GEMINI_API_KEY is not configured")
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite an existing rerun: {OUTPUT}")

    prompt = (ORIGINAL_PASS / "prompt_sent.txt").read_text(encoding="utf-8")
    original_manifest = json.loads(
        (ORIGINAL_PASS / "page_manifest.json").read_text(encoding="utf-8")
    )
    original_result = json.loads(
        (ORIGINAL_PASS / "result.json").read_text(encoding="utf-8")
    )
    candidates = original_manifest["activeCandidates"]

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "prompt_sent.txt").write_text(
        prompt, encoding="utf-8", newline="\n"
    )

    client = genai.Client(api_key=api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=Decision.model_json_schema(),
            temperature=0.0,
            candidate_count=1,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.LOW,
                include_thoughts=True,
            ),
        ),
    )
    write_json(
        OUTPUT / "raw_response.json",
        response.model_dump(mode="json", exclude_none=True),
    )

    answer, thoughts = response_parts(response)
    (OUTPUT / "answer_returned.json").write_text(
        answer + "\n", encoding="utf-8", newline="\n"
    )
    (OUTPUT / "returned_thoughts.txt").write_text(
        "\n\n".join(thoughts).strip() + "\n",
        encoding="utf-8",
        newline="\n",
    )
    decision = Decision.model_validate_json(answer)
    validate_decision(decision, "Alaungpaya", candidates)
    write_json(OUTPUT / "result.json", decision.model_dump(mode="json"))

    usage = (
        response.usage_metadata.model_dump(mode="json")
        if response.usage_metadata is not None
        else {}
    )
    status = mistake_status(decision)
    manifest = {
        "status": status.split(":", 1)[0].lower(),
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "parent": "Alaungpaya",
        "originalPass": str(ORIGINAL_PASS),
        "controlledComparison": {
            "samePrompt": True,
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "originalThinkingLevel": "minimal",
            "rerunThinkingLevel": "low",
            "originalIncludeThoughts": False,
            "rerunIncludeThoughts": True,
            "temperature": 0.0,
        },
        "usage": usage,
        "returnedThoughtParts": len(thoughts),
        "mistakeAssessment": status,
    }
    write_json(OUTPUT / "run_manifest.json", manifest)

    original_same = " | ".join(original_result["s"]) or "none"
    original_uncertain = " | ".join(original_result["u"]) or "none"
    rerun_same = " | ".join(decision.s) or "none"
    rerun_uncertain = " | ".join(decision.u) or "none"
    thought_text = "\n\n".join(thoughts).strip() or "No thought summary was returned."
    review = f"""# Alaungpaya one-page low-thinking rerun

{status}

- Model: `{MODEL}`
- Prompt: byte-for-byte identical to original pass 2
- Original thinking: `MINIMAL`, thoughts not returned
- Rerun thinking: `LOW`, returned thoughts enabled
- Prompt tokens: {int(usage.get('prompt_token_count') or 0)}
- Answer tokens: {int(usage.get('candidates_token_count') or 0)}
- Thinking tokens: {int(usage.get('thoughts_token_count') or 0)}
- Total tokens: {int(usage.get('total_token_count') or 0)}

## Original decision

- Same: {original_same}
- Uncertain: {original_uncertain}

## Rerun decision

- Same: {rerun_same}
- Uncertain: {rerun_uncertain}

## Returned thought summary

{thought_text}
"""
    (OUTPUT / "REVIEW.md").write_text(
        review, encoding="utf-8", newline="\n"
    )
    print(review)


if __name__ == "__main__":
    main()
