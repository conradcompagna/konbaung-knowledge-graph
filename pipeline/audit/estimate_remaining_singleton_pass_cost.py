#!/usr/bin/env python3
"""Estimate a singleton cascade from the completed pass and current pruned graph."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from run_konbaung_binary_resolution_production import binary_prompt


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "konbaung_nonsingleton_top50_frequency_wave_gemini_trial_20260902_first100"
SOURCE = ROOT / "konbaung_entity_resolution_candidates_20260902_v4_nonsingleton_top50"
INPUT_RATE = 0.25 / 1_000_000
OUTPUT_RATE = 1.50 / 1_000_000


# Load a CSV as dictionaries while preserving the canonical row order.
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# Fit exact prior token counts to prompt length and candidate count for local forecasting.
def fit_prompt_token_model() -> tuple[np.ndarray, dict[str, float]]:
    observations: list[tuple[float, float, float]] = []
    errors: list[float] = []
    for path in RUN.rglob("page_manifest.json"):
        page = json.loads(path.read_text(encoding="utf-8"))
        candidate_count = page.get("activeCandidatesSent", page.get("candidatesSent"))
        observations.append(
            (
                float(page["promptCharacters"]),
                float(candidate_count),
                float(page["usage"]["prompt_token_count"]),
            )
        )
    matrix = np.asarray([[1.0, chars, count] for chars, count, _ in observations])
    targets = np.asarray([tokens for _, _, tokens in observations])
    coefficients, *_ = np.linalg.lstsq(matrix, targets, rcond=None)
    predictions = matrix @ coefficients
    residuals = targets - predictions
    metrics = {
        "observations": float(len(observations)),
        "meanAbsoluteErrorTokens": float(np.mean(np.abs(residuals))),
        "rootMeanSquaredErrorTokens": float(np.sqrt(np.mean(residuals**2))),
        "rSquared": float(1.0 - np.sum(residuals**2) / np.sum((targets - np.mean(targets)) ** 2)),
    }
    return coefficients, metrics


# Predict input tokens with the regression calibrated on every completed Gemini call.
def predict_prompt_tokens(coefficients: np.ndarray, prompt: str, candidates: int) -> float:
    return max(0.0, float(coefficients @ np.asarray([1.0, len(prompt), candidates])))


# Reproduce frequency-wave pruning under the conservative assumption that every call says no.
def simulate_no_match_cascade(
    roster: list[dict[str, str]],
    pages: dict[str, list[dict[str, str]]],
    entity_to_id: dict[str, str],
    coefficients: np.ndarray,
) -> dict[str, float]:
    active = {row["id"] for row in roster}
    calls = 0
    waves = 0
    prompt_tokens = 0.0
    candidate_rows = 0
    while active:
        used: set[str] = set()
        selected: list[tuple[dict[str, str], list[dict[str, str]]]] = []
        for parent in roster:
            parent_id = parent["id"]
            if parent_id not in active:
                continue
            candidates = [
                candidate
                for candidate in pages[parent_id]
                if entity_to_id[candidate["entity"]] in active
            ]
            footprint = {parent_id}
            footprint.update(entity_to_id[candidate["entity"]] for candidate in candidates)
            if not footprint.isdisjoint(used):
                continue
            selected.append((parent, candidates))
            used.update(footprint)
        if not selected:
            raise RuntimeError("No singleton page could be selected from a non-empty roster.")
        waves += 1
        for parent, candidates in selected:
            prompt = binary_prompt(parent["entity"], candidates)
            prompt_tokens += predict_prompt_tokens(coefficients, prompt, len(candidates))
            candidate_rows += len(candidates)
        calls += len(selected)
        active.difference_update(parent["id"] for parent, _ in selected)
    return {
        "waves": float(waves),
        "calls": float(calls),
        "candidateRowsSent": float(candidate_rows),
        "promptTokens": prompt_tokens,
    }


# Calculate current graph sizes, an all-no ceiling, and empirical pruning scenarios.
def main() -> None:
    index = read_csv(SOURCE / "index.csv")
    entity_to_row = {row["entity"]: row for row in index}
    if len(entity_to_row) != len(index):
        raise RuntimeError("Exact entity strings are not unique in the source index.")
    entity_to_id = {entity: row["id"] for entity, row in entity_to_row.items()}
    remaining = read_csv(RUN / "remaining_singletons_after_nonsingleton_completion.csv")
    remaining_names = {row["entity"] for row in remaining}
    pages: dict[str, list[dict[str, str]]] = {}
    initial_prompt_characters = 0
    initial_candidate_rows = 0
    for parent in remaining:
        candidates = [
            row
            for row in read_csv(SOURCE / "entities" / parent["file"])
            if row["entity"] in remaining_names
        ]
        pages[parent["id"]] = candidates
        initial_candidate_rows += len(candidates)
        initial_prompt_characters += len(binary_prompt(parent["entity"], candidates))

    coefficients, model_metrics = fit_prompt_token_model()
    no_match = simulate_no_match_cascade(remaining, pages, entity_to_id, coefficients)

    clusters = json.loads(
        (RUN / "resolved_clusters_after_nonsingleton_completion.json").read_text(encoding="utf-8")
    )
    top_twenty_eligible_aliases = 0
    eligible_aliases = 0
    for cluster in clusters:
        parent_file = entity_to_row[cluster["parentEntity"]]["file"]
        source_candidates = read_csv(SOURCE / "entities" / parent_file)
        rank_by_entity = {
            row["entity"]: rank for rank, row in enumerate(source_candidates, start=1)
        }
        for alias in cluster["members"][1:]:
            if int(entity_to_row[alias]["mentions"]) <= 1:
                continue
            eligible_aliases += 1
            if rank_by_entity[alias] <= 20:
                top_twenty_eligible_aliases += 1

    prior_calls = int(
        json.loads((RUN / "full_nonsingleton_run_manifest.json").read_text())["totalCalls"]
    )
    top_twenty_aliases_per_call = top_twenty_eligible_aliases / prior_calls
    expected_calls = len(remaining) / (1.0 + top_twenty_aliases_per_call)
    expected_fraction = expected_calls / len(remaining)
    expected_prompt_tokens = no_match["promptTokens"] * expected_fraction

    prior_prompt_tokens = 0
    prior_output_tokens = 0
    empty_output_tokens: list[int] = []
    positive_output_tokens: list[int] = []
    for path in RUN.rglob("page_manifest.json"):
        page = json.loads(path.read_text(encoding="utf-8"))
        output_tokens = int(page["usage"]["candidates_token_count"])
        result = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))
        prior_prompt_tokens += int(page["usage"]["prompt_token_count"])
        prior_output_tokens += output_tokens
        (positive_output_tokens if result["y"] else empty_output_tokens).append(output_tokens)
    prior_output_per_call = prior_output_tokens / prior_calls
    expected_output_tokens = expected_calls * prior_output_per_call

    all_no_output_tokens = len(remaining) * float(np.mean(empty_output_tokens))
    no_match_cost = no_match["promptTokens"] * INPUT_RATE + all_no_output_tokens * OUTPUT_RATE
    expected_cost = expected_prompt_tokens * INPUT_RATE + expected_output_tokens * OUTPUT_RATE

    print(
        json.dumps(
            {
                "remainingSingletons": len(remaining),
                "currentGraph": {
                    "candidateRows": initial_candidate_rows,
                    "averageCandidatesPerPage": initial_candidate_rows / len(remaining),
                    "promptCharactersIfAllSentNow": initial_prompt_characters,
                },
                "tokenModel": {
                    "coefficients": coefficients.tolist(),
                    **model_metrics,
                },
                "allNoConservativeCeiling": {
                    **no_match,
                    "estimatedOutputTokens": all_no_output_tokens,
                    "estimatedStandardUsd": no_match_cost,
                },
                "observedTop20PruningScenario": {
                    "priorEligibleAliasesAtAnyRank": eligible_aliases,
                    "priorEligibleAliasesWithinSourceTop20": top_twenty_eligible_aliases,
                    "top20EligibleAliasesPerCall": top_twenty_aliases_per_call,
                    "estimatedCalls": expected_calls,
                    "estimatedCallsAvoided": len(remaining) - expected_calls,
                    "estimatedPromptTokens": expected_prompt_tokens,
                    "estimatedOutputTokens": expected_output_tokens,
                    "estimatedStandardUsd": expected_cost,
                },
                "priorOutput": {
                    "averageTokensPerCall": prior_output_per_call,
                    "averageEmptyTokens": float(np.mean(empty_output_tokens)),
                    "averagePositiveTokens": float(np.mean(positive_output_tokens)),
                },
                "ratesUsdPerMillion": {"input": 0.25, "output": 1.50},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
