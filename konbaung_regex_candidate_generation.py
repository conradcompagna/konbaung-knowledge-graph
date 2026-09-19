from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\conra\Desktop\dighumproject")
SOURCE = ROOT / "konbaung_node_identity_statistical_refinement_20260725"
FIRST_PASS = ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
DATASET = ROOT / "konbaung_dataset_v3_20260718"
OUTPUT = ROOT / "konbaung_node_identity_regex_refinement_20260725"

CLUSTERS_PATH = SOURCE / "corrected_initial_clusters.jsonl"
NODE_RECORDS_PATH = FIRST_PASS / "node_records.jsonl"
TRANSLATIONS_PATH = (
    DATASET / "sentence_translations" / "translations" / "all_volumes.jsonl"
)
EXPECTED_CLUSTERS = 19_002

TITLE_WORDS = {
    "prince",
    "princess",
    "king",
    "queen",
    "governor",
    "commander",
    "minister",
    "sayadaw",
    "sawbwa",
    "lord",
}
PLACE_SUFFIXES = {"town", "city", "village", "capital"}


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def tokens(value: str) -> tuple[str, ...]:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("&", " and ")
    value = "".join(char if char.isalnum() else " " for char in value)
    return tuple(value.split())


def strip_diacritics(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    return "".join(char for char in value if not unicodedata.combining(char))


def singular(token: str) -> str:
    irregular = {
        "people": "person",
        "men": "man",
        "women": "woman",
        "children": "child",
        "monks": "monk",
        "sanghas": "sangha",
    }
    if token in irregular:
        return irregular[token]
    if len(token) <= 4:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def title_key(row: tuple[str, ...]) -> str | None:
    if len(row) < 2:
        return None
    if row[0] in TITLE_WORDS:
        title = row[0]
        rest = row[2:] if len(row) > 2 and row[1] == "of" else row[1:]
    elif row[-1] in TITLE_WORDS:
        title = row[-1]
        rest = row[:-1]
    else:
        return None
    if rest and rest[-1] in {"town", "city"}:
        rest = rest[:-1]
    if not rest:
        return None
    return f"{title}:{' '.join(rest)}"


def governor_key(row: tuple[str, ...]) -> str | None:
    reduced = list(row)
    if reduced[:2] == ["governor", "of"]:
        reduced = reduced[2:] + ["governor"]
    if not reduced or reduced[-1] != "governor":
        return None
    place = reduced[:-1]
    if place and place[-1] in {"town", "city"}:
        place = place[:-1]
    if not place:
        return None
    return "governor:" + " ".join(place)


def keys_for(value: str) -> dict[str, str]:
    row = tokens(value)
    if not row:
        return {}
    result = {
        "format_exact": " ".join(row),
        "compact_exact": "".join(row),
        "token_order_exact": " ".join(sorted(row)),
    }
    no_article = row[1:] if row[0] in {"the", "a", "an"} else row
    result["leading_article_exact"] = " ".join(no_article)
    no_conjunction = tuple(token for token in row if token not in {"and"})
    result["conjunction_format_exact"] = " ".join(no_conjunction)
    result["singular_format_exact"] = " ".join(singular(token) for token in row)
    result["singular_token_order_exact"] = " ".join(
        sorted(singular(token) for token in row)
    )
    diacritic_row = tokens(strip_diacritics(value))
    result["diacritic_exact"] = " ".join(diacritic_row)
    possible_title = title_key(row)
    if possible_title:
        result["title_order_template"] = possible_title
    possible_governor = governor_key(row)
    if possible_governor:
        result["governor_template"] = possible_governor
    if row[-1] in PLACE_SUFFIXES and len(row) > 1:
        result["place_suffix_template"] = " ".join(row[:-1])
    return {rule: key for rule, key in result.items() if key}


def candidate_context(
    cluster: dict,
    records_by_tag: dict[str, dict],
    translations: dict[str, str],
) -> list[dict]:
    contexts: list[dict] = []
    seen: set[str] = set()
    for tag in cluster["memberTags"]:
        record = records_by_tag.get(tag)
        if not record:
            continue
        for occurrence in record.get("sampleOccurrences", [])[:2]:
            sid = occurrence["sid"]
            if sid in seen:
                continue
            seen.add(sid)
            contexts.append(
                {
                    "tag": tag,
                    "sid": sid,
                    "sentenceEn": translations.get(sid, ""),
                }
            )
            if len(contexts) == 3:
                return contexts
    return contexts


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    clusters = list(jsonl(CLUSTERS_PATH))
    if len(clusters) != EXPECTED_CLUSTERS:
        raise RuntimeError(f"Expected {EXPECTED_CLUSTERS:,} clusters")
    records_by_tag = {
        row["tag"]: row for row in jsonl(NODE_RECORDS_PATH)
    }
    translations = {
        row["id"]: row["en"] for row in jsonl(TRANSLATIONS_PATH)
    }

    postings: dict[tuple[str, str], set[int]] = defaultdict(set)
    witnesses: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for cluster_index, cluster in enumerate(clusters):
        labels = {cluster["canonicalLabel"], *cluster["memberTags"]}
        for label in labels:
            for rule, key in keys_for(label).items():
                postings[(rule, key)].add(cluster_index)
                witnesses[(rule, key, cluster_index)].add(label)
            # Exact deletion signatures retrieve strings separated by one
            # insertion, deletion, substitution, or adjacent transposition.
            # They only surface review candidates; they never authorize a merge.
            compact = "".join(tokens(label))
            if 6 <= len(compact) <= 60:
                numeric_signature = ",".join(
                    token for token in tokens(label) if token.isdigit()
                )
                edit_keys = {compact}
                edit_keys.update(
                    compact[:position] + compact[position + 1 :]
                    for position in range(len(compact))
                )
                for key in edit_keys:
                    key = f"{numeric_signature}|{key}"
                    postings[("one_char_edit", key)].add(cluster_index)
                    witnesses[("one_char_edit", key, cluster_index)].add(label)

    pair_evidence: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for (rule, key), members in postings.items():
        ordered = sorted(members)
        if len(ordered) < 2:
            continue
        # Extremely broad collisions are not useful deterministic candidates.
        if len(ordered) > 30:
            continue
        for left_position, left in enumerate(ordered):
            for right in ordered[left_position + 1 :]:
                pair_evidence[(left, right)].append(
                    {
                        "rule": rule,
                        "normalizedKey": key,
                        "leftWitnesses": sorted(
                            witnesses[(rule, key, left)]
                        ),
                        "rightWitnesses": sorted(
                            witnesses[(rule, key, right)]
                        ),
                    }
                )

    rule_priority = {
        "format_exact": 0,
        "diacritic_exact": 1,
        "compact_exact": 2,
        "leading_article_exact": 3,
        "token_order_exact": 4,
        "title_order_template": 5,
        "governor_template": 6,
        "conjunction_format_exact": 7,
        "singular_format_exact": 8,
        "singular_token_order_exact": 9,
        "place_suffix_template": 10,
        "one_char_edit": 11,
    }
    candidates = []
    for (left, right), evidence in pair_evidence.items():
        evidence.sort(key=lambda row: (rule_priority[row["rule"]], row["rule"]))
        left_cluster = clusters[left]
        right_cluster = clusters[right]
        candidates.append(
            {
                "leftClusterId": left_cluster["clusterId"],
                "leftCanonical": left_cluster["canonicalLabel"],
                "leftMembers": left_cluster["memberTags"],
                "rightClusterId": right_cluster["clusterId"],
                "rightCanonical": right_cluster["canonicalLabel"],
                "rightMembers": right_cluster["memberTags"],
                "primaryRule": evidence[0]["rule"],
                "allEvidence": evidence,
                "leftContexts": candidate_context(
                    left_cluster, records_by_tag, translations
                ),
                "rightContexts": candidate_context(
                    right_cluster, records_by_tag, translations
                ),
            }
        )
    candidates.sort(
        key=lambda row: (
            rule_priority[row["primaryRule"]],
            row["leftCanonical"].casefold(),
            row["rightCanonical"].casefold(),
            row["leftClusterId"],
            row["rightClusterId"],
        )
    )

    with (OUTPUT / "regex_candidate_pool.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in candidates:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )

    counts: dict[str, int] = defaultdict(int)
    for row in candidates:
        counts[row["primaryRule"]] += 1
    report = {
        "method": "deterministic regex/string normalization only",
        "embeddingInputsUsed": 0,
        "clusters": len(clusters),
        "candidatePairs": len(candidates),
        "candidatePairsByPrimaryRule": dict(counts),
        "status": "AWAITING_MANUAL_ADJUDICATION",
    }
    (OUTPUT / "regex_candidate_generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Regex-only candidate pool",
        "",
        "This is a review pool, not an accepted merge list.",
        "",
    ]
    for row in candidates:
        lines.extend(
            [
                f"## {row['leftCanonical']} ↔ {row['rightCanonical']}",
                "",
                f"Rule: `{row['primaryRule']}`",
                "",
                f"- Left aliases: {'; '.join(row['leftMembers'])}",
                f"- Right aliases: {'; '.join(row['rightMembers'])}",
                "",
            ]
        )
    (OUTPUT / "REGEX_CANDIDATE_POOL.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
