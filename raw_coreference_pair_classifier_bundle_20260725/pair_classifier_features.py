#!/usr/bin/env python3
"""Feature engineering and deterministic gates for strict tag coreference.

This module intentionally models identity normalization, not semantic relatedness.
It is imported by both the training and scoring scripts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
from rapidfuzz import fuzz
from unidecode import unidecode

NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
    "hundred thousand million first second third fourth fifth sixth seventh eighth ninth tenth "
    "eleventh twelfth".split()
)
STOPWORDS = frozenset("the of as at in on for to from and or with by a an".split())
GENERIC_TITLES = frozenset(
    "mingyi maha min thado thadoe nemyo naymyo ne myo prince princess king queen minister governor "
    "sayadaw shin lord chief royal city town village commander wundauk myosa myoza wun bo clerk u nga "
    "maung lady sir siri thiri zeya thiha kyaw htin sithu nawrahta raza yaza pyan chi thura thihathu "
    "minkhaung minhkaung minhla hla sawbwa regiment unit troops force forces group officials people "
    "men soldiers cavalry musketeer musketeers monk monks sangha".split()
)

VARIANT_MAP = {
    "myoza": "myosa",
    "naymyo": "nemyo",
    "minhgaung": "minhkaung",
    "pawara": "pavara",
    "thurain": "thurein",
    "kyawzwa": "kyawswa",
    "kyawswar": "kyawswa",
    "hlahtwe": "hlatwe",
    "oakkyaung": "okkyaung",
    "thadoe": "thado",
    "pannadipa": "panyadipa",
    "myinkhuntaing": "myinkhontine",
}

DIRECTION_AXES = (
    (frozenset(("left", "letwe", "letwa")), frozenset(("right", "letya"))),
    (frozenset(("north", "northern", "northeast", "northwest")), frozenset(("south", "southern", "southeast", "southwest"))),
    (frozenset(("east", "eastern", "northeast", "southeast")), frozenset(("west", "western", "northwest", "southwest"))),
    (frozenset(("front", "forward")), frozenset(("rear", "back"))),
    (frozenset(("inner", "interior")), frozenset(("outer", "exterior"))),
    (frozenset(("upper", "upstream")), frozenset(("lower", "downstream"))),
    (frozenset(("elder", "older", "eldest")), frozenset(("younger", "youngest"))),
    (frozenset(("winning", "winner")), frozenset(("losing", "loser"))),
    (frozenset(("male",)), frozenset(("female",))),
)
KINSHIP = frozenset("son daughter sister brother niece nephew mother father wife husband queen consort grandfather grandmother uncle aunt".split())
GROUP_MARKERS = frozenset(
    "unit regiment battalion troop troops force forces family followers associates residents personnel staff people men soldiers "
    "cavalry musketeer musketeers commanders ministers officials monks sayadaws sangha group corps fleet".split()
)
HARD_GROUP_MARKERS = frozenset(
    "unit regiment battalion troop troops force forces family followers associates residents personnel staff group corps fleet".split()
)
EVENT_MARKERS = frozenset(
    "campaign expedition battle assault attack capture arrest ceremony festival festivals prophecy funeral consecration "
    "offering offerings merit making appointment rebellion procession ritual construction birth death".split()
)
PLACE_MARKERS = frozenset(
    "city town village fort pagoda monastery river stream palace hill mountain region state kingdom dynasty canal lake ward "
    "district route camp portico chamber face side capital".split()
)
PERSON_MARKERS = frozenset(
    "prince princess king queen minister governor sayadaw shin lord chief commander clerk sawbwa myosa myoza maung nga u "
    "lady wife son daughter sister brother niece nephew".split()
)
OBJECT_MARKERS = frozenset(
    "elephant elephants horse horses boat boats barge umbrella weapon weapons cannon cannons musket muskets statue image "
    "images tree trees robe robes coin coins textile textiles".split()
)
INSTITUTION_MARKERS = frozenset("court hluttaw government authorities monarchy dynasty state sangha".split())

CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
NONALNUM_RE = re.compile(r"[^a-z0-9]+")
POSSESSIVE_RE = re.compile(r"\b([A-Za-z]+)'s\b")
NUMERIC_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")

FEATURE_NAMES = [
    "raw_similarity", "log_rank_a", "log_rank_b", "log_min_rank", "log_max_rank", "reciprocal_rank_sum",
    "top_delta_a", "top_delta_b", "log_degree_a", "log_degree_b", "char_ratio", "token_sort_ratio",
    "token_set_ratio", "partial_ratio", "variant_char_ratio", "token_jaccard", "token_overlap_min",
    "token_overlap_max", "content_jaccard", "idf_jaccard", "variant_jaccard", "length_ratio",
    "length_diff_ratio", "token_count_diff", "shared_token_count", "symmetric_diff_count",
    "extra_generic_fraction", "exact_norm", "exact_compact", "exact_token_multiset", "exact_variant_compact",
    "exact_variant_multiset", "substring", "token_subset", "first_token_same", "last_token_same",
    "and_set_equal", "max_shared_idf", "sum_shared_idf", "rare_shared_count", "rare_jaccard",
    "has_rare_shared", "numeric_conflict", "numberword_conflict", "direction_conflict", "kinship_conflict",
    "group_mismatch", "event_mismatch", "place_person_mismatch", "type_jaccard", "type_disjoint",
    "type_equal", "mutual_top1", "mutual_top5", "mutual_top10", "mutual_top20", "mutual_top100",
]


@dataclass(frozen=True)
class CandidateRow:
    score: float
    left: str
    right: str


@dataclass(frozen=True)
class TagMeta:
    raw: str
    norm: str
    tokens: Tuple[str, ...]
    token_set: frozenset[str]
    sorted_tokens: str
    compact: str
    variant: Tuple[str, ...]
    variant_sorted: str
    variant_compact: str
    content_set: frozenset[str]
    numbers: frozenset[str]
    number_words: frozenset[str]
    kinship: frozenset[str]
    groups: frozenset[str]
    events: frozenset[str]
    places: frozenset[str]
    persons: frozenset[str]
    objects: frozenset[str]
    institutions: frozenset[str]


@dataclass
class CorpusStats:
    metas: Dict[str, TagMeta]
    degrees: Counter
    top_scores: Dict[str, float]
    token_df: Counter
    token_idf: Dict[str, float]
    band_counts: Counter
    pair_count: int
    rare_cutoff: int = 50


def iter_candidates(path: str | Path) -> Iterator[CandidateRow]:
    """Stream candidates from the markdown file without loading it into memory."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("- **"):
                continue
            marker = line.find("%** — ")
            if marker < 0:
                continue
            try:
                score = float(line[4:marker])
                left, right = line[marker + 6 :].rstrip("\n").split(" ↔ ", 1)
            except (ValueError, IndexError):
                continue
            yield CandidateRow(score, left, right)


def basic_normalize(value: str) -> str:
    value = CAMEL_RE.sub(" ", value)
    value = unidecode(value)
    value = POSSESSIVE_RE.sub(r"\1", value)
    value = value.replace("&", " and ")
    value = NUMERIC_COMMA_RE.sub("", value)
    value = NONALNUM_RE.sub(" ", value.lower())
    return " ".join(value.split())


def variant_tokens(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    out = []
    i = 0
    bigrams = {
        "min hla": "minhla",
        "min htin": "minhtin",
        "min khaung": "minkhaung",
        "min kyaw": "minkyaw",
        "kyaw swa": "kyawswa",
        "kyaw zwa": "kyawswa",
        "shwe taung": "shwetaung",
        "ne myo": "nemyo",
        "ok kyaung": "okkyaung",
        "nakhan daw": "nakhandaw",
        "myin gun": "myingun",
    }
    while i < len(tokens):
        if i + 1 < len(tokens):
            replacement = bigrams.get(tokens[i] + " " + tokens[i + 1])
            if replacement:
                out.append(replacement)
                i += 2
                continue
        out.append(VARIANT_MAP.get(tokens[i], tokens[i]))
        i += 1
    return tuple(out)


def make_meta(value: str) -> TagMeta:
    norm = basic_normalize(value)
    tokens = tuple(norm.split())
    token_set = frozenset(tokens)
    variant = variant_tokens(tokens)
    return TagMeta(
        raw=value,
        norm=norm,
        tokens=tokens,
        token_set=token_set,
        sorted_tokens=" ".join(sorted(tokens)),
        compact="".join(tokens),
        variant=variant,
        variant_sorted=" ".join(sorted(variant)),
        variant_compact="".join(variant),
        content_set=frozenset(t for t in tokens if t not in STOPWORDS),
        numbers=frozenset(t for t in tokens if t.isdigit()),
        number_words=frozenset(t for t in tokens if t in NUMBER_WORDS),
        kinship=frozenset(token_set & KINSHIP),
        groups=frozenset(token_set & GROUP_MARKERS),
        events=frozenset(token_set & EVENT_MARKERS),
        places=frozenset(token_set & PLACE_MARKERS),
        persons=frozenset(token_set & PERSON_MARKERS),
        objects=frozenset(token_set & OBJECT_MARKERS),
        institutions=frozenset(token_set & INSTITUTION_MARKERS),
    )


def build_corpus_stats(path: str | Path, rare_cutoff: int = 50) -> CorpusStats:
    tags: Set[str] = set()
    degrees: Counter = Counter()
    top_scores: Dict[str, float] = {}
    band_counts: Counter = Counter()
    pair_count = 0
    for row in iter_candidates(path):
        pair_count += 1
        tags.add(row.left)
        tags.add(row.right)
        degrees[row.left] += 1
        degrees[row.right] += 1
        top_scores.setdefault(row.left, row.score)
        top_scores.setdefault(row.right, row.score)
        band_counts[int(row.score)] += 1
    metas = {tag: make_meta(tag) for tag in tags}
    token_df: Counter = Counter()
    for meta in metas.values():
        token_df.update(meta.token_set)
    n = len(metas)
    token_idf = {token: math.log((n + 1) / (count + 1)) + 1 for token, count in token_df.items()}
    return CorpusStats(metas, degrees, top_scores, token_df, token_idf, band_counts, pair_count, rare_cutoff)


def entity_types(meta: TagMeta) -> Set[str]:
    result: Set[str] = set()
    if meta.groups:
        result.add("group")
    if meta.events:
        result.add("event")
    if meta.places:
        result.add("place")
    if meta.persons:
        result.add("person")
    if meta.objects:
        result.add("object")
    if meta.institutions:
        result.add("institution")
    if meta.numbers or meta.number_words:
        result.add("quantity")
    if not result:
        result.add("other")
    return result


def axis_conflict(left: Set[str], right: Set[str]) -> bool:
    for side_a, side_b in DIRECTION_AXES:
        if ((left & side_a) and (right & side_b)) or ((left & side_b) and (right & side_a)):
            return True
    return False


def hard_rejection_reason(left: TagMeta, right: TagMeta) -> Optional[str]:
    """Return a high-confidence contradiction, or None.

    These rules are deliberately narrow. They are vetoes, never positive evidence.
    """
    personal_prefixes = {"nga", "maung", "ma", "u"}
    left_personal = bool(left.tokens and left.tokens[0] in personal_prefixes)
    right_personal = bool(right.tokens and right.tokens[0] in personal_prefixes)
    if (
        left_personal and right_personal
        and left.sorted_tokens == right.sorted_tokens
        and left.norm != right.norm
    ):
        return "personal_name_token_order_conflict"
    if left.numbers and right.numbers and left.numbers != right.numbers:
        return "conflicting_numeric_values"
    if left.number_words and right.number_words and left.number_words != right.number_words:
        return "conflicting_number_or_ordinal_words"
    if axis_conflict(set(left.token_set), set(right.token_set)):
        return "opposed_direction_side_or_relation"
    if left.kinship and right.kinship and left.kinship != right.kinship:
        return "conflicting_kinship_roles"
    left_hard_group = bool(left.token_set & HARD_GROUP_MARKERS)
    right_hard_group = bool(right.token_set & HARD_GROUP_MARKERS)
    if left_hard_group != right_hard_group:
        return "individual_or_entity_vs_explicit_group_or_unit"
    if bool(left.events) != bool(right.events):
        return "entity_vs_event_or_action"
    left_place_person = bool(left.places) and not left.persons and bool(right.persons) and not right.places
    right_place_person = bool(right.places) and not right.persons and bool(left.persons) and not left.places
    if left_place_person or right_place_person:
        return "place_vs_person_or_office"
    return None


def deterministic_merge_reason(left: TagMeta, right: TagMeta) -> Optional[str]:
    """Very conservative string-only automatic merges.

    Token reordering is safe for collectives, objects/concepts, and formal
    titles, but *not* for short personal names such as ``Nga Htun Tha`` versus
    ``Nga Tha Htun``. Function-word deletion is limited to ``of``/``the`` so
    Burmese names such as ``Nga In`` or ``Ma Swe On`` are never erased as if
    they were English prepositions.
    """
    from collections import Counter as _Counter

    if left.norm == right.norm:
        return "exact_after_case_punctuation_and_spacing_normalization"
    if left.compact == right.compact:
        return "exact_after_word_boundary_normalization"
    if left.variant_compact == right.variant_compact:
        return "exact_after_curated_transliteration_normalization"

    personal_prefixes = {"nga", "maung", "ma", "u"}
    has_personal_prefix = bool(left.tokens and left.tokens[0] in personal_prefixes) or bool(
        right.tokens and right.tokens[0] in personal_prefixes
    )
    strong_title_tokens = {
        "mingyi", "maha", "thado", "thiri", "siri", "nemyo", "naymyo",
        "nawrahta", "sithu", "prince", "princess", "king", "queen",
        "minister", "governor", "sayadaw", "sawbwa", "lord", "chief",
    }
    both_collective = "and" in left.token_set and "and" in right.token_set
    typed_non_person_entity = bool(
        (left.objects and right.objects)
        or (left.events and right.events)
        or (left.places and right.places)
        or (left.institutions and right.institutions)
        or (left.groups and right.groups and not left.persons and not right.persons)
    )
    safe_reorder = both_collective or (typed_non_person_entity and not has_personal_prefix)

    if left.sorted_tokens == right.sorted_tokens and safe_reorder:
        return "same_normalized_tokens_in_different_order"
    if left.variant_sorted == right.variant_sorted and safe_reorder:
        return "same_tokens_after_curated_transliteration_normalization"

    # Safe syntactic alternations: 'King of Pancala' ↔ 'Pancala King',
    # 'death of Alaungmintaya' ↔ "Alaungmintaya's death".
    removable = {"of", "the"}
    stripped_left = _Counter(t for t in left.tokens if t not in removable)
    stripped_right = _Counter(t for t in right.tokens if t not in removable)
    removed_any = bool((left.token_set | right.token_set) & removable)
    if removed_any and stripped_left == stripped_right and stripped_left:
        return "same_tokens_after_removing_of_the"

    # City/town wording is safe only when a named anchor remains.
    place_words = {"city", "town"}
    base_left = _Counter(t for t in left.tokens if t not in place_words)
    base_right = _Counter(t for t in right.tokens if t not in place_words)
    base_tokens = set(base_left)
    named_anchor = any(
        t not in GENERIC_TITLES and t not in STOPWORDS and t not in GROUP_MARKERS
        and t not in PERSON_MARKERS and t not in PLACE_MARKERS
        for t in base_tokens
    )
    if (
        named_anchor
        and base_left == base_right
        and left.token_set & place_words
        and right.token_set & place_words
    ):
        return "named_city_town_wording_variant"

    # Campaign/expedition synonymy is automatic only for a named campaign.
    event_words = {"campaign", "expedition"}
    event_left = _Counter(t for t in left.tokens if t not in event_words)
    event_right = _Counter(t for t in right.tokens if t not in event_words)
    event_anchor = any(
        t not in GENERIC_TITLES and t not in STOPWORDS and t not in EVENT_MARKERS
        and t not in {"royal", "military"}
        for t in set(event_left)
    )
    if (
        event_anchor
        and event_left == event_right
        and left.token_set & event_words
        and right.token_set & event_words
    ):
        return "named_campaign_expedition_wording_variant"

    return None


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _weighted_jaccard(left: Set[str], right: Set[str], idf: Mapping[str, float]) -> float:
    union = left | right
    if not union:
        return 1.0
    return sum(idf.get(t, 1.0) for t in left & right) / sum(idf.get(t, 1.0) for t in union)


def feature_vector(
    left_tag: str,
    right_tag: str,
    score: float,
    rank_left: int,
    rank_right: int,
    stats: CorpusStats,
) -> np.ndarray:
    left = stats.metas[left_tag]
    right = stats.metas[right_tag]
    set_left = set(left.token_set)
    set_right = set(right.token_set)
    intersection = set_left & set_right
    union = set_left | set_right
    content_left = set(left.content_set)
    content_right = set(right.content_set)
    variant_left = set(left.variant)
    variant_right = set(right.variant)
    variant_intersection = variant_left & variant_right
    variant_union = variant_left | variant_right
    types_left = entity_types(left)
    types_right = entity_types(right)
    extras = set_left ^ set_right
    extra_generic = (
        _safe_div(sum(t in GENERIC_TITLES or t in STOPWORDS for t in extras), len(extras)) if extras else 1.0
    )
    shared_idfs = [stats.token_idf.get(t, 1.0) for t in intersection]
    rare_left = {t for t in set_left if stats.token_df[t] <= stats.rare_cutoff}
    rare_right = {t for t in set_right if stats.token_df[t] <= stats.rare_cutoff}

    numeric_conflict = int(bool(left.numbers and right.numbers and left.numbers != right.numbers))
    numberword_conflict = int(bool(left.number_words and right.number_words and left.number_words != right.number_words))
    direction_conflict = int(axis_conflict(set_left, set_right))
    kinship_conflict = int(bool(left.kinship and right.kinship and left.kinship != right.kinship))
    group_mismatch = int(bool(left.groups) != bool(right.groups))
    event_mismatch = int(bool(left.events) != bool(right.events))
    place_person_mismatch = int(
        (bool(left.places) and not left.persons and bool(right.persons) and not right.places)
        or (bool(right.places) and not right.persons and bool(left.persons) and not left.places)
    )

    values = [
        score / 100.0,
        math.log1p(rank_left),
        math.log1p(rank_right),
        math.log1p(min(rank_left, rank_right)),
        math.log1p(max(rank_left, rank_right)),
        1.0 / rank_left + 1.0 / rank_right,
        (stats.top_scores[left_tag] - score) / 100.0,
        (stats.top_scores[right_tag] - score) / 100.0,
        math.log1p(stats.degrees[left_tag]),
        math.log1p(stats.degrees[right_tag]),
        fuzz.ratio(left.norm, right.norm) / 100.0,
        fuzz.token_sort_ratio(left.norm, right.norm) / 100.0,
        fuzz.token_set_ratio(left.norm, right.norm) / 100.0,
        fuzz.partial_ratio(left.norm, right.norm) / 100.0,
        fuzz.ratio(" ".join(left.variant), " ".join(right.variant)) / 100.0,
        _safe_div(len(intersection), len(union)),
        _safe_div(len(intersection), min(len(set_left), len(set_right))),
        _safe_div(len(intersection), max(len(set_left), len(set_right))),
        _safe_div(len(content_left & content_right), len(content_left | content_right)),
        _weighted_jaccard(set_left, set_right, stats.token_idf),
        _safe_div(len(variant_intersection), len(variant_union)),
        _safe_div(min(len(left.norm), len(right.norm)), max(len(left.norm), len(right.norm))),
        abs(len(left.norm) - len(right.norm)) / max(1, max(len(left.norm), len(right.norm))),
        abs(len(set_left) - len(set_right)),
        len(intersection),
        len(extras),
        extra_generic,
        float(left.norm == right.norm),
        float(left.compact == right.compact),
        float(left.sorted_tokens == right.sorted_tokens),
        float(left.variant_compact == right.variant_compact),
        float(left.variant_sorted == right.variant_sorted),
        float(left.norm in right.norm or right.norm in left.norm),
        float(set_left <= set_right or set_right <= set_left),
        float(left.tokens[:1] == right.tokens[:1]),
        float(left.tokens[-1:] == right.tokens[-1:]),
        float("and" in set_left and "and" in set_right and (set_left - {"and"}) == (set_right - {"and"})),
        max(shared_idfs) if shared_idfs else 0.0,
        sum(shared_idfs),
        sum(stats.token_df[t] <= stats.rare_cutoff for t in intersection),
        _safe_div(len(rare_left & rare_right), len(rare_left | rare_right)),
        float(bool(rare_left & rare_right)),
        numeric_conflict,
        numberword_conflict,
        direction_conflict,
        kinship_conflict,
        group_mismatch,
        event_mismatch,
        place_person_mismatch,
        _safe_div(len(types_left & types_right), len(types_left | types_right)),
        float(types_left.isdisjoint(types_right)),
        float(types_left == types_right),
        float(rank_left == 1 and rank_right == 1),
        float(rank_left <= 5 and rank_right <= 5),
        float(rank_left <= 10 and rank_right <= 10),
        float(rank_left <= 20 and rank_right <= 20),
        float(rank_left <= 100 and rank_right <= 100),
    ]
    return np.asarray(values, dtype=np.float32)


def identity_evidence_gate(vector: np.ndarray) -> bool:
    """Extra lexical gate for ML-driven automatic merging.

    The indices correspond to FEATURE_NAMES and intentionally require direct identity evidence.
    """
    feature = dict(zip(FEATURE_NAMES, vector))
    return bool(
        feature["exact_compact"]
        or feature["exact_token_multiset"]
        or feature["exact_variant_compact"]
        or feature["exact_variant_multiset"]
        or feature["token_set_ratio"] >= 0.92
        or feature["variant_char_ratio"] >= 0.93
        or (feature["idf_jaccard"] >= 0.68 and feature["token_overlap_min"] >= 0.80)
    )
