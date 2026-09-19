#!/usr/bin/env python3
"""
Konbaung Chronicle KG annotation pipeline, first draft.

What this does:
1. Reads the three OCR/plaintext chronicle volumes.
2. Normalizes page text and removes page furniture, TOC/index/image-plate debris, footnotes.
3. Chunks cleaned narrative text into sentence-respecting chunks with context windows.
4. Sends each target chunk to Gemini using Pydantic/JSON-schema structured output.
5. Writes one annotation record per chunk as JSONL, with provenance and resumable execution.

Install:
    pip install google-genai pydantic tqdm python-dotenv

Run:
    export GEMINI_API_KEY="..."
    python konbaung_gemini_kg_annotator.py \
        --inputs /path/konbaung_vol1_full.txt /path/konbaung_vol2_full.txt /path/konbaung_vol3_full.txt \
        --out-dir ./konbaung_annotations \
        --model gemini-2.5-flash-lite \
        --target-words 500 \
        --context-words 120

Notes for Codex / future tuning:
- The page-filtering heuristics are intentionally conservative but not sacred.
- The script annotates only <target_text>; <previous_context> and <next_context> are for disambiguation.
- Properties are open-ended as key/value pairs. Closed class enforcement is only on entity_type and relation_type.
- Add prompt caching or batch API later if cost becomes the bottleneck.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

try:
    from google import genai
except Exception:  # pragma: no cover - useful when running --prepare-only without SDK installed
    genai = None


# ---------------------------------------------------------------------------
# Closed-class schema
# ---------------------------------------------------------------------------

EntityType = Literal[
    "PERSON",
    "COLLECTIVE_HUMAN_GROUP",
    "SOCIAL_ETHNIC_GROUP",
    "POLITY_OR_REALM",
    "ADMINISTRATIVE_TERRITORY",
    "SETTLEMENT",
    "GEOGRAPHIC_FEATURE",
    "BUILT_SITE",
    "MILITARY_FORMATION",
    "OFFICE_TITLE_RANK",
    "INSTITUTION_OR_ORGANIZATION",
    "TEXT_RECORD_WORK",
    "MATERIAL_OBJECT",
    "WEAPON_OR_MILITARY_EQUIPMENT",
    "VEHICLE_OR_TRANSPORT",
    "ANIMAL",
    "RESOURCE_SUPPLY",
    "DATE_TIME",
    "QUANTITY_MEASURE",
    "EVENT_PROCESS",
    "ABSTRACT_STATUS_NORM",
    "SUPERNATURAL_RELIGIOUS_ENTITY",
    "RITUAL_CEREMONIAL_OBJECT",
    "OMEN_SIGN_PROPHECY",
    "NATURAL_CELESTIAL_PHENOMENON",
    "BODY_PART_REMAINS",
    "HEALTH_CONDITION",
    "KNOWLEDGE_SKILL_DISCIPLINE",
    "LANGUAGE_SCRIPT",
]

RelationType = Literal[
    "HAS_NAME_ALIAS_TITLE",
    "HAS_ROLE_STATUS",
    "KINSHIP_RELATION",
    "BIRTH_ORIGIN",
    "MARRIAGE_UNION",
    "DEATH_EVENT",
    "MEMBER_PART_OF",
    "LOCATED_AT",
    "EVENT_OCCURS_AT",
    "EVENT_OCCURS_ON",
    "HAS_QUANTITY_MEASURE",
    "RULES_CONTROLS_ADMINISTERS",
    "SERVES_SUBORDINATE_TO",
    "HOLDS_OFFICE",
    "APPOINTS_ASSIGNS",
    "GRANTS_CONFERS_REWARDS",
    "TRANSFERS_CEDES_GIVES_OFFERS_DONATES",
    "RECEIVES_ACQUIRES_SEIZES",
    "OWNS_POSSESSES_HAS",
    "SENDS_DISPATCHES_ESCORTS",
    "ORDERS_COMMANDS_INSTRUCTS",
    "COMMUNICATES_REPORTS_REQUESTS",
    "KNOWS_HEARS_OBSERVES",
    "PLANS_INTENDS_DECIDES",
    "MOVES_TRAVELS",
    "STATIONS_DEPLOYS_CAMPS",
    "ORGANIZES_MOBILIZES_FORMS",
    "SUPPLIES_EQUIPS_REINFORCES",
    "ATTACKS_FIGHTS_ASSAULTS",
    "DEFENDS_RESISTS_HOLDS",
    "BESIEGES_BLOCKADES_SURROUNDS",
    "DEFEATS_CONQUERS_TAKES_PLACE",
    "RETREATS_FLEES_ESCAPES",
    "CAPTURES_ARRESTS_DETAINS",
    "KILLS_EXECUTES_CAUSES_DEATH",
    "WOUNDS_HARMS",
    "DESTROYS_BURNS_DAMAGES",
    "SUBMITS_SWEARS_ALLEGIANCE",
    "REBELS_DEFIES_BETRAYS",
    "INTERROGATES_TRIES_JUDGES",
    "PUNISHES_SANCTIONS",
    "BUILDS_FOUNDS_CONSTRUCTS",
    "REPAIRS_RENOVATES_IMPROVES",
    "NAMES_RENAMES_DESIGNATES",
    "WRITES_INSCRIBES_COMPILES",
    "PUBLISHES_PRINTS_EDITS",
    "PERFORMS_RITUAL_WORSHIPS",
    "PROHIBITS_PERMITS_REGULATES",
    "CAUSES_RESULTS_IN",
    "INTERPRETS_EXPLAINS_GLOSSES",
    "TEACHES_LEARNS_TRAINS",
    "EXPERIENCES_EMOTION_ATTITUDE",
    "EVALUATES_PRAISES_CRITICIZES",
    "ASSESSES_COLLECTS_PAYS_TAX_OR_FEE",
    "TREATS_HEALTH_CONDITION",
    "PARTICIPATES_ATTENDS",
    "PRESERVES_COPIES_ARCHIVES",
    "MAKES_PEACE_ALLIANCE_TREATY",
    "PLACES_INSTALLS_REMOVES",
    "SUCCESSION_ENTHRONEMENT_ABDICATION",
    "DEPICTS_REPRESENTS",
    "DECEIVES_LURES_MISINFORMS",
    "TRADES_EXCHANGES_BUYS_SELLS",
    "RESCUES_ASSISTS_PROTECTS",
    "RELEASES_FREES_PARDONS",
]

ENTITY_DEFINITIONS = {
    "PERSON": "individual human: king, queen, prince, monk, commander, minister, envoy, author",
    "COLLECTIVE_HUMAN_GROUP": "plural/collective human actor: soldiers, captives, townspeople, merchants, monks",
    "SOCIAL_ETHNIC_GROUP": "ethnic, national, caste, or social identity group",
    "POLITY_OR_REALM": "kingdom, state, realm, country, tributary polity, royal domain",
    "ADMINISTRATIVE_TERRITORY": "district, province, circuit, county, governed zone, command/tax division",
    "SETTLEMENT": "city, town, village, port, capital, named inhabited place",
    "GEOGRAPHIC_FEATURE": "river, creek, bank, mountain, island, forest, road, natural terrain",
    "BUILT_SITE": "palace, fort, camp, gate, wall, monastery, pagoda, stupa, market, house, dam, moat, prison",
    "MILITARY_FORMATION": "army, troop, cavalry, infantry, fleet, guard, regiment, named/numbered unit",
    "OFFICE_TITLE_RANK": "king, Uparaja, governor, commander, minister, sawbwa, rank, office, title-name",
    "INSTITUTION_OR_ORGANIZATION": "court, ministry, government, school, publisher, press, company, religious order",
    "TEXT_RECORD_WORK": "chronicle, letter, royal order, inscription, book, poem, proclamation, report, treaty, list",
    "MATERIAL_OBJECT": "portable object: regalia, treasure, clothing, tool, container, jewel, flag, seal, machine",
    "WEAPON_OR_MILITARY_EQUIPMENT": "gun, cannon, sword, spear, ammunition, ladder, shield, armor, war gear",
    "VEHICLE_OR_TRANSPORT": "boat, ship, steamer, barge, raft, cart, litter, wagon, carriage",
    "ANIMAL": "elephant, horse, cattle, bird, white elephant, named/valued animal",
    "RESOURCE_SUPPLY": "food, grain, rice, water, ammunition stores, money, gold, silver, provisions, tax goods",
    "DATE_TIME": "date, time of day, reign-year, duration, calendar/festival time",
    "QUANTITY_MEASURE": "count, distance, weight, price, troop number, age, size, volume, proportion",
    "EVENT_PROCESS": "battle, siege, ceremony, journey, rebellion, publication, construction campaign, trial, festival",
    "ABSTRACT_STATUS_NORM": "law, allegiance, rebellion, authority, punishment, loyalty, oath, service, merit, obligation",
    "SUPERNATURAL_RELIGIOUS_ENTITY": "Buddha, nat, guardian spirit, deity, demon, mythic supernatural agent",
    "RITUAL_CEREMONIAL_OBJECT": "offering, relic, ritual food, consecration object, ritual vessel/instrument",
    "OMEN_SIGN_PROPHECY": "dream, omen, portent, prophecy, astrological sign, auspicious marker",
    "NATURAL_CELESTIAL_PHENOMENON": "star, planet, earthquake, storm, wind, light, darkness, fire, thunder",
    "BODY_PART_REMAINS": "corpse, body, bone, head, face, hand, foot, royal remains",
    "HEALTH_CONDITION": "illness, disease, disability, recovery, weakness, deathbed condition",
    "KNOWLEDGE_SKILL_DISCIPLINE": "education, scripture, military skill, language skill, surveying, astrology, medicine, law, craft",
    "LANGUAGE_SCRIPT": "Burmese, Pali, English, French, Mon speech, script/writing system",
}

RELATION_DEFINITIONS = {
    "HAS_NAME_ALIAS_TITLE": "entity bears a name, alias, title, epithet, style",
    "HAS_ROLE_STATUS": "entity has social/legal/ritual/political/military status",
    "KINSHIP_RELATION": "parent, child, sibling, spouse, ancestor, descendant, in-law",
    "BIRTH_ORIGIN": "birth, birthplace, natal origin, offspring production",
    "MARRIAGE_UNION": "marriage, royal union, spouse-taking",
    "DEATH_EVENT": "death, dying, royal death, cremation/burial when death-focused",
    "MEMBER_PART_OF": "member-of, part-of, unit-of, component-of",
    "LOCATED_AT": "static location of entity",
    "EVENT_OCCURS_AT": "event happens at place",
    "EVENT_OCCURS_ON": "event happens at date/time",
    "HAS_QUANTITY_MEASURE": "entity/event has count, distance, duration, amount, price, age, size",
    "RULES_CONTROLS_ADMINISTERS": "rules, controls, governs, supervises, administers",
    "SERVES_SUBORDINATE_TO": "serves, obeys, is subordinate/vassal to",
    "HOLDS_OFFICE": "person holds office, command, governorship, title-rank",
    "APPOINTS_ASSIGNS": "appoints, assigns, stations, delegates role/task/location",
    "GRANTS_CONFERS_REWARDS": "grants title/rank/office/reward/honor/insignia/privilege",
    "TRANSFERS_CEDES_GIVES_OFFERS_DONATES": "gifts, offers, donates, cedes territory/rights/goods/people",
    "RECEIVES_ACQUIRES_SEIZES": "receives, obtains, takes, seizes, confiscates, captures control/goods",
    "OWNS_POSSESSES_HAS": "owns, possesses, has resources/weapons/dependents/animals",
    "SENDS_DISPATCHES_ESCORTS": "sends envoys/troops/messages/objects, escorts, transports",
    "ORDERS_COMMANDS_INSTRUCTS": "commands, instructs, issues royal order, directs action",
    "COMMUNICATES_REPORTS_REQUESTS": "says, reports, petitions, requests, advises, negotiates, replies",
    "KNOWS_HEARS_OBSERVES": "hears, sees, learns, realizes, notices, recognizes",
    "PLANS_INTENDS_DECIDES": "plans, intends, resolves, strategizes, decides",
    "MOVES_TRAVELS": "goes, marches, returns, flees, crosses, sails, enters, exits, arrives",
    "STATIONS_DEPLOYS_CAMPS": "stations, deploys, camps, holds position, sets ambush, places guards",
    "ORGANIZES_MOBILIZES_FORMS": "forms units, assembles, divides columns, mobilizes, recruits, reorganizes",
    "SUPPLIES_EQUIPS_REINFORCES": "supplies, arms, provisions, outfits, reinforces",
    "ATTACKS_FIGHTS_ASSAULTS": "attacks, fights, storms, charges, shoots, strikes",
    "DEFENDS_RESISTS_HOLDS": "defends, resists, holds, refuses surrender",
    "BESIEGES_BLOCKADES_SURROUNDS": "encircles, blockades, surrounds, cuts off, lays siege",
    "DEFEATS_CONQUERS_TAKES_PLACE": "wins, defeats, captures/conquers place, occupies after victory",
    "RETREATS_FLEES_ESCAPES": "retreats, flees, withdraws, escapes, evades capture",
    "CAPTURES_ARRESTS_DETAINS": "captures persons, arrests, detains, imprisons, confines",
    "KILLS_EXECUTES_CAUSES_DEATH": "kills, executes, causes death, fatal combat",
    "WOUNDS_HARMS": "wounds, injures, causes non-fatal harm",
    "DESTROYS_BURNS_DAMAGES": "burns, destroys, demolishes, damages property/site/object/records",
    "SUBMITS_SWEARS_ALLEGIANCE": "surrenders, submits, takes oath, becomes subject/vassal",
    "REBELS_DEFIES_BETRAYS": "rebels, betrays, defects, violates loyalty, refuses order",
    "INTERROGATES_TRIES_JUDGES": "questions, investigates, tries, legally examines/adjudicates",
    "PUNISHES_SANCTIONS": "punishes, penalizes, disciplines, imposes legal consequence",
    "BUILDS_FOUNDS_CONSTRUCTS": "builds, founds, establishes, digs, casts, manufactures",
    "REPAIRS_RENOVATES_IMPROVES": "repairs, restores, improves, strengthens, renovates",
    "NAMES_RENAMES_DESIGNATES": "names, renames, designates, labels formally",
    "WRITES_INSCRIBES_COMPILES": "writes, records, inscribes, compiles, authors, drafts",
    "PUBLISHES_PRINTS_EDITS": "prints, publishes, edits, republishes, issues text",
    "PERFORMS_RITUAL_WORSHIPS": "worships, consecrates, propitiates, pours libation, performs ceremony",
    "PROHIBITS_PERMITS_REGULATES": "forbids, permits, regulates, sets rule/allowed conduct",
    "CAUSES_RESULTS_IN": "causes, leads to, produces outcome",
    "INTERPRETS_EXPLAINS_GLOSSES": "interprets omen/text/event, explains meaning, glosses term/doctrine",
    "TEACHES_LEARNS_TRAINS": "teaches, learns, studies, trains in skill/scripture/language/discipline",
    "EXPERIENCES_EMOTION_ATTITUDE": "experiences fear, joy, grief, gratitude, anger, reluctance, desire",
    "EVALUATES_PRAISES_CRITICIZES": "praises, condemns, evaluates, criticizes, celebrates, laments",
    "ASSESSES_COLLECTS_PAYS_TAX_OR_FEE": "levies, assesses, collects, pays, exempts tax/fee/revenue/salary",
    "TREATS_HEALTH_CONDITION": "treats illness, recovers, worsens, diagnoses, cares for health",
    "PARTICIPATES_ATTENDS": "attends or participates in ceremony, school, audience, meeting, campaign",
    "PRESERVES_COPIES_ARCHIVES": "copies, preserves, stores, archives, keeps, collects records/artifacts",
    "MAKES_PEACE_ALLIANCE_TREATY": "makes treaty, peace, alliance, diplomatic settlement/agreement",
    "PLACES_INSTALLS_REMOVES": "places, installs, raises, lowers, enshrines, removes, sets up/takes down",
    "SUCCESSION_ENTHRONEMENT_ABDICATION": "accession, enthronement, abdication, deposition, transfer of throne",
    "DEPICTS_REPRESENTS": "depicts, represents, symbolizes, portrays scene/person/doctrine",
    "DECEIVES_LURES_MISINFORMS": "deceives, misleads, feigns, lures, tricks",
    "TRADES_EXCHANGES_BUYS_SELLS": "buys, sells, trades, exchanges, conducts commerce",
    "RESCUES_ASSISTS_PROTECTS": "rescues, assists, protects, shelters, saves",
    "RELEASES_FREES_PARDONS": "releases, frees, pardons, grants amnesty/clemency, commutes punishment",
}


# ---------------------------------------------------------------------------
# Pydantic output schema for one chunk
# ---------------------------------------------------------------------------

class PropertyKV(BaseModel):
    key: str = Field(description="Free-form property key, e.g. title, rank, date, quantity, role, uncertainty.")
    value: str = Field(description="Free-form property value grounded in the target text.")


class EntityMention(BaseModel):
    mention_id: str = Field(description="Unique within this chunk. Use e1, e2, e3...")
    surface: str = Field(description="Exact or near-exact surface form from the target text.")
    normalized_name: str = Field(description="Best normalized/canonical name for this mention within the chunk.")
    entity_type: EntityType
    evidence_quote: str = Field(description="Short quote from target text supporting this entity mention.")
    description: str = Field(description="One-sentence description grounded only in this chunk/context.")
    properties: list[PropertyKV] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "high"


class RelationMention(BaseModel):
    relation_id: str = Field(description="Unique within this chunk. Use r1, r2, r3...")
    subject_mention_id: str = Field(description="mention_id of the subject entity in entities.")
    relation_type: RelationType
    object_mention_id: str = Field(description="mention_id of the object entity in entities.")
    evidence_quote: str = Field(description="Short quote from target text supporting the relation.")
    description: str = Field(description="Brief natural-language paraphrase of the relation.")
    properties: list[PropertyKV] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "high"


class ChunkAnnotation(BaseModel):
    chunk_id: str
    entities: list[EntityMention] = Field(default_factory=list)
    relations: list[RelationMention] = Field(default_factory=list)
    uncertain_or_ambiguous: list[str] = Field(default_factory=list)
    skipped_as_nonsemantic: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# OCR/page normalization and chunking
# ---------------------------------------------------------------------------

@dataclass
class Page:
    volume_id: str
    page_num: int
    raw_text: str
    clean_text: str = ""
    drop_reason: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    volume_id: str
    start_page: int
    end_page: int
    target_text: str
    previous_context: str
    next_context: str
    source_pages: list[int]
    approx_words: int
    sha1: str


PAGE_RE = re.compile(r"\[\[BOOK=(?P<book>[^\s\]]+)\s+PAGE=(?P<page>\d{4})\]\]")
MYANMAR_DIGITS = "၀၁၂၃၄၅၆၇၈၉"
ASCII_DIGITS = "0123456789"
MYANMAR_MONTHS = (
    "တန်ခူး", "ကဆုန်", "နယုန်", "ဝါဆို", "ဝါခေါင်", "တော်သလင်း", "သီတင်းကျွတ်",
    "တန်ဆောင်မုန်း", "နတ်တော်", "ပြာသို", "တပို့တွဲ", "တပေါင်း",
)
FOOTER_PATTERNS = [
    re.compile(r"^ရာပြည့်"),
    re.compile(r"^ရာျပည္"),
    re.compile(r"^(F\s*\d|G\s*\d|F\d+[A-Z]|[A-Z]\s*\d+\s*[A-Z]?)$"),
    re.compile(r"^-?[A-Z]{1,4}-?$"),
    re.compile(r"^(JRJ|JJ|JG|GU|GO|FATA|ANOTTO|FUJI|ELECTRONIC|PAKOKKU|Mg Aung)", re.I),
    re.compile(r"^[-–—_ .·•°]+$"),
]


def myanmar_digit_to_int(s: str) -> Optional[int]:
    table = str.maketrans(MYANMAR_DIGITS, ASCII_DIGITS)
    t = s.translate(table)
    t = re.sub(r"\D", "", t)
    return int(t) if t else None


def mostly_latin_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    latin = sum(1 for c in stripped if "A" <= c <= "Z" or "a" <= c <= "z")
    my = sum(1 for c in stripped if "\u1000" <= c <= "\u109F")
    # Keep real romanized names embedded in Burmese lines; drop all-Latin OCR shrapnel.
    return latin >= 1 and my == 0 and len(stripped) <= 40


def is_standalone_page_number(line: str, page_num: int) -> bool:
    s = line.strip().strip("-–— .၊။[]()")
    if not s:
        return True
    if all(ch in MYANMAR_DIGITS + ASCII_DIGITS for ch in s):
        n = myanmar_digit_to_int(s)
        # Printed page numbers often lag/cross-map the OCR PAGE marker. Drop any short numeric line.
        return n is not None and len(s) <= 5
    return False


def looks_like_footnote_line(line: str) -> bool:
    s = line.strip()
    ascii_s = s.translate(str.maketrans(MYANMAR_DIGITS, ASCII_DIGITS)).replace("ဝ", "0")
    if re.match(rf"^[{MYANMAR_DIGITS}]+[။.)]", s):
        # Footnote/date lines are usually short and packed with Western dates or multiple note numbers.
        if any(m in s for m in ("ဇန်", "ဖေ", "မတ်", "ဧပြီ", "မေ", "ဇွန်", "ဇူ", "ဩ", "သြ", "စက်", "အောက်", "နို", "ဒီ")):
            return True
        if len(s) < 70 and re.search(r"\b1[6789]\d{2}\b|\b20\d{2}\b", ascii_s):
            return True
        if len(s) < 50 and re.fullmatch(rf"[{MYANMAR_DIGITS}{ASCII_DIGITS}ဝwW\s။.,()]+", s):
            return True
    return False


def normalize_line(line: str) -> str:
    line = line.replace("\ufeff", "").replace("\u200b", "")
    line = line.replace("`", "").replace("´", "")
    line = re.sub(r"[ \t]+", " ", line)
    # normalize common OCR spacings around Burmese punctuation
    line = re.sub(r"\s+([၊။])", r"\1", line)
    line = re.sub(r"([၊။])(?=\S)", r"\1 ", line)
    return line.strip()


def clean_page_lines(raw_text: str, page_num: int) -> str:
    cleaned: list[str] = []
    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            continue
        if line.startswith("[[BOOK="):
            continue
        if is_standalone_page_number(line, page_num):
            continue
        if any(p.search(line) for p in FOOTER_PATTERNS):
            continue
        if mostly_latin_noise(line):
            continue
        if looks_like_footnote_line(line):
            continue
        # Drop OCR-inserted footnote date clusters such as "၁။ ၁၈ ... ၂။ ..." at bottoms.
        ascii_line = line.translate(str.maketrans(MYANMAR_DIGITS, ASCII_DIGITS)).replace("ဝ", "0")
        if line.count("။") >= 2 and re.search(r"[၁၂၃၄၅၆၇၈၉]\s*။", line) and re.search(r"1[6789]\d{2}|20\d{2}", ascii_line):
            continue
        cleaned.append(line)

    # Join line-wrapped Burmese prose. Keep headings as their own line if short.
    paragraphs: list[str] = []
    buf = ""
    for line in cleaned:
        if not buf:
            buf = line
            continue
        # Headings or list/catalog lines should not always be glued to previous prose.
        prev_ends_sentence = buf.endswith(("။", "?", "!”", "။”", "။'", "။။"))
        line_is_headingish = len(line) < 55 and not line.endswith(("သည်", "၏", "၍", "ပြီး", "ရာ", "လျှင်"))
        if prev_ends_sentence and line_is_headingish:
            paragraphs.append(buf)
            buf = line
        else:
            sep = "" if buf.endswith(("-", "၊")) else " "
            buf = buf + sep + line
    if buf:
        paragraphs.append(buf)

    text = "\n".join(paragraphs)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def classify_page(page: Page, *, drop_front_matter: bool, drop_appendix: bool, body_start: int) -> Optional[str]:
    """Return drop reason if page should be excluded, otherwise None."""
    txt = page.raw_text + "\n" + page.clean_text
    my_chars = sum(1 for c in page.clean_text if "\u1000" <= c <= "\u109F")
    raw_lines = [normalize_line(l) for l in page.raw_text.splitlines() if normalize_line(l)]
    clean_lines = [l for l in page.clean_text.splitlines() if l.strip()]

    # Front matter differs by volume. This is configurable because biographies/prefaces can be valuable.
    if drop_front_matter and page.page_num < body_start:
        return "front_matter_or_toc"

    toc_markers = ("မာတိကာ", "အမှတ်စဉ်", "စာမျက်နှာ")
    if page.page_num < 25 and any(m in txt for m in toc_markers):
        return "table_of_contents"
    compact_txt = re.sub(r"\s+", "", txt)
    if "အက္ခရာဝလိအညွှန်း" in compact_txt or "ပုံအညွှန်း" in compact_txt:
        return "index_or_picture_index"
    if drop_appendix and "နောက်ဆက်တွဲ" in txt:
        return "appendix"

    # Image plates/maps/captions: low prose density, many place labels, or explicit map/caption markers.
    # Do NOT use the generic word "ပုံ" as a trigger; it occurs in real prose.
    map_markers = ("စွယ်စုံကျမ်း", "တိုက်ပွဲနေရာ", "မြန်မာစစ်ကြောင်း", "ဗြိတိသျှစစ်ကြောင်း", "ရည်ညွှန်း")
    if page.page_num >= body_start and my_chars < 180:
        return "too_little_prose"
    if any(m in txt for m in map_markers) and my_chars < 900:
        return "image_plate_or_map"
    short_raw = sum(1 for l in raw_lines if len(l) < 25)
    if my_chars < 350 and raw_lines and short_raw >= max(4, len(raw_lines) - 2):
        return "caption_or_map_labels"
    return None


def parse_volume(path: Path, *, drop_front_matter: bool, drop_appendix: bool, body_starts: dict[str, int]) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    matches = list(PAGE_RE.finditer(raw))
    if not matches:
        raise ValueError(f"No page markers found in {path}")

    pages: list[Page] = []
    in_back_matter = False
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        volume_id = m.group("book")
        page_num = int(m.group("page"))
        page = Page(volume_id=volume_id, page_num=page_num, raw_text=raw[start:end])
        page.clean_text = clean_page_lines(page.raw_text, page_num)
        body_start = body_starts.get(volume_id, 23)
        page.drop_reason = classify_page(page, drop_front_matter=drop_front_matter, drop_appendix=drop_appendix, body_start=body_start)
        if in_back_matter and page.drop_reason is None:
            page.drop_reason = "back_matter_after_index_or_picture_index"
        if page.drop_reason == "index_or_picture_index":
            in_back_matter = True
        pages.append(page)
    return pages


def approx_word_count(text: str) -> int:
    # Burmese is not reliably whitespace-tokenized. Use a conservative hybrid:
    # whitespace-ish tokens OR one pseudo-word per ~6 non-space chars, whichever is larger.
    tokens = re.findall(r"[\u1000-\u109F]+|[A-Za-z0-9]+", text)
    nonspace = re.sub(r"\s+", "", text)
    return max(len(tokens), max(1, len(nonspace) // 6))


def split_sentences_with_pages(pages: list[Page]) -> list[tuple[str, int]]:
    """Return list of (sentence_or_segment, page_num), skipping dropped pages."""
    segments: list[tuple[str, int]] = []
    sentence_re = re.compile(r"(?<=။)\s+")

    for page in pages:
        if page.drop_reason or not page.clean_text.strip():
            continue
        for para in page.clean_text.splitlines():
            para = para.strip()
            if not para:
                continue
            # Keep very short headings as separate segments.
            pieces = sentence_re.split(para)
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    continue
                if approx_word_count(piece) > 180 and "၊" in piece:
                    # Fallback split for enormous OCR-wrapped sentences.
                    subpieces = re.split(r"(?<=၊)\s+", piece)
                    segments.extend((s.strip(), page.page_num) for s in subpieces if s.strip())
                else:
                    segments.append((piece, page.page_num))
    return segments


def collect_context(segments: list[tuple[str, int]], start: int, end: int, context_words: int, direction: str) -> str:
    if context_words <= 0:
        return ""
    out: list[str] = []
    total = 0
    if direction == "before":
        idxs = range(start - 1, -1, -1)
    else:
        idxs = range(end, len(segments))
    for i in idxs:
        seg = segments[i][0]
        wc = approx_word_count(seg)
        if total + wc > context_words and out:
            break
        out.append(seg)
        total += wc
        if total >= context_words:
            break
    if direction == "before":
        out.reverse()
    return " ".join(out).strip()


def build_chunks(
    pages: list[Page],
    *,
    volume_id: str,
    target_words: int,
    max_words: int,
    context_words: int,
) -> list[Chunk]:
    segments = split_sentences_with_pages(pages)
    chunks: list[Chunk] = []
    i = 0
    chunk_idx = 0
    while i < len(segments):
        start = i
        words = 0
        chunk_segments: list[str] = []
        source_pages: list[int] = []
        while i < len(segments):
            seg, pg = segments[i]
            seg_words = approx_word_count(seg)
            # If adding this segment makes a large chunk, stop at previous sentence boundary.
            if chunk_segments and words + seg_words > target_words:
                break
            chunk_segments.append(seg)
            source_pages.append(pg)
            words += seg_words
            i += 1
            if words >= target_words:
                break
        # Hard guard: if one segment exceeded target_words, still consume it.
        if not chunk_segments and i < len(segments):
            seg, pg = segments[i]
            chunk_segments = [seg]
            source_pages = [pg]
            words = approx_word_count(seg)
            i += 1

        # If still above max words due to massive line, split by character as last resort.
        target_text = " ".join(chunk_segments).strip()
        if words > max_words and len(target_text) > 0:
            # Rare OCR monster segment; split into roughly max_words/target_words slices.
            # Codex can replace with a real Burmese segmenter later.
            pieces = re.split(r"(?<=၊)\s+", target_text)
            if len(pieces) <= 1:
                pieces = [target_text[j : j + 3200] for j in range(0, len(target_text), 3200)]
            # Insert remaining pieces back into stream by making them pseudo-segments on same page.
            target_text = pieces[0]
            remainder = [(p, source_pages[-1]) for p in pieces[1:] if p.strip()]
            segments[i:i] = remainder
            words = approx_word_count(target_text)

        end = i
        previous_context = collect_context(segments, start, end, context_words, "before")
        next_context = collect_context(segments, start, end, context_words, "after")
        start_page, end_page = min(source_pages), max(source_pages)
        sha1 = hashlib.sha1(target_text.encode("utf-8")).hexdigest()
        chunk_id = f"{volume_id}_p{start_page:04d}-{end_page:04d}_c{chunk_idx:05d}_{sha1[:8]}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                volume_id=volume_id,
                start_page=start_page,
                end_page=end_page,
                target_text=target_text,
                previous_context=previous_context,
                next_context=next_context,
                source_pages=sorted(set(source_pages)),
                approx_words=words,
                sha1=sha1,
            )
        )
        chunk_idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Prompting and API calls
# ---------------------------------------------------------------------------

def label_block(title: str, definitions: dict[str, str]) -> str:
    return title + "\n" + "\n".join(f"- {k}: {v}" for k, v in definitions.items())


SCHEMA_GUIDE = "\n\n".join([
    label_block("ENTITY LABELS", ENTITY_DEFINITIONS),
    label_block("RELATION LABELS", RELATION_DEFINITIONS),
])


EXTRACTION_PROMPT_TEMPLATE = """You are annotating a Burmese royal chronicle for a knowledge graph.

Task:
Extract every meaningful entity mention and relation-bearing proposition from <target_text> only.
Use <previous_context> and <next_context> only to resolve pronouns, titles, aliases, and omitted subjects.
Do NOT annotate entities or relations that occur only in the context blocks.

Closed-class rule:
- entity_type MUST be one of the entity labels below.
- relation_type MUST be one of the relation labels below.
- properties are open-ended key/value pairs. Use them richly for dates, roles, titles, quantities, uncertainty, aliases, direction, location, ritual context, military unit size, etc.

Extraction standards:
- Prefer recall over minimalism. Extract all historically meaningful people, groups, places, offices, texts, objects, military units, rituals, dates, quantities, and events in the target text.
- Every relation must connect two entities listed in entities by mention_id.
- Use short exact evidence_quote snippets from target_text, not from context.
- If a repeated surface form clearly refers to the same local entity, reuse normalized_name but give each important mention its own mention_id only when needed for relations.
- Do not invent facts. If uncertain, still annotate with confidence="low" and explain in uncertain_or_ambiguous.
- Ignore page numbers, publisher footers, OCR debris, and detached footnote dates if they survived cleaning.
- Keep descriptions concise and grounded.

{schema_guide}

<metadata>
chunk_id: {chunk_id}
volume_id: {volume_id}
page_range: {start_page}-{end_page}
</metadata>

<previous_context>
{previous_context}
</previous_context>

<target_text>
{target_text}
</target_text>

<next_context>
{next_context}
</next_context>
"""


def build_prompt(chunk: Chunk) -> str:
    return EXTRACTION_PROMPT_TEMPLATE.format(
        schema_guide=SCHEMA_GUIDE,
        chunk_id=chunk.chunk_id,
        volume_id=chunk.volume_id,
        start_page=chunk.start_page,
        end_page=chunk.end_page,
        previous_context=chunk.previous_context,
        target_text=chunk.target_text,
        next_context=chunk.next_context,
    )


def make_client(api_key: Optional[str] = None):
    if genai is None:
        raise RuntimeError("google-genai is not installed. Run: pip install google-genai")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def call_gemini_structured(
    client,
    *,
    model: str,
    prompt: str,
    max_retries: int,
    retry_sleep: float,
) -> ChunkAnnotation:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            # Interactions API structured output. Codex can swap this to generate_content
            # if your installed google-genai version still uses the older client.models API.
            interaction = client.interactions.create(
                model=model,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": ChunkAnnotation.model_json_schema(),
                },
            )
            return ChunkAnnotation.model_validate_json(interaction.output_text)
        except (ValidationError, Exception) as e:  # keep broad for first draft, log below
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(retry_sleep * (2**attempt) + random.random())
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_chunk_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "chunk_id" in obj:
                    done.add(obj["chunk_id"])
            except json.JSONDecodeError:
                continue
    return done


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()



def parse_body_starts(spec: str) -> dict[str, int]:
    """Parse e.g. 'konbaung_vol1:33,konbaung_vol2:23,konbaung_vol3:23'."""
    out: dict[str, int] = {}
    if not spec:
        return out
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(f"Bad --body-starts item: {item!r}")
        k, v = item.split(':', 1)
        out[k.strip()] = int(v.strip())
    return out

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def prepare_chunks(args: argparse.Namespace) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    page_reports: list[dict] = []

    for input_path_str in args.inputs:
        path = Path(input_path_str)
        if not path.exists():
            raise FileNotFoundError(path)
        pages = parse_volume(path, drop_front_matter=args.drop_front_matter, drop_appendix=args.drop_appendix, body_starts=parse_body_starts(args.body_starts))
        volume_id = pages[0].volume_id if pages else path.stem
        chunks = build_chunks(
            pages,
            volume_id=volume_id,
            target_words=args.target_words,
            max_words=args.max_words,
            context_words=args.context_words,
        )
        all_chunks.extend(chunks)
        for p in pages:
            page_reports.append(
                {
                    "input_file": str(path),
                    "input_sha256": sha256_file(path),
                    "volume_id": p.volume_id,
                    "page_num": p.page_num,
                    "drop_reason": p.drop_reason,
                    "raw_chars": len(p.raw_text),
                    "clean_chars": len(p.clean_text),
                    "clean_preview": p.clean_text[:180],
                }
            )

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "page_normalization_report.jsonl", page_reports)
    write_jsonl(out_dir / "chunks.jsonl", [asdict(c) for c in all_chunks])
    return all_chunks


def run_annotation(args: argparse.Namespace, chunks: list[Chunk]) -> None:
    out_dir = Path(args.out_dir)
    annotations_path = out_dir / "annotations.raw.jsonl"
    errors_path = out_dir / "annotation_errors.jsonl"
    prompts_dir = out_dir / "debug_prompts"

    done = load_done_chunk_ids(annotations_path) if args.resume else set()
    client = make_client(api_key=args.api_key or os.getenv("GEMINI_API_KEY"))

    if args.limit is not None:
        chunks = chunks[: args.limit]

    for chunk in tqdm(chunks, desc="Annotating chunks"):
        if chunk.chunk_id in done:
            continue
        prompt = build_prompt(chunk)
        if args.save_prompts:
            prompts_dir.mkdir(parents=True, exist_ok=True)
            (prompts_dir / f"{chunk.chunk_id}.txt").write_text(prompt, encoding="utf-8")

        try:
            ann = call_gemini_structured(
                client,
                model=args.model,
                prompt=prompt,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
            # Force chunk_id to match our provenance if model drifted.
            ann.chunk_id = chunk.chunk_id
            record = {
                "chunk_id": chunk.chunk_id,
                "volume_id": chunk.volume_id,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "source_pages": chunk.source_pages,
                "approx_words": chunk.approx_words,
                "target_sha1": chunk.sha1,
                "annotation": ann.model_dump(),
            }
            append_jsonl(annotations_path, record)
        except Exception as e:
            append_jsonl(
                errors_path,
                {
                    "chunk_id": chunk.chunk_id,
                    "volume_id": chunk.volume_id,
                    "start_page": chunk.start_page,
                    "end_page": chunk.end_page,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "target_preview": chunk.target_text[:500],
                },
            )
            if not args.continue_on_error:
                raise


def summarize_outputs(out_dir: Path) -> None:
    chunks_path = out_dir / "chunks.jsonl"
    annotations_path = out_dir / "annotations.raw.jsonl"
    pages_path = out_dir / "page_normalization_report.jsonl"

    n_chunks = sum(1 for _ in chunks_path.open("r", encoding="utf-8")) if chunks_path.exists() else 0
    n_annotations = sum(1 for _ in annotations_path.open("r", encoding="utf-8")) if annotations_path.exists() else 0
    dropped = {}
    kept_pages = 0
    if pages_path.exists():
        with pages_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                reason = obj.get("drop_reason")
                if reason:
                    dropped[reason] = dropped.get(reason, 0) + 1
                else:
                    kept_pages += 1

    summary = {
        "chunks_prepared": n_chunks,
        "chunks_annotated": n_annotations,
        "pages_kept": kept_pages,
        "pages_dropped_by_reason": dropped,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate Konbaung chronicle volumes into KG JSONL using Gemini.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .txt files with [[BOOK=... PAGE=....]] markers.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--model", default="gemini-2.5-flash-lite", help="Gemini model ID.")
    parser.add_argument("--api-key", default=None, help="Optional Gemini API key. Prefer GEMINI_API_KEY env var.")
    parser.add_argument("--target-words", type=int, default=500, help="Approx target chunk size.")
    parser.add_argument("--max-words", type=int, default=760, help="Hard-ish max chunk size before fallback split.")
    parser.add_argument("--context-words", type=int, default=120, help="Approx words before and after target text.")
    parser.add_argument("--drop-front-matter", action="store_true", default=True, help="Drop pages before main body. Default true.")
    parser.add_argument("--keep-front-matter", dest="drop_front_matter", action="store_false", help="Keep prefaces/biographies/front matter after cleaning.")
    parser.add_argument("--body-starts", default="konbaung_vol1:41,konbaung_vol2:23,konbaung_vol3:23", help="Comma map of volume_id:first_body_page. Used only when --drop-front-matter is active.")
    parser.add_argument("--drop-appendix", action="store_true", help="Drop appendix-like sections if detected.")
    parser.add_argument("--prepare-only", action="store_true", help="Only normalize/chunk; do not call Gemini.")
    parser.add_argument("--limit", type=int, default=None, help="Only annotate first N chunks, useful for pilot runs.")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip chunks already in annotations.raw.jsonl. Default true.")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--save-prompts", action="store_true", help="Write each prompt to debug_prompts/.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = prepare_chunks(args)
    if args.prepare_only:
        summarize_outputs(out_dir)
        return
    run_annotation(args, chunks)
    summarize_outputs(out_dir)


if __name__ == "__main__":
    main()
