from __future__ import annotations

import hashlib
import json
import os
import threading
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, send_file
from flask_compress import Compress
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, create_model

try:
    from .axial_store import AxialCategoryStore
    from .corpus import CorpusStore
    from .graph_store import ChronicleGraphStore
    from .public_api import install_public_api
    from .reader_bridge import BurmeseReaderBridge
except ImportError:  # Direct `python app.py` execution.
    from axial_store import AxialCategoryStore
    from corpus import CorpusStore
    from graph_store import ChronicleGraphStore
    from public_api import install_public_api
    from reader_bridge import BurmeseReaderBridge


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = APP_ROOT.parent / ".env"
TRANSLATION_MODEL = os.getenv("KONBAUNG_TRANSLATION_MODEL", "gemini-3.1-flash-lite")
TRANSLATION_CACHE_ROOT = Path(
    os.getenv("KONBAUNG_TRANSLATION_CACHE", str(APP_ROOT / "data" / "gemini_token_glosses"))
)
TRANSLATION_SCHEMA_VERSION = 4


class TokenGloss(BaseModel):
    my: str = Field(description="Exact Burmese input token.")
    en: str = Field(description="Shortest accurate contextual English gloss.")


class SegmentedPageGlosses(BaseModel):
    tokens: list[TokenGloss] = Field(description="One Burmese token and English gloss per input token.")


def exact_segmented_gloss_schema(token_count: int) -> type[SegmentedPageGlosses]:
    return create_model(
        "ExactSegmentedPageGlosses",
        __base__=SegmentedPageGlosses,
        tokens=(
            list[TokenGloss],
            Field(
                description=f"Exactly {token_count} Burmese token and English gloss pairs, in input order.",
            ),
        ),
    )


def validate_raw_gloss_fields(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {"tokens"}:
        raise ValueError("Gemini response must contain only the top-level tokens field")
    if not isinstance(payload["tokens"], list):
        raise ValueError("Gemini tokens field must be an array")
    for index, item in enumerate(payload["tokens"]):
        if not isinstance(item, dict) or set(item) != {"my", "en"}:
            raise ValueError(f"Gemini token {index} must contain only my and en")

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/chronicle-assets",
)
app.config["JSON_AS_ASCII"] = False
app.config["COMPRESS_MIMETYPES"] = [
    "application/json",
    "application/javascript",
    "text/css",
]
app.config["COMPRESS_MIN_SIZE"] = 1024
Compress(app)
corpus = CorpusStore()
graph = ChronicleGraphStore()
axial_graph = AxialCategoryStore()
# The axial store reads raw tag vectors through the graph store so that theme
# and tag panels can rank by embedding similarity without a second copy.
axial_graph.attach_embeddings(graph)
public_api = install_public_api(app, graph, axial_graph)
reader_bridge = BurmeseReaderBridge()
translation_locks_guard = threading.Lock()
translation_page_locks: dict[tuple[str, int], threading.Lock] = {}


def read_env_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        current, value = line.split("=", 1)
        if current.strip() == key:
            return value.strip().strip("\"'")
    return ""


def gemini_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip() or read_env_value(DEFAULT_ENV_FILE, "GEMINI_API_KEY")


def translation_cache_path(volume_id: str, page_number: int) -> Path:
    return TRANSLATION_CACHE_ROOT / volume_id / f"{page_number:04d}.json"


def translation_page_lock(volume_id: str, page_number: int) -> threading.Lock:
    key = (volume_id, page_number)
    with translation_locks_guard:
        return translation_page_locks.setdefault(key, threading.Lock())


def read_cached_translation(
    volume_id: str,
    page_number: int,
    page_checksum: str,
) -> dict[str, Any] | None:
    path = translation_cache_path(volume_id, page_number)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        cached.get("schemaVersion") != TRANSLATION_SCHEMA_VERSION
        or cached.get("pageChecksum") != page_checksum
        or cached.get("model") != TRANSLATION_MODEL
    ):
        return None
    return cached


def migrate_cached_gloss_pairs(
    cached_tokens: list[dict[str, Any]],
    tokens: list[dict[str, Any]],
    canonical_text: str,
) -> list[dict[str, str]] | None:
    old_text = "".join(str(token.get("my", "")) for token in cached_tokens)
    new_text = "".join(token["text"] for token in tokens)
    if old_text != canonical_text or new_text != canonical_text:
        return None

    old_ranges: list[tuple[int, int, str]] = []
    cursor = 0
    for token in cached_tokens:
        text = str(token.get("my", ""))
        old_ranges.append((cursor, cursor + len(text), str(token.get("en", "")).strip()))
        cursor += len(text)

    migrated: list[dict[str, str]] = []
    cursor = 0
    old_index = 0
    for token in tokens:
        text = token["text"]
        start = cursor
        end = start + len(text)
        cursor = end
        while old_index < len(old_ranges) and old_ranges[old_index][1] <= start:
            old_index += 1
        glosses: list[str] = []
        scan = old_index
        while scan < len(old_ranges) and old_ranges[scan][0] < end:
            gloss = old_ranges[scan][2]
            if gloss and gloss not in glosses:
                glosses.append(gloss)
            scan += 1
        migrated.append({"my": text, "en": "" if text.isspace() else " / ".join(glosses)})
    return migrated


def write_cached_translation(volume_id: str, page_number: int, record: dict[str, Any]) -> Path:
    path = translation_cache_path(volume_id, page_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def segmentation_identity(tokens: list[dict[str, Any]]) -> tuple[list[str], str]:
    token_payload = [token["text"] for token in tokens]
    digest = hashlib.sha1(
        json.dumps(token_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return token_payload, digest


def cached_gloss_payload(
    volume_id: str,
    page_number: int,
    page: dict[str, Any],
    tokens: list[dict[str, Any]],
    segmentation_hash: str,
) -> dict[str, Any] | None:
    cached = read_cached_translation(
        volume_id,
        page_number,
        page["source"]["checksum"],
    )
    if cached is None:
        return None
    if cached.get("segmentationHash") != segmentation_hash:
        migrated = migrate_cached_gloss_pairs(cached["tokens"], tokens, page["canonicalText"])
        if migrated is None:
            return None
        cached["tokens"] = migrated
        cached["segmentationHash"] = segmentation_hash
        write_cached_translation(volume_id, page_number, cached)
    cached_pairs = {
        "tokens": [
            {"my": token["my"], "en": token["en"]}
            for token in cached["tokens"]
        ]
    }
    validate_raw_gloss_fields(cached_pairs)
    cached_glosses = SegmentedPageGlosses.model_validate(cached_pairs)
    hydrated_tokens = validate_token_glosses(cached_glosses, tokens)
    if any(set(token) != {"my", "en"} for token in cached["tokens"]):
        cached["tokens"] = cached_pairs["tokens"]
        write_cached_translation(volume_id, page_number, cached)
    return {
        "ok": True,
        "tokens": hydrated_tokens,
        "model": TRANSLATION_MODEL,
        "cached": True,
        "cachePath": str(translation_cache_path(volume_id, page_number)),
    }


def align_token_glosses(
    glosses: SegmentedPageGlosses,
    tokens: list[dict[str, Any]],
) -> dict[int, TokenGloss]:
    expected = [token["text"] for token in tokens]
    returned = [gloss.my for gloss in glosses.tokens]
    matcher = SequenceMatcher(a=expected, b=returned, autojunk=False)
    aligned: dict[int, TokenGloss] = {}
    for expected_start, returned_start, length in matcher.get_matching_blocks():
        for offset in range(length):
            aligned[expected_start + offset] = glosses.tokens[returned_start + offset]
    return aligned


def materialize_token_glosses(
    aligned: dict[int, TokenGloss],
    tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    missing = [index for index in range(len(tokens)) if index not in aligned]
    if missing:
        raise ValueError(f"Gemini left {len(missing)} of {len(tokens)} tokens unglossed; first missing index {missing[0]}")
    return [
        {
            "i": index,
            "my": token["text"],
            "en": aligned[index].en.strip(),
            "startUtf16": token["startUtf16"],
            "endUtf16": token["endUtf16"],
        }
        for index, token in enumerate(tokens)
    ]


def validate_token_glosses(glosses: SegmentedPageGlosses, tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return materialize_token_glosses(align_token_glosses(glosses, tokens), tokens)


def request_token_glosses(
    client: genai.Client,
    target_tokens: list[str],
    context_tokens: list[str] | None = None,
) -> tuple[SegmentedPageGlosses, dict[str, Any]]:
    context_instruction = ""
    if context_tokens is not None:
        context_instruction = f"""

<READ_ONLY_PAGE_CONTEXT>
{json.dumps(context_tokens, ensure_ascii=False, separators=(",", ":"))}
</READ_ONLY_PAGE_CONTEXT>

Use that sequence only for context. Return pairs for TARGET_TOKENS only."""
    prompt = f"""Contextually gloss every TARGET_TOKENS item from one Burmese royal chronicle page.

Return exactly one item for every target token, in the same order, with only my and en. Copy my exactly. In en, give the shortest accurate contextual English gloss, normally one to four words. Do not omit, merge, split, normalize, respell, or reorder tokens. Preserve names, titles, offices, dates, quantities, particles, and uncertainty. For punctuation, numerals, headers, or OCR debris, give the shortest honest equivalent. Do not summarize the page or add historical information.{context_instruction}

<TARGET_TOKENS>
{json.dumps(target_tokens, ensure_ascii=False, separators=(",", ":"))}
</TARGET_TOKENS>"""
    response_schema = exact_segmented_gloss_schema(len(target_tokens))
    response = client.models.generate_content(
        model=TRANSLATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1,
            candidate_count=1,
            max_output_tokens=min(20000, max(1024, len(target_tokens) * 28)),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw_response = json.loads(response.text)
    validate_raw_gloss_fields(raw_response)
    parsed = response_schema.model_validate(raw_response)
    usage: dict[str, Any] = {}
    if getattr(response, "usage_metadata", None) is not None:
        usage = response.usage_metadata.model_dump()
    return parsed, usage


def error_response(message: str, status: int):
    return jsonify({"ok": False, "error": message}), status


@app.get("/")
def root():
    return redirect("/chronicles")


@app.get("/chronicles")
def chronicles_root():
    index = corpus.index()
    first_volume = index["volumes"][0]
    return redirect(f'/chronicles/{first_volume["id"]}/{first_volume["availablePages"][0]}')


@app.get("/chronicles/<volume_id>/<int:page_number>")
def chronicles_page(volume_id: str, page_number: int):
    if not corpus.has_page(volume_id, page_number):
        return render_template(
            "chronicles.html",
            initial_volume=volume_id,
            initial_page=page_number,
            initial_graph_open=False,
        ), 404
    return render_template(
        "chronicles.html",
        initial_volume=volume_id,
        initial_page=page_number,
        initial_graph_open=False,
    )


# The graph has its own stable top-level URL while reusing the reader's one
# canonical page, graph, and navigation state.
@app.get("/knowledge-graph")
def knowledge_graph_root():
    index = corpus.index()
    first_volume = index["volumes"][0]
    return redirect(
        f'/knowledge-graph/{first_volume["id"]}/{first_volume["availablePages"][0]}'
    )


# A page-scoped graph URL opens the same graph implementation used by the
# Chronicle reader rather than maintaining a second graph frontend.
@app.get("/knowledge-graph/<volume_id>/<int:page_number>")
def knowledge_graph_page(volume_id: str, page_number: int):
    status = 200 if corpus.has_page(volume_id, page_number) else 404
    return render_template(
        "chronicles.html",
        initial_volume=volume_id,
        initial_page=page_number,
        initial_graph_open=True,
    ), status


@app.get("/api/chronicles/index")
def chronicles_index():
    return jsonify(corpus.index())


@app.get("/api/chronicles/page/<volume_id>/<int:page_number>")
def chronicles_page_data(volume_id: str, page_number: int):
    try:
        triple_variant = request.args.get("triples", "v3")
        return jsonify(corpus.page(volume_id, page_number, triple_variant))
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/stats")
def graph_stats():
    return jsonify(graph.stats())


def category_filter_values(name: str) -> list[str]:
    return [
        item.strip()
        for value in request.args.getlist(name)
        for item in value.split(",")
        if item.strip()
    ]


def category_scope_arguments() -> dict[str, object]:
    scope = request.args.get("scope", "corpus")
    arguments: dict[str, object] = {"scope": scope}
    if scope == "page":
        volume_id = request.args.get("volumeId")
        page_number = request.args.get("pageNumber")
        if not volume_id or page_number is None:
            raise ValueError("Page scope requires volumeId and pageNumber")
        arguments.update(
            volume_id=volume_id,
            page_number=int(page_number),
        )
    elif scope == "range":
        volume_id = request.args.get("volumeId")
        start_page = request.args.get("startPage")
        end_page = request.args.get("endPage")
        if not volume_id or start_page is None or end_page is None:
            raise ValueError(
                "Range scope requires volumeId, startPage, and endPage"
            )
        arguments.update(
            volume_id=volume_id,
            start_page=int(start_page),
            end_page=int(end_page),
        )
    return arguments


@app.get("/api/graph/categories/catalog")
def graph_category_catalog():
    return jsonify(axial_graph.catalog())


@app.get("/api/graph/categories/tags")
def graph_category_filtered_tags():
    try:
        role = request.args.get("role", "subject")
        anchor_kind = "relation" if role == "relation" else "entity"
        anchor_ids = request.args.getlist("similarTo")
        anchor_labels = [
            record["tag"]
            for record in (
                graph.resolve_tags(anchor_kind, anchor_ids) if anchor_ids else []
            )
        ]

        def applied_tag_labels(name, kind):
            ids = category_filter_values(name)
            return [
                record["tag"] for record in (graph.resolve_tags(kind, ids) if ids else [])
            ]

        payload = axial_graph.filtered_tags(
            role=role,
            subject_ids=category_filter_values("subject"),
            relation_ids=category_filter_values("relation"),
            object_ids=category_filter_values("object"),
            query=request.args.get("q", ""),
            offset=int(request.args.get("offset", "0")),
            limit=int(request.args.get("limit", "100")),
            similar_to=anchor_labels,
            minimum_similarity=float(request.args.get("minSimilarity", "0")),
            subject_tag_labels=applied_tag_labels("subjectTag", "entity"),
            relation_tag_labels=applied_tag_labels("relationTag", "relation"),
            object_tag_labels=applied_tag_labels("objectTag", "entity"),
            direction=request.args.get("direction", "ab"),
            **category_scope_arguments(),
        )
        kind = "relation" if role == "relation" else "entity"
        # A tag without an embedding record has no stable ID, so it is listed
        # but cannot be checked; the frontend renders those as disabled.
        payload["items"] = [
            {
                **item,
                "id": (graph.by_label[kind].get(item["label"]) or {}).get("baseKey"),
                "kind": kind,
                "role": role,
            }
            for item in payload["items"]
        ]
        return jsonify(payload)
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/categories/patterns/similar")
def graph_category_similar_patterns():
    def applied_pattern_tags(name, kind):
        ids = category_filter_values(name)
        return [
            record["tag"] for record in (graph.resolve_tags(kind, ids) if ids else [])
        ]

    try:
        return jsonify(
            axial_graph.similar_patterns(
                subject_ids=category_filter_values("subject"),
                relation_ids=category_filter_values("relation"),
                object_ids=category_filter_values("object"),
                minimum_similarity=float(
                    request.args.get("minSimilarity", "0.9")
                ),
                limit=int(request.args.get("limit", "50")),
                direction=request.args.get("direction", "ab"),
                subject_tag_labels=applied_pattern_tags("subjectTag", "entity"),
                relation_tag_labels=applied_pattern_tags("relationTag", "relation"),
                object_tag_labels=applied_pattern_tags("objectTag", "entity"),
                **category_scope_arguments(),
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/categories/topology/filtered-tags")
def graph_category_filtered_tags_topology():
    try:
        subject_ids = category_filter_values("subjectTag")
        relation_ids = category_filter_values("relationTag")
        object_ids = category_filter_values("objectTag")
        if not subject_ids and not relation_ids and not object_ids:
            raise ValueError("Select at least one filtered tag")
        if len(subject_ids) + len(relation_ids) + len(object_ids) > 100:
            raise ValueError("Select no more than 100 filtered tags")
        subject_records = (
            graph.resolve_tags("entity", subject_ids) if subject_ids else []
        )
        relation_records = (
            graph.resolve_tags("relation", relation_ids) if relation_ids else []
        )
        object_records = (
            graph.resolve_tags("entity", object_ids) if object_ids else []
        )
        return jsonify(
            axial_graph.topology(
                subject_ids=category_filter_values("subjectCategory"),
                relation_ids=category_filter_values("relationCategory"),
                object_ids=category_filter_values("objectCategory"),
                raw_subject_tag_ids=subject_ids,
                raw_subject_tag_labels=[
                    record["tag"] for record in subject_records
                ],
                raw_relation_tag_ids=relation_ids,
                raw_relation_tag_labels=[
                    record["tag"] for record in relation_records
                ],
                raw_object_tag_ids=object_ids,
                raw_object_tag_labels=[
                    record["tag"] for record in object_records
                ],
                direction=request.args.get("direction", "ab"),
                **category_scope_arguments(),
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/categories/topology/tags")
def graph_category_tags_topology():
    try:
        kind = request.args.get("kind", "entity")
        tag_records = graph.resolve_tags(kind, category_filter_values("id"))
        return jsonify(
            axial_graph.topology(
                scope="corpus",
                raw_kind=kind,
                raw_tag_ids=[record["baseKey"] for record in tag_records],
                raw_tag_labels=[record["tag"] for record in tag_records],
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get(
    "/api/graph/categories/topology/page/<volume_id>/<int:page_number>"
)
def graph_category_page_topology(volume_id: str, page_number: int):
    try:
        return jsonify(
            axial_graph.topology(
                scope="page",
                volume_id=volume_id,
                page_number=page_number,
                entity_ids=category_filter_values("entity"),
                relation_ids=category_filter_values("relation"),
            )
        )
    except ValueError as exc:
        return error_response(str(exc), 400)


@app.get(
    "/api/graph/categories/topology/range/"
    "<volume_id>/<int:start_page>/<int:end_page>"
)
def graph_category_range_topology(
    volume_id: str,
    start_page: int,
    end_page: int,
):
    try:
        return jsonify(
            axial_graph.topology(
                scope="range",
                volume_id=volume_id,
                start_page=start_page,
                end_page=end_page,
                entity_ids=category_filter_values("entity"),
                relation_ids=category_filter_values("relation"),
            )
        )
    except ValueError as exc:
        return error_response(str(exc), 400)


@app.get("/api/graph/categories/topology/overview/<scope>")
def graph_category_overview_topology(scope: str):
    try:
        return jsonify(
            axial_graph.topology(
                scope=scope,
                entity_ids=category_filter_values("entity"),
                relation_ids=category_filter_values("relation"),
            )
        )
    except ValueError as exc:
        return error_response(str(exc), 400)


@app.get("/api/graph/categories/evidence")
def graph_category_evidence():
    try:
        raw_tag_ids = category_filter_values("tagId")
        raw_kind = request.args.get("tagKind") if raw_tag_ids else None
        raw_tag_records = (
            graph.resolve_tags(raw_kind or "", raw_tag_ids)
            if raw_tag_ids
            else []
        )
        raw_subject_ids = category_filter_values("subjectTag")
        raw_relation_ids = category_filter_values("relationTag")
        raw_object_ids = category_filter_values("objectTag")
        raw_subject_records = (
            graph.resolve_tags("entity", raw_subject_ids)
            if raw_subject_ids
            else []
        )
        raw_relation_records = (
            graph.resolve_tags("relation", raw_relation_ids)
            if raw_relation_ids
            else []
        )
        raw_object_records = (
            graph.resolve_tags("entity", raw_object_ids)
            if raw_object_ids
            else []
        )
        return jsonify(
            axial_graph.evidence(
                request.args.get("source", ""),
                request.args.get("target", ""),
                relation_ids=category_filter_values("relation"),
                raw_kind=raw_kind,
                raw_labels=[record["tag"] for record in raw_tag_records],
                raw_subject_labels=[
                    record["tag"] for record in raw_subject_records
                ],
                raw_relation_labels=[
                    record["tag"] for record in raw_relation_records
                ],
                raw_object_labels=[
                    record["tag"] for record in raw_object_records
                ],
                scope=request.args.get("scope", "corpus"),
                volume_id=request.args.get("volumeId"),
                page_number=(
                    int(request.args["pageNumber"])
                    if "pageNumber" in request.args
                    else None
                ),
                start_page=(
                    int(request.args["startPage"])
                    if "startPage" in request.args
                    else None
                ),
                end_page=(
                    int(request.args["endPage"])
                    if "endPage" in request.args
                    else None
                ),
                offset=int(request.args.get("offset", "0")),
                limit=int(request.args.get("limit", "20")),
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/search")
def graph_search():
    try:
        return jsonify(
            graph.search(
                query=request.args.get("q", ""),
                kind=request.args.get("kind", "all"),
                sort=request.args.get("sort", "frequency"),
                anchor_kind=request.args.get("anchorKind"),
                anchor_id=request.args.get("anchorId"),
                minimum_similarity=(
                    float(request.args["minSimilarity"])
                    if "minSimilarity" in request.args
                    else None
                ),
                limit=int(request.args.get("limit", "50")),
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/topology/tags")
def graph_tags_topology():
    try:
        return jsonify(
            graph.tags_topology(
                request.args.get("kind", "entity"),
                category_filter_values("id"),
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/topology/page/<volume_id>/<int:page_number>")
def graph_page_topology(volume_id: str, page_number: int):
    try:
        return jsonify(
            graph.page_topology(
                volume_id,
                page_number,
                limit=int(request.args.get("limit", "500")),
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)


@app.get(
    "/api/graph/topology/range/"
    "<volume_id>/<int:start_page>/<int:end_page>"
)
def graph_page_range_topology(
    volume_id: str,
    start_page: int,
    end_page: int,
):
    try:
        return jsonify(
            graph.page_range_topology(
                volume_id,
                start_page,
                end_page,
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)


@app.get("/api/graph/topology/overview/<scope>")
def graph_overview_topology(scope: str):
    try:
        plain_path, gzip_path = graph.overview_files(scope)
        accepts_gzip = "gzip" in request.headers.get("Accept-Encoding", "")
        response = send_file(
            gzip_path if accepts_gzip else plain_path,
            mimetype="application/json",
            conditional=True,
            etag=True,
            max_age=3600,
        )
        if accepts_gzip:
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Vary"] = "Accept-Encoding"
        return response
    except ValueError as exc:
        return error_response(str(exc), 400)
    except RuntimeError as exc:
        return error_response(str(exc), 503)


@app.get("/api/graph/topology/entity/<entity_id>")
def graph_entity_topology(entity_id: str):
    try:
        return jsonify(
            graph.entity_topology(
                entity_id,
                depth=int(request.args.get("depth", "1")),
                limit=int(request.args.get("limit", "300")),
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/topology/relation/<relation_id>")
def graph_relation_topology(relation_id: str):
    try:
        return jsonify(
            graph.relation_topology(
                relation_id,
                limit=int(request.args.get("limit", "300")),
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/topology/claim/<sentence_id>/<int:ordinal>")
def graph_claim_topology(sentence_id: str, ordinal: int):
    try:
        return jsonify(graph.claim_topology(sentence_id, ordinal))
    except ValueError as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.get("/api/graph/evidence")
def graph_evidence():
    try:
        sentence_id = request.args.get("sentenceId")
        ordinal_value = request.args.get("ordinal")
        return jsonify(
            graph.evidence(
                request.args.get("source", ""),
                request.args.get("relation", ""),
                request.args.get("target", ""),
                offset=int(request.args.get("offset", "0")),
                limit=int(request.args.get("limit", "20")),
                sentence_id=sentence_id,
                ordinal=int(ordinal_value) if ordinal_value is not None else None,
            )
        )
    except (TypeError, ValueError) as exc:
        return error_response(str(exc), 400)
    except KeyError as exc:
        return error_response(str(exc), 404)


@app.post("/api/chronicles/segment")
def chronicles_segment():
    payload = request.get_json(silent=True) or {}
    volume_id = str(payload.get("volumeId", ""))
    try:
        page_number = int(payload.get("pageNumber"))
        page = corpus.page(volume_id, page_number)
        result = reader_bridge.segment_page(page["canonicalText"])
        _, segmentation_hash = segmentation_identity(result["tokens"])
        with translation_page_lock(volume_id, page_number):
            result["cachedGlosses"] = cached_gloss_payload(
                volume_id,
                page_number,
                page,
                result["tokens"],
                segmentation_hash,
            )
        return jsonify({"ok": True, **result})
    except (TypeError, ValueError, KeyError) as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        app.logger.exception("Chronicle segmentation failed")
        return error_response(f"Segmentation failed: {exc}", 500)


@app.post("/api/chronicles/translate")
def chronicles_translate():
    payload = request.get_json(silent=True) or {}
    volume_id = str(payload.get("volumeId", ""))
    try:
        page_number = int(payload.get("pageNumber"))
        page = corpus.page(volume_id, page_number)
    except (TypeError, ValueError, KeyError) as exc:
        return error_response(str(exc), 400)

    try:
        segmentation = reader_bridge.segment_page(page["canonicalText"])
    except Exception as exc:
        app.logger.exception("Unable to segment page before translation")
        return error_response(f"Segmentation failed: {exc}", 500)
    tokens = segmentation["tokens"]
    token_payload, segmentation_hash = segmentation_identity(tokens)

    with translation_page_lock(volume_id, page_number):
        try:
            cached = cached_gloss_payload(volume_id, page_number, page, tokens, segmentation_hash)
            if cached is not None:
                return jsonify(cached)

            api_key = gemini_api_key()
            if not api_key:
                return error_response("GEMINI_API_KEY is not configured", 503)

            client = genai.Client(api_key=api_key)
            parsed, initial_usage = request_token_glosses(client, token_payload)
            aligned = align_token_glosses(parsed, tokens)
            usage_calls = [initial_usage]
            for _ in range(3):
                missing = [index for index in range(len(tokens)) if index not in aligned]
                if not missing:
                    break
                repair_tokens = [tokens[index]["text"] for index in missing]
                repair, repair_usage = request_token_glosses(client, repair_tokens, token_payload)
                repaired = align_token_glosses(repair, [tokens[index] for index in missing])
                before = len(aligned)
                for local_index, gloss in repaired.items():
                    aligned[missing[local_index]] = gloss
                usage_calls.append(repair_usage)
                if len(aligned) == before:
                    break
            glossed_tokens = materialize_token_glosses(aligned, tokens)
            record = {
                "schemaVersion": TRANSLATION_SCHEMA_VERSION,
                "pageId": page["id"],
                "volumeId": volume_id,
                "pageNumber": page_number,
                "pageChecksum": page["source"]["checksum"],
                "segmentationHash": segmentation_hash,
                "model": TRANSLATION_MODEL,
                "tokens": [{"my": token["my"], "en": token["en"]} for token in glossed_tokens],
            }
            cache_path = write_cached_translation(volume_id, page_number, record)
            return jsonify(
                {
                    "ok": True,
                    "tokens": glossed_tokens,
                    "model": TRANSLATION_MODEL,
                    "cached": False,
                    "cachePath": str(cache_path),
                    "usage": {"calls": usage_calls},
                }
            )
        except Exception as exc:
            app.logger.exception("Gemini page translation failed")
            return error_response(f"Translation failed: {exc}", 502)


@app.get("/api/chronicles/glosses/<volume_id>/<int:page_number>")
def chronicles_cached_glosses(volume_id: str, page_number: int):
    try:
        page = corpus.page(volume_id, page_number)
        segmentation = reader_bridge.segment_page(page["canonicalText"])
        tokens = segmentation["tokens"]
        _, segmentation_hash = segmentation_identity(tokens)
        with translation_page_lock(volume_id, page_number):
            cached = cached_gloss_payload(volume_id, page_number, page, tokens, segmentation_hash)
        if cached is None:
            return jsonify({"ok": True, "cached": False, "tokens": []})
        return jsonify(cached)
    except (TypeError, ValueError, KeyError) as exc:
        return error_response(str(exc), 400)
    except Exception as exc:
        app.logger.exception("Unable to load cached glosses")
        return error_response(f"Cached gloss load failed: {exc}", 500)


if __name__ == "__main__":
    reader_bridge.ensure_loaded()
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5077")), debug=False)
