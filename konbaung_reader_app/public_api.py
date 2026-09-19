from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from flask import (
    Blueprint,
    Flask,
    current_app,
    g,
    jsonify,
    render_template,
    request,
)


API_NAME = "Konbaung Chronicle Graph API"
API_VERSION = "1.0.0"
MAX_PAGE_SIZE = 100
MAX_TEXT_PAGE_SIZE = 25
MAX_TAG_FILTERS = 100
DEFAULT_RATE_LIMIT = 120
RATE_WINDOW_SECONDS = 60
VALID_VOLUMES = {"vol1", "vol2", "vol3"}


class ApiProblem(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        status: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class SlidingWindowRateLimiter:
    """Small in-process guard; a reverse proxy should enforce the global limit."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = {}

    def clear(self) -> None:
        with self._lock:
            self._events.clear()

    def check(
        self,
        key: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        now = time.time()
        boundary = now - window_seconds
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and events[0] <= boundary:
                events.popleft()
            if len(events) >= limit:
                reset = max(1, int(events[0] + window_seconds - now + 0.999))
                return False, 0, reset
            events.append(now)
            remaining = max(0, limit - len(events))
            reset = max(1, int(events[0] + window_seconds - now + 0.999))
            if len(self._events) > 10_000:
                stale = [
                    candidate
                    for candidate, candidate_events in self._events.items()
                    if not candidate_events or candidate_events[-1] <= boundary
                ]
                for candidate in stale[:2_000]:
                    self._events.pop(candidate, None)
            return True, remaining, reset


@dataclass(frozen=True)
class TripleFilters:
    subjects: tuple[str, ...] = ()
    predicates: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    subject_contains: str = ""
    predicate_contains: str = ""
    object_contains: str = ""
    subject_tag_ids: tuple[str, ...] = ()
    object_tag_ids: tuple[str, ...] = ()
    entity_tag_ids: tuple[str, ...] = ()
    relation_tag_ids: tuple[str, ...] = ()
    subject_tag_labels: tuple[str, ...] = ()
    object_tag_labels: tuple[str, ...] = ()
    entity_tag_labels: tuple[str, ...] = ()
    relation_tag_labels: tuple[str, ...] = ()
    subject_categories: tuple[str, ...] = ()
    relation_categories: tuple[str, ...] = ()
    object_categories: tuple[str, ...] = ()
    volume: str | None = None
    page: int | None = None
    start_page: int | None = None
    end_page: int | None = None
    sentence_id: str | None = None

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        for internal in (
            "subject_tag_labels",
            "object_tag_labels",
            "entity_tag_labels",
            "relation_tag_labels",
        ):
            payload.pop(internal, None)
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in payload.items()
            if value not in (None, "", (), [])
        }


class PublicGraphApi:
    def __init__(self, graph: Any, axial_graph: Any) -> None:
        self.graph = graph
        self.axial_graph = axial_graph
        self.limiter = SlidingWindowRateLimiter()
        self.records_by_id = {
            str(record["id"]): record for record in axial_graph.records
        }
        self.tag_id_by_label = {
            kind: {
                str(record["tag"]): str(record["baseKey"])
                for record in graph.records[kind]
            }
            for kind in ("entity", "relation")
        }
        self.blueprint = Blueprint(
            "public_graph_api",
            __name__,
            url_prefix="/api/v1",
        )
        self._register_routes()

    @staticmethod
    def configure_app(app: Flask) -> None:
        app.config.setdefault(
            "PUBLIC_API_RATE_LIMIT",
            int(os.getenv("PUBLIC_API_RATE_LIMIT", str(DEFAULT_RATE_LIMIT))),
        )
        app.config.setdefault(
            "PUBLIC_API_CORS_ORIGINS",
            os.getenv("PUBLIC_API_CORS_ORIGINS", "*"),
        )
        app.config.setdefault(
            "PUBLIC_API_KEYS",
            os.getenv("PUBLIC_API_KEYS", ""),
        )

    @staticmethod
    def _problem(
        message: str,
        status: int,
        code: str,
    ) -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": {"code": code, "message": message},
                }
            ),
            status,
        )

    @staticmethod
    def _values(name: str, *, maximum: int = MAX_TAG_FILTERS) -> list[str]:
        values = [
            item.strip()
            for raw in request.args.getlist(name)
            for item in raw.split(",")
            if item.strip()
        ]
        values = list(dict.fromkeys(values))
        if len(values) > maximum:
            raise ApiProblem(f"{name} accepts at most {maximum} values")
        if any(len(value) > 300 for value in values):
            raise ApiProblem(f"{name} contains a value longer than 300 characters")
        return values

    @staticmethod
    def _text(name: str, *, maximum: int = 200) -> str:
        value = request.args.get(name, "").strip()
        if len(value) > maximum:
            raise ApiProblem(f"{name} must be at most {maximum} characters")
        return value

    @staticmethod
    def _integer(
        name: str,
        *,
        default: int | None = None,
        minimum: int = 0,
        maximum: int | None = None,
    ) -> int | None:
        raw = request.args.get(name)
        if raw is None or raw == "":
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ApiProblem(f"{name} must be an integer") from exc
        if value < minimum or (maximum is not None and value > maximum):
            if maximum is None:
                raise ApiProblem(f"{name} must be at least {minimum}")
            raise ApiProblem(
                f"{name} must be between {minimum} and {maximum}"
            )
        return value

    @staticmethod
    def _boolean(name: str, default: bool = False) -> bool:
        raw = request.args.get(name)
        if raw is None:
            return default
        normalized = raw.strip().casefold()
        if normalized in {"1", "true", "yes"}:
            return True
        if normalized in {"0", "false", "no"}:
            return False
        raise ApiProblem(f"{name} must be true or false")

    @staticmethod
    def _reject_unknown(allowed: set[str]) -> None:
        unknown = sorted(set(request.args) - allowed)
        if unknown:
            raise ApiProblem(
                f"Unknown query parameter{'s' if len(unknown) != 1 else ''}: "
                + ", ".join(unknown)
            )

    def _resolve_tag_labels(
        self,
        kind: str,
        tag_ids: Iterable[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        ids = tuple(tag_ids)
        if not ids:
            return (), ()
        try:
            records = self.graph.resolve_tags(kind, ids)
        except KeyError as exc:
            raise ApiProblem(
                str(exc),
                code="tag_not_found",
                status=404,
            ) from exc
        except ValueError as exc:
            raise ApiProblem(str(exc)) from exc
        return (
            tuple(str(record["baseKey"]) for record in records),
            tuple(str(record["tag"]) for record in records),
        )

    def _parse_filters(self, *, extra_allowed: Iterable[str] = ()) -> TripleFilters:
        allowed = {
            "subject",
            "predicate",
            "object",
            "subject_contains",
            "predicate_contains",
            "object_contains",
            "subject_tag_id",
            "object_tag_id",
            "entity_tag_id",
            "relation_tag_id",
            "subject_category",
            "relation_category",
            "object_category",
            "volume",
            "page",
            "start_page",
            "end_page",
            "sentence_id",
            *extra_allowed,
        }
        self._reject_unknown(allowed)

        volume = self._text("volume", maximum=12) or None
        if volume is not None and volume not in VALID_VOLUMES:
            raise ApiProblem("volume must be vol1, vol2, or vol3")
        page = self._integer("page", minimum=1)
        start_page = self._integer("start_page", minimum=1)
        end_page = self._integer("end_page", minimum=1)
        if page is not None and (start_page is not None or end_page is not None):
            raise ApiProblem("page cannot be combined with start_page or end_page")
        if (start_page is None) != (end_page is None):
            raise ApiProblem("start_page and end_page must be provided together")
        if start_page is not None and end_page is not None and start_page > end_page:
            raise ApiProblem("start_page cannot be greater than end_page")
        if (page is not None or start_page is not None) and volume is None:
            raise ApiProblem("volume is required with page or page ranges")

        subject_categories = tuple(self._values("subject_category"))
        relation_categories = tuple(self._values("relation_category"))
        object_categories = tuple(self._values("object_category"))
        unknown_entities = (
            set(subject_categories) | set(object_categories)
        ) - self.axial_graph.entities.keys()
        unknown_relations = (
            set(relation_categories) - self.axial_graph.relations.keys()
        )
        if unknown_entities:
            raise ApiProblem(
                "Unknown entity categories: " + ", ".join(sorted(unknown_entities))
            )
        if unknown_relations:
            raise ApiProblem(
                "Unknown relation categories: " + ", ".join(sorted(unknown_relations))
            )

        subject_tag_ids, subject_tag_labels = self._resolve_tag_labels(
            "entity", self._values("subject_tag_id")
        )
        object_tag_ids, object_tag_labels = self._resolve_tag_labels(
            "entity", self._values("object_tag_id")
        )
        entity_tag_ids, entity_tag_labels = self._resolve_tag_labels(
            "entity", self._values("entity_tag_id")
        )
        relation_tag_ids, relation_tag_labels = self._resolve_tag_labels(
            "relation", self._values("relation_tag_id")
        )

        sentence_id = self._text("sentence_id", maximum=100) or None
        return TripleFilters(
            subjects=tuple(self._values("subject")),
            predicates=tuple(self._values("predicate")),
            objects=tuple(self._values("object")),
            subject_contains=self._text("subject_contains"),
            predicate_contains=self._text("predicate_contains"),
            object_contains=self._text("object_contains"),
            subject_tag_ids=subject_tag_ids,
            object_tag_ids=object_tag_ids,
            entity_tag_ids=entity_tag_ids,
            relation_tag_ids=relation_tag_ids,
            subject_tag_labels=subject_tag_labels,
            object_tag_labels=object_tag_labels,
            entity_tag_labels=entity_tag_labels,
            relation_tag_labels=relation_tag_labels,
            subject_categories=subject_categories,
            relation_categories=relation_categories,
            object_categories=object_categories,
            volume=volume,
            page=page,
            start_page=start_page,
            end_page=end_page,
            sentence_id=sentence_id,
        )

    @staticmethod
    def _contains(value: str, query: str) -> bool:
        return query.casefold() in value.casefold()

    @classmethod
    def _matches(cls, record: dict[str, Any], filters: TripleFilters) -> bool:
        if filters.subjects and record["subject"] not in filters.subjects:
            return False
        if filters.predicates and record["predicate"] not in filters.predicates:
            return False
        if filters.objects and record["object"] not in filters.objects:
            return False
        if filters.subject_contains and not cls._contains(
            record["subject"], filters.subject_contains
        ):
            return False
        if filters.predicate_contains and not cls._contains(
            record["predicate"], filters.predicate_contains
        ):
            return False
        if filters.object_contains and not cls._contains(
            record["object"], filters.object_contains
        ):
            return False
        if (
            filters.subject_tag_labels
            and record["subject"] not in filters.subject_tag_labels
        ):
            return False
        if (
            filters.object_tag_labels
            and record["object"] not in filters.object_tag_labels
        ):
            return False
        if filters.entity_tag_labels and (
            record["subject"] not in filters.entity_tag_labels
            and record["object"] not in filters.entity_tag_labels
        ):
            return False
        if (
            filters.relation_tag_labels
            and record["predicate"] not in filters.relation_tag_labels
        ):
            return False
        categories = record["categories"]
        if (
            filters.subject_categories
            and categories["s"] not in filters.subject_categories
        ):
            return False
        if (
            filters.relation_categories
            and categories["r"] not in filters.relation_categories
        ):
            return False
        if (
            filters.object_categories
            and categories["o"] not in filters.object_categories
        ):
            return False
        if filters.volume is not None and record["volumeId"] != filters.volume:
            return False
        if filters.page is not None and filters.page not in record["pages"]:
            return False
        if filters.start_page is not None and not any(
            filters.start_page <= int(page) <= int(filters.end_page or -1)
            for page in record["pages"]
        ):
            return False
        if filters.sentence_id is not None and record["sid"] != filters.sentence_id:
            return False
        return True

    def _matching_records(self, filters: TripleFilters) -> list[dict[str, Any]]:
        return [
            record
            for record in self.axial_graph.records
            if self._matches(record, filters)
        ]

    @staticmethod
    def _category_summary(category: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": category["id"],
            "tag": category["tagId"],
            "label": category["label"],
            "definition": category["definition"],
        }

    def _tag_summary(self, kind: str, label: str) -> dict[str, Any]:
        return {
            "id": self.tag_id_by_label[kind].get(label),
            "label": label,
        }

    def _triple_item(
        self,
        record: dict[str, Any],
        *,
        include_text: bool,
    ) -> dict[str, Any]:
        categories = record["categories"]
        item: dict[str, Any] = {
            "id": record["id"],
            "triple": {
                "subject": self._tag_summary("entity", record["subject"]),
                "predicate": self._tag_summary("relation", record["predicate"]),
                "object": self._tag_summary("entity", record["object"]),
            },
            "thematic": {
                "subject": self._category_summary(
                    self.axial_graph.entities[categories["s"]]
                ),
                "relation": self._category_summary(
                    self.axial_graph.relations[categories["r"]]
                ),
                "object": self._category_summary(
                    self.axial_graph.entities[categories["o"]]
                ),
            },
            "source": {
                "sentenceId": record["sid"],
                "ordinal": record["ordinal"],
                "volume": record["volumeId"],
                "ownerPage": record["ownerPage"],
                "pages": record["pages"],
                "readerUrl": (
                    f"/chronicles/{record['volumeId']}/{record['ownerPage']}"
                ),
            },
        }
        if include_text:
            item["sentence"] = {
                "my": record["sentenceMy"],
                "en": record["sentenceEn"],
            }
        return item

    @staticmethod
    def _fingerprint(resource: str, filters: TripleFilters) -> str:
        payload = json.dumps(
            {"resource": resource, "filters": filters.public()},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    @classmethod
    def _encode_cursor(cls, offset: int, fingerprint: str) -> str:
        raw = json.dumps(
            {"v": 1, "o": offset, "q": fingerprint},
            separators=(",", ":"),
        ).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @classmethod
    def _decode_cursor(cls, value: str, fingerprint: str) -> int:
        if not value:
            return 0
        if len(value) > 300:
            raise ApiProblem("cursor is invalid")
        try:
            padding = "=" * (-len(value) % 4)
            payload = json.loads(
                base64.urlsafe_b64decode(value + padding).decode("utf-8")
            )
            offset = int(payload["o"])
            if payload.get("v") != 1 or payload.get("q") != fingerprint:
                raise ValueError
            if offset < 0 or offset > 10_000_000:
                raise ValueError
            return offset
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ApiProblem(
                "cursor is invalid or belongs to a different query",
                code="invalid_cursor",
            ) from exc

    @classmethod
    def _pagination(
        cls,
        *,
        offset: int,
        limit: int,
        total: int,
        fingerprint: str,
    ) -> dict[str, Any]:
        next_offset = offset + limit
        has_more = next_offset < total
        return {
            "limit": limit,
            "returned": max(0, min(limit, total - offset)),
            "total": total,
            "hasMore": has_more,
            "nextCursor": (
                cls._encode_cursor(next_offset, fingerprint)
                if has_more
                else None
            ),
        }

    def _authorized_key(self) -> str | None:
        configured = [
            value.strip()
            for value in str(current_app.config.get("PUBLIC_API_KEYS", "")).split(",")
            if value.strip()
        ]
        if not configured:
            return None
        provided = request.headers.get("X-API-Key", "").strip()
        authorization = request.headers.get("Authorization", "")
        if not provided and authorization.casefold().startswith("bearer "):
            provided = authorization[7:].strip()
        if not provided or not any(
            secrets.compare_digest(provided, candidate) for candidate in configured
        ):
            raise ApiProblem(
                "A valid API key is required",
                code="authentication_required",
                status=401,
            )
        return hashlib.sha256(provided.encode("utf-8")).hexdigest()[:20]

    def _register_routes(self) -> None:
        bp = self.blueprint

        @bp.before_request
        def public_api_guard():
            if request.method == "OPTIONS":
                return None
            exempt = {
                "public_graph_api.landing",
                "public_graph_api.docs",
                "public_graph_api.openapi",
                "public_graph_api.health",
            }
            if request.endpoint in exempt:
                return None
            try:
                api_key = self._authorized_key()
            except ApiProblem as exc:
                return self._problem(exc.message, exc.status, exc.code)
            limit = max(
                1,
                int(current_app.config.get("PUBLIC_API_RATE_LIMIT", DEFAULT_RATE_LIMIT)),
            )
            remote = request.remote_addr or "unknown"
            rate_key = f"key:{api_key}" if api_key else f"ip:{remote}"
            allowed, remaining, reset = self.limiter.check(
                rate_key,
                limit=limit,
                window_seconds=RATE_WINDOW_SECONDS,
            )
            g.public_api_rate = {
                "limit": limit,
                "remaining": remaining,
                "reset": reset,
            }
            if not allowed:
                return self._problem(
                    "Rate limit exceeded; retry after the indicated delay",
                    429,
                    "rate_limit_exceeded",
                )
            return None

        @bp.after_request
        def public_api_headers(response):
            origins = str(
                current_app.config.get("PUBLIC_API_CORS_ORIGINS", "*")
            ).strip()
            request_origin = request.headers.get("Origin", "")
            allowed_origins = {
                origin.strip() for origin in origins.split(",") if origin.strip()
            }
            if "*" in allowed_origins:
                response.headers["Access-Control-Allow-Origin"] = "*"
            elif request_origin and request_origin in allowed_origins:
                response.headers["Access-Control-Allow-Origin"] = request_origin
                response.headers.add("Vary", "Origin")
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Accept, Content-Type, Authorization, X-API-Key"
            )
            response.headers["Access-Control-Expose-Headers"] = (
                "X-RateLimit-Limit, X-RateLimit-Remaining, Retry-After"
            )
            response.headers["X-Content-Type-Options"] = "nosniff"
            if response.status_code == 401:
                response.headers["WWW-Authenticate"] = "Bearer"
            rate = getattr(g, "public_api_rate", None)
            if rate:
                response.headers["X-RateLimit-Limit"] = str(rate["limit"])
                response.headers["X-RateLimit-Remaining"] = str(rate["remaining"])
                if response.status_code == 429:
                    response.headers["Retry-After"] = str(rate["reset"])
            if request.method == "GET" and response.status_code == 200:
                endpoint = request.endpoint or ""
                if endpoint.endswith(("openapi", "categories", "stats")):
                    response.cache_control.public = True
                    response.cache_control.max_age = 3600
                elif endpoint.endswith(("tags", "similar_tags")):
                    response.cache_control.public = True
                    response.cache_control.max_age = 300
                elif endpoint.endswith(("triples", "thematic_patterns", "claim")):
                    response.cache_control.public = True
                    response.cache_control.max_age = 60
            return response

        @bp.errorhandler(ApiProblem)
        def api_problem(exc: ApiProblem):
            return self._problem(exc.message, exc.status, exc.code)

        @bp.get("")
        @bp.get("/")
        def landing():
            return jsonify(
                {
                    "ok": True,
                    "name": API_NAME,
                    "version": API_VERSION,
                    "readOnly": True,
                    "documentation": "/api/v1/docs",
                    "openapi": "/api/v1/openapi.json",
                    "endpoints": {
                        "triples": "/api/v1/triples",
                        "thematicPatterns": "/api/v1/thematic-patterns",
                        "tags": "/api/v1/tags",
                        "similarTags": "/api/v1/tags/{kind}/{tag_id}/similar",
                        "categories": "/api/v1/categories",
                        "stats": "/api/v1/stats",
                    },
                    "limits": {
                        "maximumPageSize": MAX_PAGE_SIZE,
                        "maximumTagFilters": MAX_TAG_FILTERS,
                        "requestsPerMinute": int(
                            current_app.config.get(
                                "PUBLIC_API_RATE_LIMIT", DEFAULT_RATE_LIMIT
                            )
                        ),
                    },
                }
            )

        @bp.get("/health")
        def health():
            return jsonify(
                {
                    "ok": True,
                    "status": "ready",
                    "version": API_VERSION,
                    "tripleCount": len(self.axial_graph.records),
                }
            )

        @bp.get("/stats")
        def stats():
            self._reject_unknown(set())
            graph_stats = self.graph.stats()
            return jsonify(
                {
                    "ok": True,
                    "version": API_VERSION,
                    "counts": {
                        "triples": len(self.axial_graph.records),
                        "entityTags": len(self.graph.records["entity"]),
                        "relationTags": len(self.graph.records["relation"]),
                        "entityCategories": len(self.axial_graph.entities),
                        "relationCategories": len(self.axial_graph.relations),
                        "thematicPatterns": len(self.axial_graph.pattern_keys),
                    },
                    "embeddings": graph_stats["embeddingSort"],
                    "generatedAt": graph_stats["generatedAt"],
                }
            )

        @bp.get("/categories")
        def categories():
            self._reject_unknown({"kind"})
            kind = request.args.get("kind", "all")
            if kind not in {"all", "entity", "relation"}:
                raise ApiProblem("kind must be all, entity, or relation")
            payload: dict[str, Any] = {
                "ok": True,
                "version": API_VERSION,
            }
            if kind in {"all", "entity"}:
                payload["entities"] = [
                    self._category_summary(item)
                    for item in self.axial_graph._catalog["entities"]
                ]
            if kind in {"all", "relation"}:
                payload["relations"] = [
                    self._category_summary(item)
                    for item in self.axial_graph._catalog["relations"]
                ]
            return jsonify(payload)

        @bp.get("/tags")
        def tags():
            self._reject_unknown({"q", "kind", "limit"})
            query = self._text("q")
            kind = request.args.get("kind", "all")
            limit = int(self._integer("limit", default=25, minimum=1, maximum=100) or 25)
            try:
                result = self.graph.search(
                    query=query,
                    kind=kind,
                    sort="frequency",
                    limit=limit,
                )
            except ValueError as exc:
                raise ApiProblem(str(exc)) from exc
            return jsonify(
                {
                    "ok": True,
                    "version": API_VERSION,
                    "query": {"q": query, "kind": kind},
                    "data": result["results"],
                    "count": len(result["results"]),
                }
            )

        @bp.get("/tags/<kind>/<tag_id>/similar")
        def similar_tags(kind: str, tag_id: str):
            self._reject_unknown({"minimum_similarity", "limit"})
            if kind not in {"entity", "relation"}:
                raise ApiProblem("kind must be entity or relation")
            try:
                minimum_similarity = float(
                    request.args.get("minimum_similarity", "0.90")
                )
            except ValueError as exc:
                raise ApiProblem("minimum_similarity must be a number") from exc
            if not -1 <= minimum_similarity <= 1:
                raise ApiProblem("minimum_similarity must be between -1 and 1")
            limit = int(self._integer("limit", default=50, minimum=1, maximum=100) or 50)
            try:
                result = self.graph.search(
                    kind=kind,
                    sort="gemini",
                    anchor_kind=kind,
                    anchor_id=tag_id,
                    minimum_similarity=minimum_similarity,
                    limit=limit,
                )
            except KeyError as exc:
                raise ApiProblem(
                    str(exc), code="tag_not_found", status=404
                ) from exc
            except ValueError as exc:
                raise ApiProblem(str(exc)) from exc
            return jsonify(
                {
                    "ok": True,
                    "version": API_VERSION,
                    "anchor": result["anchor"],
                    "minimumSimilarity": minimum_similarity,
                    "data": result["results"],
                    "count": len(result["results"]),
                }
            )

        @bp.get("/triples")
        def triples():
            filters = self._parse_filters(
                extra_allowed={"limit", "cursor", "include_text"}
            )
            limit = int(
                self._integer(
                    "limit", default=25, minimum=1, maximum=MAX_PAGE_SIZE
                )
                or 25
            )
            include_text = self._boolean("include_text", False)
            if include_text and limit > MAX_TEXT_PAGE_SIZE:
                raise ApiProblem(
                    f"limit cannot exceed {MAX_TEXT_PAGE_SIZE} when include_text=true"
                )
            fingerprint = self._fingerprint("triples", filters)
            offset = self._decode_cursor(request.args.get("cursor", ""), fingerprint)
            matching = self._matching_records(filters)
            selected = matching[offset : offset + limit]
            return jsonify(
                {
                    "ok": True,
                    "version": API_VERSION,
                    "query": filters.public(),
                    "data": [
                        self._triple_item(record, include_text=include_text)
                        for record in selected
                    ],
                    "pagination": self._pagination(
                        offset=offset,
                        limit=limit,
                        total=len(matching),
                        fingerprint=fingerprint,
                    ),
                }
            )

        @bp.get("/triples/<claim_id>")
        def claim(claim_id: str):
            self._reject_unknown({"include_text"})
            if len(claim_id) > 150:
                raise ApiProblem("claim ID is invalid")
            record = self.records_by_id.get(claim_id)
            if record is None:
                raise ApiProblem(
                    f"Unknown claim: {claim_id}",
                    code="claim_not_found",
                    status=404,
                )
            include_text = self._boolean("include_text", True)
            return jsonify(
                {
                    "ok": True,
                    "version": API_VERSION,
                    "data": self._triple_item(record, include_text=include_text),
                }
            )

        @bp.get("/thematic-patterns")
        def thematic_patterns():
            filters = self._parse_filters(extra_allowed={"limit", "cursor"})
            limit = int(
                self._integer(
                    "limit", default=50, minimum=1, maximum=MAX_PAGE_SIZE
                )
                or 50
            )
            matching = self._matching_records(filters)
            counts: Counter[tuple[str, str, str]] = Counter(
                (
                    record["categories"]["s"],
                    record["categories"]["r"],
                    record["categories"]["o"],
                )
                for record in matching
            )
            ordered = sorted(
                counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            fingerprint = self._fingerprint("thematic-patterns", filters)
            offset = self._decode_cursor(request.args.get("cursor", ""), fingerprint)
            selected = ordered[offset : offset + limit]
            data = []
            for (source_id, relation_id, object_id), count in selected:
                data.append(
                    {
                        "subject": self._category_summary(
                            self.axial_graph.entities[source_id]
                        ),
                        "relation": self._category_summary(
                            self.axial_graph.relations[relation_id]
                        ),
                        "object": self._category_summary(
                            self.axial_graph.entities[object_id]
                        ),
                        "tripleCount": count,
                    }
                )
            return jsonify(
                {
                    "ok": True,
                    "version": API_VERSION,
                    "query": filters.public(),
                    "matchingTripleCount": len(matching),
                    "data": data,
                    "pagination": self._pagination(
                        offset=offset,
                        limit=limit,
                        total=len(ordered),
                        fingerprint=fingerprint,
                    ),
                }
            )

        @bp.get("/openapi.json")
        def openapi():
            return jsonify(self._openapi_schema())

        @bp.get("/docs")
        def docs():
            return render_template(
                "api_docs.html",
                api_name=API_NAME,
                api_version=API_VERSION,
                rate_limit=int(
                    current_app.config.get(
                        "PUBLIC_API_RATE_LIMIT", DEFAULT_RATE_LIMIT
                    )
                ),
            )

    @staticmethod
    def _openapi_schema() -> dict[str, Any]:
        def filter_parameter(
            name: str,
            description: str,
            repeated: bool,
            schema_type: str,
        ) -> dict[str, Any]:
            parameter: dict[str, Any] = {
                "name": name,
                "in": "query",
                "required": False,
                "description": description,
                "schema": (
                    {"type": "array", "items": {"type": schema_type}}
                    if repeated
                    else {"type": schema_type}
                ),
            }
            if repeated:
                parameter["style"] = "form"
                parameter["explode"] = True
            return parameter

        filter_parameters = [
            filter_parameter(name, description, repeated, schema_type)
            for name, description, repeated, schema_type in (
                ("subject", "Exact raw subject label; repeat for OR", True, "string"),
                ("predicate", "Exact raw predicate label; repeat for OR", True, "string"),
                ("object", "Exact raw object label; repeat for OR", True, "string"),
                ("subject_contains", "Case-insensitive subject substring", False, "string"),
                ("predicate_contains", "Case-insensitive predicate substring", False, "string"),
                ("object_contains", "Case-insensitive object substring", False, "string"),
                ("subject_tag_id", "Exact raw subject embedding tag ID", True, "string"),
                ("object_tag_id", "Exact raw object embedding tag ID", True, "string"),
                ("entity_tag_id", "Subject OR object embedding tag ID", True, "string"),
                ("relation_tag_id", "Raw predicate embedding tag ID", True, "string"),
                ("subject_category", "Thematic subject category ID", True, "string"),
                ("relation_category", "Thematic relation category ID", True, "string"),
                ("object_category", "Thematic object category ID", True, "string"),
                ("volume", "vol1, vol2, or vol3", False, "string"),
                ("page", "Source page; requires volume", False, "integer"),
                ("start_page", "Inclusive page-range start", False, "integer"),
                ("end_page", "Inclusive page-range end", False, "integer"),
                ("sentence_id", "Exact source sentence ID", False, "string"),
            )
        ]
        pagination_parameters = [
            {
                "name": "limit",
                "in": "query",
                "schema": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            {
                "name": "cursor",
                "in": "query",
                "schema": {"type": "string"},
            },
        ]
        response = {
            "200": {
                "description": "Successful response",
                "content": {
                    "application/json": {
                        "schema": {"type": "object"}
                    }
                },
            },
            "400": {"$ref": "#/components/responses/BadRequest"},
            "429": {"$ref": "#/components/responses/RateLimited"},
        }
        return {
            "openapi": "3.1.0",
            "info": {
                "title": API_NAME,
                "version": API_VERSION,
                "description": (
                    "Read-only access to the V3 Konbaung Chronicle triples, "
                    "thematic categories, and Gemini embedding-neighbor scores. "
                    "No arbitrary SPARQL execution is exposed."
                ),
            },
            "servers": [{"url": "/api/v1"}],
            "paths": {
                "/": {
                    "get": {
                        "summary": "API discovery document",
                        "responses": {"200": response["200"]},
                    }
                },
                "/health": {
                    "get": {
                        "summary": "Readiness check",
                        "responses": {"200": response["200"]},
                    }
                },
                "/stats": {
                    "get": {
                        "summary": "Dataset and embedding statistics",
                        "responses": response,
                    }
                },
                "/categories": {
                    "get": {
                        "summary": "List thematic categories",
                        "parameters": [
                            {
                                "name": "kind",
                                "in": "query",
                                "schema": {
                                    "type": "string",
                                    "enum": ["all", "entity", "relation"],
                                },
                            }
                        ],
                        "responses": response,
                    }
                },
                "/tags": {
                    "get": {
                        "summary": "Search raw entity or relation tags",
                        "parameters": [
                            {
                                "name": "q",
                                "in": "query",
                                "schema": {"type": "string", "maxLength": 200},
                            },
                            {
                                "name": "kind",
                                "in": "query",
                                "schema": {
                                    "type": "string",
                                    "enum": ["all", "entity", "relation"],
                                },
                            },
                            *pagination_parameters[:1],
                        ],
                        "responses": response,
                    }
                },
                "/tags/{kind}/{tag_id}/similar": {
                    "get": {
                        "summary": "Find embedding-nearest tags",
                        "parameters": [
                            {
                                "name": "kind",
                                "in": "path",
                                "required": True,
                                "schema": {
                                    "type": "string",
                                    "enum": ["entity", "relation"],
                                },
                            },
                            {
                                "name": "tag_id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "minimum_similarity",
                                "in": "query",
                                "schema": {
                                    "type": "number",
                                    "minimum": -1,
                                    "maximum": 1,
                                    "default": 0.9,
                                },
                            },
                            *pagination_parameters[:1],
                        ],
                        "responses": {
                            **response,
                            "404": {"$ref": "#/components/responses/NotFound"},
                        },
                    }
                },
                "/triples": {
                    "get": {
                        "summary": "Filter provenance-preserving raw triples",
                        "parameters": [
                            *filter_parameters,
                            *pagination_parameters,
                            {
                                "name": "include_text",
                                "in": "query",
                                "schema": {"type": "boolean", "default": False},
                            },
                        ],
                        "responses": response,
                    }
                },
                "/triples/{claim_id}": {
                    "get": {
                        "summary": "Retrieve one claim by ID",
                        "parameters": [
                            {
                                "name": "claim_id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            },
                            {
                                "name": "include_text",
                                "in": "query",
                                "schema": {"type": "boolean", "default": True},
                            },
                        ],
                        "responses": {
                            **response,
                            "404": {"$ref": "#/components/responses/NotFound"},
                        },
                    }
                },
                "/thematic-patterns": {
                    "get": {
                        "summary": "Aggregate filtered triples by thematic pattern",
                        "parameters": [*filter_parameters, *pagination_parameters],
                        "responses": response,
                    }
                },
                "/openapi.json": {
                    "get": {
                        "summary": "OpenAPI 3.1 schema",
                        "responses": {"200": response["200"]},
                    }
                },
            },
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key",
                        "description": (
                            "Optional deployment authentication. It is required only "
                            "when PUBLIC_API_KEYS is configured."
                        ),
                    },
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "description": (
                            "Alternative way to send a configured public API key."
                        ),
                    },
                },
                "responses": {
                    "BadRequest": {
                        "description": "Invalid query",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                    "NotFound": {
                        "description": "Resource not found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                    "RateLimited": {
                        "description": "Request quota exceeded",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                },
                "schemas": {
                    "Error": {
                        "type": "object",
                        "required": ["ok", "error"],
                        "properties": {
                            "ok": {"const": False},
                            "error": {
                                "type": "object",
                                "required": ["code", "message"],
                                "properties": {
                                    "code": {"type": "string"},
                                    "message": {"type": "string"},
                                },
                            },
                        },
                    }
                },
            },
        }


def install_public_api(
    app: Flask,
    graph: Any,
    axial_graph: Any,
) -> PublicGraphApi:
    PublicGraphApi.configure_app(app)
    public_api = PublicGraphApi(graph, axial_graph)
    app.register_blueprint(public_api.blueprint)
    return public_api
