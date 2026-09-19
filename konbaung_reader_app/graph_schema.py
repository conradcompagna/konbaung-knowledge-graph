from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
GRAPH_ROOT = APP_ROOT / "data" / "konbaung_knowledge_graph_v3"
GRAPH_DB = GRAPH_ROOT / "oxigraph"
GRAPH_EXPORT = GRAPH_ROOT / "konbaung_knowledge_graph_v3.nq"
GRAPH_MANIFEST = GRAPH_ROOT / "manifest.json"

V3_SENTENCE_ROOT = (
    APP_ROOT / "data" / "konbaung_historiography_v3_canonical_20260724"
)
PAGE_ROOT = APP_ROOT / "static" / "data" / "konbaung"
EMBEDDING_ROOT = (
    PROJECT_ROOT / "konbaung_v3_node_edge_clustering_first_pass_20260724"
)
OCCURRENCES_PATH = (
    PROJECT_ROOT / "konbaung_v3_eight_view_embeddings_20260724"
    / "occurrences.jsonl"
)

KG = "urn:konbaung:vocab:"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
DCTERMS = "http://purl.org/dc/terms/"
XSD = "http://www.w3.org/2001/XMLSchema#"

SOURCE_GRAPH = "urn:konbaung:graph:v3-source"
DIRECT_GRAPH = "urn:konbaung:graph:v3-direct-relations"
EMBEDDING_GRAPH = "urn:konbaung:graph:gemini-embeddings"
SCHEMA_GRAPH = "urn:konbaung:graph:schema"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entity_uri(base_key: str) -> str:
    return f"urn:konbaung:entity:{base_key}"


def relation_uri(base_key: str) -> str:
    return f"urn:konbaung:relation:{base_key}"


def claim_uri(occurrence_id: str) -> str:
    return f"urn:konbaung:claim:{occurrence_id}"


def sentence_uri(sentence_id: str) -> str:
    return f"urn:konbaung:sentence:{sentence_id}"


def page_uri(volume_id: str, page_number: int) -> str:
    return f"urn:konbaung:page:{volume_id}:{int(page_number)}"


def volume_uri(volume_id: str) -> str:
    return f"urn:konbaung:volume:{volume_id}"

