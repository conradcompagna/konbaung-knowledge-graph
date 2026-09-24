from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Iterator

from pyoxigraph import Literal, NamedNode, Quad, RdfFormat, Store

try:
    from .build_graph_overview import build_overview
except ImportError:
    from build_graph_overview import build_overview

try:
    from .graph_schema import (
        DCTERMS,
        DIRECT_GRAPH,
        EMBEDDING_GRAPH,
        EMBEDDING_ROOT,
        GRAPH_DB,
        GRAPH_EXPORT,
        GRAPH_MANIFEST,
        GRAPH_ROOT,
        KG,
        OCCURRENCES_PATH,
        PAGE_ROOT,
        RDF,
        RDFS,
        SCHEMA_GRAPH,
        SOURCE_GRAPH,
        V3_SENTENCE_ROOT,
        claim_uri,
        entity_uri,
        page_uri,
        read_jsonl,
        relation_uri,
        sentence_uri,
        sha256,
        volume_uri,
    )
except ImportError:  # Direct `python build_graph_database.py` execution.
    from graph_schema import (
        DCTERMS,
        DIRECT_GRAPH,
        EMBEDDING_GRAPH,
        EMBEDDING_ROOT,
        GRAPH_DB,
        GRAPH_EXPORT,
        GRAPH_MANIFEST,
        GRAPH_ROOT,
        KG,
        OCCURRENCES_PATH,
        PAGE_ROOT,
        RDF,
        RDFS,
        SCHEMA_GRAPH,
        SOURCE_GRAPH,
        V3_SENTENCE_ROOT,
        claim_uri,
        entity_uri,
        page_uri,
        read_jsonl,
        relation_uri,
        sentence_uri,
        sha256,
        volume_uri,
    )


RDF_TYPE = NamedNode(RDF + "type")
RDF_SUBJECT = NamedNode(RDF + "subject")
RDF_PREDICATE = NamedNode(RDF + "predicate")
RDF_OBJECT = NamedNode(RDF + "object")
RDFS_LABEL = NamedNode(RDFS + "label")
SOURCE = NamedNode(SOURCE_GRAPH)
DIRECT = NamedNode(DIRECT_GRAPH)
EMBEDDINGS = NamedNode(EMBEDDING_GRAPH)
SCHEMA = NamedNode(SCHEMA_GRAPH)


def node(value: str) -> NamedNode:
    return NamedNode(value)


def quad(
    subject: str | NamedNode,
    predicate: str | NamedNode,
    object_: str | NamedNode | Literal,
    graph: NamedNode,
) -> Quad:
    return Quad(
        subject if isinstance(subject, NamedNode) else node(subject),
        predicate if isinstance(predicate, NamedNode) else node(predicate),
        object_ if isinstance(object_, (NamedNode, Literal)) else node(object_),
        graph,
    )


def schema_quads() -> Iterator[Quad]:
    classes = {
        "Dataset": "Chronicle graph dataset",
        "Volume": "Chronicle volume",
        "Page": "Source page",
        "Sentence": "Canonical translated sentence",
        "Claim": "Provenanced subject-predicate-object assertion",
        "RawEntity": "Undisambiguated entity label",
        "RawRelation": "Undisambiguated directed relation label",
    }
    properties = {
        "volume": "volume",
        "volumeId": "volume identifier",
        "pageNumber": "source page number",
        "ownerPage": "sentence owner page",
        "appearsOnPage": "physical page occurrence",
        "sentence": "supporting sentence",
        "tripleOrdinal": "triple ordinal within sentence",
        "burmeseText": "Burmese sentence text",
        "englishTranslation": "English sentence translation",
        "decision": "annotation decision",
        "justification": "annotation decision justification",
        "normalizedLabel": "normalized label",
        "frequency": "corpus occurrence frequency",
        "embeddingIndex": "row in the embedding matrix",
        "embeddingModel": "embedding model",
        "embeddingView": "embedding input view",
        "embeddingFile": "embedding matrix path",
        "sourceRecordHash": "source record SHA-256",
    }
    for local, label in classes.items():
        resource = node(KG + local)
        yield quad(resource, RDF_TYPE, node(RDFS + "Class"), SCHEMA)
        yield quad(resource, RDFS_LABEL, Literal(label, language="en"), SCHEMA)
    for local, label in properties.items():
        resource = node(KG + local)
        yield quad(resource, RDF_TYPE, node(RDF + "Property"), SCHEMA)
        yield quad(resource, RDFS_LABEL, Literal(label, language="en"), SCHEMA)


def build() -> dict[str, object]:
    if GRAPH_ROOT.exists():
        raise FileExistsError(
            f"Graph database already exists: {GRAPH_ROOT}. "
            "Move or remove that exact derived directory before rebuilding."
        )
    staging = GRAPH_ROOT.with_name(GRAPH_ROOT.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Stale graph build directory exists: {staging}")
    staging.mkdir(parents=True)
    database_path = staging / GRAPH_DB.name
    export_path = staging / GRAPH_EXPORT.name
    manifest_path = staging / GRAPH_MANIFEST.name

    node_records = list(read_jsonl(EMBEDDING_ROOT / "node_records.jsonl"))
    edge_records = list(read_jsonl(EMBEDDING_ROOT / "edge_records.jsonl"))
    node_by_tag = {record["tag"]: record for record in node_records}
    edge_by_tag = {record["tag"]: record for record in edge_records}
    if len(node_records) != 23_890 or len(node_by_tag) != 23_890:
        raise RuntimeError("Expected exactly 23,890 unique raw entity labels")
    if len(edge_records) != 11_886 or len(edge_by_tag) != 11_886:
        raise RuntimeError("Expected exactly 11,886 unique raw relation labels")

    page_index = json.loads((PAGE_ROOT / "index.json").read_text(encoding="utf-8"))
    v3_index = json.loads((V3_SENTENCE_ROOT / "index.json").read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    direct_edges: set[tuple[str, str, str]] = set()

    def all_quads() -> Iterator[Quad]:
        for item in schema_quads():
            counts["schemaStatements"] += 1
            yield item

        dataset = node("urn:konbaung:dataset:v3")
        yield quad(dataset, RDF_TYPE, node(KG + "Dataset"), SOURCE)
        yield quad(
            dataset,
            RDFS_LABEL,
            Literal("Konbaung Chronicle V3 knowledge graph", language="en"),
            SOURCE,
        )
        yield quad(
            dataset,
            node(DCTERMS + "created"),
            Literal(datetime.now(timezone.utc).isoformat()),
            SOURCE,
        )
        counts["sourceStatements"] += 3

        for volume in page_index["volumes"]:
            volume_id = volume["id"]
            volume_resource = node(volume_uri(volume_id))
            volume_rows = (
                quad(volume_resource, RDF_TYPE, node(KG + "Volume"), SOURCE),
                quad(
                    volume_resource,
                    node(DCTERMS + "identifier"),
                    Literal(volume_id),
                    SOURCE,
                ),
                quad(
                    volume_resource,
                    RDFS_LABEL,
                    Literal(volume["label"], language="en"),
                    SOURCE,
                ),
                quad(
                    volume_resource,
                    node(DCTERMS + "isPartOf"),
                    dataset,
                    SOURCE,
                ),
            )
            for item in volume_rows:
                counts["sourceStatements"] += 1
                yield item
            counts["volumes"] += 1

            for page_number in volume["availablePages"]:
                page_path = PAGE_ROOT / "pages" / volume_id / f"{int(page_number):04d}.json"
                page = json.loads(page_path.read_text(encoding="utf-8"))
                page_resource = node(page_uri(volume_id, page_number))
                page_rows = [
                    quad(page_resource, RDF_TYPE, node(KG + "Page"), SOURCE),
                    quad(
                        page_resource,
                        node(DCTERMS + "identifier"),
                        Literal(page["id"]),
                        SOURCE,
                    ),
                    quad(
                        page_resource,
                        node(KG + "volume"),
                        volume_resource,
                        SOURCE,
                    ),
                    quad(
                        page_resource,
                        node(KG + "pageNumber"),
                        Literal(int(page_number)),
                        SOURCE,
                    ),
                ]
                if page.get("summary"):
                    page_rows.append(
                        quad(
                            page_resource,
                            node(DCTERMS + "description"),
                            Literal(page["summary"], language="en"),
                            SOURCE,
                        )
                    )
                checksum = page.get("source", {}).get("checksum")
                if checksum:
                    page_rows.append(
                        quad(
                            page_resource,
                            node(DCTERMS + "source"),
                            Literal(str(checksum)),
                            SOURCE,
                        )
                    )
                for item in page_rows:
                    counts["sourceStatements"] += 1
                    yield item
                counts["pages"] += 1

        for volume_id in ("vol1", "vol2", "vol3"):
            sentence_data = json.loads(
                (V3_SENTENCE_ROOT / "sentences" / f"{volume_id}.json").read_text(encoding="utf-8")
            )["sentences"]
            for sentence in sentence_data.values():
                sentence_resource = node(sentence_uri(sentence["sid"]))
                rows = [
                    quad(
                        sentence_resource,
                        RDF_TYPE,
                        node(KG + "Sentence"),
                        SOURCE,
                    ),
                    quad(
                        sentence_resource,
                        node(DCTERMS + "identifier"),
                        Literal(sentence["sid"]),
                        SOURCE,
                    ),
                    quad(
                        sentence_resource,
                        node(KG + "volume"),
                        node(volume_uri(volume_id)),
                        SOURCE,
                    ),
                    quad(
                        sentence_resource,
                        node(KG + "volumeId"),
                        Literal(volume_id),
                        SOURCE,
                    ),
                    quad(
                        sentence_resource,
                        node(KG + "ownerPage"),
                        node(page_uri(volume_id, sentence["ownerPage"])),
                        SOURCE,
                    ),
                    quad(
                        sentence_resource,
                        node(KG + "burmeseText"),
                        Literal(sentence["my"], language="my"),
                        SOURCE,
                    ),
                    quad(
                        sentence_resource,
                        node(KG + "englishTranslation"),
                        Literal(sentence["en"], language="en"),
                        SOURCE,
                    ),
                    quad(
                        sentence_resource,
                        node(KG + "decision"),
                        Literal(sentence["decision"]),
                        SOURCE,
                    ),
                ]
                if sentence.get("justification"):
                    rows.append(
                        quad(
                            sentence_resource,
                            node(KG + "justification"),
                            Literal(sentence["justification"], language="en"),
                            SOURCE,
                        )
                    )
                for page_number in sentence["pages"]:
                    rows.append(
                        quad(
                            sentence_resource,
                            node(KG + "appearsOnPage"),
                            node(page_uri(volume_id, page_number)),
                            SOURCE,
                        )
                    )
                for item in rows:
                    counts["sourceStatements"] += 1
                    yield item
                counts["sentences"] += 1

        for kind, records, class_name, uri_function in (
            ("node", node_records, "RawEntity", entity_uri),
            ("relation", edge_records, "RawRelation", relation_uri),
        ):
            for record in records:
                resource = node(uri_function(record["baseKey"]))
                rows = (
                    quad(resource, RDF_TYPE, node(KG + class_name), EMBEDDINGS),
                    quad(resource, RDFS_LABEL, Literal(record["tag"]), EMBEDDINGS),
                    quad(
                        resource,
                        node(KG + "normalizedLabel"),
                        Literal(record["normalizedTag"]),
                        EMBEDDINGS,
                    ),
                    quad(
                        resource,
                        node(KG + "frequency"),
                        Literal(int(record["frequency"])),
                        EMBEDDINGS,
                    ),
                    quad(
                        resource,
                        node(KG + "embeddingIndex"),
                        Literal(int(record["index"])),
                        EMBEDDINGS,
                    ),
                    quad(
                        resource,
                        node(KG + "embeddingModel"),
                        Literal("gemini-embedding-2"),
                        EMBEDDINGS,
                    ),
                    quad(
                        resource,
                        node(KG + "embeddingView"),
                        Literal("raw tag"),
                        EMBEDDINGS,
                    ),
                    quad(
                        resource,
                        node(KG + "embeddingFile"),
                        Literal(f"{kind}_base_vectors.npy"),
                        EMBEDDINGS,
                    ),
                )
                for item in rows:
                    counts["embeddingStatements"] += 1
                    yield item
                counts["entities" if kind == "node" else "relations"] += 1

        for occurrence in read_jsonl(OCCURRENCES_PATH):
            subject_record = node_by_tag[occurrence["subject"]]
            relation_record = edge_by_tag[occurrence["predicate"]]
            object_record = node_by_tag[occurrence["object"]]
            subject = node(entity_uri(subject_record["baseKey"]))
            predicate = node(relation_uri(relation_record["baseKey"]))
            object_ = node(entity_uri(object_record["baseKey"]))
            claim = node(claim_uri(occurrence["occurrenceId"]))
            sentence = node(sentence_uri(occurrence["sid"]))
            rows = [
                quad(claim, RDF_TYPE, node(KG + "Claim"), SOURCE),
                quad(claim, RDF_TYPE, node(RDF + "Statement"), SOURCE),
                quad(claim, RDF_SUBJECT, subject, SOURCE),
                quad(claim, RDF_PREDICATE, predicate, SOURCE),
                quad(claim, RDF_OBJECT, object_, SOURCE),
                quad(claim, node(KG + "sentence"), sentence, SOURCE),
                quad(
                    claim,
                    node(KG + "tripleOrdinal"),
                    Literal(int(occurrence["tripleOrdinal"])),
                    SOURCE,
                ),
                quad(
                    claim,
                    node(KG + "ownerPage"),
                    node(
                        page_uri(
                            occurrence["volumeId"],
                            int(occurrence["ownerPage"]),
                        )
                    ),
                    SOURCE,
                ),
                quad(
                    claim,
                    node(KG + "sourceRecordHash"),
                    Literal(occurrence["sourceRecordHash"]),
                    SOURCE,
                ),
            ]
            for page_number in occurrence["pageOccurrences"]:
                rows.append(
                    quad(
                        claim,
                        node(KG + "appearsOnPage"),
                        node(page_uri(occurrence["volumeId"], page_number)),
                        SOURCE,
                    )
                )
            for item in rows:
                counts["sourceStatements"] += 1
                yield item
            direct_key = (
                subject_record["baseKey"],
                relation_record["baseKey"],
                object_record["baseKey"],
            )
            if direct_key not in direct_edges:
                direct_edges.add(direct_key)
                counts["directStatements"] += 1
                yield quad(subject, predicate, object_, DIRECT)
            counts["claims"] += 1

    store = Store(database_path)
    try:
        store.bulk_extend(all_quads())
        store.flush()
        if counts["claims"] != 27_129:
            raise RuntimeError(f"Expected 27,129 claim resources, found {counts['claims']:,}")
        if counts["sentences"] != int(v3_index["totals"]["canonicalSentences"]):
            raise RuntimeError("Canonical sentence count changed during graph import")
        if len(store) != sum(
            counts[key]
            for key in (
                "schemaStatements",
                "sourceStatements",
                "embeddingStatements",
                "directStatements",
            )
        ):
            raise RuntimeError("RDF store statement count does not match importer count")
        store.dump(output=export_path, format=RdfFormat.N_QUADS)
        overview = build_overview(staging / "overview.json")
        manifest = {
            "schemaVersion": 1,
            "database": "Oxigraph embedded RDF dataset",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "namedGraphs": {
                "schema": SCHEMA_GRAPH,
                "source": SOURCE_GRAPH,
                "directRelations": DIRECT_GRAPH,
                "embeddings": EMBEDDING_GRAPH,
            },
            "counts": dict(sorted(counts.items())),
            "statements": len(store),
            "embeddingSort": {
                "model": "gemini-embedding-2",
                "view": "raw tag",
                "dimensions": 768,
                "nodeMatrix": str(EMBEDDING_ROOT / "node_base_vectors.npy"),
                "relationMatrix": str(EMBEDDING_ROOT / "edge_base_vectors.npy"),
            },
            "sourceHashes": {
                str(V3_SENTENCE_ROOT / "index.json"): sha256(V3_SENTENCE_ROOT / "index.json"),
                str(OCCURRENCES_PATH): sha256(OCCURRENCES_PATH),
                str(EMBEDDING_ROOT / "node_records.jsonl"): sha256(
                    EMBEDDING_ROOT / "node_records.jsonl"
                ),
                str(EMBEDDING_ROOT / "edge_records.jsonl"): sha256(
                    EMBEDDING_ROOT / "edge_records.jsonl"
                ),
                str(EMBEDDING_ROOT / "node_base_vectors.npy"): sha256(
                    EMBEDDING_ROOT / "node_base_vectors.npy"
                ),
                str(EMBEDDING_ROOT / "edge_base_vectors.npy"): sha256(
                    EMBEDDING_ROOT / "edge_base_vectors.npy"
                ),
            },
            "portableExport": GRAPH_EXPORT.name,
            "overview": overview,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        del store
        shutil.rmtree(staging)
        raise
    del store
    staging.rename(GRAPH_ROOT)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
