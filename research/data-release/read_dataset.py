"""Read a claim, source sentence and linked vectors from the Konbaung V3 ZIP."""
import argparse
import hashlib
import io
import json
import zipfile

import numpy as np


def records(archive, path):
    with archive.open(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="Path to Konbaung_Chronicle_V3_Data.zip")
    parser.add_argument("--triple-id", help="Read this claim instead of the first")
    parser.add_argument("--verify", action="store_true", help="Verify all 23 original payload hashes")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        if args.verify:
            manifest = json.loads(archive.read("manifest.json"))
            for entry in manifest["files"]:
                with archive.open(entry["path"]) as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest != entry["sha256"]:
                    raise ValueError(f"Checksum mismatch: {entry['path']}")
            print(f"Verified {len(manifest['files'])} original payloads.")

        triple = next((r for r in records(archive, "data/triples.jsonl")
                       if args.triple_id is None or r["triple_id"] == args.triple_id), None)
        if triple is None:
            parser.error(f"Claim not found: {args.triple_id}")
        sentence = next(r for r in records(archive, "data/sentences.jsonl")
                        if r["sentence_id"] == triple["sentence_id"])
        link = next(r for r in records(archive, "embeddings/triple_context_links.jsonl")
                    if r["triple_id"] == triple["triple_id"])
        cache = {}

        def matrix(path):
            if path not in cache:
                cache[path] = np.load(io.BytesIO(archive.read(path)), allow_pickle=False)
            return cache[path]

        vectors = {role: matrix(triple[role]["embedding_matrix"])[triple[role]["embedding_row"]]
                   for role in ("subject", "predicate", "object")}
        context = matrix("embeddings/triple_context.npy")[link["vector_rows"]]
        result = {
            "triple_id": triple["triple_id"],
            "claim": [triple[role]["tag"] for role in ("subject", "predicate", "object")],
            "sentence_id": sentence["sentence_id"],
            "volume_id": sentence["volume_id"],
            "page_numbers": sentence["page_numbers"],
            "burmese": sentence["burmese"],
            "english_translation": sentence["english_translation"],
            "vector_shapes": {key: list(value.shape) for key, value in vectors.items()},
            "context_vector_rows": link["vector_rows"],
            "context_vector_shape": list(context.shape),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
