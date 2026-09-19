from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_READER_ROOT = Path(__file__).resolve().parents[2] / "burmese-neural-reader"


def utf16_offset(text: str, codepoint_offset: int) -> int:
    return len(text[:codepoint_offset].encode("utf-16-le")) // 2


def _append_unique(values: list[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in values:
        values.append(text)


def _dictionary_display_fields(base: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    romanizations: list[str] = []
    parts_of_speech: list[str] = []
    definitions: list[str] = []
    _append_unique(romanizations, base.get("roman", ""))

    for raw_pos in re.split(r"\s*[,;/|]\s*", str(base.get("pos", ""))):
        _append_unique(parts_of_speech, raw_pos)

    for raw_sense in base.get("senses", []) or []:
        line = str(raw_sense or "").strip("\r\n")
        if not line.strip():
            continue
        columns = line.split("\t")
        if len(columns) >= 4:
            _append_unique(romanizations, columns[1])
            for raw_pos in re.split(r"\s*[,;/|]\s*", columns[2].strip()):
                _append_unique(parts_of_speech, raw_pos)
            definition = " ".join(part.strip() for part in columns[3:] if part.strip())
        else:
            definition = line.strip()
        definition = re.sub(r"^\s*\d+\s*[.)-]?\s*", "", definition).strip()
        embedded = list(
            re.finditer(
                r"(?<!\S)(n|v|adj|adv|pron|part|conj|interj|int|prep|postp|aux)\s+\d+\s+",
                definition,
                flags=re.IGNORECASE,
            )
        )
        if not embedded:
            _append_unique(definitions, definition)
            continue
        _append_unique(definitions, definition[: embedded[0].start()])
        for match_index, match in enumerate(embedded):
            _append_unique(parts_of_speech, match.group(1).lower())
            end = embedded[match_index + 1].start() if match_index + 1 < len(embedded) else len(definition)
            _append_unique(definitions, definition[match.end() : end])

    return romanizations, parts_of_speech, definitions


class BurmeseReaderBridge:
    """Read-only adapter around Burmese Neural Reader's DP segmenter and dictionaries."""

    def __init__(self, reader_root: Path | None = None) -> None:
        configured = os.getenv("BURMESE_READER_ROOT", "").strip()
        self.reader_root = Path(configured) if configured else (reader_root or DEFAULT_READER_ROOT)
        self._module: ModuleType | None = None
        self._lock = threading.Lock()

    def ensure_loaded(self) -> ModuleType:
        if self._module is not None:
            return self._module
        with self._lock:
            if self._module is not None:
                return self._module
            module_path = self.reader_root / "newserverPDF21split.py"
            if not module_path.exists():
                raise FileNotFoundError(f"Burmese Neural Reader module not found: {module_path}")

            # Importing a Python file normally creates __pycache__. The external reader tree
            # is a read-only dependency for this app, so bytecode writes are disabled.
            previous = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(self.reader_root))
            try:
                name = "burmese_neural_reader_readonly"
                spec = importlib.util.spec_from_file_location(name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Unable to load {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                module.load_dictionary()
                self._module = module
            finally:
                sys.dont_write_bytecode = previous
                if sys.path and sys.path[0] == str(self.reader_root):
                    sys.path.pop(0)
            return self._module

    def segment_page(self, canonical_text: str) -> dict[str, Any]:
        module = self.ensure_loaded()
        # Do not strip, normalize, or rewrite the canonical page string.
        segmentation_text = module.normalize_burmese_for_segmentation(canonical_text)
        if segmentation_text != canonical_text:
            raise ValueError("The external segmentation transform changed canonical page text")

        segments: list[str] = []
        island_spans: list[tuple[int, int]] = []
        for run in re.findall(r"\s+|\S+", segmentation_text):
            if run.isspace():
                segments.append(run)
                continue
            run_start = len(segments)
            run_segments, _, _ = module._dp_segment_text_only(run)
            segments.extend(run_segments)
            island_spans.append((run_start, len(segments)))
        offsets = module._build_segment_offsets(segmentation_text, segments)
        if offsets is None:
            raise ValueError("The external segmenter returned tokens that could not be mapped to canonical text")

        entries: list[dict[str, Any]] = []
        token_records: list[dict[str, Any]] = []
        for index, (segment, (start, end)) in enumerate(zip(segments, offsets)):
            key = module.normalize_headword(segment)
            base = module.DICT[key] if key and key in module.DICT else None
            if base:
                romanizations, parts_of_speech, definitions = _dictionary_display_fields(base)
                entry = {
                    "head": segment.strip() or segment,
                    "romanizations": romanizations,
                    "partsOfSpeech": parts_of_speech,
                    "definitions": definitions,
                    "known": True,
                }
            else:
                entry = {
                    "head": segment.strip() or segment,
                    "romanizations": [],
                    "partsOfSpeech": [],
                    "definitions": [],
                    "known": False,
                }
            entries.append(entry)
            token_records.append(
                {
                    "id": f"t{index}",
                    "index": index,
                    "text": segment,
                    "startUtf16": utf16_offset(canonical_text, start),
                    "endUtf16": utf16_offset(canonical_text, end),
                }
            )

        reconstructed = "".join(
            canonical_text[offsets[i][0] : offsets[i][1]] for i in range(len(offsets))
        )
        return {
            "canonicalTextLengthUtf16": utf16_offset(canonical_text, len(canonical_text)),
            "tokens": token_records,
            "dictionary": entries,
            "islandSpans": island_spans,
            "diagnostics": {
                "tokenCount": len(token_records),
                "tokenTextMatchesCanonicalSlices": all(
                    segment == canonical_text[start:end]
                    for segment, (start, end) in zip(segments, offsets)
                ),
                "concatenatedTokenTextLength": len(reconstructed),
                "externalReaderRoot": str(self.reader_root),
                "pipelinesUsed": ["dp_segmenter", "layered_dictionary"],
            },
        }
