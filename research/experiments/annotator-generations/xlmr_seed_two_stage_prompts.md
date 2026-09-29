# Konbaung XLM-R Seed Annotation Prompts

Purpose: generate supervised seed data for two XLM-R style models:

1. Entity/event span tagger: `text -> spans with entity labels`
2. Relation classifier: `text + marked spans -> positive relation labels between span IDs`

The LLM output is candidate training data only. Offsets, BIO/BILOU conversion, negative relation-pair generation, validation, and RDF/Neo4j writing happen later in code.

---

## Shared Input Blocks

Use the same page/context wrapper for both calls:

```text
<metadata>
job_id: {job_id}
volume_id: {volume_id}
page_num: {page_num}
</metadata>

<PREVIOUS_CONTEXT annotate="no">
{previous_context}
</PREVIOUS_CONTEXT>

<TARGET_PAGE_TEXT annotate="yes">
{line_numbered_target_page_text}
</TARGET_PAGE_TEXT>

<NEXT_CONTEXT annotate="no">
{next_context}
</NEXT_CONTEXT>
```

Previous/next context is read-only. Use it only for speaker resolution, omitted subjects, pronouns, incomplete sentences, and continuity. Never create spans or evidence from context. All returned `tx` strings must come from `TARGET_PAGE_TEXT`.

---

## Call 1: Entity/Event Span Tagging

```text
Tag entity and event spans in one Burmese royal chronicle page.

You are creating supervised training data for an XLM-R entity/event span tagger, not a knowledge graph summary. Mark explicit text spans in TARGET_PAGE_TEXT that should become graph-node candidates.

Use only schema entity labels. Include people, groups, polities, places, offices, titles, objects, dates, named works, omens, abstract statuses, and event/process spans when they are expressed in the target page.

Tag each explicit mention that should train the model, including repeated mentions when they occur in different places. Do not tag page numbers, headers, footers, publisher text, punctuation-only spans, generic particles, pronouns, or OCR debris.

Prefer the longest clear span for one semantic unit. Avoid overlapping spans unless the shorter span has a different label and is genuinely needed. Event spans may include the main verb phrase and its essential complements, but should not swallow an entire paragraph.

For each span, return:
- id: E1, E2, E3...
- label: one schema entity label
- ln: starting line number in TARGET_PAGE_TEXT
- tx: exact Burmese span text copied from TARGET_PAGE_TEXT
- i: occurrence number of tx starting from that line, usually 1

Return JSON only.
```

Structured output shape:

```json
{
  "spans": [
    {
      "id": "E1",
      "label": "PERSON",
      "ln": 1,
      "tx": "exact Burmese span",
      "i": 1
    }
  ]
}
```

Validation expected after the call:

```text
Every tx must map exactly, or whitespace-normalized exactly, to TARGET_PAGE_TEXT.
Every id must be unique.
Every label must be in the entity schema.
Offsets are computed locally after validation.
```

---

## Call 2: Relation Tagging Over Span IDs

Input to this call includes the same page/context wrapper plus the validated entity spans from Call 1:

```text
<ENTITY_SPANS_FROM_CALL_1>
{entity_spans_json}
</ENTITY_SPANS_FROM_CALL_1>
```

Prompt:

```text
Tag relations between the provided entity/event span IDs for one Burmese royal chronicle page.

You are creating supervised training data for an XLM-R relation classifier. Use the full TARGET_PAGE_TEXT plus the provided spans to identify positive relations. Return positive relations only; NO_RELATION examples will be generated later by code.

Use only schema relation labels. Each relation must connect two provided span IDs. Do not invent new spans. Direction matters: s is the source/subject span ID, o is the target/object span ID.

Use previous/next context only to resolve omitted subjects, pronouns, speakers, and continuity. Do not create relations whose evidence is only in context. Every evidence tx must come from TARGET_PAGE_TEXT.

If a relation depends on an omitted subject, link it to the best explicit span ID from TARGET_PAGE_TEXT when the context clearly resolves it. If no provided span can serve as an argument, skip the relation rather than inventing an implicit node.

Avoid relation spam. Tag relations that express the main semantic connections in the page: command, report, movement, attack, defense, defeat, retreat, construction, naming, reward, office/status, kinship, succession, cause/result, event date/place, and similar graph-relevant links.

For each relation, return:
- id: R1, R2, R3...
- s: source span ID from Call 1
- p: one schema relation label
- o: target span ID from Call 1
- ln: starting line number for evidence in TARGET_PAGE_TEXT
- tx: exact Burmese evidence text copied from TARGET_PAGE_TEXT
- i: occurrence number of tx starting from that line, usually 1

Return JSON only.
```

Structured output shape:

```json
{
  "relations": [
    {
      "id": "R1",
      "s": "E1",
      "p": "ORDERS_COMMANDS_INSTRUCTS",
      "o": "E2",
      "ln": 1,
      "tx": "exact Burmese evidence text",
      "i": 1
    }
  ]
}
```

Validation expected after the call:

```text
Every s/o must be a valid span ID from Call 1.
Every p must be in the relation schema.
Every tx must map exactly, or whitespace-normalized exactly, to TARGET_PAGE_TEXT.
Relation direction is preserved as given.
Negative relation pairs are generated later from unlinked candidate span pairs.
```
