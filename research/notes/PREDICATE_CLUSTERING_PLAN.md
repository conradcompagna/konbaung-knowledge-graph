# Konbaung V3 Predicate Clustering Plan

## Goal

Turn V3’s 11,886 raw predicate strings into a smaller, coherent vocabulary that can later become RDF properties.

Do not modify V1, V2, V3, the reader, or any source triple. The result is a new crosswalk layered over the original data.

## Corpus

- 27,129 canonical triple occurrences
- 11,886 distinct predicate strings
- 8,979 predicates occur once
- 11,267 occur five times or fewer
- Burmese sentence and English translation are available for every triple
- Cross-page duplicates have already been resolved in the canonical V3 snapshot

Because most predicates are rare, fixed-k clustering or predicate-frequency rules will not work well.

## What we produce

Each original triple receives:

```text
occurrence_id
subject
raw_predicate
object
sentence_my
sentence_en
relation_family
canonical_predicate
subject_role
object_role
mapping_confidence
review_status
```

There are two semantic levels:

1. **Relation family:** broad category such as title bestowal, appointment, military action, movement, kinship, or diplomacy.
2. **Canonical predicate:** a specific directed relation with defined subject and object roles.

Related predicates are not necessarily equivalent. For example, these belong to one family but remain distinct:

- grantor → bestowed title on → recipient
- recipient → received title → title
- recipient → received title from → grantor

## Method

### 1. Freeze and extract

- Hash the canonical V3 files.
- Create one immutable staging row per triple.
- Preserve the exact S–P–O, Burmese sentence, English translation, SID, and page provenance.
- Give every occurrence a deterministic ID.

### 2. Create semantic representations

Embed every occurrence using:

```text
Subject: …
Predicate: …
Object: …
English sentence: …
Burmese sentence: …
```

Also retain a predicate-only representation so long sentence topics cannot dominate the relation.

Use `gemini-embedding-2`, 768 dimensions, through the Batch API. Never silently truncate long sentences; split oversized contexts into chunks that repeat the complete S–P–O frame.

### 3. Find candidate groups

Combine:

- exact and near lexical matches;
- predicate-embedding neighbors;
- complete S–P–O and bilingual-context neighbors.

Use these similarities to propose:

- broad relation families;
- narrower canonical-predicate groups;
- possible inverses;
- possible polysemous predicates;
- malformed predicates containing names, titles, or other leaked arguments.

Do not force every item into a group. `UNRESOLVED` is valid.

### 4. Review a pilot

First run a deterministic pilot of roughly 1,500–2,000 occurrences covering:

- all three volumes;
- common predicates;
- singleton predicates;
- close but non-equivalent relations;
- inverse relations;
- long sentences and malformed cases.

Produce a readable Markdown review containing the actual triples and both sentence texts.

The pilot must show that:

- real synonyms appear together;
- related but differently directed relations remain separate;
- bilingual context improves rather than confuses the grouping;
- the proposed canonical granularity is useful for historical queries.

### 5. Build the vocabulary iteratively

After approving the pilot:

1. Review common predicates and create the initial canonical vocabulary.
2. Assign rarer predicates to that vocabulary when appropriate.
3. Create new canonical predicates only when an existing one is genuinely inadequate.
4. Split different uses of the same raw predicate with occurrence-level overrides.
5. Leave uncertain cases unresolved.

Machine clustering proposes mappings; reviewed decisions define the accepted vocabulary.

### 6. Validate and freeze

Before RDF work:

- confirm all 27,129 source triples remain present and unchanged;
- audit accepted clusters for semantic equivalence and correct direction;
- verify that no inverse relations were merged;
- report accepted, proposed, and unresolved coverage separately;
- freeze the predicate registry and occurrence crosswalk as a versioned release.

## Use of the old pass

The old relation list may supply extra synonym candidates, but it is not the target ontology and does not affect V3 frequencies. Nothing from it is accepted automatically, and neither old database is changed.

## Immediate next step

Implement only extraction and the pilot. Deliver:

1. the lossless occurrence table;
2. the predicate-frequency and anomaly report;
3. pilot embeddings and candidate clusters;
4. a readable pilot review;
5. the projected full-run cost.

After we approve the pilot’s granularity, run the full clustering pass. Entity resolution and RDF conversion come afterward.
