<KONBAUNG_POWER_RELATION_ENRICHMENT>

## Project

You are assisting a historical-sociological study of power in the Konbaung dynasty of Burma/Myanmar.

The project is building a large database of open-coded subject-predicate-object triples from Burmese chronicles. The database records concrete power relations: who commanded, appointed, reported, served, attacked, punished, taxed, recruited, granted, transferred, submitted to, resisted, or otherwise acted upon whom or what.

The research question is:

**How did power operate in the Konbaung dynasty as a lived social system?**

The triples are intentionally open coded. Their relation labels and entity tags will later be clustered with embeddings into a smaller closed-class system for quantitative analysis.

## Existing annotations

A first annotation pass has already captured most relevant content.

For every sentence, you will receive the Burmese sentence, an English translation, and the triples already extracted from that sentence.

Treat those existing triples as the current database record. Do not reproduce, revise, criticize, or replace them.

Your only job is to identify important power relations that the first pass genuinely missed.

## Required procedure

Process every supplied sentence in order.

For each sentence, return:

1. `sid`: the supplied sentence ID.
2. `analysis`: one freeform paragraph of no more than 100 words.
3. `T`: only genuinely missing triples, or an empty array when coverage is sufficient.

The `analysis` must briefly explain:

- how the sentence helps answer the research question;
- what the existing triples already demonstrate;
- whether the existing coverage is sufficient;
- what important power relation remains missing, if any.

Write one paragraph, not separate subsections.

## Deciding whether to add a triple

Compare the meaning of the sentence with the meaning of the existing triples.

Add a triple only when it records a distinct power relation that the existing triples do not already communicate.

Do not add a triple merely because it:

- uses different wording;
- uses a narrower or broader predicate;
- adds a place, date, route, quantity, title, purpose, manner, or descriptive detail to an existing event;
- isolates one member of a group already covered by the same action;
- restates the same event from another grammatical perspective;
- converts an existing action into a near-synonymous order, execution, consequence, or description;
- makes an existing triple more specific without adding a separate power relation.

Use a conservative standard:

**If the existing triples already communicate the relevant power relationship, set `T` to an empty array.**

It is better to return no new triple than a redundant or marginal one.

Read long sentences especially carefully because they may contain several independent reports, commands, transfers, appointments, responses, or outcomes. Length alone does not justify additions.

## Triple format

Each new triple contains `s`, `p`, and `o`.

Use a concise English relation label in `ALL_CAPS_WITH_UNDERSCORES`.

Each subject and object contains:

- `my`: Burmese referent;
- `en`: compact English gloss;
- `tag`: concise open-coded analytical tag;
- `source`: `sentence`, `page`, or `inferred`.

Use `source: "sentence"` only when the exact Burmese referent appears in the assigned sentence.

Use `source: "page"` only when the exact Burmese referent appears elsewhere in the supplied page.

Use `source: "inferred"` only when the referent is absent from the supplied Burmese text but is grammatically required.

When `source` is `sentence` or `page`, copy an exact Burmese span.

## Consistency rule

The analysis and triples must agree.

- If the analysis says coverage is sufficient, `T` must be `[]`.
- If the analysis identifies a missing relation, `T` must contain only the triple or triples representing it.
- Do not mention a missing relation and then omit it from `T`.
- Do not output a triple that the analysis says is already covered.

## Output

Return one result record for every supplied sentence, preserving sentence order.

Return only JSON conforming to the supplied response schema.

Do not return any page summary, translation, evidence block, confidence score, correction to an existing triple, or field not present in the schema.

## Input

{{ENRICHMENT_INPUT}}

</KONBAUNG_POWER_RELATION_ENRICHMENT>
