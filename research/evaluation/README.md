# Extraction evaluation

[`Konbaung_Extraction_Evaluation.xlsx`](Konbaung_Extraction_Evaluation.xlsx) contains
every judgment from the evaluation of the extracted claims.

Two independent random samples of 61 pages each were drawn from the 1,215 pages
(122 pages; 1,201 sentences; 2,808 triples). ChatGPT (OpenAI) judged each triple
against the Burmese sentence and its surrounding narrative, and separately listed the
relationships stated in each sentence to measure how many the extraction captured.
The extraction itself was done by Gemini 3.1 Flash Lite.

| Sheet | Contents |
|---|---|
| Summary | Counts, precision and recall for each sample and combined, with definitions and 95% bootstrap intervals |
| Triples | All 2,808 triples: verdict (supported, soft error, hard error, unresolved), the judge's reasoning, suggested corrections, and the Burmese sentence with its English translation |
| Recall | All 1,522 source relationships identified by the judge, whether each was covered, partially covered, incorrect or missing, and the matching triple IDs |
| Sentences | All 1,201 sampled sentences in Burmese and English |

The Summary sheet's figures are formulas over the Triples and Recall sheets.
