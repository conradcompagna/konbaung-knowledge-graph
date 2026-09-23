from common import *
import html, re, hashlib


def table(df, cols=None):
    if cols is not None:
        df = df[cols]

    def fmt(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4g}"
        return str(v).replace("|", " / ").replace("\n", " ")

    return (
        "| "
        + " | ".join(map(str, df.columns))
        + " |\n|"
        + "|".join(["---"] * len(df.columns))
        + "|\n"
        + "\n".join(
            "| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)
        )
        + "\n"
    )


def write(name, txt):
    (O / name).write_text(txt, encoding="utf-8")


summary = json.loads((RESULTS / "10_audit_summary.json").read_text())
families = pd.read_csv(RESULTS / "10_family_audit.csv").fillna("")
methods = pd.read_csv(RESULTS / "10_requested_method_checklist.csv").fillna("")
validation = json.loads((RESULTS / "11_validation.json").read_text())
audit = (
    f"""# Audit of existing Konbaung analyses

The supplied catalogue contains **26 families and 380 bullet items**, including overlapping metrics, alternative algorithms and proposed applications. The audit found extensive earlier work, added a historically focused battery of missing analyses, and records the unrun alternatives explicitly. Coverage of a family does not imply that every named algorithm in it has been executed.

## What was inspected

The article folder contains **14 top-level ZIP packages**. Including nested members and loose files, the inventory has **3,350 entries**, **70 Word-document occurrences**, **154 Python-script occurrences representing 127 distinct scripts**, and 29 PDFs (117 pages of extracted text). Repeated archive copies are not counted as independent studies. Reports, scripts, result tables, correction notes and withdrawal lists were cross-checked. A keyword hit was used to locate evidence, not to certify a completed test. The original project was also searched for the missing advanced implementations and used to retrieve the intact source.

The primary source for the new work is `../../DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831.zip` in the original `dighumproject` directory. The copy nested in the large final reproducibility package is truncated. Forty-one complete members were initially recovered with CRC and manifest checks; the intact original subsequently supplied the missing relation context/fused matrices. The original and recovered canonical triples, sentences, taxonomy and base vectors match byte for byte. No original inputs were edited.

## Coverage by family

"""
    + table(families[["family", "family_name", "prior_status", "new_work"]])
    + """
## Exact previous evidence and new outputs

Each entry below names the archived evidence rather than relying on a report title. `!/` separates an archive from its member path.

"""
)
for r in families.itertuples():
    audit += (
        f"### {r.family} {r.family_name}\n\n**Before:** {r.prior_status}. {r.prior_coverage}\n\n"
    )
    if r.prior_evidence:
        audit += (
            "**Located evidence:**\n\n"
            + "".join("- `" + p + "`\n" for p in r.prior_evidence.split(" | "))
            + "\n"
        )
    audit += "**New work:** " + r.new_work + "\n\n"
    if r.new_outputs:
        audit += (
            "**Outputs:** "
            + ", ".join(f"[{Path(p).name}]({p})" for p in r.new_outputs.split(";"))
            + ".\n\n"
        )
    if r.remaining_limits:
        audit += "**Boundary:** " + r.remaining_limits + "\n\n"
audit += """## Previously attempted but withdrawn or qualified

The earlier archives explicitly withdraw or limit several attractive results. They remain classified as attempted, not missing:

- **Titling causes subsequent activity:** exposure imbalance and identity uncertainty undermine the earlier pre/post activation result. Do not reinstate it from page order.
- **Nearest-prior-title semantic continuity:** equal-sized random prior sets perform similarly; chronological inheritance was not established.
- **Positive title/appointment/access/appanage sentence association:** early results were sensitive to sentence density; later matched-page analyses replace them.
- **R12–R64 semantic proximity as independent validation:** these categories share raw predicates, so part of the proximity is circular.
- **Exact title strings or embedding neighbors as identities:** recurrent titles, homonyms and circular title bridges can merge distinct people.
- **Sovereign share correlated with an HHI including the sovereign:** the quantities are mechanically coupled.

The controlling files are `WITHDRAWN_OR_SUPERSEDED_RESULTS.md`, `WITHDRAWN_OR_NOT_USED.md`, and `CLEAR_ERRORS_AND_CAUTIONS.md` in the indexed archives. Their exact extracted paths are searchable in `inventory.csv`.

## Method-level enumeration and remaining work

The [380-item checklist](results/10_requested_method_checklist.csv) distinguishes previous evidence, new execution, partial execution, missing data prerequisites and unrun alternatives. It deliberately does not claim an exhaustive benchmark of every centrality, community algorithm, tensor factorization or neural model. Exact equivalence algorithms, RESCAL/DEDICOM, supra-adjacency multilayer models, general closed/maximal subgraph mining, and additional embedding/GNN architectures remain listed as unrun variants.

Historical event-time paths, REMs, survival models and dynamic actor/network processes require validated event order, identities, exposure or network-state observations. Geographic distance/autocorrelation/gravity models require validated geolocation. Those inputs are not supplied by narrative page order or the existing three-tier place-string gazetteer. The new results therefore do not relabel page windows as elapsed historical time, or mentions as complete population risk sets.

The inventory is [CSV](inventory.csv) and [JSON](inventory.json); the [method-search evidence](results/10_archive_method_search_hits.csv) records source paths and line numbers. The HTML reader provides a filterable version of the method checklist.
"""
write("AUDIT_OF_EXISTING_TESTS.md", audit)

ll = pd.read_csv(RESULTS / "01_loglinear_models.csv")
cells = pd.read_csv(RESULTS / "01_threeway_cells.csv")
supported = cells[(cells.observed >= 20) & (cells.expected_all_pairwise >= 5)].head(12)
q = pd.read_csv(RESULTS / "02_layer_qap.csv")
q = q[q.null_model == "ontology_stratified"]
sens = pd.read_csv(RESULTS / "06_layer_qap_sensitivity.csv")
noking = sens[sens.specification == "without_sovereign"]
q = q.merge(
    noking[["r1", "r2", "observed", "q_two_sided"]],
    on=["r1", "r2"],
    suffixes=("_all", "_no_sovereign"),
)
perm = json.loads((RESULTS / "01_conditional_permutation.json").read_text())
cp = json.loads((RESULTS / "03_static_cp_fit.json").read_text())
stability = pd.read_csv(RESULTS / "06_tensor_component_stability.csv")
mot = pd.read_csv(RESULTS / "09_category_conditioned_chain_null.csv")
focal = [
    "allegiance -> authorization",
    "information -> authorization",
    "authorization -> execution",
    "resistance -> coercion",
]
rules = pd.read_csv(RESULTS / "04_length2_horn_rules.csv")
sem = pd.read_csv(RESULTS / "03_semantic_structure_qap.csv")
pred = pd.read_csv(RESULTS / "08_link_prediction.csv")
p1 = pd.read_csv(RESULTS / "05_p1_models.csv")
reigns = pd.read_csv(RESULTS / "06_adjacent_reign_tensor_comparison.csv")
results = (
    f"""# New analyses of power relations in the Konbaung triples

The new analyses support **differentiated but connected forms of power in the chronicle**. Royal allocation, status-bearing, military operations, upward information/resource flows and religious patronage have distinguishable structures. Important layer associations remain after removing the sovereign category. The stronger claim that these form recurrent causal chains is not supported by the stricter motif nulls or the sparse held-out rule results.

## Data and units

The verified source has **27,129 canonical triple occurrences**, **11,282 sentence/translation records**, **10,498 triple-bearing sentences**, **23,890 distinct entity strings**, **11,886 predicate strings**, **52 entity categories** and **81 relation categories**. There are 1,215 annotation pages and 1,206 owner pages with triples. Sixty sentences lack canonical V3 annotation and have not been filled with invented claims.

Three objects are kept distinct: directed graphs of **analytical categories**, the earlier conservative **named-person undirected graph**, and **exact-tag graphs scoped within an owner page**. The 52/81 axial categories are supervised analytical codes assigned by a model; the embedding-derived semantic communities are separate exploratory products. A category such as Sovereign is not one person; an identical title string is not necessarily one person. Subject/object direction is the extractor's grammatical/semantic encoding, and has not been globally converted into grantor-to-recipient direction. Foreign and rival polities mentioned by the chronicle are included. Thus the broad patterns concern the recorded political world, not exclusively Konbaung officeholders.

All resampling uses fixed seeds. Where used for new inference, pages stay together or permutations are explicitly constrained. P-values describe the chosen randomization/model, not the probability that a historical claim is true. Related tests receive BH correction within the stated test family. The analyses are exploratory and were not preregistered; the 15 focal layer pairs and seven motif families were fixed in code before their tests were run.

## 1 Three-way organization exceeds marginal frequencies

The complete S × P × O table has 6,050 occupied category combinations. Three nested expected-count specifications were fitted:

"""
    + table(ll[["model", "G2", "iterations", "max_margin_error", "converged"]])
    + f"""
The all-pairwise fit controls the SP, PO and SO margins. Iterative proportional fitting reached an absolute marginal discrepancy below **0.001 occurrence**. The largest residuals often belong to singletons with tiny expectations; they are not treated as headline discoveries. The following repeated cells have observed counts at least 20 and expected counts at least 5:

"""
    + table(supported[["S", "P", "O", "observed", "expected_all_pairwise", "observed_expected"]])
    + f"""
For example, commander × appointment/delegation × armed unit has **48 occurrences versus 15.92 expected**, about **3.02 times expected** after all pairwise margins. Ministers × assignment × policy/objective has **21 versus 6.16**. These are candidate institutional specializations, not individually significant cells: the displayed ratios and Pearson residuals are descriptive and are not leverage-studentized residuals.

An independent randomization shuffled object categories **within owner page and predicate**, preserving SP/PO and page endpoint margins. The global conditional S–O statistic was **G² = {perm["observed"]:,.2f}**, compared with null mean **{perm["null_mean"]:,.2f}**, with one-sided **p = {perm["p_greater"]}** across 1,999 draws. This tests endpoint association given predicate/page; it **does not isolate a three-way interaction after controlling SO**. No sparse-table asymptotic chi-square p-value is reported.

Source checks clarify the meaning. `vol2_s003750` describes a minister and commander being assigned to supervise canal restoration; `vol3_s002648` assigns tax collection and fiscal oversight to named ministers. In the military cell, `APPOINTED_AS_LEADER_OF` and `APPOINTED_BY_KING_TO_COMMAND` encode an officer linked to a unit, not necessarily an officer independently appointing troops. Royal women and princes also have overrepresented appanage-to-territory associations; `vol1_s003103` explicitly connects Ratanamahe's reporting to title, palace position and the Kyaukmaw fief. These passages support differentiated assignments and benefits, while constraining claims about who initiated them.

## 2 Relation layers remain connected beyond the king

All 81 layers were measured on the same 52 categories. The 3,240 pairwise descriptive comparisons are separate from the 15 focal inferential comparisons. QAP correlates log occurrence counts on corresponding directed category pairs; 1,999 permutations relabel one layer within four explicit ontology strata.

"""
    + table(
        q[
            [
                "r1",
                "r2",
                "observed_all",
                "q_two_sided_all",
                "observed_no_sovereign",
                "q_two_sided_no_sovereign",
            ]
        ]
    )
    + """
Appointment–officeholding and command–implementation remain strongly associated after E01 is removed. Reporting–removal, ritual–titling and tax–tribute do not retain the same corrected support after that removal. This distinction is more informative than a blanket assertion that everything forms one royal power network. Association surviving exclusion does not imply actor autonomy or successful command transmission.

![Relation layers and sovereign exclusion](figures/01_layer_coupling.png)

The page-bootstrap bars are percentile ranges of 999 within-volume page resamples; they describe resampling variability and can be biased for this sparse, nonlinear statistic. QAP supplies the reported inferential comparison. Register exclusion and exclusion of both E01 and E30 are separate sensitivity tables. Undefined correlations receive no p-value.

## 3 Joint factors distinguish recurring repertoires

Nonnegative CP was fitted to square-root counts, with ranks 3, 5 and 7 and two starting seeds compared on withheld pages. Rank 7 gave the lowest selection-set negative log likelihood, about **9.596 per claim**, versus **9.914** for a marginal-independence baseline. Because that holdout selected the rank, it is a selection diagnostic rather than an unbiased final test score. The full-data relative reconstruction error is **"""
    + f"{cp['relative_error']:.3f}"
    + """**, so considerable structure remains unexplained. A Tucker 8 × 8 × 8 fit is also supplied.

The seven fitted components were interpreted as military action, status-bearing, court performance, upward flows, royal allocation, religious patronage and governing operations. Their factor loadings, not these labels, are the primary outputs.

![Joint subject relation object factors](figures/02_tensor_roles.png)

The military, upward-flow, royal-allocation and governing components recur strongly in the alternate-seed and register-exclusion fits. The court-performance component **does not**: its cross-fit similarity is only about 0.14 by seed and 0.11 after register exclusion. Its boundaries should not be treated as established. Religious patronage is moderately stable. Excluding R12 and R64 still leaves recognizable military, upward-flow, allocation and patronage components. The status-bearing/royal-allocation distinction partly reflects receive versus bestow encoding; it is not itself evidence that the model discovered a new institutional duality.

The four-mode fit adds the 12 reign/crisis segments, scaling each segment to occurrences per 1,000 claims. It is an exploratory map of changing textual composition. It does not reconstruct event hazards, reign durations, or continuous political evolution.

## 4 Labelled paths are real configurations but not established mechanisms

Within owner pages, the raw exact-tag graph has **26,550 unique non-self labelled edges**, **18,228 two-edge paths**, **526 transitive edge configurations** and **108 oriented cyclic configurations**. Predicate families were specified as authorization, execution, information, coercion, resistance, allegiance and other.

Weak nulls shuffling labels within pages make several paths look strongly enriched. But preserving the ordered subject/object category pair within each page removes the corrected support for the focal patterns:

"""
    + table(
        mot[(mot.pattern.isin(focal)) & (mot.unit == "path_count")][
            ["pattern", "observed", "null_mean", "p_greater", "q_greater"]
        ]
    )
    + """
![Effect of stronger motif controls](figures/03_motif_null_sensitivity.png)

These results show that the apparent paths can largely follow from who tends to send and receive each type of relation. Of information-to-authorization paths, **89.3%** have the sovereign category as intermediate; for allegiance-to-authorization it is **96.5%**. Counting each supporting page once yields the same negative conclusion under the stronger null. The paths are spatial configurations within text pages, not time-respecting event sequences.

The separate unlabelled category triad test uses 1,000 saved draws in two directed-degree-preserving MCMC chains, with symmetric edge-switch proposals, triangle reversals and rejection self-loops. Degree invariants and triad totals pass. Several triads differ from that null, but these are category-graph structure diagnostics. The saved chain means and autocorrelations should accompany any use of their significance profiles; sampling is approximate and complete mixing is not guaranteed.

## 5 Rule mining supplies few generalizable rules

One-edge implications and length-two Horn rules were enumerated with page-scoped entities. Supports count distinct page/subject/object bindings. Standard confidence, head coverage and PCA confidence are explicit. A training support of at least three bindings leaves only six length-two candidates:

"""
    + table(
        rules[
            [
                "body1",
                "body2",
                "head",
                "train_support",
                "train_pages",
                "train_confidence",
                "test_body_pairs",
                "test_support",
                "test_confidence",
            ]
        ]
    )
    + """
Most are concentrated on one training page; held-out support is usually zero. For the assignment–fortification–deployment rule (R11/R46/R44), there are three training supporting bindings across three pages and one supporting test binding among four test body pairs. That is a source-reading lead, not a general historical law. A high PCA confidence based on a tiny denominator does not override this limitation. Every rule and its candidate passages remain available, including failures.

## 6 Graph position and statistical network models

The conservative archived named-person graph contains **599 nodes, 443 edges and 162 components**; its largest component contains **64 nodes**. It has **427 bridges**. Additional paths, betweenness and articulation measures are supplied, but this fragmentation is also a statement about extraction and conservative identity resolution. It is not evidence that actual Konbaung society was disconnected.

Category-level analyses add Burt constraint/effective size, Gould–Fernandez brokerage, degrees and centralities, communities, reachability, flow hierarchy, generalized trophic levels, cores, rich club and nestedness. Quantities such as current flow, hitting time, inverse-frequency distance and minimum cuts are mathematical graph diagnostics, not historical quantities of influence, money, travel time or coercive capacity.

For 30 actor/organization categories, the regularized p1 model estimates recorded directed adjacency with sender, receiver and reciprocity terms. Held-out dyad negative log likelihood falls from **1.384** for density alone to **0.876** for sender/receiver effects and **0.850** after reciprocity. The fitted reciprocity multiplier is about **3.63**, conditional on this category model. This is not a causal ERGM of historical people. A coordinate-ascent Bernoulli SBM selects three blocks under its explicit description-score penalty; sparse categories group together partly because of observability. Model adequacy simulations and all candidate partitions are supplied.

## 7 Semantics agree with structure but learned completion adds no advantage

The relation base/context/fused centroids correlate with endpoint-structure similarity by **0.371 / 0.476 / 0.414**. QAP remains positive within broad relation domains and in a Freedman–Lane residual QAP controlling frequency difference and shared raw predicates. The nine tests have corrected q = 0.001 at the current permutation resolution. This is internal agreement between model-assisted descriptions of the same corpus, not independent historical validation; contextual embeddings contain the source sentences from which the triples were extracted.

On a separate split of 6,050 distinct category facts, the 1,210 test facts are disjoint from training and validation. DistMult and ComplEx both underperform the predicate–object frequency baseline:

"""
    + table(pred[["model", "filter", "MRR", "hits1", "hits10", "facts"]])
    + """
![Semantic consistency and predictive checks](figures/04_semantics_and_prediction.png)

No predicted fact has been added to the database. This diagnostic does not support deploying more elaborate link prediction to fill gaps in the historical record.

## 8 Reign comparisons are about recorded composition

Adjacent reign/crisis segments were compared with Jensen–Shannon divergence on the complete S-P-O distribution. Pages are permuted within volume, retaining page blocks. Some early comparisons with very little source exposure are null; later comparisons often differ:

"""
    + table(reigns[["reign1", "reign2", "pages1", "pages2", "observed", "q_greater"]])
    + """
These contrasts do not distinguish political change from narrative selection, compilation, register composition or extraction differences. The data do not provide complete repeated network states for TERGM/SAOM, or validated office spells for survival analysis.

## Reproduction and inspection

The script order is recorded in [RUN_ANALYSES.ps1](RUN_ANALYSES.ps1). The exact source archive and input hashes are in [original_source_verification.json](original_source_verification.json), [input_provenance.json](input_provenance.json), and the original source manifest. The environment, seeds and **"""
    + str(validation["checks_passed"])
    + """ passing invariants** are recorded in [11_validation.json](results/11_validation.json) and [11_environment.json](results/11_environment.json). Four figures are supplied as PNG and SVG. The [selected source passages](results/09_selected_source_passages.csv) retain Burmese and English; the HTML source reader links them to sentence IDs.

The code uses [TensorLy's documented decomposition routines](https://tensorly.org/dev/modules/api.html) and [NetworkX graph definitions](https://networkx.org/documentation/stable/reference/algorithms/index.html). The rule measures follow the support/standard/PCA-confidence distinctions in the supplied [AMIE reference](https://suchanek.name/work/publications/vldbj2015.pdf). The implementation is bounded custom rule enumeration, not AMIE software. The custom degree null, conditioning schemes, transformations and all departures from full historical-process models are stated in the scripts and scope JSONs.
"""
)
write("NEW_ANALYSES_AND_RESULTS.md", results)

interpretation = """# Preliminary historical interpretation

The database suggests that power in the political world narrated by the Konbaung chronicles was **asymmetrically distributed through differentiated relationships**. The sovereign is a privileged source of recognized standing, allocation and command, while military action, administration, information, revenue and religious authority involve distinct combinations of other actors. A useful preliminary formulation is: **royal centrality was enacted through the organization of intermediaries and their relationships to offices, people, resources and recognized rank**.

This goes beyond counting how often the king appears. Appointment and officeholding, command and implementation, and petition and reporting retain related endpoint structures after the sovereign category is removed. Those patterns suggest an organized division of political work in the narrative. They do not by themselves establish how effectively the king controlled it, or how independently intermediaries acted.

The joint analysis also distinguishes several recurring repertoires: military operations, the allocation of positions and benefits, status-bearing, upward reporting/tribute/submission, and religious patronage. Power therefore appears through different kinds of action and entitlement, with actors occupying different positions in each. Titles belong in this account as one way of recording and recognizing standing, connected to offices and benefits. Their causal priority over capacity has not been established. The earlier title-activation inference was withdrawn, and the new analyses do not restore it.

Concrete passages help interpret the statistics. The appointment/delegation category often links commanders to armed units, or ministers to an assigned project. In `vol2_s003750`, royal orders assign a minister and commander to canal restoration. In `vol3_s002648`, fiscal oversight is allocated to named ministers. These passages show assignments of responsibility; the grammatical subject of an extracted triple can be the appointed official rather than the appointing sovereign. They should not be counted automatically as independent subordinate acts of appointment.

Royal women and princes also appear in patterned connections to towns and appanages. For example, `vol1_s003103` links Ratanamahe's reporting to a title, palace position and the Kyaukmaw fief. This supports including royal women in the analysis of allocated status and resources. It does not support treating all receipt-voice appanage triples as evidence that their recipients exercised the sovereign's allocative power.

Information, allegiance and resources are directed upward in many recorded relations, while appointments, grants and commands are directed outward or recorded as received. This makes the court a point where different political relationships meet. But the most attractive feedback-loop claim needs restraint: information-to-authorization and resistance-to-coercion paths cease to be independently enriched once their endpoint-category structure is preserved. Their presence is evidence of a relational repertoire; it is not proof of a general sequence in which reports reliably produce decisions or resistance reliably produces sanctions. Sparse held-out rule supports reinforce that limit.

The database can therefore tell us **which kinds of actors are connected through which instruments of power, how those combinations recur, how narrative attention changes across reigns, and where the source supplies exceptions or competing forms of authority**. It can make generalizations more precise, expose counterexamples and direct close reading. It cannot on its own measure the full reach or effectiveness of royal control, reconstruct a complete social network, or establish that one recorded relation caused another.

The strongest preliminary answer is consequently a historical hypothesis about the organization and representation of power: **Konbaung rule is narrated as a monarchy whose privileged authority was exercised through differentiated assignments, recognitions, obligations and intermediaries, with room for negotiation, overlapping authority and resistance.** How far this narrative structure describes everyday political practice remains a question for the linked Burmese passages and comparison with other sources.

Evidence and qualifications are in [New analyses and results](NEW_ANALYSES_AND_RESULTS.md). Existing studies and withdrawn claims are separated in [Audit of existing tests](AUDIT_OF_EXISTING_TESTS.md).
"""
write("PRELIMINARY_HISTORICAL_INTERPRETATION.md", interpretation)

readme = f"""# Konbaung analysis audit and new results

The audit covers all 26 families in the pasted catalogue. The new analyses investigate how different actors, relations and recipients organize the chronicle's account of power. The strongest results concern differentiated roles and relation-layer structure; stronger causal claims fail several of the new controls.

- [Preliminary historical interpretation](PRELIMINARY_HISTORICAL_INTERPRETATION.md)
- [Audit of existing tests](AUDIT_OF_EXISTING_TESTS.md)
- [New analyses and results](NEW_ANALYSES_AND_RESULTS.md)
- [Interactive report and searchable enumeration](index.html)
- [26-family audit CSV](results/10_family_audit.csv)
- [380-item method checklist CSV](results/10_requested_method_checklist.csv)
- [Source passages in Burmese and English](source_passages.html)

**Scope:** {summary["method_status_counts"].get("Not run alternative", 0)} named alternatives remain explicitly unrun, and historical event-time/spatial models require additional validated inputs. This is a completed audit and a substantial new test battery, not a claim that every algorithm in the catalogue has been executed.

**Source:** the intact `DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831.zip` in the original `dighumproject` directory. Its canonical data match the article archive. Originals were left unchanged. Reproducible code, raw result tables, source hashes, seeds and {validation["checks_passed"]} passing checks are included.

Run [RUN_ANALYSES.ps1](RUN_ANALYSES.ps1) from this directory to reproduce the new analyses using the verified extracted inputs. The tensor package is in `vendor`; the installed scientific Python versions are recorded in `results/11_environment.json`. `scripts/inventory.py`, `prepare.py`, and `supplement_inventory.py` document the source inventory/extraction phase. The original archive is required to repeat source verification. No paid API calls are required.
"""
write("READ_ME_FIRST.md", readme)


# Dependency-free HTML reader; source prose is escaped before rendering.
def inline(s):
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(
        r"<code>(vol[123]_s\d+)</code>", r'<a href="source_passages.html#\1"><code>\1</code></a>', s
    )
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def mdhtml(text):
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("| "):
            chunk = []
            while i < len(lines) and lines[i].startswith("|"):
                chunk.append(lines[i])
                i += 1
            out.append(
                '<div class="tablewrap"><table><thead><tr>'
                + "".join(
                    "<th>" + inline(c.strip()) + "</th>" for c in chunk[0].strip("|").split("|")
                )
                + "</tr></thead><tbody>"
            )
            for row in chunk[2:]:
                out.append(
                    "<tr>"
                    + "".join(
                        "<td>" + inline(c.strip()) + "</td>" for c in row.strip("|").split("|")
                    )
                    + "</tr>"
                )
            out.append("</tbody></table></div>")
            continue
        m = re.match(r"(#+) (.*)", line)
        if m:
            level = min(4, len(m[1]))
            out.append(f"<h{level}>" + inline(m[2]) + f"</h{level}>")
            i += 1
            continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if m:
            out.append(
                f'<figure><img alt="{html.escape(m[1])}" src="{html.escape(m[2])}"></figure>'
            )
            i += 1
            continue
        if line.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append("<li>" + inline(lines[i][2:]) + "</li>")
                i += 1
            out.append("</ul>")
            continue
        para = [line]
        i += 1
        while (
            i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ", "!["))
        ):
            para.append(lines[i])
            i += 1
        out.append("<p>" + inline(" ".join(para)) + "</p>")
    return "\n".join(out)


css = """body{font:17px/1.65 Georgia,serif;color:#182c34;background:#f8f8f5;margin:0}header,main{max-width:1160px;margin:auto;padding:26px 32px}header{border-bottom:1px solid #bfcbd0}nav{display:flex;gap:12px;flex-wrap:wrap}button,input,select{font:15px system-ui;padding:9px 12px;border:1px solid #a4b5bb;background:white;border-radius:3px}button{cursor:pointer}button.active{background:#215e75;color:white}h1,h2,h3{font-family:system-ui;line-height:1.2;color:#182c34}h1{font-size:32px}h2{font-size:24px;margin-top:46px}h3{font-size:19px;margin-top:32px}a{color:#155975}p{max-width:980px}table{border-collapse:collapse;font:13px/1.45 system-ui;min-width:650px;width:100%}td,th{padding:9px;border:1px solid #d2dbdf;text-align:left;vertical-align:top}th{background:#e8eff2;position:sticky;top:0}tr:nth-child(even){background:#f1f4f5}.tablewrap{overflow:auto;max-height:700px;margin:22px 0}code{font:12px ui-monospace,monospace;overflow-wrap:anywhere;background:#edf1f3;padding:1px 3px}img{max-width:100%;height:auto}figure{margin:28px 0}.tab[hidden]{display:none}.tools{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}.muted{font:14px system-ui;color:#526771}details{border-bottom:1px solid #c5d1d6;padding:15px 0}summary{cursor:pointer;font-family:system-ui}blockquote{margin:12px 0;padding-left:18px;border-left:3px solid #a8bac3}"""
methodtable = methods[["family", "family_name", "requested_method", "status", "detail"]].to_html(
    index=False, escape=True, table_id="methodtable"
)
sections = (
    '<section id="interpretation" class="tab">'
    + mdhtml(interpretation)
    + '</section><section id="results" class="tab" hidden>'
    + mdhtml(results)
    + '</section><section id="audit" class="tab" hidden>'
    + mdhtml(audit)
    + "</section>"
)
options = "".join(
    "<option>" + html.escape(x) + "</option>" for x in sorted(methods.status.unique())
)
sections += (
    '<section id="methods" class="tab" hidden><h1>Method enumeration</h1><p>All 380 bullet items from the supplied catalogue, including overlapping metrics and unrun alternatives.</p><div class="tools"><input id="search" aria-label="Search methods" placeholder="Search methods or families"><select id="status" aria-label="Filter status"><option value="">All statuses</option>'
    + options
    + '</select><span id="count" class="muted"></span></div><div class="tablewrap">'
    + methodtable
    + "</div></section>"
)
js = """document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(s=>s.hidden=s.id!==b.dataset.tab);document.querySelectorAll('nav button').forEach(x=>x.classList.toggle('active',x===b));window.scrollTo(0,0)});const filter=()=>{let n=0;document.querySelectorAll('#methodtable tbody tr').forEach(r=>{const ok=r.textContent.toLowerCase().includes(document.querySelector('#search').value.toLowerCase())&&(!document.querySelector('#status').value||r.cells[3].textContent===document.querySelector('#status').value);r.hidden=!ok;if(ok)n++});document.querySelector('#count').textContent=n+' items shown'};document.querySelector('#search').oninput=filter;document.querySelector('#status').onchange=filter;filter();"""
write(
    "index.html",
    '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Konbaung power relations audit and results</title><style>'
    + css
    + '</style><header><p class="muted">11 September 2026 · Verified canonical corpus · Reproducible analysis</p><nav><button class="active" data-tab="interpretation">Historical interpretation</button><button data-tab="results">New results</button><button data-tab="audit">Existing work audit</button><button data-tab="methods">380-item checklist</button><a href="source_passages.html">Source passages</a></nav></header><main>'
    + sections
    + "</main><script>"
    + js
    + "</script></html>",
)
passages = pd.read_csv(RESULTS / "09_selected_source_passages.csv").fillna("")
content = '<h1>Selected source passages</h1><p>Canonical Burmese and supplied English translations. These are model-assisted translations; the source is retained for historical reading. Sentence IDs connect to the result ledgers. <a href="index.html">Return to the report</a>.</p>'
for r in passages.itertuples():
    content += (
        '<details id="'
        + html.escape(r.sentence_id)
        + '"><summary>'
        + html.escape(f"{r.sentence_id} · {r.volume} page {r.page}")
        + '</summary><p lang="my">'
        + html.escape(r.burmese)
        + "</p><p>"
        + html.escape(r.english_translation)
        + "</p></details>"
    )
write(
    "source_passages.html",
    '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Konbaung source passages</title><style>'
    + css
    + "</style><main>"
    + content
    + "</main></html>",
)
write(
    "RUN_ANALYSES.ps1",
    """$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$analysisScripts = @('01_contingency.py','02_graph_layers.py','03_tensor_semantics.py','04_paths_rules_motifs.py','04b_degree_null.py','05_network_models.py','06_robustness_comparison.py','07_brokerage_paths_hierarchy.py','08_predictive_check.py','09_motif_sensitivity_sources.py','10_audit_catalogue.py','11_validate.py','12_figures.py','13_write_reports.py','14_finalize.py')
foreach ($analysisScript in $analysisScripts) {
    Write-Host "Running $analysisScript"
    & python (Join-Path 'scripts' $analysisScript)
    if ($LASTEXITCODE -ne 0) { throw "Analysis failed: $analysisScript" }
}
""",
)
# Repair provenance metadata from the initial recovery before the intact source was found.
recovery = json.loads((O / "source_recovery.json").read_text())
recovery["nested_member"] = (
    "Konbaung_Status_Power_FINAL_Statistical_Reproducibility_UPDATED/02_CANONICAL_SOURCE/DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831.zip"
)
recovery["resolution"] = "Intact original archive located and verified; used for new analyses."
(O / "source_recovery.json").write_text(json.dumps(recovery, indent=2))
print("Reports written", flush=True)
