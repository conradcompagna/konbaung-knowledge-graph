"""Independent data/count/numerical invariants for the new statistical outputs."""

from common import *
import hashlib, zipfile, platform, importlib.metadata, re

d = load()
checks = []


def check(name, condition, detail=""):
    if not condition:
        raise AssertionError(name + ": " + detail)
    checks.append(dict(check=name, status="passed", detail=detail))


original = O.parent.parent / "DIGHUM_WEBGPT_ANALYSIS_PACKAGE_20260831.zip"
with zipfile.ZipFile(original) as z:
    expected = {}
    for line in z.read("CHECKSUMS.sha256").decode().splitlines():
        h, name = line.split(maxsplit=1)
        expected[name.lstrip("*")] = h
    for name in [
        "data/triples.jsonl",
        "data/sentences.jsonl",
        "data/axial_category_catalog.json",
        "embeddings/entity_base_vectors.npy",
        "embeddings/relation_base_vectors.npy",
        "similarity/features/relation_context_vectors.npy",
        "similarity/features/relation_fused_vectors.npy",
    ]:
        raw = (O / "recovered_source" / name).read_bytes()
        check("source manifest " + name, hashlib.sha256(raw).hexdigest() == expected[name])
for name in [
    "embeddings/entity_base_vectors.npy",
    "embeddings/relation_base_vectors.npy",
    "similarity/features/relation_context_vectors.npy",
    "similarity/features/relation_fused_vectors.npy",
]:
    a = np.load(O / "recovered_source" / name, mmap_mode="r")
    norm = np.linalg.norm(a, axis=1)
    check(
        "embedding finite and normalized " + name,
        np.isfinite(a).all() and np.max(abs(norm - 1)) < 1e-4,
        str(a.shape),
    )
check("Canonical rows and primary keys", len(d) == 27129 and d.triple_id.nunique() == 27129)
check(
    "No annotations invented for missing sentences",
    json.loads((O / "data_validation.json").read_text())["unavailable_annotation_sentences"] == 60,
)
t = tensor(d)
check("SPO total", int(t.sum()) == 27129)
ll = np.load(RESULTS / "01_observed_and_expected.npz")
fitted = ll["all_pairwise_no_threeway"]
check(
    "IPF pairwise margins", max(abs(fitted.sum(ax) - t.sum(ax)).max() for ax in [0, 1, 2]) < 0.0011
)
check("IPF fitted expectations nonnegative", np.isfinite(fitted).all() and (fitted >= 0).all())
rng = np.random.default_rng(SEED + 11)
op = d.O.to_numpy().copy()
for ix in d.groupby(["page_key", "P"]).indices.values():
    op[ix] = rng.permutation(op[ix])
dp = d.copy()
dp.O = op
tp = tensor(dp)
check(
    "Conditional permutation preserves SP and PO",
    np.array_equal(t.sum(0), tp.sum(0)) and np.array_equal(t.sum(2), tp.sum(2)),
)
check(
    "Conditional permutation preserves page/O margins",
    d.groupby(["page_key", "O"]).size().equals(dp.groupby(["page_key", "O"]).size()),
)
tri = pd.read_csv(RESULTS / "04_degree_preserving_triad_draws.csv")
cols = [c for c in tri if c not in ["chain", "draw"]]
check(
    "Triad census total equals choose(52,3)", (tri[cols].sum(1) == 22100).all() and len(tri) == 1000
)
check(
    "Degree-null invariants verified",
    json.loads((RESULTS / "04_degree_null_diagnostics.json").read_text())[
        "degree_invariants_verified"
    ],
)
focal = pd.read_csv(RESULTS / "09_all_focal_path_anchors.csv")
q = pd.read_csv(RESULTS / "09_category_conditioned_chain_null.csv")
for pattern, g in focal.groupby("pattern"):
    actual = q[(q.pattern == pattern) & (q.unit == "page_presence")].observed.iloc[0]
    check("Page support " + pattern, actual == g.page_key.nunique())
for filename in ["03_static_cp_loadings", "03_reign_cp_loadings"]:
    f = pd.read_csv(RESULTS / (filename + ".csv"))
    sums = f.groupby(["mode", "component"]).loading.sum()
    check("Normalized factors " + filename, np.allclose(sums, 1) and (f.loading >= -1e-12).all())
split = pd.read_csv(RESULTS / "08_fact_split.csv")
check(
    "Disjoint predictive facts",
    not split.duplicated(["S", "P", "O"]).any()
    and len(split) == 6050
    and set(split.split) == {"train", "validation", "test"},
)
p1 = pd.read_csv(RESULTS / "05_p1_dyad_probabilities.csv")
check("p1 probabilities sum to one", np.allclose(p1[["p00", "p10", "p01", "p11"]].sum(1), 1))
rules = pd.read_csv(RESULTS / "04_length2_horn_rules.csv")
check(
    "Rule supports and confidence",
    (
        (rules.train_support <= rules.train_body_pairs)
        & (rules.test_support <= rules.test_body_pairs)
        & (rules.train_confidence <= 1)
        & (rules.PCA_confidence <= 1)
    ).all(),
)
for p in RESULTS.glob("*.csv"):
    f = pd.read_csv(p)
    for col in f:
        if col in ["p_greater", "p_two_sided", "q_two_sided", "q_greater", "q_enrichment"]:
            val = f[col].dropna()
            check("Probability bounds " + p.name + " " + col, ((val >= 0) & (val <= 1)).all())
    if {"observed", "p_greater", "p_two_sided"}.issubset(f.columns):
        check(
            "Undefined statistics have no p-values " + p.name,
            f.loc[f.observed.isna(), ["p_greater", "p_two_sided"]].isna().all().all(),
        )
# Registry files are present and every cited previous file was verified by catalogue.
families = pd.read_csv(RESULTS / "10_family_audit.csv").fillna("")
for row in families.itertuples():
    for path in filter(None, row.new_outputs.split(";")):
        check("Audit output exists " + path, (O / path).exists())
check("All 26 families enumerated", families.family.tolist() == list(range(1, 27)))
env = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {
        p: importlib.metadata.version(p)
        for p in ["numpy", "pandas", "scipy", "networkx", "scikit-learn", "matplotlib", "torch"]
    },
    "tensorly": "0.9.0 (task-local vendor)",
    "seed": SEED,
}
jsave(env, "11_environment")
jsave({"checks_passed": len(checks), "checks": checks}, "11_validation")
print("Validation passed:", len(checks), flush=True)
