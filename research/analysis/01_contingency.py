"""S x P x O models and an owner-page/predicate constrained randomization."""

from common import *
from scipy.special import xlogy


def dev(x, m):
    return float(2 * (xlogy(x, np.divide(x, m, out=np.ones_like(x), where=m > 0)) - x + m).sum())


def ipf(x, margin_axes, tol=1e-3, maxiter=6000):
    m = np.ones_like(x) * x.sum() / x.size
    targets = [x.sum(axis=a, keepdims=True) for a in margin_axes]
    for it in range(maxiter):
        for axes, target in zip(margin_axes, targets):
            cur = m.sum(axis=axes, keepdims=True)
            m *= np.divide(target, cur, out=np.zeros_like(target), where=cur > 0)
        err = max(
            float(np.abs(m.sum(axis=a, keepdims=True) - t).max())
            for a, t in zip(margin_axes, targets)
        )
        if err < tol:
            break
    return m, dict(
        iterations=it + 1,
        max_margin_error=err,
        absolute_margin_tolerance_counts=tol,
        converged=err < tol,
    )


def conditional_expected(x):
    a = x.sum(2, keepdims=True)
    b = x.sum(0, keepdims=True)
    c = x.sum((0, 2), keepdims=True)
    return np.divide(a * b, c, out=np.zeros_like(x), where=c > 0)


d = load()
x = tensor(d)
N = x.sum()
rows = []
models = {}
for name, axes in [
    ("mutual_independence", [(1, 2), (0, 2), (0, 1)]),
    ("S_independent_O_given_P", [(2,), (0,)]),
    ("all_pairwise_no_threeway", [(2,), (0,), (1,)]),
]:
    m, meta = ipf(x, axes)
    models[name] = m
    rows.append(
        dict(
            model=name,
            G2=dev(x, m),
            positive_observed_cells=int((x > 0).sum()),
            positive_expected_cells=int((m > 0).sum()),
            **meta,
        )
    )
save(pd.DataFrame(rows), "01_loglinear_models")
np.savez_compressed(
    RESULTS / "01_observed_and_expected.npz", observed=x, **models, entities=E, relations=R
)
m = models["all_pairwise_no_threeway"]
idx = np.argwhere(x > 0)
cells = []
for s, p, o in idx:
    exp = m[s, p, o]
    obs = x[s, p, o]
    cells.append(
        dict(
            S=E[s],
            subject_category=LABEL[E[s]],
            P=R[p],
            relation=LABEL[R[p]],
            O=E[o],
            object_category=LABEL[E[o]],
            observed=obs,
            expected_all_pairwise=exp,
            observed_expected=obs / exp if exp else None,
            pearson_residual=(obs - exp) / np.sqrt(exp) if exp else None,
        )
    )
cells = pd.DataFrame(cells)
save(cells.sort_values("pearson_residual", ascending=False), "01_threeway_cells")
# O is reassigned only within the same owner page AND axial predicate. This keeps
# SP, PO, page relation counts, and all page S/O marginals fixed, not SO margins.
rng = np.random.default_rng(SEED)
groups = [np.asarray(v) for v in d.groupby(["page_key", "P"]).indices.values() if len(v) > 1]
s = d.S.map(dict(zip(E, range(52)))).to_numpy()
p = d.P.map(dict(zip(R, range(81)))).to_numpy()
o = d.O.map(dict(zip(E, range(52)))).to_numpy()
base = (s * 81 + p) * 52
expect = conditional_expected(x)
obs = dev(x, expect)
null = []
for b in range(1999):
    op = o.copy()
    for ix in groups:
        op[ix] = rng.permutation(o[ix])
    xp = np.bincount(base + op, minlength=x.size).reshape(x.shape).astype(float)
    null.append(dev(xp, expect))
rr = permutation_summary(obs, null)
rr.update(
    test="S-O association conditional on predicate and owner-page exchangeability",
    exchangeable_groups=len(groups),
    potentially_movable_occurrences=sum(len(i) for i in groups),
    actual_category_variable_groups=sum(len(set(o[i])) > 1 for i in groups),
    null_preserves="SP, PO, owner-page S/O and P margins; does not preserve SO; not a test specifically of the three-way interaction",
)
jsave(rr, "01_conditional_permutation")
save(pd.DataFrame({"G2": null}), "01_conditional_null_draws")
# Descriptive robustness under extraction multiplicity and registers.
sens = []
for name, z in [
    ("all", d),
    ("sentence_category_deduplicated", d.drop_duplicates(["sentence_id", "S", "P", "O"])),
    ("without_registers", d[d.register.isin(["", "NaN"])]),
] + [(v, g) for v, g in d.groupby("volume")]:
    xx = tensor(z)
    e = conditional_expected(xx)
    sens.append(
        dict(
            specification=name,
            n=len(z),
            conditional_mutual_information_bits=dev(xx, e) / (2 * len(z) * np.log(2)),
        )
    )
save(pd.DataFrame(sens), "01_association_sensitivity")
# Source anchors for strong, repeatedly supported cells. Selection is descriptive.
chosen = (
    cells[(cells.observed >= 20) & (cells.expected_all_pairwise >= 5)]
    .sort_values("pearson_residual", ascending=False)
    .head(25)
)
anchors = []
for r in chosen.itertuples():
    z = d[(d.S == r.S) & (d.P == r.P) & (d.O == r.O)].drop_duplicates("page_key").head(4)
    for t in z.itertuples():
        anchors.append(
            dict(
                test="threeway",
                S=r.S,
                P=r.P,
                O=r.O,
                triple_id=t.triple_id,
                sentence_id=t.sentence_id,
                page_key=t.page_key,
                subject=t.s,
                predicate=t.p,
                object=t.o,
            )
        )
save(pd.DataFrame(anchors), "01_source_anchors")
print("Contingency completed", rr, flush=True)
