"""Descriptive p1 exponential-family and Bernoulli block models on actor categories.

These are models of recorded category adjacency, not of historical tie formation.
"""

from common import *
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from sklearn.cluster import KMeans
import networkx as nx

d = load()
codes = E[:23] + E[45:]
indices = [E.index(e) for e in codes]
a = tensor(d).sum(1)[np.ix_(indices, indices)]
y = (a > 0).astype(float)
np.fill_diagonal(y, 0)
n = len(y)
ii, jj = np.triu_indices(n, 1)
forward = y[ii, jj]
reverse = y[jj, ii]
rng = np.random.default_rng(SEED + 5)
train = rng.random(len(ii)) > 0.2


def design(effects):
    if effects:
        D = np.zeros((len(ii), 1 + 2 * (n - 1)))
        Dr = D.copy()
        D[:, 0] = Dr[:, 0] = 1
        # Reference-category coding, regularized estimates for separated categories.
        for k in range(n - 1):
            D[:, 1 + k] = ii == k
            Dr[:, 1 + k] = jj == k
            D[:, n + k] = jj == k
            Dr[:, n + k] = ii == k
        return D, Dr
    return np.ones((len(ii), 1)), np.ones((len(ii), 1))


def fit(effects, recip, keep):
    D, Dr = design(effects)
    k = D.shape[1]
    lam = 0.05

    def objective(beta):
        eta = D @ beta[:k]
        etar = Dr @ beta[:k]
        rho = beta[-1] if recip else 0
        logits = np.column_stack([np.zeros(len(ii)), eta, etar, eta + etar + rho])
        logz = logsumexp(logits, axis=1)
        prob = np.exp(logits - logz[:, None])
        p1 = prob[:, 1] + prob[:, 3]
        p2 = prob[:, 2] + prob[:, 3]
        ll = forward * eta + reverse * etar + forward * reverse * rho - logz
        grad = D[keep].T @ (p1[keep] - forward[keep]) + Dr[keep].T @ (p2[keep] - reverse[keep])
        if recip:
            grad = np.r_[grad, (prob[keep, 3] - forward[keep] * reverse[keep]).sum()]
        return -ll[keep].sum() + lam * np.dot(beta[1:], beta[1:]) / 2, grad + lam * np.r_[
            0, beta[1:]
        ]

    result = minimize(
        objective,
        np.zeros(k + int(recip)),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7},
    )
    beta = result.x
    eta = D @ beta[:k]
    etar = Dr @ beta[:k]
    rho = beta[-1] if recip else 0
    logits = np.column_stack([np.zeros(len(ii)), eta, etar, eta + etar + rho])
    logz = logsumexp(logits, axis=1)
    prob = np.exp(logits - logz[:, None])
    ll = forward * eta + reverse * etar + forward * reverse * rho - logz
    return result, prob, ll, rho


models = []
full = None
for name, eff, rec in [
    ("density_only", False, False),
    ("sender_receiver", True, False),
    ("sender_receiver_reciprocity", True, True),
]:
    fittrain, _, ll, _ = fit(eff, rec, train)
    fitall, prob, llfull, rho = fit(eff, rec, np.ones(len(ii), bool))
    models.append(
        dict(
            model=name,
            parameters=len(fitall.x),
            penalty=0.05,
            converged=fitall.success,
            gradient_max_abs=float(np.max(np.abs(fitall.jac))),
            training_converged=fittrain.success,
            heldout_dyad_nll=float(-ll[~train].mean()),
            full_log_likelihood=float(llfull.sum()),
            reciprocity_log_odds=rho,
            reciprocity_odds_multiplier=np.exp(rho),
        )
    )
    if rec:
        full = prob
save(pd.DataFrame(models), "05_p1_models")
save(
    pd.DataFrame(
        {
            "category_a": [codes[i] for i in ii],
            "category_b": [codes[i] for i in jj],
            "a_to_b": forward,
            "b_to_a": reverse,
            "train_dyad": train,
            "p00": full[:, 0],
            "p10": full[:, 1],
            "p01": full[:, 2],
            "p11": full[:, 3],
        }
    ),
    "05_p1_dyad_probabilities",
)
# Model adequacy, including triads it was not fitted to reproduce.
g = nx.from_numpy_array(y, create_using=nx.DiGraph)
obs = nx.triadic_census(g)
draw = []
for b in range(999):
    u = rng.random(len(ii))
    state = (u[:, None] > full.cumsum(1)).sum(1)
    new = np.zeros_like(y)
    new[ii, jj] = np.isin(state, [1, 3])
    new[jj, ii] = np.isin(state, [2, 3])
    draw.append(nx.triadic_census(nx.from_numpy_array(new, create_using=nx.DiGraph)))
draw = pd.DataFrame(draw)
save(draw, "05_p1_gof_draws")
rr = []
for k, v in obs.items():
    rr.append(dict(triad=k, **permutation_summary(v, draw[k])))
rr = pd.DataFrame(rr)
rr["q_two_sided"] = bh(rr.p_two_sided)
save(rr, "05_p1_triad_adequacy")
# Bernoulli SBM with explicit latent memberships, coordinate-ascent profile ML.
mask = ~np.eye(n, dtype=bool)


def blockscore(labels, k):
    idx = labels[:, None] * k + labels[None, :]
    m = np.bincount(idx[mask], minlength=k * k)
    count = np.bincount(idx[mask], weights=y[mask], minlength=k * k)
    prob = np.clip(np.divide(count, m, out=np.zeros_like(count), where=m > 0), 1e-9, 1 - 1e-9)
    ll = float((count * np.log(prob) + (m - count) * np.log1p(-prob)).sum())
    return ll, prob.reshape(k, k)


profile = np.concatenate([y, y.T], axis=1)
solutions = []
best = None
for k in range(2, 7):
    for seed in [SEED, SEED + 1, SEED + 2]:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(profile)
        ll, _ = blockscore(labels, k)
        for sweep in range(30):
            changed = 0
            for node in rng.permutation(n):
                old = labels[node]
                if (labels == old).sum() <= 2:
                    continue
                cur = ll
                new = old
                for c in range(k):
                    labels[node] = c
                    val, _ = blockscore(labels, k)
                    if val > cur + 1e-8:
                        cur = val
                        new = c
                labels[node] = new
                if old != new:
                    changed += 1
                    ll = cur
            if not changed:
                break
        # Discrete membership penalty is made explicit, not called ordinary BIC.
        criterion = -2 * ll + k * k * np.log(mask.sum()) + 2 * n * np.log(k)
        row = dict(
            k=k,
            seed=seed,
            log_likelihood=ll,
            penalized_description_score=criterion,
            sweeps=sweep + 1,
            stopped_without_moves=changed == 0,
        )
        solutions.append(row)
        if best is None or criterion < best[0]:
            best = (criterion, labels.copy(), k)
save(pd.DataFrame(solutions), "05_sbm_selection")
save(
    pd.DataFrame(
        {"entity_id": codes, "label": [LABEL[c] for c in codes], "block": best[1], "k": best[2]}
    ),
    "05_sbm_memberships",
)
ll, b = blockscore(best[1], best[2])
save(
    pd.DataFrame(
        [
            dict(sender_block=i, receiver_block=j, edge_probability=b[i, j])
            for i in range(best[2])
            for j in range(best[2])
        ]
    ),
    "05_sbm_probabilities",
)
jsave(
    dict(
        categories=codes,
        nodes=n,
        edges=int(y.sum()),
        risk_set="all ordered distinct pairs of the listed 30 political/social actor categories",
        statistical_scope="p1 is an exact dyad-independent exponential-family likelihood with weak L2 regularization for separated category effects; GOF draws conditional on fitted parameters, no claim of process causality or parameter uncertainty. SBM is fitted by coordinate-ascent profile likelihood and can reach local optima. Categories are analytical aggregates, not distinct historical people. Neither fit is a temporal ERGM or a causal historical ERGM.",
    ),
    "05_model_scope",
)
print("Network models completed", models, flush=True)
