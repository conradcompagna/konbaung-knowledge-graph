from common import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)
F = O / "figures"
F.mkdir(exist_ok=True)


def finish(fig, name):
    fig.savefig(F / (name + ".png"), dpi=190, bbox_inches="tight")
    fig.savefig(F / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


short = {
    "R11": "Appointment",
    "R12": "Title conferral",
    "R13": "Officeholding",
    "R14": "Appanage",
    "R23": "Command",
    "R24": "Implementation",
    "R25": "Petition",
    "R26": "Reporting",
    "R29": "Accountability",
    "R21": "Removal",
    "R33": "Tax collection",
    "R34": "Tribute",
    "R36": "Labour mobilization",
    "R44": "Deployment",
    "R54": "Rebellion",
    "R52": "Punishment",
    "R76": "Noncompliance",
    "R09": "Ritual",
    "R72": "Faction / patronage",
    "R73": "Fragmentation",
}
q = pd.read_csv(RESULTS / "02_layer_qap.csv")
q = q[q.null_model == "ontology_stratified"].reset_index(drop=True)
sens = pd.read_csv(RESULTS / "06_layer_qap_sensitivity.csv")
sens = sens[sens.specification == "without_sovereign"].set_index(["r1", "r2"])
ci = pd.read_csv(RESULTS / "06_layer_bootstrap_intervals.csv").set_index(["r1", "r2"])
fig, ax = plt.subplots(figsize=(10.4, 7.7))
yy = np.arange(len(q))[::-1]
for y, row in zip(yy, q.itertuples()):
    ss = sens.loc[(row.r1, row.r2)]
    cc = ci.loc[(row.r1, row.r2)]
    ax.plot([cc.lower95, cc.upper95], [y + 0.12, y + 0.12], color="#215e75", lw=2, alpha=0.6)
    ax.scatter(row.observed, y + 0.12, c="#215e75", s=45)
    ax.scatter(
        ss.observed,
        y - 0.12,
        facecolors="#bc722f" if ss.q_two_sided < 0.05 else "white",
        edgecolors="#bc722f",
        s=45,
    )
ax.set_yticks(yy, [short[r.r1] + " / " + short[r.r2] for r in q.itertuples()])
ax.set_xlim(-0.03, 0.8)
ax.set_xlabel("Correlation of log occurrence counts on shared directed category pairs")
ax.set_title(
    "Some relation layers remain coupled without the sovereign category",
    loc="left",
    fontweight="bold",
    pad=18,
)
ax.axvline(0, color="#bbbbbb", lw=0.8)
ax.grid(axis="x", alpha=0.18)
ax.scatter([], [], c="#215e75", label="All categories; line = page bootstrap 95% interval")
ax.scatter([], [], c="#bc722f", label="Sovereign category removed")
ax.scatter([], [], facecolors="white", edgecolors="#bc722f", label="Removed-sovereign QAP q ≥ .05")
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.text(
    0.12,
    0.015,
    "15 prespecified comparisons. QAP shuffles category labels within ontology strata.\nCoupling describes the encoded relation layers; it is not evidence of a causal process.",
    fontsize=9,
    color="#444444",
)
fig.subplots_adjust(left=0.33, bottom=0.13, top=0.90)
finish(fig, "01_layer_coupling")

loads = pd.read_csv(RESULTS / "03_static_cp_loadings.csv")
selected = {
    "subject": ["E01", "E02", "E03", "E05", "E07", "E08", "E16", "E17", "E20", "E23"],
    "relation": [
        "R05",
        "R09",
        "R11",
        "R12",
        "R13",
        "R23",
        "R25",
        "R26",
        "R30",
        "R34",
        "R44",
        "R46",
        "R47",
        "R49",
        "R64",
    ],
    "object": [
        "E01",
        "E02",
        "E05",
        "E07",
        "E08",
        "E20",
        "E24",
        "E25",
        "E26",
        "E27",
        "E30",
        "E31",
        "E35",
        "E40",
    ],
}
lab = {
    "E01": "Sovereign",
    "E02": "Princes / heirs",
    "E03": "Royal women",
    "E05": "Ministers",
    "E07": "Commanders",
    "E08": "Armed units",
    "E16": "Frontier rulers",
    "E17": "Foreign powers",
    "E20": "Monks",
    "E23": "State institutions",
    "E24": "Territory / towns",
    "E25": "Palace / court",
    "E26": "Strategic space",
    "E27": "Sacred sites",
    "E30": "Title / office / rank",
    "E31": "Regalia / sacra",
    "E35": "Ritual / ceremony",
    "E40": "Policy / objective",
    **short,
    "R05": "Religious patronage",
    "R30": "Oath / submission",
    "R46": "Fortification",
    "R47": "Combat",
    "R49": "Military outcome",
    "R64": "Status attribution",
}
fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.3), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
for ax, (mode, codes) in zip(axes, selected.items()):
    t = (
        loads[loads["mode"] == mode]
        .pivot(index="code", columns="component", values="loading")
        .reindex(codes)
    )
    im = ax.imshow(t.values, vmin=0, vmax=0.73, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(codes)), [lab[c] for c in codes])
    ax.set_xticks(range(7), [f"C{k}" for k in range(1, 8)])
    ax.set_title(mode.capitalize(), loc="left", fontweight="bold")
    ax.set_xlabel("Latent component")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
fig.suptitle(
    "A joint model separates several repertoires of recorded power",
    x=0.02,
    ha="left",
    fontweight="bold",
    fontsize=15,
)
fig.subplots_adjust(left=0.10, right=0.94, bottom=0.18, top=0.89, wspace=0.70)
cax = fig.add_axes([0.957, 0.27, 0.012, 0.48])
fig.colorbar(im, cax=cax, label="Within-mode normalized loading")
fig.text(
    0.02,
    0.035,
    "C1 military action · C2 status-bearing · C3 court performance · C4 upward flows · C5 royal allocation · C6 sacred patronage · C7 governing operations\nComponent labels are interpretations. C3 is unstable across seeds/register exclusions. Grammatical direction contributes to the C2/C5 split.\nThe model has substantial residual error; seven components are not seven objectively established institutions.",
    fontsize=9,
    color="#444444",
)
finish(fig, "02_tensor_roles")

patterns = [
    "allegiance -> authorization",
    "information -> authorization",
    "authorization -> execution",
    "resistance -> coercion",
]
weak = pd.read_csv(RESULTS / "04_labelled_motif_tests.csv")
strong = pd.read_csv(RESULTS / "09_category_conditioned_chain_null.csv")
fig, ax = plt.subplots(figsize=(10.7, 5.0))
y = np.arange(4)[::-1]
for i, p in enumerate(patterns):
    a = weak[(weak.motif == "chain") & (weak.pattern == p)].iloc[0]
    b = strong[(strong.unit == "path_count") & (strong.pattern == p)].iloc[0]
    ax.barh(y[i] + 0.17, a.observed / a.null_mean, height=0.29, color="#215e75")
    ax.barh(y[i] - 0.17, b.observed / b.null_mean, height=0.29, color="#b9c6cc")
    ax.text(
        b.observed / b.null_mean + 0.08,
        y[i] - 0.17,
        f"q = {b.q_greater:.2f}",
        va="center",
        fontsize=9,
    )
ax.axvline(1, c="black", lw=1)
ax.set_yticks(y, [p.replace(" -> ", " → ").capitalize() for p in patterns])
ax.set_xlabel("Observed path count / null mean")
ax.set_title(
    "Apparent path enrichment does not survive the stricter null",
    loc="left",
    fontweight="bold",
    pad=17,
)
ax.legend(
    handles=[
        Patch(facecolor="#215e75", label="Shuffle relation families within page"),
        Patch(facecolor="#b9c6cc", label="Also preserve subject/object category pairs"),
    ],
    frameon=False,
    loc="lower right",
    fontsize=9,
)
ax.set_xlim(0, 6.2)
fig.subplots_adjust(left=0.32, bottom=0.19, top=0.85)
fig.text(
    0.04,
    0.03,
    "Arrows are encoded graph paths, not sequences of dated events.\nThese configurations do not independently establish an information-to-decision or resistance-to-punishment mechanism.",
    fontsize=9,
    color="#444444",
)
finish(fig, "03_motif_null_sensitivity")

sem = pd.read_csv(RESULTS / "03_semantic_structure_qap.csv")
pred = pd.read_csv(RESULTS / "08_link_prediction.csv")
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
a = sem[sem.null_model == "within_relation_domain"]
axes[0].bar(a["view"], a.observed, color=["#6c96a5", "#215e75", "#8cabb5"])
axes[0].set_ylim(0, 0.65)
axes[0].set_ylabel("Semantic / endpoint-structure correlation")
axes[0].set_title("Internal semantic consistency", loc="left", fontweight="bold")
p = pred[pred["filter"] == "unfiltered"]
axes[1].bar(
    ["DistMult", "ComplEx", "Frequency\nbaseline"], p.MRR, color=["#a7bdc6", "#6c96a5", "#bc722f"]
)
axes[1].set_ylim(0, 0.3)
axes[1].set_ylabel("Mean reciprocal rank")
axes[1].set_title("No predictive advantage here", loc="left", fontweight="bold")
fig.subplots_adjust(wspace=0.40, bottom=0.25, top=0.86)
fig.text(
    0.03,
    0.03,
    "Left: contextual vectors share source text with the extraction and are not independent historical validation.\nRight: object-category ranking on 1,210 held-out category facts. No predictions are added as historical facts.",
    fontsize=9,
    color="#444444",
)
finish(fig, "04_semantics_and_prediction")
print("Created four figures in PNG and SVG", flush=True)
