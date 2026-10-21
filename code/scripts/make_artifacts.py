"""Build LaTeX tables, number macros and figures for the chapter from the results csv files."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from faithaal.data import load_recordings  # noqa: E402
from faithaal.ladder import LEVELS, authority  # noqa: E402

res, tab, fig = ROOT / "results", ROOT / "chapter" / "tables", ROOT / "chapter" / "figures"
ref = pd.read_csv(res / "reference.csv")
llm = pd.read_csv(res / "summary.csv") if (res / "summary.csv").exists() else None
recs = load_recordings(ROOT / "data" / "adl_noerror")

def ci(r, k, d=2):
    return f"{r[k]:.{d}f} [{r[k + '_lo']:.{d}f}, {r[k + '_hi']:.{d}f}]"

# ---- number macros
n_ev = pd.Series([len(r.events) for r in recs])
nums = {
    "NRec": len(recs), "NPart": len({r.participant for r in recs}), "NEvMin": int(n_ev.min()), "NEvMed": int(n_ev.median()),
    "NEvMax": int(n_ev.max()),
}
fr = ref.set_index("model")
for key, name in (("Faith", "reference-faithful"), ("Pos", "reference-positional"), ("Rand", "reference-random")):
    r = fr.loc[name]
    nums[f"Acc{key}"] = f"{100 * r.accuracy:.1f}"
    nums[f"F{key}"] = f"{r.F:.3f}"
    nums[f"F{key}Lo"] = f"{r.F_lo:.3f}"
    nums[f"F{key}Hi"] = f"{r.F_hi:.3f}"
    nums[f"S{key}"] = f"{r.S:.2f}"
    nums[f"Rem{key}"] = f"{r.delta_remove:.3f}"
    nums[f"Val{key}"] = f"{r.delta_value:.3f}"
    nums[f"Shift{key}"] = f"{r.delta_shift:.3f}"
with open(tab / "numbers.tex", "w") as fh:
    for k, v in nums.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- table: reference agents
names = {"reference-faithful": "Faithful (cites top contributions)", "reference-positional": "Positional (cites first events)",
         "reference-random": "Random (cites random events)"}
with open(tab / "reference.tex", "w") as fh:
    fh.write("\\begin{tabular}{lcccc}\n\\toprule\nCiting rule & Accuracy & $F$ [95\\% CI] & $S$ & Noise floor \\\\\n\\midrule\n")
    for _, r in ref.iterrows():
        fh.write(f"{names[r.model]} & {100 * r.accuracy:.1f}\\% & {ci(r, 'F', 3)} & {r.S:.2f} & {r.noise:.2f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")

# ---- figure: cited vs uncited flip rates per family
fams = ["remove", "value", "shift"]
fig_, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, (_, r) in zip(axes, ref.iterrows()):
    x = range(len(fams))
    ax.bar([i - 0.2 for i in x], [r[f"cited_{f}"] for f in fams], 0.4, label="cited", color="#1b6ca8")
    ax.bar([i + 0.2 for i in x], [r[f"uncited_{f}"] for f in fams], 0.4, label="uncited (matched)", color="#bdbdbd")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fams, fontsize=8)
    ax.set_title(names[r.model].split(" (")[0], fontsize=9)
axes[0].set_ylabel("conclusion-change rate")
axes[0].legend(fontsize=7, frameon=False)
fig_.tight_layout()
fig_.savefig(fig / "reference_flip_rates.pdf")
plt.close(fig_)

# ---- LLM table, ladder table and scatter (only when LLM results exist)
if llm is not None:
    f_ref = fr.loc["reference-faithful", "F"]
    with open(tab / "llm.tex", "w") as fh:
        fh.write("\\begin{tabular}{llccccc}\n\\toprule\nModel & Prompt & Accuracy [95\\% CI] & $F$ [95\\% CI] & $S$ & Noise & Level \\\\\n\\midrule\n")
        for _, r in llm.iterrows():
            lv = authority(r.accuracy_lo, r.F_lo, r.noise, r.parse_rate, f_ref)
            fh.write(f"{r.model.split('/')[-1]} & {r.variant} & {ci(r, 'accuracy')} & {ci(r, 'F')} & {r.S:.2f} & {r.noise:.2f} & {lv} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    fig_, ax = plt.subplots(figsize=(3.6, 3.0))
    for v, c in (("plain", "#bdbdbd"), ("enriched", "#1b6ca8")):
        d = llm[llm.variant == v]
        ax.scatter(d.accuracy, d.F, label=v, color=c, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("recognition accuracy")
    ax.set_ylabel("faithfulness $F$")
    ax.legend(frameon=False, fontsize=8)
    fig_.tight_layout()
    fig_.savefig(fig / "accuracy_vs_faithfulness.pdf")
print("artifacts written; llm results present:", llm is not None)
