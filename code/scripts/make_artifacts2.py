"""Tables, number macros and a figure for the free-living and explain-only experiments."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from faithaal.ladder import authority  # noqa: E402

res, tab, fig = ROOT / "results", ROOT / "chapter" / "tables", ROOT / "chapter" / "figures"
NAME = {"openrouter:openai/gpt-6-luna": "gpt-6-luna", "openrouter:deepseek/deepseek-v4.1-flash-1017": "deepseek-v4.1-flash",
        "rule-faithful": "rule: faithful", "rule-positional": "rule: positional", "rule-random": "rule: random"}


def ci(r, k, d=2):
    return f"{r[k]:.{d}f} [{r[k + '_lo']:.{d}f}, {r[k + '_hi']:.{d}f}]"


acc = pd.read_csv(res / "decision_accuracy.csv").set_index("task")
macros: dict[str, str] = {}

# ---- free-living protocol results (language model decides)
h = pd.read_csv(res / "home_summary.csv")
f_ref_home = pd.read_csv(res / "reference_home.csv").set_index("model").loc["reference-faithful", "F"]
with open(tab / "home_llm.tex", "w") as fh:
    fh.write("\\begin{tabular}{llcccccc}\n\\toprule\nAgent & Prompt & Accuracy [95\\% CI] & $F$ [95\\% CI] & $S$ & Noise & Level \\\\\n\\midrule\n")
    a = acc.loc["home"]
    fh.write(f"logistic regression & -- & {a.accuracy:.2f} [{a.lo:.2f}, {a.hi:.2f}] & -- & -- & -- & -- \\\\\n")
    for _, r in h.sort_values(["model", "variant"]).iterrows():
        lv = authority(r.accuracy_lo, r.F_lo, r.noise, r.parse_rate, f_ref_home)
        fh.write(f"{NAME[r.model]} & {r.variant} & {ci(r, 'accuracy')} & {ci(r, 'F')} & {r.S:.2f} & {r.noise:.2f} & {lv} \\\\\n")
        key = ("HLuna" if "luna" in r.model else "HDeep") + ("Enr" if r.variant == "enriched" else "Pl")
        macros[f"{key}Acc"] = f"{100 * r.accuracy:.1f}"
        macros[f"{key}F"] = f"{r.F:.3f}"
        macros[f"{key}FLo"], macros[f"{key}FHi"] = f"{r.F_lo:.3f}", f"{r.F_hi:.3f}"
    fh.write("\\bottomrule\n\\end{tabular}\n")
macros["HomeRefAcc"] = f"{100 * acc.loc['home'].accuracy:.1f}"
macros["HomeRefAccLo"], macros["HomeRefAccHi"] = f"{100 * acc.loc['home'].lo:.1f}", f"{100 * acc.loc['home'].hi:.1f}"
macros["HomeRefF"] = f"{f_ref_home:.3f}"
macros["HomeMaxAccLo"] = f"{h.accuracy_lo.max():.2f}"

# ---- explain-only tables and ladder levels
panels = {}
with open(tab / "explain.tex", "w") as fh:
    fh.write("\\begin{tabular}{lllcccc}\n\\toprule\nExplainer & Prompt & Gap [95\\% CI] & Precision & Chance & Level A & Level B \\\\\n\\midrule\n")
    for task, title in (("adl", "Scripted ADL (decision accuracy {:.1f}\\%)"), ("home", "Free-living segments (decision accuracy {:.1f}\\%)")):
        s = pd.read_csv(res / f"explain_summary_{task}.csv")
        panels[task] = s
        fh.write(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title.format(100 * acc.loc[task].accuracy)}}}}} \\\\\n")
        s["order"] = s.explainer.str.startswith("rule").map({True: 0, False: 1})
        for _, r in s.sort_values(["order", "explainer", "variant"]).iterrows():
            v = "--" if r.variant == "-" else r.variant
            fh.write(f"{NAME[r.explainer]} & {v} & {ci(r, 'gap', 3)} & {r.precision:.2f} & {r.precision_chance:.2f} & "
                     f"{int(r.level_policy_A)} & {int(r.level_policy_B)} \\\\\n")
        fh.write("\\midrule\n" if task == "adl" else "")
    fh.write("\\bottomrule\n\\end{tabular}\n")
for task, s in panels.items():
    t = "A" if task == "adl" else "H"
    for _, r in s.iterrows():
        if r.explainer.startswith("rule"):
            key = f"X{t}{r.explainer.split('-')[1].capitalize()}"
        else:
            key = f"X{t}" + ("Luna" if "luna" in r.explainer else "Deep") + ("Enr" if r.variant == "enriched" else "Pl")
        macros[f"{key}Gap"] = f"{r.gap:.3f}"
        macros[f"{key}GapLo"], macros[f"{key}GapHi"] = f"{r.gap_lo:.3f}", f"{r.gap_hi:.3f}"
        macros[f"{key}Prec"] = f"{r.precision:.2f}"
        macros[f"{key}Chance"] = f"{r.precision_chance:.2f}"
macros["DecAccAdl"] = f"{100 * acc.loc['adl'].accuracy:.1f}"
macros["DecAccAdlLo"] = f"{100 * acc.loc['adl'].lo:.1f}"
with open(tab / "extra_numbers.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- figure: explainer faithfulness gap with intervals
fig_, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
for ax, (task, s) in zip(axes, panels.items()):
    s = s.copy()
    s["label"] = [NAME[e] + ("" if v == "-" else f"\n{v}") for e, v in zip(s.explainer, s.variant)]
    s["rule"] = s.explainer.str.startswith("rule")
    s = s.sort_values(["rule", "gap"], ascending=[False, False])
    colors = ["#1b6ca8" if not r else "#9e9e9e" for r in s.rule]
    ax.barh(range(len(s)), s.gap, xerr=[s.gap - s.gap_lo, s.gap_hi - s.gap], color=colors, capsize=2)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.label, fontsize=6)
    ax.invert_yaxis()
    ax.axvline(0, color="k", linewidth=0.6)
    ax.set_title("Scripted ADL" if task == "adl" else "Free-living segments", fontsize=9)
    ax.set_xlabel("faithfulness gap", fontsize=8)
fig_.tight_layout()
fig_.savefig(fig / "explain_gap.pdf")
print("written", len(macros), "macros")
