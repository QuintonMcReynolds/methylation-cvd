"""Figures + README tables from reports/*.json. Palette: dataviz reference slots (validated pairs), neutral grey for
baselines, and the leaky result drawn hatched so it can't be mistaken for a real result."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
SURFACE, INK, INK2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#8a8984"
TASK_C = {"generalized_vs_none": "#2a78d6", "any_sa_vs_none": "#eb6834"}
TASK_L = {"generalized_vs_none": "generalized vs none (n = 282)", "any_sa_vs_none": "any atherosclerosis vs none (n = 391)"}
CFG_L = {"age_sex_only": "age + sex only", "elastic_net_nested_adjusted": "elastic net (nested, age-adjusted)",
         "elastic_net_k1000_unadjusted": "elastic net (not age-adjusted)", "random_forest_k1000_adjusted": "random forest (age-adjusted)",
         "lightgbm_k1000_adjusted": "LightGBM (age-adjusted)", "LEAKY_selection_outside_cv": "LEAKY: CpGs picked before CV"}


def exp(t):
    return json.loads((REP / f"experiments_{t}.json").read_text())


def auc_figure():
    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURFACE)
    cfgs = list(CFG_L)
    h = 0.36
    for j, t in enumerate(TASK_C):
        e = exp(t)
        for i, c in enumerate(cfgs):
            y = i + (j - 0.5) * (h + 0.04)
            v = e[c]["auc_mean"]
            ax.barh(y, v - 0.5, left=0.5, height=h, color=TASK_C[t], hatch="///" if c.startswith("LEAKY") else None,
                    edgecolor=SURFACE if not c.startswith("LEAKY") else TASK_C[t], alpha=0.45 if c.startswith("LEAKY") else 1)
            ax.text(v + 0.004, y, f"{v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels([CFG_L[c] for c in cfgs])
    ax.invert_yaxis()
    ax.axvline(0.5, color=GREY, lw=1)
    ax.set_xlim(0.5, 0.95)
    ax.set_xlabel("cross-validated AUC (0.5 = chance)", color=INK2, fontsize=9)
    ax.set_title("Subclinical atherosclerosis from blood methylation\n", loc="left", fontsize=11, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TASK_C[t], label=TASK_L[t]) for t in TASK_C], frameon=False, fontsize=8.5, loc="lower left",
              bbox_to_anchor=(0, 1.0), ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "auc_by_model.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def incr_figure():
    fig, ax = plt.subplots(figsize=(7.4, 3.0), facecolor=SURFACE)
    rows = []
    for t in TASK_C:
        d = json.loads((REP / f"incremental_{t}.json").read_text())
        for m in ("lightgbm", "elastic_net"):
            rows.append((t, m, d[m]["increment_vs_age_sex"]))
    for i, (t, m, r) in enumerate(rows):
        ax.errorbar(r["diff"], i, xerr=[[r["diff"] - r["ci95"][0]], [r["ci95"][1] - r["diff"]]], fmt="o", color=TASK_C[t], ms=7,
                    capsize=3, lw=2)
    ax.axvline(0, color=GREY, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{'generalized' if t.startswith('gen') else 'any SA'} · {'LightGBM' if m == 'lightgbm' else 'elastic net'}" for t, m, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("AUC gained by adding methylation to age + sex (95 % bootstrap CI)", color=INK2, fontsize=9)
    ax.set_title("Does methylation add anything beyond age?", loc="left", fontsize=11, color=INK, pad=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "increment.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------- README tables
def t_models():
    rows = ["| model | generalized vs none (282) | any atherosclerosis vs none (391) |", "|---|---|---|"]
    a, b = exp("generalized_vs_none"), exp("any_sa_vs_none")
    for c, n in CFG_L.items():
        bold = "**" if c.startswith("LEAKY") else ""
        rows.append(f"| {bold}{n}{bold} | {a[c]['auc_mean']:.3f} ± {a[c].get('auc_sd', 0):.3f} | {b[c]['auc_mean']:.3f} ± {b[c].get('auc_sd', 0):.3f} |")
    pa, pb = a["permutation"], b["permutation"]
    rows.append("")
    rows.append(f"Label-permutation null (elastic net, 50 shuffles): mean AUC {pa['null_mean']:.2f} / {pb['null_mean']:.2f}, 95th percentile "
                f"{pa['null_95pct']:.2f} / {pb['null_95pct']:.2f}; observed beats every shuffle (p = {pa['p_value']:.2f}, the smallest possible with 50).")
    return "\n".join(rows)


def t_incr():
    rows = ["| target · model | methylation only (age-adjusted, no age/sex input) | age + sex | age + sex + methylation | gain [95 % CI] | p (one-sided) |",
            "|---|---|---|---|---|---|"]
    for t in TASK_C:
        d = json.loads((REP / f"incremental_{t}.json").read_text())
        for m, ml in (("lightgbm", "LightGBM"), ("elastic_net", "elastic net")):
            r = d[m]
            i = r["increment_vs_age_sex"]
            rows.append(f"| {'generalized' if t.startswith('gen') else 'any atherosclerosis'} · {ml} | {r['methylation_only_adjusted_auc']:.2f} | "
                        f"{d['age_sex_auc']:.2f} | {r['age_sex_plus_methylation_auc']:.2f} | **{i['diff']:+.3f}** [{i['ci95'][0]:+.3f}, {i['ci95'][1]:+.3f}] | {i['p_one_sided']:.3f} |")
    return "\n".join(rows)


def t_prod():
    log = json.loads((REP / "production_generalized_vs_none.json").read_text())
    rows = ["| step | what the monitor saw | action |", "|---|---|---|"]
    for e in log:
        if e["event"] == "register":
            rows.append(f"| register v{e['version']} as @champion | trained on {e['n_train']} samples | — |")
        elif e["event"] == "batch":
            seen = (f"MMD p = {e['batch_mmd_p']:.3f}; {e['model_cpgs_shifted_fdr05']} / {e['model_cpgs_total']} model CpGs shifted; "
                    f"{e['outlier_samples']} / {e['n']} outlier samples")
            act = e.get("action", "released")
            if "after_renormalisation" in e:
                a = e["after_renormalisation"]
                act += f" → after: MMD p = {a['batch_mmd_p']:.2f}, {a['model_cpgs_shifted_fdr05']} CpGs shifted, released"
            rows.append(f"| batch {e['batch']}{' (**injected reagent-lot artefact**)' if e['injected_artifact'] else ''} | {seen} | {act} |")
        elif e["event"] == "challenger":
            rows.append(f"| challenger v{e['version']} (launch data + {e['n_train'] - log[0]['n_train']} feedback labels) | AUC on batch 4: champion "
                        f"{e['auc_champion']:.2f}, challenger {e['auc_challenger']:.2f}; difference {e['auc_diff']:+.2f} [{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}] | "
                        f"{'promoted' if e['promoted'] else '**not promoted**: 28 labelled samples cannot show it is not worse'} |")
    cal = json.loads((REP / "drift_calibration.json").read_text())
    rows.append("")
    rows.append(f"Drift calibration ({cal['clean']['n']} clean and {cal['artifact']['n']} artefact batches of 28 held-out samples): hold rate "
                f"{100 * cal['clean']['hold_rate']:.1f} % on clean batches, {100 * cal['artifact']['hold_rate']:.0f} % on artefact batches.")
    return "\n".join(rows)


GEN = {"models": t_models, "incremental": t_incr, "production": t_prod}

if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    auc_figure()
    incr_figure()
    p = ROOT / "README.md"
    if p.exists():
        s = p.read_text()
        for name, fn in GEN.items():
            s = re.sub(rf"(<!-- gen:{name} -->\n).*?(<!-- /gen:{name} -->)", lambda m, fn=fn: m.group(1) + fn() + "\n" + m.group(2), s, flags=re.S)
        p.write_text(s)
    print(sorted(x.name for x in FIG.glob("*.png")))
