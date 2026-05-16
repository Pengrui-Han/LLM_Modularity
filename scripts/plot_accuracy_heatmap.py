"""Annotated horizontal accuracy heatmap (Nature / PNAS style) for both-correct accuracy.

Reads docs/accuracy_table_both_correct.csv and renders a clean
6 models (rows) x 45 tasks (cols) heatmap with cell-value annotations,
a top domain-block strip (coloured bands with domain names), thin within-domain
separators, model names on the left, task names rotated 90° below, and a
single right-side colour bar.

Outputs:
  results/figures/accuracy_heatmap_both_correct.png
  results/figures/accuracy_heatmap_both_correct.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV  = os.path.join(ROOT, "docs/accuracy_table_both_correct.csv")
OUT_PNG = os.path.join(ROOT, "results/figures/accuracy_heatmap_both_correct.png")
OUT_PDF = os.path.join(ROOT, "results/figures/accuracy_heatmap_both_correct.pdf")

MODELS = ["Mistral-Small-24B", "Qwen2.5-32B", "OLMo-2-32B",
          "Llama-3.1-70B", "Qwen2.5-72B", "Mistral-Large-123B"]
DOMAIN_ORDER  = ["Language", "Formal", "Physics", "Social"]
DOMAIN_COLORS = {"Language": "#C0392B", "Formal": "#2471A3",
                 "Physics":  "#E67E22", "Social":  "#27AE60"}

# Canonical short task names — kept in sync with scripts/run_overlap_chord.R
TASK_SHORT_NAMES = {
    # Language (8)
    "anaphor_gender_agreement": "Anaphor",
    "det_noun_agreement_irregular": "DetN-Irr",
    "det_noun_agreement_regular": "DetN-Reg",
    "det_noun_agreement_with_adjective": "DetN-Adj",
    "hypernymy": "Hyper",
    "npi": "NPI",
    "subject_verb_agreement": "S-V",
    "wug": "Wug",
    # Formal — Arithmetic
    "add_sub_2op_symbolic": "Add2-Sym",
    "add_sub_2op_verbal": "Add2-Vrb",
    "add_sub_3op_symbolic": "Add3-Sym",
    "add_sub_3op_verbal": "Add3-Vrb",
    "mul_div_2op_symbolic": "Mul2-Sym",
    "mul_div_2op_verbal": "Mul2-Vrb",
    "mul_div_3op_symbolic": "Mul3-Sym",
    "mul_div_3op_verbal": "Mul3-Vrb",
    # Formal — Logic
    "logic_propositional_1": "PropL-NL",
    "logic_propositional_symbolic": "PropL-Sym",
    "logic_syllogism_1": "Syll-NL",
    "logic_syllogism_symbolic": "Syll-Sym",
    # Formal — Code
    "code_A": "CodeA",
    "code_B": "CodeB",
    "code_conditional": "CodeCond",
    "code_list": "CodeList",
    "code_loop": "CodeLoop",
    # Formal — Algorithmic
    "simple_equation": "Eq",
    "number_sequence": "NumSeq",
    "number_sorting": "NumSort",
    # Physics (9)
    "phys_newton": "Newton",
    "phys_prost": "PROST",
    "physics_brightness": "Brightness",
    "physics_buoyancy": "Buoyancy",
    "physics_elasticity": "Elasticity",
    "physics_solubility": "Solubility",
    "physics_speed": "Speed",
    "physics_stability": "Stability",
    "physics_temperature": "Temperature",
    # Social (9)
    "agent": "Agent",
    "desires_goals": "Desires",
    "emotion_fewshot": "EmotionFS",
    "norm_appropriate": "NormApp",
    "norm_moral": "NormMoral",
    "primary_emotions": "PrimEmo",
    "secondary_emotions": "SecEmo",
    "social_interactions": "SocInt",
    "social_relations": "SocRel",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---------- load + sort ------------------------------------------------------
df = pd.read_csv(CSV)
df["__d"] = df["Domain"].map({d: i for i, d in enumerate(DOMAIN_ORDER)})
df = df.sort_values(["__d", "Task"]).reset_index(drop=True)

n_tasks = len(df)
n_mods  = len(MODELS)
data    = df[MODELS].values.T.astype(float)   # rows=models, cols=tasks

# domain block boundaries on x-axis (column indices where domain changes)
domain_at_col = df["Domain"].values
boundaries = []
for i in range(1, n_tasks):
    if domain_at_col[i] != domain_at_col[i - 1]:
        boundaries.append(i - 0.5)

# domain block extents: (domain, x_start, x_end)
blocks = []
start = 0
for i in range(1, n_tasks + 1):
    if i == n_tasks or domain_at_col[i] != domain_at_col[i - 1]:
        blocks.append((domain_at_col[i - 1], start, i - 1))
        start = i

# ---------- figure -----------------------------------------------------------
fig = plt.figure(figsize=(14.5, 5.0))
gs = gridspec.GridSpec(
    2, 3,
    width_ratios=[1.55, 12.0, 0.30],   # model labels | main | colorbar
    height_ratios=[0.55, 4.5],          # domain strip | main+labels
    wspace=0.02, hspace=0.04,
)
ax_strip = fig.add_subplot(gs[0, 1])     # top domain strip (over main only)
ax_lab   = fig.add_subplot(gs[1, 0])     # left model labels
ax_main  = fig.add_subplot(gs[1, 1])     # heatmap + bottom task labels
ax_cb    = fig.add_subplot(gs[1, 2])     # colorbar

# Top domain strip: coloured rectangles + horizontal domain text
ax_strip.set_xlim(-0.5, n_tasks - 0.5)
ax_strip.set_ylim(0, 1)
for d, s, e in blocks:
    ax_strip.add_patch(Rectangle((s - 0.5, 0.05), e - s + 1, 0.55,
                                  facecolor=DOMAIN_COLORS[d], edgecolor="white",
                                  linewidth=1.0))
    ax_strip.text((s + e) / 2, 0.78, d, ha="center", va="center",
                  fontsize=11, fontweight="bold", color=DOMAIN_COLORS[d])
ax_strip.set_xticks([]); ax_strip.set_yticks([])
for s in ("top", "bottom", "left", "right"):
    ax_strip.spines[s].set_visible(False)

# Model labels on the left
ax_lab.set_xlim(0, 1); ax_lab.set_ylim(n_mods - 0.5, -0.5)
for i, m in enumerate(MODELS):
    ax_lab.text(0.98, i, m, ha="right", va="center", fontsize=10)
ax_lab.set_xticks([]); ax_lab.set_yticks([])
for s in ("top", "bottom", "left", "right"):
    ax_lab.spines[s].set_visible(False)

# Main heatmap
vmin, vmax = 50, 100
im = ax_main.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax,
                    aspect="auto", interpolation="nearest")

# Cell annotations
for i in range(n_mods):
    for j in range(n_tasks):
        v = data[i, j]
        if not np.isfinite(v):
            continue
        rel = max(0.0, min(1.0, (v - vmin) / max(vmax - vmin, 1)))
        text_color = "white" if rel < 0.55 else "black"
        ax_main.text(j, i, f"{v:.0f}", ha="center", va="center",
                     color=text_color, fontsize=7.5)

# Vertical separators between domain blocks
for b in boundaries:
    ax_main.axvline(b, color="white", linewidth=2.0, zorder=5)

# Bottom: task labels rotated 90 degrees
ax_main.set_xticks(np.arange(n_tasks))
task_labels = [TASK_SHORT_NAMES.get(t, t) for t in df["Task"].values]
missing = [t for t in df["Task"].values if t not in TASK_SHORT_NAMES]
if missing:
    print(f"WARN: no short name for {len(missing)} task(s): {missing}")
ax_main.set_xticklabels(task_labels, rotation=90, ha="center",
                        fontsize=7.5)
ax_main.set_yticks([])
for s in ("top", "right"):
    ax_main.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax_main.spines[s].set_linewidth(1.0)
ax_main.tick_params(axis="x", which="major", direction="out", length=2.5,
                    width=0.8, pad=2)

# Align strip xlim with main heatmap (gridspec handles it but be explicit)
ax_strip.set_xlim(ax_main.get_xlim())

# Colour bar
cbar = fig.colorbar(im, cax=ax_cb)
cbar.set_label("Both-correct accuracy (%)", fontsize=10)
cbar.ax.tick_params(labelsize=9, length=3, width=0.8)
cbar.outline.set_linewidth(0.8)

fig.suptitle("Per-task both-correct accuracy (the set used for AP)",
             fontsize=12, y=1.005)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF,           bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PNG}\nSaved: {OUT_PDF}")
