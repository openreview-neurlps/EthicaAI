"""
Publication-Quality Figures for EthicaAI NeurIPS 2026 Paper
============================================================
Generates:
  1. Floor Spectrum Heatmap (fig_floor_spectrum.pdf)
  2. Byzantine Sensitivity line plot (fig_byzantine.pdf)

Usage:
    python generate_pub_figures.py
"""

import json
import os
import sys
import numpy as np

# ---------- matplotlib backend & imports ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    print("WARNING: seaborn not installed; falling back to plain matplotlib.")

# ---------- paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, os.pardir, os.pardir)  # NeurIPS2026_final_submission
FIG_DIR = os.path.normpath(os.path.join(BASE_DIR, "paper", "figures"))
os.makedirs(FIG_DIR, exist_ok=True)

SPECTRUM_JSON = os.path.normpath(
    os.path.join(BASE_DIR, "code", "outputs", "mild_tpsd_spectrum",
                 "mild_tpsd_spectrum_results.json"))
BYZANTINE_JSON = os.path.normpath(
    os.path.join(BASE_DIR, "code", "outputs", "byzantine_sensitivity",
                 "byzantine_sensitivity_results.json"))

# ---------- style ----------
DPI = 300
COL_WIDTH = 3.25     # NeurIPS single-column width (inches)
TEXT_WIDTH = 6.75     # NeurIPS full text width (inches)

# Try to enable LaTeX rendering; fall back gracefully
try:
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    # Quick smoke-test: render a tiny string
    fig_test = plt.figure(figsize=(0.1, 0.1))
    fig_test.text(0.5, 0.5, r"$\phi$")
    fig_test.savefig(os.devnull if os.name != "nt" else "NUL",
                     format="png")
    plt.close(fig_test)
    LATEX_OK = True
except Exception:
    matplotlib.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
    })
    LATEX_OK = False

if HAS_SNS:
    sns.set_context("paper", font_scale=0.9)
    sns.set_style("whitegrid", {
        "axes.edgecolor": "0.15",
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
    })

# Colorblind-friendly palette (Okabe-Ito inspired)
CB_COLORS = {
    "Selfish":  "#E69F00",   # orange
    "IPPO":     "#56B4E9",   # sky blue
    "Floor":    "#009E73",   # bluish green
    "MACCL":    "#CC79A7",   # reddish purple
}

CB_MARKERS = {
    "Selfish": "v",
    "IPPO":    "s",
    "Floor":   "D",
    "MACCL":   "o",
}

# ---------- helpers ----------
def _label(s: str) -> str:
    """LaTeX-safe label."""
    if LATEX_OK:
        mapping = {
            "phi1": r"$\varphi_1$",
            "f_crit": r"$f_{\mathrm{crit}}$",
            "beta": r"$\beta$",
            "survival": r"Survival Rate",
        }
        return mapping.get(s, s)
    mapping = {
        "phi1": "\u03c6\u2081",
        "f_crit": "f_crit",
        "beta": "\u03b2",
        "survival": "Survival Rate",
    }
    return mapping.get(s, s)


# ================================================================
# Figure 1 -- Floor Spectrum Heatmap
# ================================================================
def fig_floor_spectrum():
    print("Generating Floor Spectrum Heatmap ...")
    with open(SPECTRUM_JSON) as f:
        data = json.load(f)

    cfg = data["config"]
    phi1_vals = np.array(cfg["phi1_values"])
    fcrit_vals = np.array(cfg["f_crit_values"])

    # Build survival matrix: rows = f_crit (bottom-to-top), cols = phi1
    Z = np.zeros((len(fcrit_vals), len(phi1_vals)))
    theorem2_boundary = {}  # f_crit -> theorem2_phi1

    for fkey, fdata in data["results"].items():
        fc = fdata["f_crit"]
        fi = list(fcrit_vals).index(fc)
        theorem2_boundary[fc] = fdata["theorem2_phi1"]
        for pkey, pdata in fdata["sweeps"].items():
            p1 = pdata["phi1"]
            pi = list(phi1_vals).index(p1)
            Z[fi, pi] = pdata["survival_mean"]

    # Custom red-to-green colormap
    cmap = LinearSegmentedColormap.from_list(
        "surv", ["#d73027", "#fee08b", "#1a9850"], N=256)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.78))

    im = ax.imshow(
        Z, origin="lower", aspect="auto", cmap=cmap,
        vmin=0.0, vmax=1.0,
        extent=[
            phi1_vals[0] - 0.05, phi1_vals[-1] + 0.05,
            -0.5, len(fcrit_vals) - 0.5,
        ],
    )

    # Y ticks = f_crit values
    ax.set_yticks(range(len(fcrit_vals)))
    ax.set_yticklabels([f"{v:.2f}" for v in fcrit_vals])

    # X ticks = phi1 values
    ax.set_xticks(phi1_vals)
    ax.set_xticklabels([f"{v:.1f}" for v in phi1_vals], fontsize=7)

    ax.set_xlabel(_label("phi1"), fontsize=9)
    ax.set_ylabel(_label("f_crit"), fontsize=9)

    # Annotate cells with survival %
    for fi, fc in enumerate(fcrit_vals):
        for pi, p1 in enumerate(phi1_vals):
            val = Z[fi, pi]
            txt_color = "white" if val < 0.45 else "black"
            ax.text(p1, fi, f"{val*100:.0f}",
                    ha="center", va="center", fontsize=6.5,
                    fontweight="bold", color=txt_color)

    # Overlay Theorem 2 boundary as a line
    th2_phis = []
    th2_yidx = []
    for fi, fc in enumerate(fcrit_vals):
        if fc in theorem2_boundary:
            th2_phis.append(theorem2_boundary[fc])
            th2_yidx.append(fi)

    ax.plot(th2_phis, th2_yidx, color="black", linewidth=1.8,
            linestyle="--", marker="x", markersize=5, markeredgewidth=1.5,
            label="Thm 2 boundary" if not LATEX_OK else r"Thm\,2 boundary",
            zorder=5)

    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.85,
              edgecolor="0.4")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(_label("survival"), fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.4)
    outpath = os.path.join(FIG_DIR, "fig_floor_spectrum.pdf")
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ================================================================
# Figure 2 -- Byzantine Sensitivity
# ================================================================
def fig_byzantine():
    print("Generating Byzantine Sensitivity plot ...")
    with open(BYZANTINE_JSON) as f:
        data = json.load(f)

    beta_vals = np.array(data["config"]["beta_sweep"]) * 100  # to %

    # Map JSON method names -> plot labels
    name_map = {
        "Selfish REINFORCE": "Selfish",
        "IPPO":              "IPPO",
        "Commitment Floor":  "Floor",
        "MACCL":             "MACCL",
    }

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.72))

    for json_name, label in name_map.items():
        mdata = data["results"][json_name]
        means = []
        stds = []
        for b in data["config"]["beta_sweep"]:
            key = f"beta_{b:.2f}"
            means.append(mdata[key]["survival_mean"])
            stds.append(mdata[key]["survival_std"])
        means = np.array(means)
        stds = np.array(stds)

        color = CB_COLORS[label]
        marker = CB_MARKERS[label]

        ax.plot(beta_vals, means, color=color, marker=marker,
                markersize=5, linewidth=1.6, label=label, zorder=3)
        ax.fill_between(beta_vals, means - stds, np.minimum(means + stds, 1.0),
                        color=color, alpha=0.15, zorder=2)

    ax.set_xlabel(r"Byzantine fraction $\beta$ (\%)" if LATEX_OK
                  else "Byzantine fraction \u03b2 (%)", fontsize=9)
    ax.set_ylabel(_label("survival"), fontsize=9)
    ax.set_xlim(-2, 52)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks(beta_vals)
    ax.set_xticklabels([f"{int(b)}" for b in beta_vals], fontsize=7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.tick_params(axis="y", labelsize=7)

    ax.legend(fontsize=7, loc="center right", framealpha=0.85,
              edgecolor="0.4")
    if HAS_SNS:
        sns.despine(ax=ax, top=True, right=True)

    fig.tight_layout(pad=0.4)
    outpath = os.path.join(FIG_DIR, "fig_byzantine.pdf")
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return outpath


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print(f"LaTeX rendering: {'enabled' if LATEX_OK else 'disabled (fallback)'}")
    print(f"Seaborn: {'yes' if HAS_SNS else 'no'}")
    print(f"Output directory: {FIG_DIR}\n")

    fig_floor_spectrum()
    fig_byzantine()

    print("\nDone. All figures saved.")
