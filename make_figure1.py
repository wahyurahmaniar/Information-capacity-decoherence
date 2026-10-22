#!/usr/bin/env python3
"""Figure 1: conceptual schematic of Assumptions 2-3 (ICD).
Revised per referee comment 24: gray region labelled CONJECTURED capacity-limited regime;
diagonalization marked POSTULATED, not shown as established dynamics. No '--' in any label."""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
mpl.rcParams.update({"text.usetex": False, "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False, "savefig.dpi": 300})

BLUE, ORANGE, GRAY, RED = "#5580a3", "#dca262", "#9a9a9a", "#b5293a"
ORTEXT, GRTEXT = "#b07b32", "#6f6f6f"

fig = plt.figure(figsize=(11.0, 7.4))
gs = GridSpec(2, 1, height_ratios=[2.6, 1.4], hspace=0.80,
              left=0.055, right=0.93, top=0.88, bottom=0.085)
axT = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])

def ypos(level): return np.linspace(0.07, 0.93, 2 ** level)
blue_nodes = {L: ypos(L) for L in range(4)}
for L in range(3):
    ys, yc = blue_nodes[L], blue_nodes[L + 1]
    for i, yp in enumerate(ys):
        for child in (yc[2*i], yc[2*i+1]):
            axT.plot([L, L+1], [yp, child], color=BLUE, lw=1.6, zorder=2)
for L in range(4):
    axT.plot(np.full(2**L, L), blue_nodes[L], "o", color=BLUE, ms=6.5, zorder=3)
axT.plot([0], [0.5], "o", color="black", ms=12, zorder=4)
leaves, orange_y = blue_nodes[3], ypos(4)
for i, yp in enumerate(leaves):
    for child in (orange_y[2*i], orange_y[2*i+1]):
        axT.plot([3, 4], [yp, child], color=ORANGE, lw=1.5, zorder=2)
axT.plot(np.full(16, 4), orange_y, "o", color=ORANGE, ms=5.5, zorder=3)
for yp in orange_y:
    axT.plot([4, 6], [yp, yp], color=GRAY, lw=1.1, alpha=0.7, zorder=1)
for xx in (5, 6):
    axT.plot(np.full(16, xx), orange_y, "x", color=GRAY, ms=5.0, alpha=0.85,
             zorder=2, markeredgewidth=1.2)
axT.annotate("", xy=(6.55, -0.02), xytext=(-0.15, -0.02),
             arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.2), annotation_clip=False)
for t in range(1, 7):
    axT.axvline(t, color="0.7", ls=(0, (4, 4)), lw=0.8, ymax=0.97, zorder=0)
    axT.text(t, -0.085, rf"$t={t}\,\tau_{{\mathrm{{ML}}}}$", ha="center", va="top",
             color="0.35", fontsize=11)
axT.text(3.0, -0.21, r"time   $(\tau_{\mathrm{ML}}\ \mathrm{intervals})$", ha="center",
         va="top", color="0.35", fontsize=13, style="italic")
axT.text(-0.20, 0.5, r"$|\psi(0)\rangle$", ha="right", va="center", fontsize=16)
axT.text(0.55, 1.10, "coherent branching", color=BLUE, fontsize=12.5, ha="center",
         style="italic", fontweight="bold")
axT.text(0.55, 1.02, r"$\log_2 b$ bits per $\tau_{\mathrm{ML}}$", color=BLUE, fontsize=11, ha="center")
axT.text(3.55, 1.10, "capacity saturating", color=ORTEXT, fontsize=12.5, ha="center",
         style="italic", fontweight="bold")
axT.text(5.30, 1.135, "conjectured", color=GRTEXT, fontsize=12, ha="center",
         style="italic", fontweight="bold")
axT.text(5.30, 1.065, "capacity-limited regime", color=GRTEXT, fontsize=12, ha="center",
         style="italic", fontweight="bold")
axT.text(5.30, 0.99, r"(postulated) $|\psi\rangle\!\to\!\rho_{\mathrm{diag}}$",
         color=GRTEXT, fontsize=10.5, ha="center")
axT.set_xlim(-0.7, 6.7); axT.set_ylim(-0.03, 1.20); axT.axis("off")
axT.text(3.0, -0.36, "(a)  Branching tree of coherent histories",
         ha="center", va="top", fontsize=12)

IAV, IBEK = 0.72, 1.0          # available capacity and the full Bekenstein capacity
STEP = IAV / 4.0               # branching information added per tau_ML

axB.add_patch(plt.Rectangle((0.5, IAV), 6.55, IBEK - IAV, facecolor="#cfd8e0",
                            alpha=0.9, edgecolor="none", zorder=1))
axB.text(0.60, (IAV + IBEK) / 2.0, r"$I_{\mathrm{past}}+I_{\mathrm{present}}$",
         ha="left", va="center", fontsize=10.5, color="#44586a")
axB.add_patch(plt.Rectangle((4.46, 0.0), 2.6, IBEK, facecolor=GRAY, alpha=0.32,
                            edgecolor="none", zorder=2))

heights, cols = [STEP, 2*STEP, 3*STEP, 4*STEP], [BLUE, BLUE, ORANGE, ORANGE]
for t, (h, c) in enumerate(zip(heights, cols), start=1):
    axB.bar(t, h, width=0.92, color=c, edgecolor="white", lw=0.8, zorder=3)
for t in (5, 6):
    axB.bar(t, t * STEP, width=0.92, facecolor="none", edgecolor=ORANGE, lw=1.3,
            ls=(0, (4, 2.4)), zorder=3)
axB.plot([4.46, 6.46], [4.46 * STEP, 6.46 * STEP], color=ORTEXT, lw=1.5,
         ls=(0, (5, 3)), zorder=6)
axB.text(6.56, 6.46 * STEP, "branching\ncontinues", ha="left",
         va="center", fontsize=10.0, color=ORTEXT, style="italic", linespacing=1.15)

axB.add_patch(plt.Rectangle((0.5, 0.0), 6.55, IBEK, fill=False, edgecolor="black",
                            lw=1.6, zorder=4))
axB.plot([0.5, 7.05], [IAV, IAV], color=RED, lw=1.3, ls=(0, (2, 2)), zorder=5)
axB.axvline(4.46, color=RED, ls=(0, (5, 4)), lw=1.8, zorder=5)
axB.text(4.56, 1.03, r"$t=\tau_{\mathrm{cap}}$", ha="left", va="bottom",
         fontsize=11.5, color=RED, style="italic")
axB.text(0.55, 1.12, r"accumulated branching information $I_{\mathrm{fut}}(t)$",
         ha="left", va="bottom", fontsize=12.0)
axB.text(7.16, IAV, r"$I_{\mathrm{avail}}$", ha="left", va="center", fontsize=13, color=RED)
axB.text(7.16, IBEK + 0.02, r"$I_{\mathrm{Bek}}$", ha="left", va="bottom", fontsize=14)
for t in range(1, 7):
    axB.text(t, -0.06, f"{t}", ha="center", va="top", fontsize=11, color="0.35")
axB.text(3.0, -0.26, r"time  $t/\tau_{\mathrm{ML}}$", ha="center", va="top",
         fontsize=12, color="0.35", style="italic")
axB.set_xlim(-0.7, 8.7); axB.set_ylim(-0.02, 1.30); axB.axis("off")
axB.text(4.0, -0.50, r"(b)  Capacity reservoir filling toward $I_{\mathrm{avail}}$",
         ha="center", va="top", fontsize=12)
fig.savefig("new_fig1.png", bbox_inches="tight", facecolor="white")
fig.savefig("new_fig1.eps")
print("wrote new_fig1.png / new_fig1.eps")
