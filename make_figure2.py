#!/usr/bin/env python3
"""Figure 2: the dynamical completion.

(a) analytic decay law of the block completion, C(t) = 2^{-[I_fut - I_avail]_+},
    for three available capacities;
(b) growth of the history-space coherence number and of the effective branch
    number under closed unitary evolution of the six-site chains, with an
    illustrative coherence cap that the simulation does not enforce.
Panel (b) is read from fig2_data.json, produced by fig2_data.py.
"""
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "text.usetex": False, "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "font.size": 9.5, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.6, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "ps.fonttype": 42, "pdf.fonttype": 42, "savefig.dpi": 600,
})

BLUE_L, BLUE_M, ORANGE, BLACK, TEAL = "#7fb3e0", "#2c5f8e", "#d35400", "black", "#1a7f6b"
NPLOT, CAP = 7, 32.0

fig = plt.figure(figsize=(6.70, 2.85))
axA = fig.add_axes([0.085, 0.235, 0.355, 0.730])
axB = fig.add_axes([0.600, 0.235, 0.355, 0.730])

# ---- (a) analytic decay law
t = np.linspace(0.8, 1.3, 600)
for iav, c, ls in ((10, BLUE_L, (0, (1.6, 1.8))), (30, BLUE_M, (0, (6, 3))), (100, ORANGE, "-")):
    axA.plot(t, 2.0 ** (-iav * np.clip(t - 1.0, 0, None)), color=c, lw=1.6, ls=ls,
             label=r"$I_{\rm avail}=%d$ bits" % iav)
axA.axvline(1.0, color="0.45", lw=1.0, ls=(0, (4, 2, 1, 2)))
axA.text(1.012, 1.25, r"$t=\tau_{\rm cap}$", fontsize=8.8, color="0.35",
         va="center", ha="left")
axA.set_yscale("log")
axA.set_xlim(0.8, 1.3); axA.set_ylim(1e-6, 4.0)
axA.set_xlabel(r"$t/\tau_{\rm cap}$")
axA.set_ylabel(r"Normalized coherence  $\mathcal{C}(t)$")
axA.legend(loc="lower left", frameon=False, labelspacing=0.3, handlelength=2.2,
           borderaxespad=0.4,
           title=r"$\mathcal{C}=2^{-[I_{\rm future}(t)-I_{\rm avail}]_+}$",
           title_fontsize=8.8)

# ---- (b) unconstrained growth of the coherence number
d = json.load(open("fig2_data.json"))
n = np.arange(1, NPLOT + 1)
for key, col, lab in (("ordered", BLACK, "ordered XX"), ("rf_W4", ORANGE, r"random field, $W/J=4$")):
    y = 2.0 ** np.array(d[key]["log2_N_coh"][:NPLOT])
    h = 2.0 ** np.array(d[key]["H_n"][:NPLOT])
    axB.plot(n, y, color=col, lw=1.1, marker="o", ms=4.4, mfc=col, mec=col, label=lab)
    axB.plot(n, h, color=col, lw=0.9, ls=(0, (4, 2.4)), marker="o", ms=4.4,
             mfc="white", mec=col, mew=1.0)
axB.axhline(CAP, color=TEAL, lw=1.2, ls=(0, (5, 2, 1, 2)))
axB.text(0.75, CAP * 1.22, r"$2^{I_{\rm avail}}=32$ (not enforced)",
         fontsize=8.2, color=TEAL, va="bottom", ha="left")
axB.set_yscale("log")
axB.set_xlim(0.6, NPLOT + 0.4); axB.set_ylim(0.8, 9e3)
axB.set_xticks(n)
axB.set_xlabel(r"Coarse-graining steps  $n=t/\tau_{\rm ML}$")
axB.set_ylabel("Effective number of histories")
hB = [Line2D([], [], color="0.3", lw=1.1, marker="o", ms=4.4, mfc="0.3", mec="0.3",
             label=r"$\mathcal{N}_{\rm coh}$"),
      Line2D([], [], color="0.3", lw=0.9, ls=(0, (4, 2.4)), marker="o", ms=4.4,
             mfc="white", mec="0.3", label=r"$2^{H_n}$"),
      Line2D([], [], color=BLACK, lw=1.1, label="ordered XX"),
      Line2D([], [], color=ORANGE, lw=1.1, label=r"random field, $W/J=4$")]
axB.legend(handles=hB, loc="upper left", frameon=False, labelspacing=0.28,
           handlelength=2.0, handletextpad=0.5, borderaxespad=0.3)

for ax, lab in ((axA, "(a)"), (axB, "(b)")):
    bb = ax.get_position()
    fig.text(bb.x0 + bb.width / 2.0, 0.015, lab, ha="center", va="bottom", fontsize=10)

fig.savefig("new_fig2.png")
fig.savefig("new_fig2.eps")
fig.savefig("new_fig2_check.png", dpi=150)
print("wrote new_fig2.png / new_fig2.eps")
for key in ("ordered", "rf_W4"):
    g = np.diff(np.array(d[key]["log2_N_coh"]))
    hh = np.diff(np.array(d[key]["H_n"]))
    print(f"  {key}: N_coh rate {np.mean(g[-3:]):.3f} bits/interval | "
          f"history rate {np.mean(hh[-3:]):.3f} bits/interval")
