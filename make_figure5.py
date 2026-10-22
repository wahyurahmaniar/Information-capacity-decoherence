#!/usr/bin/env python3
"""Figure 5: branching rate and complexity factor against disorder strength.

(a) history-entropy rate h_hist tau_ML from the decoherent-histories functional,
    with the ordered XX baseline; error bars are standard deviations over the
    twelve disorder realizations.
(b) complexity factor normalized to the ordered chain, from the history
    determination (filled) and from the uncalibrated OTOC proxy (open).
Every plotted point is a tabulated value of Table 7 or Table 8.
"""
import json
import os

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

W = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
H_ORD, CHI_ORD = 1.201, 0.172                      # Table 8 baselines

# Table 8: history-entropy rate (mean, standard deviation) and complexity factor
hist = {"rf": [1.013, 0.611, 0.398, 0.247], "rc": [0.853, 0.713, 0.798, 0.969],
        "hb": [0.931, 0.562, 0.386, 0.389]}
hstd = {"rf": [0.151, 0.240, 0.234, 0.247], "rc": [0.356, 0.372, 0.361, 0.309],
        "hb": [0.170, 0.337, 0.271, 0.297]}
chi = {"rf": [0.145, 0.087, 0.057, 0.035], "rc": [0.138, 0.150, 0.216, 0.328],
       "hb": [0.133, 0.081, 0.055, 0.056]}
edyn8 = {"rf": [0.500, 0.500, 0.500, 0.500], "rc": [0.566, 0.736, 0.945, 1.181],
         "hb": [0.500, 0.500, 0.500, 0.500]}
# Table 7: normalized operator-spreading rate lambda_L / lambda_L,ord, with its
# realization standard deviation. This is the chi ratio a unit-proportionality
# insertion of the proxy would assign, since E_dyn cancels from it.
prox = {"rf": [1.00, 0.78, 0.52, 0.35], "rc": [0.88, 0.91, 0.79, 0.69],
        "hb": [0.98, 0.81, 0.62, 0.48]}
pstd = {"rf": [0.04, 0.19, 0.27, 0.29], "rc": [0.31, 0.20, 0.25, 0.21],
        "hb": [0.06, 0.14, 0.20, 0.21]}

# The entries above are the printed values of Tables 7 and 8. When the scripts
# that produce those tables have been run, their output is compared against the
# values plotted here and any disagreement beyond the printed precision stops
# the figure from being written.
_TOY = {"rf": "random_fields", "rc": "random_couplings",
        "hb": "heisenberg_random_fields"}


def _check(path, rows, label):
    """Compare plotted literals against a results file. Returns a status line."""
    if not os.path.exists(path):
        return f"  {label}: {path} absent, values unchecked"
    with open(path) as f:
        res = json.load(f)
    bad = []
    for key, Ws, entries in rows:
        for i, W in enumerate(Ws):
            row = res[key(W)]
            for name, table, field, tol in entries:
                want, got = table[i], row[field]
                if abs(want - got) > tol:
                    bad.append(f"{name}[W={W}]: table {want:g}, script {got:.4f}")
    if bad:
        raise SystemExit(f"{label}: plotted values disagree with {path}\n  "
                         + "\n  ".join(bad))
    return f"  {label}: checked against {path}, OK"


_status = []
for _k, _stem in _TOY.items():
    _status.append(_check(
        "toy_model_results.json",
        [(lambda W, s=_stem: f"{s}_W{W}", (1, 2, 3, 4),
          [(f"prox[{_k}]", prox[_k], "ratio_mean", 0.005),
           (f"pstd[{_k}]", pstd[_k], "ratio_std", 0.005)])],
        f"Table 7 ({_k})"))
for _k in ("rf", "rc", "hb"):
    _status.append(_check(
        "history_check.json",
        [(lambda W, s=_k: f"{s}_W{W}", (1, 2, 3, 4),
          [(f"hist[{_k}]", hist[_k], "rate_mean", 0.0015),
           (f"hstd[{_k}]", hstd[_k], "rate_std", 0.0015),
           (f"edyn8[{_k}]", edyn8[_k], "E_dyn_mean", 0.0015)])],
        f"Table 8 ({_k})"))

col = {"rf": "#c1392b", "rc": "#2c5f8e", "hb": "#2f7a46"}
mk = {"rf": "o", "rc": "s", "hb": "^"}
name = {"rf": "random field", "rc": "random couplings", "hb": "Heisenberg"}


fig = plt.figure(figsize=(6.70, 2.85))
axA = fig.add_axes([0.075, 0.235, 0.370, 0.730])
axB = fig.add_axes([0.600, 0.235, 0.370, 0.730])

axA.axhline(H_ORD, color="0.3", lw=1.1)
axA.text(0.08, H_ORD + 0.035, "ordered baseline", fontsize=8.4, color="0.3", va="bottom")
for k in ("rf", "rc", "hb"):
    y = np.concatenate([[H_ORD], hist[k]])
    e = np.concatenate([[0.0], hstd[k]])
    axA.errorbar(W, y, yerr=e, color=col[k], marker=mk[k], ms=4.6, lw=1.0,
                 capsize=2.4, elinewidth=0.8, mfc=col[k], mec=col[k])
axA.set_xlim(-0.25, 4.25); axA.set_ylim(0.0, 1.70)
axA.set_xticks([0, 1, 2, 3, 4])
axA.set_xlabel(r"Disorder strength $W/J$")
axA.set_ylabel(r"$h_{\rm hist}\tau_{\rm ML}$  (bits per interval)")

axB.axhline(1.0, color="0.3", lw=1.1)
for k in ("rf", "rc", "hb"):
    chi_h = np.concatenate([[1.0], np.array(chi[k]) / CHI_ORD])
    # the rate carries the realization spread; E_dyn is held at its tabulated mean
    err = np.concatenate([[0.0], (np.array(chi[k]) / CHI_ORD)
                          * (np.array(hstd[k]) / np.array(hist[k]))])
    pr = np.concatenate([[1.0], prox[k]])
    axB.errorbar(W, chi_h, yerr=err, color=col[k], marker=mk[k], ms=4.6, lw=1.1,
                 capsize=2.4, elinewidth=0.8, mfc=col[k], mec=col[k], zorder=3)
    axB.plot(W, pr, color=col[k], marker=mk[k], ms=4.6, lw=0.9, ls=(0, (3, 2)),
             mfc="white", mec=col[k], mew=1.0, zorder=2)
axB.set_xlim(-0.25, 4.25); axB.set_ylim(0.0, 2.95)
axB.set_xticks([0, 1, 2, 3, 4])
axB.set_xlabel(r"Disorder strength $W/J$")
axB.set_ylabel(r"$\chi/\chi_{\rm ordered}$")

var = [Line2D([], [], ls="none", marker=mk[k], ms=4.6, mfc=col[k], mec=col[k],
              label=name[k]) for k in ("rf", "rc", "hb")]
det = [Line2D([], [], ls="-", lw=1.1, color="0.3", marker="o", ms=4.6, mfc="0.3",
              mec="0.3", label="history determination"),
       Line2D([], [], ls=(0, (3, 2)), lw=0.9, color="0.3", marker="o", ms=4.6,
              mfc="white", mec="0.3", label=r"OTOC proxy, $\lambda_L/\lambda_{L,\mathrm{ord}}$")]
axA.legend(handles=var, loc="upper right", frameon=False, labelspacing=0.3,
           handletextpad=0.5, borderaxespad=0.3)
# panel (b) names both the determination and the variant, so it reads on its own
axB.legend(handles=det + var, loc="upper left", frameon=False, labelspacing=0.28,
           handlelength=2.0, handletextpad=0.5, borderaxespad=0.25, ncol=1)

for ax, lab in ((axA, "(a)"), (axB, "(b)")):
    bb = ax.get_position()
    fig.text(bb.x0 + bb.width / 2.0, 0.015, lab, ha="center", va="bottom", fontsize=10)

fig.savefig("new_fig5.png")
fig.savefig("new_fig5.eps")
print("wrote new_fig5.png / new_fig5.eps")
for _line in _status:
    print(_line)
for k in ("rf", "rc", "hb"):
    r = np.array(chi[k]) / CHI_ORD
    # consistency: chi must reproduce (E_dyn/E_cap) * h_hist tau_ML to rounding
    chk = (np.array(edyn8[k]) / 3.494) * np.array(hist[k])
    assert np.allclose(chk, chi[k], atol=6e-3), (k, chk, chi[k])
    print(f"  {k}: chi/chi_ord = {np.round(r, 3)}   proxy = {prox[k]}")
