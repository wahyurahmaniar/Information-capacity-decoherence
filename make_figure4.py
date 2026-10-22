"""Figure 4: history quantities of Lemma 2 versus the OTOC proxy index.

(a) ordered chain, per-step increments against coarse-graining step n
    (support growth, history-entropy increment of the left-half magnetization
    family, history-entropy increment of the configuration-basis family),
    with the ceilings log2 m = 2, log2 20, log2 64 = 6 and the OTOC index.
(b) history-entropy rate (Table 8) and OTOC index (Table 7) against W/J.

Panel (a) is recomputed from history_check.py; panel (b) uses the tabulated
values of Tables 7 and 8, which history_check.py reproduces for the history
rates.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.2,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": False,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
})

res = json.load(open("history_check.json"))
o = res["ordered"]
n = np.arange(1, 9)
logN = np.array(o["log2_N_hist"])
support_growth = np.diff(np.concatenate([[0.0], logN]))
dH = np.array(o["increments"])
fine = np.array(res["ordered_fine_family"]["increments"][:8])

L2M, L2S, L2D = 2.0, np.log2(20.0), 6.0
LAM_ORD = 14.05

# Table 7 (OTOC index, means) and Table 8 (history rate, mean and std)
W = np.array([1, 2, 3, 4])
lam = {"rf": [14.00, 10.92, 7.39, 4.89], "rc": [11.65, 9.72, 6.63, 4.67], "hb": [13.78, 11.33, 8.79, 6.71]}
hist = {"rf": [1.013, 0.611, 0.398, 0.247], "rc": [0.853, 0.713, 0.798, 0.969], "hb": [0.931, 0.562, 0.386, 0.389]}
hstd = {"rf": [0.151, 0.240, 0.234, 0.247], "rc": [0.356, 0.372, 0.361, 0.309], "hb": [0.170, 0.337, 0.271, 0.297]}
H_ORD = 1.201

# cross-check panel (b) history means against the recomputation
for key, tag in [("rf", "rf"), ("rc", "rc"), ("hb", "hb")]:
    for i, w in enumerate(W):
        rec = res[f"{tag}_W{w}"]
        assert abs(rec["rate_mean"] - hist[key][i]) < 6e-4, (key, w)
        assert abs(rec["rate_std"] - hstd[key][i]) < 6e-4, (key, w)
assert abs(o["rate_last3"] - H_ORD) < 6e-4

col = {"ord": "black", "rf": "#ff0000", "rc": "#0000ff", "hb": "#00a000"}
mk = {"rf": "o", "rc": "s", "hb": "^"}
off = {"rf": -0.14, "rc": 0.0, "hb": 0.14}

fig = plt.figure(figsize=(6.70, 2.85))
axA = fig.add_axes([0.070, 0.235, 0.352, 0.735])
axB = fig.add_axes([0.578, 0.235, 0.352, 0.735])

ref_style = dict(color="0.35", lw=1.0)
LS_M, LS_S, LS_D = (0, (5, 3)), (0, (6, 2, 1.5, 2)), (0, (1.2, 1.8))
for ax in (axA, axB):
    ax.axhline(L2M, ls=LS_M, **ref_style)
    ax.axhline(L2S, ls=LS_S, **ref_style)
    ax.axhline(L2D, ls=LS_D, **ref_style)

# panel (a)
axA.axhline(LAM_ORD, color="#e000e0", lw=1.3)
C_SUP, C_MAG, C_CFG = "#ff8000", "#8000c0", "#00b0f0"   # orange, purple, cyan: distinct from panel (b)
axA.plot(n, support_growth, ls="-", lw=1.0, color=C_SUP, marker="s", ms=5.0, mfc=C_SUP, mec=C_SUP)
axA.plot(n, dH, ls="-", lw=1.0, color=C_MAG, marker="o", ms=5.0, mfc=C_MAG, mec=C_MAG)
axA.plot(n, fine, ls="-", lw=1.0, color=C_CFG, marker="^", ms=5.4, mfc=C_CFG, mec=C_CFG)
axA.set_xlim(0.5, 8.5)
axA.set_ylim(0, 18.4)
axA.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
axA.set_xticks(n)
axA.set_xlabel(r"Coarse-graining step $n$")
axA.set_ylabel("Bits per interval")
hA = [
    Line2D([], [], color=C_SUP, lw=1.0, marker="s", ms=5.0, mfc=C_SUP, mec=C_SUP, label="support growth"),
    Line2D([], [], color=C_MAG, lw=1.0, marker="o", ms=5.0, mfc=C_MAG, mec=C_MAG, label="entropy, magnetization family"),
    Line2D([], [], color=C_CFG, lw=1.0, marker="^", ms=5.4, mfc=C_CFG, mec=C_CFG, label="entropy, configuration family"),
    Line2D([], [], color="#e000e0", lw=1.3, label="OTOC index"),
]
legA = axA.legend(handles=hA, loc="center", bbox_to_anchor=(0.5, 0.50), ncol=1, frameon=False,
                  handlelength=2.2, labelspacing=0.4, borderaxespad=0.2)

# panel (b)
for key in ["rf", "rc", "hb"]:
    x = np.concatenate([[0.0], W + off[key]])
    yh = np.concatenate([[H_ORD], hist[key]])
    yl = np.concatenate([[LAM_ORD], lam[key]])
    axB.plot(x, yl, ls=(0, (3, 2)), lw=0.8, color=col[key])
    axB.plot(W + off[key], lam[key], ls="none", marker=mk[key], ms=5.0, mfc="white", mec=col[key], mew=1.1)
    axB.plot(x, yh, ls="-", lw=0.9, color=col[key])
    axB.plot(W + off[key], hist[key], ls="none", marker=mk[key], ms=4.4, mfc=col[key], mec=col[key])
axB.plot([0], [LAM_ORD], ls="none", marker="D", ms=4.8, mfc="white", mec="black", mew=1.1)
axB.plot([0], [H_ORD], ls="none", marker="D", ms=4.8, mfc="black", mec="black")
axB.set_xlim(-0.4, 4.4)
axB.set_ylim(0, 18.4)
axB.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
axB.set_xticks([0, 1, 2, 3, 4])
axB.set_xlabel(r"Disorder strength $W/J$")
axB.set_ylabel("Bits per interval")
lab_kw = dict(fontsize=8.2, ha="left", va="bottom", color="0.25")
axB.text(-0.33, L2M + 0.18, "family ceiling", **lab_kw)
axB.text(-0.33, L2S + 0.18, "sector ceiling", **lab_kw)
axB.text(-0.33, L2D + 0.18, "full-space ceiling", **lab_kw)
hB = [
    Line2D([], [], ls=(0, (3, 2)), lw=0.8, color="0.25", marker="o", ms=4.6, mfc="white", mec="0.25", label="OTOC index"),
    Line2D([], [], ls="none", marker="o", ms=4.6, mfc=col["rf"], mec=col["rf"], label="random field"),
    Line2D([], [], ls="-", lw=0.9, color="0.25", marker="o", ms=4.6, mfc="0.25", mec="0.25", label="history rate"),
    Line2D([], [], ls="none", marker="s", ms=4.6, mfc=col["rc"], mec=col["rc"], label="random couplings"),
    Line2D([], [], ls="none", marker="D", ms=4.4, mfc="black", mec="black", label="ordered chain"),
    Line2D([], [], ls="none", marker="^", ms=5.0, mfc=col["hb"], mec=col["hb"], label="Heisenberg"),
]
legB = axB.legend(handles=hB, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=2, frameon=False,
                  handlelength=2.0, labelspacing=0.30, columnspacing=1.1, borderaxespad=0.1)

# right-hand axis: the proxy index is dimensionless, not an entropy increment
for ax in (axA, axB):
    tw = ax.twinx()
    tw.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16])
    tw.set_ylim(0, 18.4)
    tw.tick_params(axis="y", direction="in", labelsize=9)
    tw.set_ylabel("Uncalibrated OTOC index\n(dimensionless)", fontsize=9.0, linespacing=1.15)
    for sp in tw.spines.values():
        sp.set_linewidth(0.8)

for ax, lab in ((axA, "(a)"), (axB, "(b)")):
    bb = ax.get_position()
    fig.text(bb.x0 + bb.width / 2.0, 0.015, lab, ha="center", va="bottom", fontsize=10)

fig.savefig("new_fig4.png", dpi=600)
fig.savefig("new_fig4.eps")
fig.savefig("new_fig4_check.png", dpi=150)

# overlap audit: legend boxes versus plotted data points (display coordinates)
fig.canvas.draw()
r = fig.canvas.get_renderer()
def pts(ax, xs, ys):
    return ax.transData.transform(np.column_stack([xs, ys]))
for ax, leg, sets in [
    (axA, legA, [(n, support_growth), (n, dH), (n, fine)]),
    (axB, legB, [(W + off[k], lam[k]) for k in lam] + [(W + off[k], np.array(hist[k])) for k in hist]
     + [([0], [LAM_ORD]), ([0], [H_ORD])]),
]:
    bb = leg.get_window_extent(r)
    hits = 0
    for xs, ys in sets:
        for px, py in pts(ax, xs, ys):
            if bb.x0 - 4 <= px <= bb.x1 + 4 and bb.y0 - 4 <= py <= bb.y1 + 4:
                hits += 1
    # reference lines crossing the legend box
    lines = [L2M, L2S, L2D] + ([LAM_ORD] if ax is axA else [])
    yl = [ax.transData.transform((0, y))[1] for y in lines]
    crossing = [round(y, 1) for y, yp in zip(lines, yl) if bb.y0 - 2 <= yp <= bb.y1 + 2]
    print("legend overlap hits:", hits, " reference lines crossing legend:", crossing)
