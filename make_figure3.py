#!/usr/bin/env python3
"""Figure 3: four decoherence timescales vs mass.
Revised per referee comment 5: ALL four curves are now computed from one explicit,
stated parameter set (listed in Table 2), so the figure is reproducible. CSL is
labelled a compact-object benchmark. No '--' in any text label.
Reference model (illustrative; swap values and re-run to match a different choice):
  rho=2.0e3 kg/m^3 ; R=(3M/4 pi rho)^(1/3) ; N=M/m_C, m_C=12 u ; E_dyn=3 N kB T, T=4 K ;
  E_cap=M c^2 ; tau_env=1/(n_bg sigma v_th), N2 gas @300K, P=1e-8 Pa ;
  tau_grav=hbar R/(G M^2) ; tau_CSL=m_nuc^2/(lambda_CSL M^2), lambda_CSL=1e-8 s^-1."""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams.update({"text.usetex": False, "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False, "savefig.dpi": 300})

GREEN, GRAV_RED, PURPLE, NAVY = "#3c7a4e", "#be2f37", "#9b59b6", "#2c5f8e"
MFC, MEC, YELL = "#f0a830", "#9c6c17", "#f8f6e9"

# physical constants (SI)
hbar, kB, c, G, amu = 1.054571e-34, 1.380649e-23, 2.99792458e8, 6.674e-11, 1.660539e-27
# reference parameters
rho, m_C, T = 2.0e3, 12*amu, 4.0           # bulk density, constituent mass, internal temp
P, Tg, mgas = 1e-8, 300.0, 28*amu          # UHV pressure, gas temp, N2 mass
lam, m_nuc = 1e-8, amu                      # CSL benchmark rate, nucleon reference mass

M = np.logspace(0, 12, 600)                 # mass in amu
Mk = M*amu                                  # mass in kg
R = (3*Mk/(4*np.pi*rho))**(1/3)
n_bg = P/(kB*Tg); v_th = np.sqrt(3*kB*Tg/mgas)
tau_env  = 1.0/(n_bg*np.pi*R**2*v_th)
tau_grav = hbar*R/(G*Mk**2)
tau_csl  = m_nuc**2/(lam*Mk**2)
N = Mk/m_C; E_dyn = 3*N*kB*T; E_cap = Mk*c**2
cap = {be: (np.pi**2/np.log(2))*R/(c*(E_dyn/E_cap)*be) for be in (0.01, 1.0, 100.0)}

fig, ax = plt.subplots(figsize=(11.4, 6.6))
YLO, YHI = 1e-10, 1e14
MS100 = 100.0 * m_C / amu
adm = M >= MS100
env = np.minimum.reduce([tau_env[adm], tau_grav[adm], tau_csl[adm], cap[100.0][adm]])
ax.fill_between(M[adm], YLO, np.clip(env, YLO, YHI), color=YELL, zorder=0)

h_env,  = ax.plot(M, tau_env,  color=GREEN,    lw=2.2, zorder=3)
h_grav, = ax.plot(M, tau_grav, color=GRAV_RED, lw=2.2, zorder=3)
h_csl,  = ax.plot(M, tau_csl,  color=PURPLE,   lw=2.2, zorder=3)
# Lemma 2 caps the per-interval increment at log2 d, so a given eta_b log2 b is
# admissible only above the mass that supplies that many accessible dimensions:
# with one accessible level pair per constituent, log2 d >= N = M/m_C.
STYLE = {0.01: (2.0, (0, (1.5, 1.8))), 1.0: (2.0, (0, (6, 3))), 100.0: (2.2, "-")}
hcap = {}
for be, (lw, ls) in STYLE.items():
    Mstar = be * m_C / amu                      # mass in amu at which log2 d = eta_b log2 b
    ok = M >= Mstar
    hcap[be], = ax.plot(M[ok], cap[be][ok], color=NAVY, lw=lw, ls=ls, zorder=4)
    if (~ok).any():
        ax.plot(M[~ok], cap[be][~ok], color=NAVY, lw=lw * 0.8, ls=ls, alpha=0.28, zorder=4)
        ax.plot([Mstar], [np.interp(Mstar, M, cap[be])], marker="|", color=NAVY,
                ms=13, mew=1.8, zorder=6)
h_d, h_a, h_s = hcap[0.01], hcap[1.0], hcap[100.0]
ax.annotate("below the tick, Lemma 2 forbids this branching rate:\n"
            r"$\eta_b\log_2 b\leq\log_2 d$ needs $M\gtrsim\eta_b\log_2 b\,m_C$",
            xy=(100.0 * m_C / amu, np.interp(100.0 * m_C / amu, M, cap[100.0])),
            xytext=(2.4e1, 4e-9), fontsize=10.5, color=NAVY, ha="left", va="bottom",
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2), zorder=7)

systems = [("NV$^-$ center", 30, 2.0e-3, (-34, 34), "center"),
           (r"$^{87}$Rb", 87, 2.0e-3, (-6, -30), "center"),
           (r"$C_{60}$", 720, 2.0e-3, (2, 34), "center"),
           ("TPPF152", 1.6e4, 2.0e-3, (10, -30), "left"),
           (r"Yb$^+$ ion", 170, 0.4, (16, 10), "left"),
           ("Nanoparticle", 1.0e6, 2.0e-7, (16, 18), "left")]
ax.text(1.4e0, 2.2e-10, "markers: representative experimental systems at their own\n"
        "preparation conditions, shown for context, not fits to the curves",
        fontsize=10.5, color=MEC, ha="left", va="bottom", style="italic", zorder=7)
for name, m, t, off, ha in systems:
    ax.plot(m, t, "o", mfc=MFC, mec=MEC, mew=1.2, ms=9, zorder=6)
    ax.annotate(name, (m, t), textcoords="offset points", xytext=off,
                ha=ha, va="center", fontsize=11, zorder=7,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.8,
                                shrinkA=1.0, shrinkB=5.0))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e0, 1e12); ax.set_ylim(YLO, YHI)
ax.set_xlabel(r"mass $M$ (amu)", fontsize=15)
ax.set_ylabel(r"time (s)", fontsize=15)
ax.tick_params(labelsize=12)
ax.grid(True, which="major", color="0.85", lw=0.6, zorder=0)

leg = ax.legend(
    [h_env, h_grav, h_csl, h_d, h_a, h_s],
    [r"$\tau_{\mathrm{env}}$ (collisional, UHV $10^{-8}\,$Pa)",
     r"$\tau_{\mathrm{grav}}$ (Penrose)",
     r"$\tau_{\mathrm{CSL}}$ (compact-object benchmark, $\lambda_{\mathrm{CSL}}=10^{-8}\,\mathrm{s}^{-1}$)",
     r"$\tau_{\mathrm{cap}},\ \eta_b\log_2 b=0.01$",
     r"$\tau_{\mathrm{cap}},\ \eta_b\log_2 b=1$",
     r"$\tau_{\mathrm{cap}},\ \eta_b\log_2 b=100$"],
    loc="upper right", ncol=2, fontsize=10.5, framealpha=0.95,
    handlelength=2.6, columnspacing=1.3, borderpad=0.7)
leg.get_frame().set_edgecolor("0.7")
fig.tight_layout()
fig.savefig("new_fig3.png", bbox_inches="tight", facecolor="white")
fig.savefig("new_fig3.eps")
print("wrote new_fig3.png / new_fig3.eps")
# quick sanity print
for be in (0.01,1.0,100.0):
    print(f"  tau_cap(M=720amu, eta_b log2 b={be}) = {(np.pi**2/np.log(2))*((3*720*amu/(4*np.pi*rho))**(1/3))/(c*(3*(720*amu/m_C)*kB*T/(720*amu*c**2))*be):.2e} s")
