"""Re-derive every quantitative statement added or changed in the R2 revision."""
import json, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("hc", "history_check.py"); hc = importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)
r = json.load(open("history_check.json")); o = r["ordered"]
ok = []
def chk(name, cond): ok.append((name, bool(cond))); print(("PASS " if cond else "FAIL ") + name)
chk("N_hist 4 -> 36786", o["N_hist"][0] == 4 and o["N_hist"][-1] == 36786)
g = np.diff(np.concatenate([[0], o["log2_N_hist"]]))
chk("support growth 1.8-2.0 bits/step", round(g.min(), 1) == 1.8 and round(g.max(), 1) == 2.0)
inc = np.array(o["increments"])
chk("entropy increments 0.80-1.38", f"{inc.min():.2f}" == "0.80" and f"{inc.max():.2f}" == "1.38")
chk("all increments <= log2 m = 2", inc.max() <= 2.0)
chk("rate 1.20 (1.201)", f"{o['rate_last3']:.3f}" == "1.201")
chk("configuration family 3.51 < log2 20", f"{r['ordered_fine_family']['rate_asymptotic']:.2f}" == "3.51" and r['ordered_fine_family']['rate_asymptotic'] < np.log2(20))
chk("log2 20 = 4.32", f"{np.log2(20):.2f}" == "4.32")
chk("kappa ceiling 0.31", f"{np.log2(20)/14.05:.2f}" == "0.31")
chk("Gamma_ICD = 0.265", f"{np.log(2)*o['rate_last3']/np.pi:.3f}" == "0.265")
chk("log2 N_coh <= H_1/2", all(a <= b + 1e-12 for a, b in zip(o["log2_N_coh"], o["renyi_half"][:6])))
# Table 7 proxy ratios from the published chi values (chi proportional to lambda_L per realization)
chi = {"rf": [(2.00, .25), (1.56, .67), (1.06, .86), (0.70, .86)], "rc": [(1.76, .70), (1.82, .75), (1.59, .86), (1.38, .81)],
       "hb": [(1.97, .28), (1.62, .53), (1.26, .65), (0.96, .65)]}
lam = {"rf": [3.09, 2.41, 1.63, 1.08], "rc": [2.72, 2.81, 2.45, 2.13], "hb": [3.04, 2.50, 1.94, 1.48]}
tab = {"rf": [(1.00, .12), (0.78, .33), (0.53, .43), (0.35, .43)], "rc": [(0.88, .35), (0.91, .37), (0.79, .43), (0.69, .40)],
       "hb": [(0.98, .14), (0.81, .26), (0.63, .32), (0.48, .32)]}
good = True
for k in tab:
    for i in range(4):
        m1 = round(lam[k][i] / 3.10, 2); m2 = round(chi[k][i][0] / 2.01, 2); s = round(chi[k][i][1] / 2.01, 2)
        if not (abs(tab[k][i][0] - m1) < 0.0051 and abs(tab[k][i][0] - m2) < 0.011 and abs(tab[k][i][1] - s) < 0.0051):
            good = False; print("  mismatch", k, i, tab[k][i], m1, m2, s)
chk("Table 7 ratio column consistent with lambda_L and chi of R1", good)
chk("chi_proxy = 0.6486 lambda_L (E_dyn cancels)", abs(np.pi / (2 * 3.494 * np.log(2)) * 3.10 - 2.01) < 0.005)
chk("Lambda range 4.67-14.05 all > log2 20", min(4.67, 4.89, 6.71) > np.log2(20))
# Heisenberg zero disorder (R1 statement)
Hh = hc.h_xx([1.0] * 5, heis=True); rr = hc.history_run(Hh); HH = rr["H"]
ih = [HH[0]] + [HH[k] - HH[k - 1] for k in range(1, len(HH))]
chk("Heisenberg W=0 rate 1.28", f"{np.mean(ih[-3:]):.2f}" == "1.28")
# N = 4 example of Proposition 2
N, Nc = 4, 2; C = (Nc - 1) / (N - 1)
rhoB = np.zeros((4, 4)); rhoB[:2, :2] = 0.25; rhoB[2:, 2:] = 0.25
rhoU = (1 - C) * np.eye(4) / 4 + C * np.full((4, 4), 0.25)
l1 = lambda m: np.abs(m).sum() - np.abs(np.diag(m)).sum()
chk("N=4: equal l1, C=1/3", abs(l1(rhoB) - l1(rhoU)) < 1e-12 and abs(l1(rhoB) / 3 - 1 / 3) < 1e-12)
chk("N=4: purities 1/2 and 1/3", abs(np.trace(rhoB @ rhoB) - .5) < 1e-12 and abs(np.trace(rhoU @ rhoU) - 1 / 3) < 1e-12)
chk("N=4: visibilities", abs(4 * rhoB[0, 1] - 1) < 1e-12 and abs(4 * rhoB[0, 2]) < 1e-12 and abs(4 * rhoU[0, 2] - 1 / 3) < 1e-12)
chk("both states positive", np.linalg.eigvalsh(rhoB).min() > -1e-12 and np.linalg.eigvalsh(rhoU).min() > -1e-12)
# random check of Proposition 2 bound on random uniform-diagonal states
rng = np.random.default_rng(1); viol = 0
for _ in range(2000):
    n = int(rng.integers(2, 9)); V = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    V /= np.linalg.norm(V, axis=0); rho = (V.conj().T @ V) / n      # Gram matrix with unit-norm columns -> diag 1/n
    if np.trace(rho @ rho).real > (1 + l1(rho)) / n + 1e-12: viol += 1
chk("Proposition 2 bound on 2000 random states", viol == 0)
# random check of Lemma 2 on random families
viol = 0
for trial in range(40):
    d = 6; Hm = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); Hm = Hm + Hm.conj().T
    w, U0 = np.linalg.eigh(Hm); U = (U0 * np.exp(-1j * w * 0.7)) @ U0.conj().T
    parts = [[0, 1], [2, 3, 4], [5]]; P = [np.diag([1.0 if i in p else 0.0 for i in range(d)]) for p in parts]
    psi = rng.normal(size=d) + 1j * rng.normal(size=d); psi /= np.linalg.norm(psi)
    br = [psi]; Hprev = 0.0
    for n in range(5):
        br = [Pk @ (U @ b) for b in br for Pk in P]
        p = np.array([np.vdot(b, b).real for b in br]); p = p[p > 1e-15]
        Hn = -(p * np.log2(p)).sum()
        if Hn - Hprev > np.log2(3) + 1e-9 or abs(p.sum() - 1) > 1e-9: viol += 1
        Hprev = Hn
chk("Lemma 2 on 40 random three-outcome families", viol == 0)
json.dump(ok, open("numcheck_r2.json", "w"))
print("ALL PASS" if all(c for _, c in ok) else "SOME FAIL")
