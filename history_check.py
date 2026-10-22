"""Re-derive the decoherent-histories branching quantities of the six-site chains.

Family: projectors onto the left-half magnetization M_L = S^z_1+S^z_2+S^z_3,
applied at t_k = k tau_ML (k = 1..n), tau_ML = pi hbar / (2 E_dyn), E_dyn = Delta E
in the domain-wall state |uuuddd>.  Branch weights p_alpha = ||C_alpha psi||^2,
cutoff 1e-12, entropy rate = mean of the last three increments of H_n.

Reports, per step n, the three history quantities that Reviewer #2 asks to be
separated: the support count N_hist(n), the history entropy H_n, and the
increment H_n - H_{n-1}, together with the ceilings log2 m, log2 d_sector,
log2 d.  Also evaluates the finest family (configuration basis of the
zero-magnetization sector), for which the history process is a Markov chain.
"""
import json
import sys
import numpy as np

N = 6
J = 1.0
NSTEP = 8
CUT = 1e-12

sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
I2 = np.eye(2, dtype=complex)


def site_op(op, i):
    mats = [I2] * N
    mats[i] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


SX = [site_op(sx, i) for i in range(N)]
SY = [site_op(sy, i) for i in range(N)]
SZ = [site_op(sz, i) for i in range(N)]


def h_xx(couplings, fields=None, heis=False):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        H += couplings[i] * (SX[i] @ SX[i + 1] + SY[i] @ SY[i + 1])
        if heis:
            H += couplings[i] * (SZ[i] @ SZ[i + 1])
    if fields is not None:
        for i in range(N):
            H += fields[i] * SZ[i]
    return H


# basis ordering: kron(site0, site1, ...), |0> = up
def basis_index(bits):
    idx = 0
    for b in bits:
        idx = 2 * idx + b
    return idx


psi0 = np.zeros(2**N, dtype=complex)
psi0[basis_index([0, 0, 0, 1, 1, 1])] = 1.0  # |uuuddd>

ML = SZ[0] + SZ[1] + SZ[2]
ml_vals = np.real(np.diag(ML))
ML_LEVELS = sorted(set(np.round(ml_vals, 6)))
PROJ_DIAG = [np.isclose(ml_vals, v).astype(float) for v in ML_LEVELS]  # diagonal projectors

Mtot = np.real(np.diag(sum(SZ)))
SECTOR = np.where(np.isclose(Mtot, 0.0))[0]  # 20 states


def evolve_op(H, t):
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w * t)) @ V.conj().T


def edyn(H, psi):
    e1 = np.real(psi.conj() @ H @ psi)
    e2 = np.real(psi.conj() @ H @ H @ psi)
    return np.sqrt(max(e2 - e1**2, 0.0))


def bandwidth(H):
    w = np.linalg.eigvalsh(H)
    return w[-1] - w[0]


def shannon(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def history_run(H, nstep=NSTEP, cut=CUT, keep_vectors_upto=0):
    """Return per-step support count, entropy, discarded weight, (optional) branch vectors."""
    ed = edyn(H, psi0)
    tau = np.pi / (2 * ed)
    U = evolve_op(H, tau)
    branches = [psi0.copy()]
    out = {"E_dyn": ed, "tau_ML": tau, "N_hist": [], "H": [], "discarded": [], "renyi_half": []}
    discarded = 0.0
    vec_store = {}
    for n in range(1, nstep + 1):
        new = []
        for b in branches:
            v = U @ b
            for P in PROJ_DIAG:
                w = P * v
                nrm = np.real(np.vdot(w, w))
                if nrm >= cut:
                    new.append(w)
                else:
                    discarded += nrm
        branches = new
        p = np.array([np.real(np.vdot(b, b)) for b in branches])
        out["N_hist"].append(len(branches))
        out["H"].append(shannon(p))
        out["renyi_half"].append(float(2 * np.log2(np.sqrt(p).sum())))
        out["discarded"].append(discarded)
        if n <= keep_vectors_upto:
            vec_store[n] = np.array(branches)
    out["vectors"] = vec_store
    return out


def coherence_number(B):
    """N_coh = 1 + sum_{a != a'} |<B_a'|B_a>| for branch vectors as rows of B."""
    G = B.conj() @ B.T
    l1 = np.abs(G).sum() - np.abs(np.diag(G)).sum()
    return 1.0 + float(l1)


def fine_family_rate(H, nstep=40):
    """Configuration basis of the M=0 sector: Markov chain with T_ab = |<b|U|a>|^2."""
    ed = edyn(H, psi0)
    tau = np.pi / (2 * ed)
    U = evolve_op(H, tau)
    Us = U[np.ix_(SECTOR, SECTOR)]
    T = np.abs(Us) ** 2  # T[b, a] = P(b | a)
    a0 = np.where(SECTOR == basis_index([0, 0, 0, 1, 1, 1]))[0][0]
    p = np.zeros(len(SECTOR))
    p[a0] = 1.0
    incs = []
    rowH = np.array([shannon(T[:, a]) for a in range(len(SECTOR))])
    for n in range(1, nstep + 1):
        incs.append(float(p @ rowH))  # H(alpha_n | alpha_{n-1})
        p = T @ p
    return incs


def main():
    res = {}
    Hord = h_xx([J] * (N - 1))
    res["E_cap_ordered_bandwidth"] = bandwidth(Hord)
    r = history_run(Hord, keep_vectors_upto=6)
    H = r["H"]
    incs = [H[0]] + [H[k] - H[k - 1] for k in range(1, len(H))]
    rate = float(np.mean(incs[-3:]))
    ncoh = [coherence_number(r["vectors"][n]) for n in range(1, 7)]
    res["ordered"] = {
        "E_dyn": r["E_dyn"], "tau_ML": r["tau_ML"], "N_hist": r["N_hist"],
        "log2_N_hist": [float(np.log2(x)) for x in r["N_hist"]], "H_n": H,
        "increments": incs, "rate_last3": rate, "discarded": r["discarded"][-1],
        "renyi_half": r["renyi_half"], "N_coh_n1to6": ncoh,
        "log2_N_coh": [float(np.log2(x)) for x in ncoh],
    }
    fine = fine_family_rate(Hord)
    res["ordered_fine_family"] = {"increments": fine[:10], "rate_last3_of_8": float(np.mean(fine[5:8])),
                                  "rate_asymptotic": float(np.mean(fine[-5:])),
                                  "ceiling_log2_sector": float(np.log2(len(SECTOR)))}
    res["sector_dim"] = int(len(SECTOR))

    # disordered variants, own random stream (seeds 0..11), for qualitative cross-check only
    Ecap = res["E_cap_ordered_bandwidth"]
    for name in ["rf", "rc", "hb"]:
        for W in [1, 2, 3, 4]:
            rates, eds, maxinc = [], [], []
            for seed in range(12):
                rng = np.random.default_rng(seed)
                if name == "rf":
                    Hh = h_xx([J] * (N - 1), fields=rng.uniform(-W, W, N))
                elif name == "rc":
                    Hh = h_xx(J + rng.uniform(-W, W, N - 1))
                else:
                    Hh = h_xx([J] * (N - 1), fields=rng.uniform(-W, W, N), heis=True)
                rr = history_run(Hh)
                HH = rr["H"]
                inc = [HH[0]] + [HH[k] - HH[k - 1] for k in range(1, len(HH))]
                rates.append(np.mean(inc[-3:]))
                eds.append(rr["E_dyn"])
                maxinc.append(max(inc))
            res[f"{name}_W{W}"] = {"rate_mean": float(np.mean(rates)), "rate_std": float(np.std(rates)),
                                   "E_dyn_mean": float(np.mean(eds)), "max_increment": float(max(maxinc))}
    json.dump(res, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
