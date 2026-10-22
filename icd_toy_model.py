"""ICD toy model: organization-dependent coherence in a disordered spin chain.

Exact diagonalization of an N = 6 spin-1/2 chain, giving per Hamiltonian:

    dynamical energy uncertainty   Delta E = sqrt(<H^2> - <H>^2)
    OTOC growth rate               lambda_L  (boundary operators S^z_1, S^z_N)
    Loschmidt echo decay rate      Gamma_L
    proxy branching factor         log2 b = lambda_L * tau_ML / ln 2
    complexity factor              chi = eta_b (E_dyn/E_cap) log2 b
    capacity-limited time          tau_cap = (pi^2 / ln 2) / chi, in units of R/c

These are the uncalibrated proxy diagnostics of Table 7. They are NOT the
history-entropy rate: lambda_L is an effective early-time growth rate, and the
paper inserts it into the budget only through the calibration relation of
Eq. (45). The physically defined branching rate is computed by history_check.py.

E_cap is taken as the spectral bandwidth of the ordered Hamiltonian, a
toy-model normalization rather than a Bekenstein capacity scale.
"""
from functools import reduce

import numpy as np
import scipy.linalg as la

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
SY = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
SZ = np.array([[0.5, 0], [0, -0.5]], dtype=complex)


def kron_all(ops):
    return reduce(np.kron, ops)


def site_op(op, site, N):
    """`op` on one site, identity elsewhere."""
    ops = [I2] * N
    ops[site] = op
    return kron_all(ops)


# --------------------------------------------------------------- Hamiltonians
def build_H_ordered(N, J):
    """H = J sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1}), open boundaries."""
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += J * (site_op(SX, i, N) @ site_op(SX, i + 1, N)
                  + site_op(SY, i, N) @ site_op(SY, i + 1, N))
    return H


def build_H_disordered(N, J, W, seed=None):
    """Ordered XX chain plus random longitudinal fields h_i ~ U[-W, W]."""
    if seed is not None:
        np.random.seed(seed)
    H = build_H_ordered(N, J)
    h = np.random.uniform(-W, W, size=N)
    for i in range(N):
        H += h[i] * site_op(SZ, i, N)
    return H, h


def build_H_random_couplings(N, J_mean, J_disorder, seed=None):
    """XX chain with J_i ~ J_mean + U[-J_disorder, J_disorder]."""
    if seed is not None:
        np.random.seed(seed)
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    Js = J_mean + np.random.uniform(-J_disorder, J_disorder, size=N - 1)
    for i in range(N - 1):
        H += Js[i] * (site_op(SX, i, N) @ site_op(SX, i + 1, N)
                      + site_op(SY, i, N) @ site_op(SY, i + 1, N))
    return H, Js


def build_H_heisenberg_random_fields(N, J, W, seed=None):
    """Heisenberg chain plus random longitudinal fields."""
    if seed is not None:
        np.random.seed(seed)
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for i in range(N - 1):
        H += J * (site_op(SX, i, N) @ site_op(SX, i + 1, N)
                  + site_op(SY, i, N) @ site_op(SY, i + 1, N)
                  + site_op(SZ, i, N) @ site_op(SZ, i + 1, N))
    h = np.random.uniform(-W, W, size=N)
    for i in range(N):
        H += h[i] * site_op(SZ, i, N)
    return H, h


# ------------------------------------------------- dynamical quantities
def dynamical_energy_scale(H, psi):
    """Delta E = sqrt(<H^2> - <H>^2) in the given state."""
    psi = psi / np.linalg.norm(psi)
    EH = float((psi.conj() @ H @ psi).real)
    EH2 = float((psi.conj() @ H @ H @ psi).real)
    return float(np.sqrt(max(EH2 - EH ** 2, 0.0)))


def otoc(H, W_op, V_op, psi, t):
    """C(t) = <psi| [W(t), V]^dag [W(t), V] |psi>."""
    U = la.expm(-1j * H * t)
    Wt = U.conj().T @ W_op @ U
    comm = Wt @ V_op - V_op @ Wt
    return float(np.real(psi.conj() @ (comm.conj().T @ comm) @ psi))


def otoc_series(H, W_op, V_op, psi, ts):
    """C(t) on a grid, via the spectral decomposition of H.

    Identical to `otoc` for Hermitian H, and much faster over a grid because
    the eigendecomposition is reused instead of exponentiating at every t.
    """
    E, V = np.linalg.eigh(H)
    Wv = V.conj().T @ W_op @ V
    Vv = V.conj().T @ V_op @ V
    pv = V.conj().T @ psi
    out = np.empty(len(ts))
    for k, t in enumerate(ts):
        ph = np.exp(-1j * E * t)
        Wt = (ph.conj()[:, None] * Wv) * ph[None, :]
        comm = Wt @ Vv - Vv @ Wt
        out[k] = float(np.real(pv.conj() @ (comm.conj().T @ comm) @ pv))
    return out


def loschmidt_echo(H, dH, psi, t):
    """|<psi(t) | psi'(t)>|^2 with psi' evolved under H + dH."""
    psi_f = la.expm(-1j * H * t) @ psi
    psi_fp = la.expm(-1j * (H + dH) * t) @ psi
    return float(abs(psi_fp.conj() @ psi_f) ** 2)


def initial_state_domain_wall(N):
    """|up ... up down ... down>, the state every run starts from."""
    psi0 = np.zeros(2 ** N, dtype=complex)
    psi0[sum(2 ** i for i in range(N // 2))] = 1.0
    return psi0


# --------------------------------------------------- framework quantities
def compute_chi_for_H(H, psi0, ts_otoc, W_op, V_op, eta_b=1.0, E_cap=None):
    """Proxy diagnostics for one Hamiltonian.

    lambda_L is the slope of log C(t) over the early-time window in which the
    OTOC lies between 1e-8 and 0.1, that is, after it has risen above numerical
    noise and before it saturates. E_cap defaults to the spectral bandwidth.
    """
    DeltaE = dynamical_energy_scale(H, psi0)
    eigs = np.linalg.eigvalsh(H)
    bandwidth = float(eigs[-1] - eigs[0])
    if E_cap is None:
        E_cap = bandwidth

    otoc_vals = otoc_series(H, W_op, V_op, psi0 / np.linalg.norm(psi0), ts_otoc)
    mask = (otoc_vals > 1e-8) & (otoc_vals < 0.1)
    slope = (np.polyfit(ts_otoc[mask], np.log(otoc_vals[mask]), 1)[0]
             if mask.sum() > 3 else 0.0)
    lambda_L = float(max(slope, 0.0))

    tau_ML = np.pi / (2 * DeltaE) if DeltaE > 0 else np.inf
    log2_b = lambda_L * tau_ML / np.log(2)
    chi = eta_b * (DeltaE / E_cap) * log2_b
    tau_cap = (np.pi ** 2 / np.log(2)) / chi if chi > 0 else np.inf
    return {"DeltaE": DeltaE, "bandwidth": bandwidth, "lambda_L": lambda_L,
            "tau_ML": float(tau_ML), "log2_b": float(log2_b),
            "chi": float(chi), "tau_cap": float(tau_cap)}


def table7(n_disorder=12, Ws=(1, 2, 3, 4), N=6, J=1.0, ts=None):
    """Regenerate the OTOC proxy diagnostics of Table 7.

    Disordered rows are means over `n_disorder` realizations, seeds
    0..n_disorder-1, with plain (ddof=0) realization standard deviations.
    Lambda_OTOC is formed from the column means, lambda_L_bar * tau_ML(E_dyn_bar)
    / ln 2, so it is the index of the mean chain rather than the mean of the
    per-realization indices; the two differ only where E_dyn varies across
    realizations, that is, for the random-coupling variant.
    """
    if ts is None:
        ts = np.linspace(0.1, 5.0, 30)
    psi0 = initial_state_domain_wall(N)
    W_op, V_op = site_op(SZ, 0, N), site_op(SZ, N - 1, N)

    builders = {
        "random_fields": lambda W, s: build_H_disordered(N, J, W, seed=s)[0],
        "random_couplings": lambda W, s: build_H_random_couplings(N, J, W, seed=s)[0],
        "heisenberg_random_fields":
            lambda W, s: build_H_heisenberg_random_fields(N, J, W, seed=s)[0],
    }

    ordered = compute_chi_for_H(build_H_ordered(N, J), psi0, ts, W_op, V_op)
    lam_ord = ordered["lambda_L"]
    rows = {"ordered": ordered}

    for name, build in builders.items():
        for W in Ws:
            rs = [compute_chi_for_H(build(W, s), psi0, ts, W_op, V_op)
                  for s in range(n_disorder)]
            lam = float(np.mean([r["lambda_L"] for r in rs]))
            ed = float(np.mean([r["DeltaE"] for r in rs]))
            ratios = np.array([r["lambda_L"] for r in rs]) / lam_ord
            rows[f"{name}_W{W}"] = {
                "E_dyn": ed, "lambda_L": lam,
                "Lambda_OTOC": lam * (np.pi / (2 * ed)) / np.log(2),
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios)),
            }
    return rows


if __name__ == "__main__":
    import json

    N, J = 6, 1.0
    psi0 = initial_state_domain_wall(N)
    W_op, V_op = site_op(SZ, 0, N), site_op(SZ, N - 1, N)
    ts = np.linspace(0.1, 5.0, 30)

    # the spectral scan must agree with the direct expm definition
    H0 = build_H_ordered(N, J)
    p0 = psi0 / np.linalg.norm(psi0)
    direct = np.array([otoc(H0, W_op, V_op, p0, t) for t in ts])
    assert np.allclose(direct, otoc_series(H0, W_op, V_op, p0, ts), atol=1e-12)

    rows = table7()
    o = rows["ordered"]
    print("Ordered baseline (Table 7 first row, Table 8 first row):")
    for k in ("DeltaE", "bandwidth", "lambda_L", "log2_b", "chi", "tau_cap"):
        print(f"  {k:10s} = {o[k]:.4f}")

    print("\nDisordered rows: E_dyn, lambda_L, Lambda_OTOC, ratio +- std")
    for k, r in rows.items():
        if k == "ordered":
            continue
        print(f"  {k:28s} {r['E_dyn']:.3f}  {r['lambda_L']:.2f}  "
              f"{r['Lambda_OTOC']:5.2f}  {r['ratio_mean']:.2f}+-{r['ratio_std']:.2f}")

    with open("toy_model_results.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("\nwrote toy_model_results.json")
