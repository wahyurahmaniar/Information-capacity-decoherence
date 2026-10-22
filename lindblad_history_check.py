"""History-space image of the dephasing generator, Eq. (39), for the ordered chain.
Compares |D_open(a,a')| / |D_unitary(a,a')| over off-diagonal history pairs."""
import numpy as np, importlib.util, itertools
from scipy.linalg import expm
spec = importlib.util.spec_from_file_location("hc", "history_check.py"); hc = importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)
S = hc.SECTOR; d = len(S)
H = hc.h_xx([1.0]*5)[np.ix_(S, S)]
psi = hc.psi0[S]
P = [np.diag(p[S]) for p in hc.PROJ_DIAG]
P = [p for p in P if np.trace(p).real > 0]
tau = np.pi
I = np.eye(d)
def liouv(G):
    # vec(rho) column-stacking: vec(A rho B) = (B^T kron A) vec(rho)
    L = -1j*(np.kron(I, H) - np.kron(H.T, I))
    D = sum(np.kron(p.T, p) for p in P) - np.kron(I, I)
    return L + G*D
def run(G, n):
    V = expm(liouv(G)*tau)
    rho0 = np.outer(psi, psi.conj())
    # recursive: operators X_{a,a'} for histories
    hist = {((),()): rho0}
    for k in range(n):
        new = {}
        for (a, ap), X in hist.items():
            Y = (V @ X.reshape(-1, order='F')).reshape(d, d, order='F')
            for i, pi in enumerate(P):
                for j, pj in enumerate(P):
                    Z = pi @ Y @ pj
                    if np.abs(Z).max() > 1e-14:
                        new[(a+(i,), ap+(j,))] = Z
        hist = new
    D = {k: np.trace(v) for k, v in hist.items()}
    return D
n = 3
Du = run(0.0, n)
for G in [0.1, 0.3]:
    Do = run(G, n)
    ratios = {}
    for (a, ap), val in Du.items():
        if a != ap and abs(val) > 1e-6:
            r = abs(Do.get((a, ap), 0))/abs(val)
            ham = sum(x != y for x, y in zip(a, ap))
            ratios.setdefault(ham, []).append(r)
    print("Gamma", G, "exp(-G*tau)=", round(np.exp(-G*tau), 4))
    for h in sorted(ratios):
        arr = np.array(ratios[h])
        print("  Hamming", h, "pairs", len(arr), "ratio min/median/max", round(arr.min(), 4), round(np.median(arr), 4), round(arr.max(), 4))

print("---- physical rate for the ordered chain ----")
G = np.log(2)*1.2011536530959628/np.pi
print("Gamma_ICD =", G, "Gamma*tau =", G*np.pi, "uniform factor per interval", np.exp(-G*np.pi))
Do = run(G, 3)
ratios = {}
l1u = l1o = 0.0
for (a, ap), val in Du.items():
    if a != ap:
        l1u += abs(val); l1o += abs(Do.get((a, ap), 0))
        if abs(val) > 1e-6:
            ham = sum(x != y for x, y in zip(a, ap))
            ratios.setdefault(ham, []).append(abs(Do.get((a, ap), 0))/abs(val))
for h in sorted(ratios):
    arr = np.array(ratios[h])
    q = np.percentile(arr, [10, 50, 90])
    print(" Hamming", h, "pairs", len(arr), "10/50/90 pct", np.round(q, 3), "min/max", round(arr.min(), 4), round(arr.max(), 3))
print(" aggregate l1 ratio", l1o/l1u)
