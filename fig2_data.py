"""Per-step coherence number and effective branch number for Figure 2(b).
Disorder averages are taken on log2, matching the per-realization growth rate."""
import json
import numpy as np
import history_check as hc

NSTEP = 8
out = {}
r = hc.history_run(hc.h_xx([hc.J] * (hc.N - 1)), nstep=NSTEP, keep_vectors_upto=NSTEP)
out["ordered"] = {
    "log2_N_coh": [float(np.log2(hc.coherence_number(r["vectors"][n]))) for n in range(1, NSTEP + 1)],
    "H_n": [float(h) for h in r["H"]],
}
L, LH = [], []
for seed in range(12):
    rng = np.random.default_rng(seed)
    H = hc.h_xx([hc.J] * (hc.N - 1), fields=rng.uniform(-4.0, 4.0, hc.N))
    rr = hc.history_run(H, nstep=NSTEP, keep_vectors_upto=NSTEP)
    L.append([float(np.log2(hc.coherence_number(rr["vectors"][n]))) for n in range(1, NSTEP + 1)])
    LH.append([float(h) for h in rr["H"]])
out["rf_W4"] = {"log2_N_coh": np.mean(L, axis=0).tolist(), "H_n": np.mean(LH, axis=0).tolist()}
json.dump(out, open("fig2_data8.json", "w"), indent=1)
for k, v in out.items():
    d = np.diff(v["log2_N_coh"])
    print(k, "log2 N_coh", np.round(v["log2_N_coh"], 2).tolist())
    print(k, "late slope (mean of last 3 increments)", round(float(np.mean(d[-3:])), 3))
