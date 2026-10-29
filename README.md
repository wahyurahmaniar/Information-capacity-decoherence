# Information Capacity Decoherence

Code reproducing the numerical results and figures of

> W. Rahmaniar and B. Ramzan,
> Information Capacity Decoherence: A Temporal-Memory Bound on Quantum Coherence,
> submitted to Annals of Physics.

Everything here is exact diagonalization of six-site spin-1/2 chains together
with the decoherent-histories construction defined in the paper. There are no
fitted parameters and no experimental data. The scripts depend only on NumPy,
SciPy and Matplotlib.

## Install and run

```bash
pip install -r requirements.txt

python history_check.py > history_check.json   # Table 8, and inputs to Figures 2(b) and 4
python fig2_data.py                            # per-step coherence numbers for Figure 2(b)
python lindblad_history_check.py               # history-space image of the generator, Section 6.4
python icd_toy_model.py                        # Table 7, the OTOC proxy diagnostics
python numcheck_r2.py                          # 20 checks of the analytic statements

python make_figure1.py   # new_fig1.eps/.png   capacity constraint and its completion (schematic)
python make_figure2.py   # new_fig2.eps/.png   decay law, and unconstrained coherence-number growth
python make_figure3.py   # new_fig3.eps/.png   four decoherence timescales against mass
python make_figure4.py   # new_fig4.eps/.png   history quantities against the OTOC proxy index
python make_figure5.py   # new_fig5.eps/.png   branching rate and complexity factor against disorder
```

`history_check.py` and `fig2_data.py` each take a few minutes, since they
propagate branch vectors for eight and seven coarse-grainings over twelve
disorder realizations. Their results are committed as `history_check.json` and
`fig2_data.json`, so the figure scripts can be run on their own.

## What each file does

| File | Purpose |
|---|---|
| `history_check.py` | Branch vectors of the left-half magnetization family at intervals `tau_ML`; per-step support count `N_hist`, history entropy `H_n`, per-step increment, Renyi-1/2 entropy, coherence number. Produces Table 8 and the ceilings of Figure 4(a). |
| `fig2_data.py` | Coherence number and effective branch number per coarse-graining step, for the ordered chain and the random-field chain at `W/J = 4`. Disorder averages are taken on `log2`, so the quoted growth rate is the mean per-realization rate. |
| `lindblad_history_check.py` | Applies the dephasing generator of Eq. (38) to the ordered chain and compares the resulting history-space decoherence functional with the unitary one, giving the suppression factors quoted in Section 6.4. |
| `icd_toy_model.py` | Out-of-time-order correlator, Loschmidt echo and dynamical energy scale for the four Hamiltonians; the proxy branching factor `log2 b = lambda_L tau_ML / ln 2`, the complexity factor and `tau_cap`. Produces Table 7 and writes `toy_model_results.json`. |
| `numcheck_r2.py` | Independent re-derivation of every quantitative statement in the paper: the chain-rule ceiling of Lemma 2 on random families, Proposition 1's bracket, Proposition 2's purity bound on random states, the pairwise visibilities, the calibration constants, and the arithmetic of Tables 7 and 8. Prints `ALL PASS`. |
| `make_figure1.py` … `make_figure5.py` | The five figures, each written as both `.eps` (for the journal) and `.png`. Figure text is Times New Roman where available, falling back to a metric-compatible serif. |

## Model

Six sites, open boundaries, `J = 1`, initial state the domain wall
`|up up up down down down>`. Four Hamiltonians:

- ordered XX chain,
- XX chain with random longitudinal fields, `h_i ~ U[-W, W]`,
- XX chain with random couplings,
- Heisenberg chain with random fields.

The dynamical energy is the energy uncertainty in the initial state, and the
sampling interval is `tau_ML = pi hbar / (2 E_dyn)`.

In `icd_toy_model.py`, `lambda_L` is the slope of `log C(t)` over the window in
which the OTOC lies between `1e-8` and `0.1`, that is, above numerical noise and
below saturation. `otoc` evaluates the definition directly with a matrix
exponential and `otoc_series` evaluates it from the spectral decomposition; the
two agree to machine precision, and the script asserts this on startup.

## Reproducibility

Disorder averages run over twelve realizations, seeds 0 to 11, so every number
in the paper is reproducible exactly. Standard deviations are plain `numpy.std`
(`ddof = 0`) throughout.

The two determinations use different random generators:

- `history_check.py`, `fig2_data.py` and `lindblad_history_check.py` use
  `numpy.random.default_rng(seed)`.
- `icd_toy_model.py` uses the legacy `numpy.random.seed(seed)`.

The random-field and Heisenberg variants hold `E_dyn` at the ordered value for
every realization, so the generator makes no difference to them. The
random-coupling variant draws its couplings from the stream, so its mean
`E_dyn` differs by a few per cent between Table 7 and Table 8. The paper refers
to these as independent random streams.

## Consistency checks

`make_figure5.py` plots the published values of Tables 7 and 8 as written in the
paper. Before drawing, it compares each of them against `toy_model_results.json`
and `history_check.json` and reports the outcome:

```
  Table 7 (rf): checked against toy_model_results.json, OK
  Table 8 (rf): checked against history_check.json, OK
```

Any entry that differs from the computed value by more than the printed
precision stops the script and is listed. Running `icd_toy_model.py` and
`history_check.py` first enables both checks; if a results file is absent the
script says so and draws the figure anyway. The script additionally verifies
that every complexity factor reproduces `(E_dyn / E_cap) * h_hist * tau_ML`.

`numcheck_r2.py` is the independent check of the analytic statements and of the
table arithmetic, and prints `ALL PASS`.

## License

MIT. See `LICENSE`.
