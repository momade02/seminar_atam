# The Incremental Value of Deep Quantile Networks for VaR

### Do neural networks add value to a realistic volatility engine?

Seminar paper, B560 *Advanced Topics in Asset Management* - University of Tübingen,
Chair of Finance (Prof. Dr. Monika Gehde-Trapp, M.Sc. Tom Ernst). Submitted January 2026.

**Result:** Conditional on a well-specified EGARCH(1,1)-skew-t volatility
engine, a compact Deep Quantile Network does *not* improve one-day VaR forecasts for the
EURO STOXX 50. Additionally, it degrades the economic interpretability of the volatility-to-VaR
mapping.

---

## Motivation

Most published evidence for neural VaR models feeds the network high-dimensional predictor
sets, so it implicitly performs both the volatility modelling *and* the quantile mapping.
Risk desks are not in that position: they already run validated volatility engines and would
realistically only swap out the final mapping layer.

This paper isolates that layer. A single EGARCH(1,1)-skew-t model is estimated once on
pre-test data and then **frozen**. All three VaR engines see exactly the same
one-dimensional volatility state, so any performance difference is attributable to the
functional form of the mapping and nothing else.

### Research questions

1. **RQ1 — Accuracy.** Given a one-dimensional volatility state, can a compact DQN
   materially improve one-day VaR forecasts (hit rates, coverage tests, pinball loss) over a
   semi-parametric EGARCH benchmark and a linear quantile regression?
2. **RQ2 — Tail severity.** How do the engines differ in realized shortfall and tail-severity
   metrics? Do VaR gains come at the cost of deeper, more concentrated tail losses?
3. **RQ3 — Plausibility.** Is the DQN's mapping from volatility to VaR monotone and smooth,
   or irregular in ways that conflict with standard volatility-risk intuition?

## Repository contents

| File | Description |
| --- | --- |
| `seminar_paper.pdf` | Full paper (13 pages) with all tables, figures and references. |
| `code_notebook.ipynb` | Complete analysis: data prep, diagnostics, three VaR engines, backtests, interpretability plots. 78 cells, outputs included. |
| `data_eur_stoxx_50.csv` | EURO STOXX 50 daily OHLC, 1996–2024. |
| `README.md` | This file. |

## Data

EURO STOXX 50 price index, daily close-to-close, **2 January 1996 – 31 December 2024**.
7,450 price observations → **7,449 daily log-returns**.

Semicolon-delimited with comma decimal separator; the notebook handles the conversion on
load. Columns: `date`, `close`, `open`, `high`, `low`. Only `close` is used.

*Provider: — to be added.*

**Chronological split, no shuffling:**

| Split | Period | Observations | Used for |
| --- | --- | --- | --- |
| Pre-test | 1996-01-02 – 2015-12-31 | 5,138 | EGARCH estimation, residual quantiles, LQR fitting, DQN training and CV |
| Test | 2016-01-04 – 2024-12-31 | 2,311 | Backtesting and all diagnostics |

The test period is touched only at evaluation time. Volatility standardisation uses pre-test
moments, so no future information enters the state variable.

### Why EGARCH-skew-t

Preliminary diagnostics motivate the specification rather than assuming it:

| Property | Test | Result |
| --- | --- | --- |
| Stationarity | ADF | t = −14.89, p < 0.0001 |
| Heavy tails | Jarque–Bera | 11,152, p < 0.0001 (skew −0.23, kurtosis 8.98) |
| Volatility clustering | ARCH-LM (lag 10) | 1,232.67, p < 0.0001 |
| Leverage | Engle–Ng sign/size bias | F = 35.11, p < 0.0001 |

Stationary, heavy-tailed, conditionally heteroscedastic and asymmetric — which is exactly
the case for EGARCH with skewed Student-t innovations.

## Method

### The frozen backbone

`arch_model(mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="skewt")` is fitted by maximum
likelihood on the pre-test sample (5,138 obs). Parameters are then **fixed** and the full
sample is filtered with `.fix()` to recover conditional volatility σ̂ₜ across the whole period
without refitting.

The single state variable is standardised log-volatility:

```
x_σ,t = (log σ̂_t − μ_pre) / s_pre
```

with `μ_pre` and `s_pre` computed on pre-test data only.

### Three engines, one state

All estimate one-day VaR at **α ∈ {1%, 2.5%, 5%}**.

| Engine | Mapping | Estimation |
| --- | --- | --- |
| **Semi-parametric EGARCH** (benchmark) | `VaR = μ̂ + σ̂_t · q̂_α` | Empirical α-quantile of pre-test standardized residuals (filtered historical simulation) |
| **LQR** | `Q_α(r_t \| x_σ,t) = β₀ + β₁ x_σ,t` | `statsmodels.QuantReg`, pinball loss on pre-test |
| **DQN** | Feed-forward MLP, one network per α | PyTorch, Adam, pinball loss + monotonicity penalty |

The LQR is the bridge model: if the DQN cannot beat it, non-linearity in the volatility–VaR
relationship is not economically meaningful.

### DQN configuration

Deliberately compact, to avoid handing the network enough capacity to overfit a
one-dimensional input:

- **Grid:** hidden dim ∈ {32, 64} × layers ∈ {1, 2} × lr ∈ {1e-3, 3e-4, 1e-4}
- **Selection:** 4-fold expanding-window time-series CV on the pre-test sample, scored on
  *unpenalized* pinball loss; early stopping (patience 20, max 200 epochs)
- **Regularisation:** L2 weight decay 1e-4, gradient clipping at norm 1.0, and a
  differentiable **monotonicity penalty** (λ = 0.10) penalising positive ∂VaR/∂x —
  economically implausible slopes
- **Final fit:** best configuration retrained on the full pre-test sample for the
  CV-averaged epoch count

All three α levels independently selected the same architecture — **64 hidden units,
2 layers**:

| α | Architecture | Learning rate | Epochs | CV pinball |
| --- | --- | --- | --- | --- |
| 1.0% | 64 × 2 | 3e-4 | 56 | 0.000405 |
| 2.5% | 64 × 2 | 1e-3 | 40 | 0.000870 |
| 5.0% | 64 × 2 | 1e-3 | 32 | 0.001510 |

Because the three networks are trained independently, raw forecasts can cross
(VaR₁% > VaR₅%). Crossings are negligible for the benchmark and LQR; for the DQN they are
corrected by row-wise sorting (Chernozhukov et al., 2010), which preserves the set of
predicted risks while enforcing VaR₁% ≤ VaR₂.₅% ≤ VaR₅%.

### Evaluation, in three layers

1. **Statistical adequacy** — Kupiec LR_uc (unconditional coverage), Christoffersen LR_cc
   (joint coverage and independence), Engle–Manganelli Dynamic Quantile test.
2. **Forecast accuracy** — pinball loss as a proper scoring rule; Diebold–Mariano tests with
   Newey–West HAC standard errors on the loss differential.
3. **Economic diagnostics** — realized shortfall and tail-severity quantiles as ES proxies;
   the volatility-to-VaR "money plot" with data-scarce regions shaded; numerical gradients
   ∂VaR/∂x to test monotonicity; volatility-bucketed performance including crisis bucket B5.

Layer 3 is the point of the paper. Backtests establish adequacy but say nothing about whether
a model is defensible under model-risk governance.

## Results

### Coverage and accuracy (2,311 out-of-sample days)

DM statistic tests H₀: L_benchmark = L_model. **Negative ⇒ the model loses to the benchmark.**
Bold marks rejection at the 5% level.

| Model | α | Hit rate | LR_uc | LR_cc | DQ | Pinball | DM vs benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Benchmark** | 1.0% | 1.17% | 0.428 | 0.531 | 0.086 | 4.12e-4 | — |
| | 2.5% | 2.55% | 0.871 | 0.900 | **0.023** | 7.89e-4 | — |
| | 5.0% | 4.15% | 0.055 | 0.137 | 0.700 | 12.81e-4 | — |
| **LQR** | 1.0% | 1.04% | 0.853 | 0.764 | **0.014** | 4.25e-4 | −1.00 (p = 0.32) |
| | 2.5% | 2.60% | 0.768 | 0.849 | **0.000** | 8.18e-4 | **−2.65 (p = 0.008)** |
| | 5.0% | 4.93% | 0.882 | 0.972 | **0.000** | 13.24e-4 | **−3.74 (p < 0.001)** |
| **DQN** | 1.0% | 1.64% | **0.004** | **0.009** | 0.078 | 4.28e-4 | −1.71 (p = 0.088) |
| | 2.5% | 2.94% | 0.185 | 0.415 | **0.028** | 7.86e-4 | 0.44 (p = 0.661) |
| | 5.0% | 4.24% | 0.086 | 0.228 | **0.005** | 12.96e-4 | **−2.36 (p = 0.018)** |

The benchmark passes both coverage tests at every level. The LQR gets hit rates right but
fails every DQ test and loses significantly on pinball loss at 2.5% and 5% — accurate
frequency, misspecified dynamics. The DQN fails coverage outright at 1% (1.64% against a 1.0%
nominal target), fails DQ at 2.5% and 5%, and beats the benchmark on loss only once,
insignificantly.

### Tail severity

| Model | α | Realized shortfall | Median violation | Violation Q10 |
| --- | --- | --- | --- | --- |
| Benchmark | 1.0% | −3.69% | −3.03% | −6.45% |
| LQR | 1.0% | −3.86% | −3.18% | −7.64% |
| DQN | 1.0% | −3.28% | −2.64% | −4.70% |

The DQN's *milder* breaches at 1% are not better tail protection. It generates more
exceedances (1.64% vs 1.17%), so its breach set includes more moderate loss days — a
**breach-set selection effect**, not improved tail-risk efficiency. Tables 3 and 4 have to be
read jointly; conditional severity alone is not a ranking criterion.

> **Caveat on the crisis bucket.** In the highest-volatility bucket B5 at α = 1% there are
> only 2 exceedances in 206 observations, occurring on the same March-2020 dates for all
> three models. B5 hit rate and realized shortfall are therefore mechanically identical
> across models — computed from the same two returns. This does not mean the VaR forecasts
> agree; it means the effective tail sample is too small to support inference there.

### Mapping plausibility (RQ3)

The decisive result. At α = 2.5%, the benchmark traces a stable, monotone lower envelope of
the return cloud and the LQR gives a linear approximation. The DQN, even with the
monotonicity penalty active, bends erratically in intermediate-to-high volatility regions.
Its numerical gradient repeatedly jumps and intermittently crosses zero — implying states
where *higher* volatility maps to a *less* negative VaR.

The scatter overlay explains why: the heteroscedasticity cone means tail observations thin
out precisely in the high-volatility range where the network has to extrapolate. It is
fitting noise where it matters most.

### Conclusion

Under a fixed one-dimensional EGARCH state, the DQN delivers no robust improvement, and its
irregular mapping is hard to defend under model-risk governance. The takeaway is not that
DQNs are useless for VaR — Chronopoulos et al. (2023) find gains with high-dimensional
predictor sets — but that in a fixed-engine architecture, where only the mapping layer is
replaced, the incremental value is small relative to the interpretability and governance
burden.

## Running the notebook

Python 3.12. No pinned environment is committed; install current versions:

```bash
pip install numpy pandas scipy statsmodels arch torch matplotlib seaborn
jupyter lab code_notebook.ipynb
```

Run **top to bottom** — cells share state (`features`, `res_pre`, the `alpha_to_col_*` dicts)
and the later sections will fail on a cold kernel. `data_eur_stoxx_50.csv` is read by
relative path, so start Jupyter from the repository root.

Seeds are fixed at 42 for NumPy and PyTorch, with cuDNN determinism enabled when CUDA is
available. The DQN search is the slow step: 12 configurations × 4 CV folds × 3 α levels =
144 training runs plus 3 final fits. CPU is sufficient for a one-dimensional input.

The notebook writes no files — figures and tables render inline, and the committed outputs
are the record of the reported results.

## Notes and limitations

**Stated in the paper.** One index, one-day horizon, one volatility specification, a
one-dimensional state, and a modest hyperparameter search. These are design choices rather
than oversights: constraining the DQN to the same information set as the benchmark is what
makes the comparison a test of the *mapping layer* instead of the information set. The paper
notes explicitly that this limits the DQN's potential.

**Not addressed.** Expected Shortfall is evaluated only through realized-shortfall proxies,
not a formal ES backtest — a gap given that the regulatory motivation in the introduction is
the FRTB shift from VaR to ES. Longer horizons, multiple indices, and a rolling-refit
volatility engine are all left open.

**Reproducibility gaps.** No `requirements.txt` or lockfile, so package versions are not
pinned. Nothing is serialised — no saved figures, fitted models or forecast series — so every
number in the paper requires a full rerun to regenerate. Determinism holds for a clean
top-to-bottom execution; re-running individual training cells out of order advances the RNG
state and shifts results.

## Key references

- Chronopoulos, I., Raftapostolos, A. & Kapetanios, G. (2023). Forecasting Value-at-Risk using deep neural network quantile regression. *Journal of Financial Econometrics*, 22(3), 636–669.
- Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347–370.
- Hansen, B. E. (1994). Autoregressive conditional density estimation. *International Economic Review*, 35(3), 705–730.
- Barone-Adesi, G., Giannopoulos, K. & Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5), 583–602.
- Chernozhukov, V., Fernández-Val, I. & Galichon, A. (2010). Quantile and probability curves without crossing. *Econometrica*, 78(3), 1093–1125.
- Engle, R. F. & Manganelli, S. (2004). Dynamic quantile tests, in: CAViaR — conditional autoregressive value at risk by regression quantiles. *Journal of Business & Economic Statistics*, 22(4), 367–381.

Full bibliography in `seminar_paper.pdf`.
