# Growth Microscope — can we explain what grew?

A deterministic, reproducible browser instrument for **causal explanations of task-grown conductive networks**, following [GrowingMatrix](https://github.com/anttiluode/GrowingMatrix), [Unruh](https://github.com/anttiluode/Unruh), [Limasieni](https://github.com/anttiluode/Limasieni), and the methodological challenge raised by [Goodfire's *Neural Networks Are Already Clean*](https://already-clean.goodfire.com/).

**[Open the browser experiment](https://anttiluode.github.io/Growth-Microscope/)** (no install or backend). Or serve locally: `python -m http.server 8000` and open `http://localhost:8000`.

## Why this exists

Growing a network that solves a task is not the same as understanding the role of its components. We test a narrower, falsifiable question: **given a trained network, can an explanation predict the consequence of disrupting a region on unseen but related tasks, and outperform simpler weight/activity controls?** The software reveals negative results rather than claiming every causal fingerprint is superior.

There are three pre-defined *needs*: **Bridge** (horizontal routes), **Fork** (one source, three sinks), **Cross** (diagonal and horizontal demands). A single 11×9 lattice has 178 weak candidate edges and no node birth. Each need includes **three training source–sink pairs** and three *different*, spatially shifted held-out pairs; test pairs are related demands, not new task families or independent random samples. Training starts from identical material.

## Run locally

Node 20+ is used for tests and CLI; the static browser app runs with modern JavaScript and no dependencies.

```sh
npm test
npm run bench
npm run bench -- 640
node benchmark.js 320 --save
```

`benchmark-results.json` is ignored by git. The browser offers 40-step growth, 320-step target, pause/auto, checkpoint restore, region ablation analysis, frozen predictions, reveal, and complete JSON experiment export.

## Equations and interventions

The fast field is the passive Kirchhoff response on a lattice with no diagonals:

\[
L(h)=B\operatorname{diag}(g_0+h)B^T,\qquad L(h)v=b,\quad g_0=0.06.
\]

We fix one node's potential as gauge and solve the grounded reduced Laplacian with diagonally preconditioned conjugate gradients. For each demand (source `+1`, sink `-1`), the task loss is effective resistance \(R=b^Tv=v_s-v_t\). The network grows additive edge conductances under a strict budget \(\sum h_e\le24\) by an exact gradient

\[
\frac{\partial R}{\partial h_e}=-(v_a-v_b)^2.
\]

The normalized growth step is projected into \(0\le h_e\le2\) using a bisection dual threshold. Every objective and held-out resistance is calculated using the full network. Train and held-out data are disjoint *terminal configurations* but strongly related within the same structural need.

The **microscope** divides edges by midpoint into 12 pre-specified rectangular regions (4 columns × 3 rows). At a checkpoint, each region is tested by cutting **50% of its *recruited* conductance**, retaining the fixed baseline scaffold. The measured harm is the *increase* in mean resistance.

We construct four predictions, each calibrated by only the **training** intervention effects (except the fingerprint, which is directly transferred):

| Prediction | Knows | Does not know |
|---|---|---|
| **Train causal fingerprint** | Actual 50%-region ablation responses on train terminal pairs | Any held-out field or held-out cut |
| **Weight only** | Total grown material in region, and training-cut calibration | Terminal-specific test field |
| **Held-out activity** | Current magnitude in each region from one *uncut held-out electrical probe*, and training-cut calibration | Held-out cut results |
| **Held-out adjoint sensitivity** | Exact first-order held-out gradient measured without any damage (plus training calibration) | Held-out finite-cut outcomes |

After prediction, reveal the 12 **actual** held-out 50%-cut responses. Score Pearson correlation, RMSE, and how many of the top three most damaging regions are identified. A first-order derivative is expected to be strong on moderate interventions because the entire passive network law is already known. This is a structural sanity check, **not blind semantic interpretation or automatically discovered circuits**.

The browser's per-region functional fingerprint has three bars, corresponding to the three **training** pairs; a region is labelled `Shared` if every pair's harm is at least 60% of its largest pair's harm, or `Need 1/2/3` for the dominant pair. Regions are manually selected: a `Shared` label is not evidence of a learned transferable neuron concept.

## Pre-recorded baseline (320 steps, deterministic)

Run `npm run bench` to reproduce. A simple material-only predictor can nearly match causal fingerprints on Bridge and is **better on Cross**, which is a meaningful negative control.

| Need | Held-out Pearson r: train fingerprint | Weight-only r | Activity r | Adjoint r |
|---|---:|---:|---:|---:|
| Bridge | 0.987 | 0.986 | 0.987 | 0.993 |
| Fork | 0.951 | 0.942 | 0.984 | 0.992 |
| Cross | 0.825 | **0.849** | 0.888 | 0.999 |

**A sharper negative finding:** moved-endpoint generalization gets worse with prolonged growth. The minimum held-out resistance occurs at step 10 for Bridge (5.727 vs 10.101 at step 320) and Fork (6.407 vs 9.746), and at step 5 for Cross (10.747 vs 14.530). These minima were found *post hoc on the held-out curves*, so they are not deployable early-stopping rules without a separate validation split. They demonstrate how quickly task-specialized conductance can become brittle under small endpoint shifts.\n\nThese are **one deterministic lattice**, 12 correlated regions and chosen endpoint configurations, not an estimated generalization distribution. The adjoint's advantage is expected because it receives held-out unablated dynamics that the pure train fingerprint does not. The expected next test is held-out demand families, multiple topology seeds, causal region discovery independent of rectangular bins, and interventions on nonlinear recurrent states.

## Relation to yesterday's work

- **GrowingMatrix** optimizes the substrate. This project independently reconstructs a small version of its passive routing setup specifically for interpretation and evaluation. Its 11×9 lattice is **not** a direct re-run of GrowingMatrix's 17×11 source or its numerical results.
- **Limasieni** shows why ports and observation geometry determine the measured operator. Here each probe changes source/sink, and the exposed per-edge meaning changes accordingly. **No transformer or attention operator is trained.**
- **Unruh** studies a slow state that feeds back on the fast field; this v1 has structural plasticity during training, but the trained network is **static** during held-out tests.
- **Goodfire** motivates tracking native components and causal roles before replacing them with abstract features. This is a controllable illustration, not a replication of a trained vision network or evidence about neuron interpretability.

**The scientific claim is deliberately narrow:** exact physics plus controlled interventions can determine when an explanation of a grown network predicts unseen related consequences—and when a simpler explanation does just as well.
