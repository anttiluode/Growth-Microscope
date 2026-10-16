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

These are **one deterministic lattice**, 12 correlated regions and chosen endpoint configurations, not an estimated generalization distribution. The adjoint's advantage is expected because it receives held-out unablated dynamics that the pure train fingerprint does not. The expected next test is held-out demand families, multiple topology seeds, causal region discovery independent of rectangular bins, and interventions on nonlinear recurrent states.

## Relation to yesterday's work

- **GrowingMatrix** optimizes the substrate. This project independently reconstructs a small version of its passive routing setup specifically for interpretation and evaluation. Its 11×9 lattice is **not** a direct re-run of GrowingMatrix's 17×11 source or its numerical results.
- **Limasieni** shows why ports and observation geometry determine the measured operator. Here each probe changes source/sink, and the exposed per-edge meaning changes accordingly. **No transformer or attention operator is trained.**
- **Unruh** studies a slow state that feeds back on the fast field; this v1 has structural plasticity during training, but the trained network is **static** during held-out tests.
- **Goodfire** motivates tracking native components and causal roles before replacing them with abstract features. This is a controllable illustration, not a replication of a trained vision network or evidence about neuron interpretability.

**The scientific claim is deliberately narrow:** exact physics plus controlled interventions can determine when an explanation of a grown network predicts unseen related consequences—and when a simpler explanation does just as well.

## Gate 2 — prospective model selection and causal explanation transfer (October 16)

**[Open Gate 2 in the browser](https://anttiluode.github.io/Growth-Microscope/gate2.html)** or run `npm run bench:gate2` and `npm test`.

V1 reported a large, visually legible specialization/versatility trade-off but inspected a single three-pair held-out set while describing it. Gate 2 removes that selection mistake with a **predefined train / augmentation / validation / stress / random-sweep separation**. The original V1 held-out pairs now become **validation only**. Three new stress pairs and a deterministic seed-20261016 24-pair disjoint distribution form independent final tests. The source code contains every endpoint, so blinding here is *procedural*, not cryptographic.

Two networks start with identical weak scaffolds and use the **same 24-unit material cap**, learning rate and exactly **three Kirchhoff solves per training update**:

* `specialist`: repeated exact three original training pairs;
* `diverse`: deterministically cycles through 7 training/augmentation endpoint pairs, drawing 3 per update. It has broader experience but not more field solves.

The selection protocol is fixed before test: evaluate checkpoints at 0, 10, 20, 40, 80, 160, 320 steps (up to selected horizon), choose **lowest validation resistance**, freeze the chosen conductances, and only then evaluate test resistance. Thus a favorable early step is no longer cherry-picked from test. Both arms also receive an equal-*actual*-material uniform control. The uniform baseline remains a weak optimization competitor and no result claims superior topology optimization.

### Frozen reference observations (320-step runs)

| Need | Specialist validation-picked step | Specialist 24-case test ↓ | Specialist 320-step test ↓ | Diverse picked step | Diverse 24-case test ↓ | Diverse 320-step test ↓ | Equal-material uniform test ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bridge | 10 | 11.062 | 17.224 | 10 | 10.298 | 14.673 | **9.937** |
| Fork | 10 | **9.238** | 13.522 | 40 | 10.470 | 11.098 | 10.022 |
| Cross | 10 | 14.493 | 15.618 | 10 | 12.391 | 14.522 | **11.715** |

Lower means better transport. **Validation-based stopping beats the final 320-step specialist on all three tasks**, confirming measurable late specialization on this selected demand distribution, but neither diverse training nor early stopping consistently beats uniform conductive material on the broad unseen split. These are three deterministic toy tasks, not three random seeds or an empirical confidence interval.

### Damage explanations, genuinely frozen before reveal

Gate 2 tests a held-out 50% region-conductance cut on **24 unseen endpoint pairs** using only predictions created from the *original training pairs*. Four training-only explanations compete: exact finite **train causal effect**, regional material, training current flow, and first-order training sensitivity. Scalar predictor gains are calibrated on the training cut effects, **never** on test. The browser **Commit predictions** step saves these estimates plus a digest of the precise conductance state. **Reveal new tests** then measures actual cut effects and scores correlation, RMSE, and top-3 region overlap; it rejects intervening changes to the state. The gate never reports statistical significance from 12 predefined spatial regions.

The input task, physics equations, and rectangular intervention regions are still supplied. This is a controlled causal-attribution laboratory, not discovery of semantic neurons. An improved next gate should learn candidate groups from unlabeled behavior and test whether they predict intervention effects beyond simpler structural and gradient models in unrelated task families; strong topology optimizers and cost-matched adaptive routing remain necessary baselines.

## Gate 3 (October 16): Discover candidate mechanisms before naming them

**[Open Gate 3 — functional groups](https://anttiluode.github.io/Growth-Microscope/gate3.html)** | run locally with `python -m http.server 8000` and visit `/gate3.html`.

Gate 2 prescribed 12 rectangular spatial regions. Gate 3 instead asks whether **edge groups inferred from causal behavior**, with no spatial positions in the clustering input, explain unseen interventions better than simpler models. **It does not assume that they do.**

### Frozen protocol

1. Grow the same fixed 11×9 lattice under **specialist** or **diverse** policies, choose a checkpoint on validation only (as Gate 2).
2. With the medium frozen, halve each of its **178 recruited edge conductances** separately (scaffold `g0=0.06` untouched), and record the effect on resistance under each of **3 TRAIN source/sink configurations**. These 3-dimensional *finite causal signatures*, not the edge coordinates or rectangular region names, feed balanced deterministic 12-way k-means. Each discovered group has 14–15 edges to prevent tiny singleton groups. Group colors can identify noncontiguous edges: colors are not necessarily physical circuits.
3. For every edge, predict its future finite-cut damage using (i) group average training harm per unit grown material, or (ii) material-only, train flow and exact training first derivative. Use *the same* observed training-edge effects to calibrate controls. Compare groupings obtained from causal signatures, weight ranks, flow ranks, the old rectangular regions and shuffled edge assignments. Freeze all predictions before any test cut.
4. Select approximately 49–51 edge probes using **30 largest training-material edges plus 24 evenly spaced edge IDs**, with duplicate IDs removed. This favors material-heavy edges; it is not an unbiased test-edge sample. On each selected edge, halve conductance and score the resistance difference on **12 previously excluded deterministic source/sink test pairs**. No held-out case or gradient is used to fit the predictions. Score RMSE, Pearson correlation, top-five effect retrieval and within-group standard deviation on the same held-out probe edges. Test score computation is in a separate explicit Reveal step in the browser.
5. Evaluate **20 shuffled partitions**, each with the identical training causal signatures for calibration, and report how many have lower prediction RMSE than the functionally discovered grouping. The random baseline is decided before reveal and has no further test-time optimization.

From the terminal:

```bash
npm test                         # 30 checks including previous two gates
npm run bench:gate3               # both growth arms × all three tasks
python -m http.server 8000        # navigate to /gate3.html
```

The frozen deterministic output from `npm run bench:gate3` is committed as [`results/gate3-reference.json`](results/gate3-reference.json); the complete browser receipt additionally exports the original per-edge probe fingerprints, memberships, predicted harms and measured effects.

### What the reference run establishes—and does not

At the validation-selected checkpoints for all six combinations (3 tasks × 2 growth arms), discovered causal groups are **informative but do not win the lowest held-out RMSE**. The simpler material, flow, spatial, or shuffled groups can do at least as well. A high correlation can coexist with a calibration error; inspecting both RMSE and top-five coverage is essential. The graph morphology and the easy physics baselines already explain much of the held-out damage.

The fixed-Gate-2 checkpoints are often early because late specialized structures perform poorly on shifted terminal validation cases; group evolution plots track the growth of edges **retrospectively assigned** to the selected checkpoint's groups, not persistent cluster identities across training. The empirical test is confined to Kirchhoff routing with known physics and closely related endpoint shifts. This is not evidence that Goodfire's semantic-neuron results generalize to networks grown in this way.

**Future honest gate:** use mechanisms whose roles genuinely change with context (e.g., nonlinear state-dependent routing), then ask whether unsupervised causal groups transfer to a separately defined task family and outperform a physically informed task-agnostic model at matched measurement cost. First verify that the hidden mechanism is identifiable from the allowed probes.
