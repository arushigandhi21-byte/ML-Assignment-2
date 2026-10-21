# ML Assignment 2 — Bagging, Boosting and Bias–Variance on Face Recognition

**Arushi Gandhi · Roll number M24MA2001 · seed 242001**

Decision-tree baseline, bagging (written from scratch), multiclass SAMME AdaBoost (written from scratch), and a measured bias–variance study on the Olivetti faces dataset (40 identities × 10 images, 64×64), using raw flattened pixels only.

## Setup

Python 3.12 was used.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used (numpy 2.2.6, scikit-learn 1.6.1, scipy 1.18.1, matplotlib 3.10.3, joblib 1.6.0).

## Reproduce everything

From the repository root:

```bash
bash run_all.sh
```

This runs every step below in order and regenerates all JSON results, logs and figures in `results/` (~35 min on a 10-core laptop; the Q3 refits take ~23 min). Each step can also be run on its own:

| Step | Command | Outputs |
|---|---|---|
| Data split + masked set | `.venv/bin/python -m common.data` | `results/split.npz`, `figures/masked_examples.png` |
| Q1(a) tree baseline | `.venv/bin/python -m Q1.q1a_tree_baseline` | `q1a.json`, `figures/q1a_depth.png` |
| Q1(b,c) bagging sweep | `.venv/bin/python -m Q1.q1bc_bagging` | `q1bc.json`, `figures/q1b_*.png`, `figures/q1c_*.png` |
| Q1(d) tree vs bagging | `.venv/bin/python -m Q1.q1d_tree_vs_bagging` | `q1d.json`, `figures/q1d_*.png` |
| Q2(a) toy SAMME round | `.venv/bin/python -m Q2.q2a_toy_round` | `q2a.json`, `logs/q2a_toy_round.log` |
| Q2(b) boosting grid | `.venv/bin/python -m Q2.q2b_boosting_grid` | `q2b.json`, `figures/q2b_*.png` |
| Q2(c) boosting vs bagging | `.venv/bin/python -m Q2.q2c_boost_vs_bag` | `q2c.json`, `figures/q2c_*.png` |
| Q2(d) clean vs masked | `.venv/bin/python -m Q2.q2d_masked_comparison` | `q2d.json`, `figures/q2d_masked_face.png` |
| SAMME sanity check | `.venv/bin/python -m Q2.check_against_sklearn` | `logs/q2_check_against_sklearn.log` |
| Q3 refits (30×) | `.venv/bin/python -m Q3.q3_refits` | `q3_probs.npz`, `q3_struct.json` |
| Q3(a–d) analysis | `.venv/bin/python -m Q3.q3_analysis` | `q3.json`, `figures/q3*.png` |
| Q3(c) coverage | `.venv/bin/python -m Q3.q3c_coverage` | `q3c_coverage.json` |
| Frozen config | `.venv/bin/python -m common.export_config` | `configs/frozen.json` |

All paths in the Outputs column are under `results/` unless they start with `configs/`. Every script also writes its console output to `results/logs/<script>.log`.

On macOS with a python.org build of Python, the dataset download can fail with `CERTIFICATE_VERIFY_FAILED`. `run_all.sh` sets `SSL_CERT_FILE` to certifi's bundle to avoid this. When running scripts by hand, prefix them with `SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi;print(certifi.where())')`, or run `common.data` once, because later scripts use the cached copy in `data/`.

## Repository layout

```
common/      config (seed, paths, mask), data split + masking, bagging, SAMME,
             bias-variance measurement, shared helpers
Q1/ Q2/ Q3/  one script per sub-question + NOTES.md with the results and their interpretation
configs/     frozen.json - every selected hyper-parameter (generated from the results)
results/     split.npz, *.json results, logs/, figures/
run_all.sh   exact reproduction commands
```

## Protocol

- **Seed:** all digits of the roll number in order → `numpy.random.default_rng(242001)`.
- **Split:** within each identity, 10 images are randomly split into 6 train / 2 validation / 2 test (240/80/80). It is saved once to `results/split.npz` and loaded by every script.
- **Masked test set:** a copy of the clean test images with rows and columns 24–39 (the centred 16×16 block) set to 0. Models are trained only on clean images.
- **Model selection** uses validation accuracy only. Ties go to the simpler model (fewer leaves / smaller B / shallower learner, then fewer rounds). Chosen settings are then frozen (`configs/frozen.json`).
- **Libraries:** only `DecisionTreeClassifier` is taken from scikit-learn. Bagging (`common/bagging.py`) and SAMME (`common/samme.py`) are implemented here. `sklearn.ensemble.AdaBoostClassifier` appears only in `Q2/check_against_sklearn.py`, which verifies that the SAMME weight update reproduces sklearn's learner weights; it is not used for any reported result.
- **Weak learners not better than random** (SAMME, ε ≥ 1 − 1/K): the learner is rejected, refit once with a new random state, and boosting stops if it is rejected again. ε = 0 is clipped to 1e-10 and boosting stops.
- **Determinism:** all randomness is drawn from generators seeded with 242001, and parallel tree fitting draws every random number before fitting, so results are identical across runs. Only wall-clock training times (Q2c) vary slightly between runs.

## Frozen choices and main results

| | Setting | Clean test acc | Masked test acc |
|---|---|---|---|
| Single tree | `max_depth=None, min_samples_leaf=4` | 0.600 | 0.163 |
| Bagging | 200 trees of the frozen tree | 0.875 | 0.625 |
| SAMME | depth-3 learners, 50 rounds | 0.250 | 0.275 |

Bias–variance over 30 within-identity resamples (clean test set, probability-vector definition):

| | Bias² | Variance | Test acc (mean ± sd) |
|---|---|---|---|
| Single tree | 0.400 | 0.664 | 0.468 ± 0.052 |
| Bagging | 0.532 | 0.021 | 0.781 ± 0.041 |
| SAMME | 0.919 | 0.016 | 0.163 ± 0.049 |

See `Q1/NOTES.md`, `Q2/NOTES.md` and `Q3/NOTES.md` for every table and its interpretation.
