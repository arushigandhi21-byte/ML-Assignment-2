# Q3 — results and interpretation

Numbers come from `results/q3.json`, `results/q3_struct.json`, `results/q3c_coverage.json` and `results/logs/q3_*.log`.

## How to reproduce
```
.venv/bin/python -m Q3.q3_refits      # 30 refits of every model, ~23 min on 10 cores -> results/q3_probs.npz
.venv/bin/python -m Q3.q3_analysis    # tables + figures for (a)-(d), seconds
.venv/bin/python -m Q3.q3c_coverage   # supporting measurement for (c), ~1 min
```

## Setup
- 30 resampled training sets. For each identity, draw 6 faces **with replacement** from its 6 original training faces. Every identity is always present, but on average only **160.2 of 240** original faces appear in a set (theory 240·(1−(5/6)^6) = 159.6).
- The same 30 sets are used for every model (paired comparison). Each repetition also draws its own model seed, so the variance includes both the training-set change and the model's own randomness (bootstrap draws, tie-breaking).
- No re-tuning. Frozen configs: tree (None, msl 4); bagging B=200 of that tree; SAMME depth 3, 50 rounds. No boosting learner was ever rejected (50 rounds fitted in all 30 refits).
- p_r(x): tree = `predict_proba` (padded to 40 columns); bagging = vote fractions; boosting = q. Fixed clean test set, N = 80, R = 30.
- `common/biasvar.py` checks the exact identity MSE = Bias² + Variance every time. MSE here means mean ‖p_r − e‖².

**Key point for interpreting the numbers.** Bias² and variance are measured on **probability vectors**, not on 0/1 errors. A soft vote (bagging: max share ≈ 0.33; boosting: max q ≈ 0.11) is far from the one-hot target even when its argmax is correct, so it gets a large Bias². A hard one-hot output (single tree, max p = 1.0) that flips between classes gets a large Variance. Keep this in mind for every comparison below.

## (a) Selected models
| model | Bias² | Variance | MSE | test acc (mean ± sd over 30) | label flip rate | mean max p |
|---|---|---|---|---|---|---|
| single tree | **0.400** | 0.664 | 1.064 | 0.468 ± 0.052 | 0.507 | 1.000 |
| bagging (B=200) | 0.532 | 0.021 | **0.554** | **0.781 ± 0.041** | **0.172** | 0.333 |
| boosting (d3, T50) | 0.919 | **0.016** | 0.935 | 0.163 ± 0.049 | 0.738 | 0.110 |

(label flip rate = fraction of (face, refit) pairs whose predicted label differs from that face's most common label over the 30 refits)

- **Tree:** variance dominates (0.664 > 0.400). On half of all (face, refit) pairs the label differs from the modal label. Its p̄ is the average of 30 one-hot votes, effectively "bagging over resamples", which is why its Bias² looks low.
- **Bagging:** removes 97% of the tree's variance (0.664 → 0.021) and halves the MSE. Its Bias² is *higher* than the tree's (0.532 vs 0.400) despite +31 points of accuracy. Its vote shares are soft (≈0.33 for the winner), so ‖p̄ − e‖² stays large even when the plurality is right.
- **Boosting:** lowest variance (0.016) but by far the highest bias (0.919, close to the 0.975 that a uniform vector would get). Its q vectors are almost flat, so they cannot move much between refits. Its labels are still the *least* stable (flip rate 0.738): with such flat q, tiny shifts reorder the argmax.
- Accuracy across the 30 refits is lower than for the frozen models (tree 0.600, bagging 0.875, boosting 0.250), because each resample contains only ~160 distinct faces instead of 240.

## (b) Depth study (min_samples_leaf = 4 throughout; bagging = 100 trees)
The selected depth is **None (unlimited)**. It already grows to the maximum depth the data allows (33.8 levels on resampled sets), so a "much deeper" setting does not exist. The three depths discussed are **shallow = 4, intermediate = 16, selected = None (also the deepest)**; 2 and 8 are included to complete the curve.

| depth | tree Bias² | tree Var | tree acc | bag Bias² | bag Var | bag acc | var removed | % removed | acc gain (paired) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.916 | 0.039 | 0.059 ± 0.012 | 0.935 | 0.011 | 0.139 ± 0.060 | 0.028 | 71.3% | +0.080 ± 0.064 |
| **4** | 0.876 | 0.071 | 0.089 ± 0.016 | 0.897 | 0.013 | 0.248 ± 0.063 | 0.058 | 81.5% | +0.159 ± 0.062 |
| 8 | 0.805 | 0.132 | 0.144 ± 0.026 | 0.822 | 0.016 | 0.425 ± 0.073 | 0.116 | 88.2% | +0.281 ± 0.071 |
| **16** | 0.673 | 0.251 | 0.250 ± 0.034 | 0.688 | 0.021 | 0.611 ± 0.067 | 0.230 | 91.6% | **+0.361 ± 0.070** |
| **None** | 0.400 | 0.664 | 0.468 ± 0.052 | 0.532 | 0.025 | 0.768 ± 0.041 | **0.639** | **96.3%** | +0.300 ± 0.049 |

- **Depth → single tree:** Bias² falls monotonically (0.916 → 0.400) and variance rises monotonically (0.039 → 0.664). From 4 → 16 → None, bias drops by 0.20, then 0.27, while variance grows 3.5×, then 2.6×. At depth None variance overtakes bias.
- **Bagging removes variance at every depth.** The absolute amount grows with depth (0.058 at 4, 0.230 at 16, 0.639 at None), and so does the share (81.5% → 91.6% → 96.3%). After bagging, variance is nearly flat in depth (0.013–0.025): almost all of the tree's depth-driven variance is averaged away.
- **Bagging raises Bias² slightly at every depth** (+0.02 at 4, +0.015 at 16, +0.13 at None). Each bootstrap member sees only ~63% of the (already resampled) faces, and the vote share is softer than one-hot. The +0.13 at None is the cost of turning a hard 0/1 output into a soft vote.

## (c) Is the deepest tree where bagging helps most? **No.**
Largest mean accuracy gain: **depth 16 (+0.361)**, not None (+0.300). Over the 30 paired refits, depth 16 gives the largest gain of all five depths in **21/30**, None in 5/30, depth 8 in 4/30. Head to head, gain(16) − gain(None) = **+0.061 ± 0.077**, larger in **24/30** refits (1 tie), paired t-test p = 1.7e-4, Wilcoxon p = 8.3e-4. This is not chance.

Why, from the measurements:

| depth | leaves / tree (= max identities nameable) | ids a bagged tree actually names | true id nameable by a member | member acc | ensemble names | disagreement | mean samples / leaf |
|---|---|---|---|---|---|---|---|
| 4 | 5.5 | 5.2 | 13% | 0.090 | 13 ids | 0.947 | 44.3 |
| 16 | 20.8 | 20.0 | 50% | 0.248 | 35 ids | 0.891 | 11.7 |
| None | 39.2 | 36.2 | 91% | 0.354 | 37 ids | 0.835 | 6.1 |

(leaf occupancy / disagreement from the 30 refits; the "nameable / names" columns from `Q3/q3c_coverage.py`, original training set. 1-sample leaves = 0 at every depth because min_samples_leaf = 4.)

1. **Depth 16 has a hard ceiling that bagging breaks.** A depth-16 tree has ~21 leaves and names only ~20 of the 40 identities, so for half the test faces the true identity is *impossible* for a single tree (single-tree acc 0.25). Because the trees are chain-shaped (Q1a: one identity peeled off per split), different bootstrap samples peel off *different* identities. Disagreement is high (0.891) and the ensemble covers 35 identities. Bagging here fixes **variance (91.6% removed) and part of the single tree's structural, leaf-count limitation**. The gain is the largest.
2. **At depth None the single tree is already complete.** Its leaves hold ~6 faces (one identity each) and it can name 36–40 identities. The remaining error is mostly the tree's variance. Bagging removes 96% of it, the largest absolute amount (0.639), but that variance is measured on one-hot vectors, and many of the flips are *between two wrong identities*, which removing them does not convert into accuracy. Members are also more alike (disagreement 0.835, the lowest), so there is less independent error to average out. Gain +0.30.
3. **At shallow depth (2–8) diversity is useless.** Disagreement is highest (0.95–0.96), but members are right only 6–14% of the time and can name the true identity for only 8–25% of faces. Averaging cannot create a plurality for a class the trees rarely predict. Gain +0.08 to +0.28.
- So the bagging gain is **not** monotone in depth. It needs members that are (i) right often enough for the plurality to land on the truth and (ii) different enough that their errors scatter. Here that peaks at depth 16, before the tree becomes so complete that members agree and the remaining variance is mostly "wrong-to-wrong". Note that the largest **variance reduction** (None) is not the largest **accuracy gain** (16).
- The best **final accuracy** is still at depth None (0.768 vs 0.611). The question is about the largest *improvement*, not the best ensemble.

## (d) Final table
| model | clean test acc | masked test acc | drop | Bias² (clean) | Variance (clean) |
|---|---|---|---|---|---|
| single tree | 0.468 ± 0.052 | 0.240 ± 0.166 | 0.228 (−49%) | 0.400 | 0.664 |
| bagging (B=200) | **0.781 ± 0.041** | **0.556 ± 0.130** | 0.225 (−29%) | 0.532 | 0.021 |
| boosting (d3, T50) | 0.163 ± 0.049 | 0.166 ± 0.058 | −0.003 | 0.919 | **0.016** |

(mean ± sd over the 30 refits; frozen models on the original split: tree 0.600/0.163, bagging 0.875/0.625, boosting 0.250/0.275. Masked Bias²/Variance: tree 0.802/0.719, bagging 0.695/0.021, boosting 0.919/0.018.)

- **Most stable: bagging.** It has the lowest accuracy sd (0.041), the lowest label flip rate (0.172), and its variance is near the minimum (0.021). Boosting's variance is marginally lower (0.016), but only because its q vectors are compressed near uniform (max ≈ 0.11). Its labels flip most often (0.738) and its accuracy sd (0.049) is higher. Low probability-variance is not the same as stable decisions. Earlier plots agree: bagging's vote margin is flat in B (Q1c), and boosting's evidence for an identity sits in a few learners (Q2d).
- **Most sensitive to the missing region: the single tree.** It loses 49% of its accuracy (0.468 → 0.240; frozen tree 0.600 → 0.163). Its Bias² doubles under the mask (0.400 → 0.802), and its masked accuracy is the least predictable across refits (sd 0.166). Whether a refit survives depends on whether its few split pixels fall inside the block. Bagging loses the same absolute amount (0.225) but only 29% relative. Its masked variance is unchanged (0.021), because the mask moves every refit in the same direction rather than making the ensemble erratic. Boosting does not drop, but it is close to chance at 0.16 in both conditions; it has little to lose (Q2d: flat q, flips in both directions).
- **Is lowest variance = best accuracy? No.** The lowest-variance model, boosting (0.016), has the worst accuracy (0.163), because its Bias² is 0.919. The best model, bagging (0.781), wins because it combines low variance with moderate bias. MSE ranks the models correctly here (bagging 0.554 < boosting 0.935 < tree 1.064), but neither component alone does: the tree has the *lowest* Bias² and is not the best. Q3(c) gives the same lesson: the depth with the largest variance reduction is not the depth with the largest accuracy gain.
