# Q1 — results and interpretation

All numbers come from `results/q1a.json`, `results/q1bc.json`, `results/q1d.json` and the logs in `results/logs/`.

## How to reproduce
```
.venv/bin/python -m common.data              # split (already saved, only rerun if missing)
.venv/bin/python -m Q1.q1a_tree_baseline     # ~5 s
.venv/bin/python -m Q1.q1bc_bagging          # ~20 s (10 cores)
.venv/bin/python -m Q1.q1d_tree_vs_bagging   # ~20 s
```

## (a) Single tree baseline
Frozen: **max_depth=None, min_samples_leaf=4.** Val 0.5375 ties with msl = 1 and 2; the tie is broken toward fewer leaves.

| | train | val | test | leaves | mean samples/leaf | frac. 1-sample leaves | actual depth |
|---|---|---|---|---|---|---|---|
| selected | 0.996 | 0.5375 | 0.600 | 40 | 6.0 | 0.000 | 31 |

What changes with depth (all from the grid):
- Leaves = depth + 1 for depth ≤ 16 (3, 5, 9, 18). The tree is a **chain**: each split separates roughly one identity (a pure leaf of its 6 training faces) from the rest. All faces not yet separated sit in one big leaf (192 faces at depth 8, 138 at depth 16).
- So "mean samples per leaf" (80 → 48 → 26.7 → 13.3 → 6) falls mainly because that big leaf shrinks, not because all leaves get smaller.
- Train accuracy goes 0.075 → 0.125 → 0.225 → 0.45 → 1.0. A tree with L leaves can name at most L classes, so a depth-limited tree cannot reach 40 identities at all. This is **bias** from too few leaves.
- At unlimited depth the tree is pure and fits training nearly perfectly, but val accuracy stops at 0.54: a 0.46 gap. That gap is **variance / memorisation**: each leaf is decided by a few pixel thresholds learned from about 6 images.
- min_samples_leaf has almost no effect. The natural leaf size is already 6 (one identity), so the constraint rarely binds. With msl=1 one leaf holds a single image (frac 0.024 = 1/41); msl=2 and msl=4 remove it.

## (b) Bagging from scratch
- Code: `common/bagging.py`. Bootstrap = 240 draws with replacement. The tree has the frozen params and its own random_state drawn from the seeded rng. The ensembles over B are **nested** (the B-tree model is the first B of the 200 trees).
- Distinct images per bootstrap: observed **151.4 ± 5.0** (over 200 bootstraps). Theory: n(1−(1−1/n)^n) = **151.89 = 0.633 n** (sd 4.83). This matches within 0.5 images. As n→∞ the fraction tends to 1 − 1/e ≈ 0.632.

| B | train | val | test | distinct (mean) |
|---|---|---|---|---|
| 1 | 0.679 | 0.338 | 0.375 | 153.0 |
| 5 | 0.917 | 0.562 | 0.588 | 152.6 |
| 10 | 0.988 | 0.613 | 0.700 | 152.4 |
| 25 | 1.000 | 0.775 | 0.825 | 151.4 |
| 50 | 1.000 | 0.787 | 0.838 | 151.5 |
| 100 | 1.000 | 0.800 | 0.838 | 151.5 |
| 200 | 1.000 | 0.825 | 0.875 | 151.4 |

- One bootstrap tree (B=1) is **worse** than the single tree (val 0.338 vs 0.538). It sees only ~63% of the distinct images, about 3.8 of 6 per identity. The mean individual bootstrap tree scores 0.356 val. The ensemble still reaches 0.825, so averaging more than makes up for the weaker members.

## (c) OOB, disagreement, margin → final B = 200
| B | OOB acc | OOB coverage | disagreement (val) | mean margin (val) | val ties |
|---|---|---|---|---|---|
| 1 | 0.264 | 0.362 | – | 1.000 | 0 |
| 5 | 0.340 | 0.883 | 0.784 | 0.255 | 20 |
| 10 | 0.353 | 0.992 | 0.782 | 0.263 | 13 |
| 25 | 0.550 | 1.000 | 0.802 | 0.257 | 2 |
| 50 | 0.667 | 1.000 | 0.800 | 0.262 | 3 |
| 100 | 0.767 | 1.000 | 0.803 | 0.260 | 1 |
| 200 | 0.771 | 1.000 | 0.803 | 0.255 | 1 |

Final B = **200** (best val 0.825, unique max). Note that 100 → 200 is only +2 val images.

**Do they tell the same story? No, because they measure different things:**
- **Disagreement is flat at about 0.80** from B=5 onward. It is a property of the *distribution* of trees (how different two random bootstrap trees are), not of B. Adding trees only estimates it more precisely. Two trees disagree on 80% of faces: the members are very diverse, which is exactly what makes averaging useful.
- **Mean margin is flat at about 0.26** once B ≥ 5. The vote fractions converge to their limits, and the winner holds only about 26 points more vote share than the runner-up. B=1 has margin 1 trivially (one vote). So the ensemble's "certainty" does not grow with B, yet accuracy does. Accuracy rises because with more trees the plurality is more often the true class, and exact ties disappear (20 tied val faces at B=5, 1 at B=200).
- **OOB keeps rising until B≈100** and stays below val. Each training image is out-of-bag for only ~37% of the trees, so the OOB vote at B uses about 0.37·B trees. OOB at B=200 (0.771) is roughly the accuracy of a ~74-tree ensemble. For small B it is also computed on part of the data only (coverage 0.36 at B=1). OOB is therefore a pessimistic, lagging estimate. It agrees with val that 100–200 trees are needed; it does not indicate overfitting.
- Summary: disagreement and margin say "the ensemble's character is fixed by B≈10–25". Accuracy and OOB say "more votes still help, slowly, up to 100–200". There is no conflict once you see that the first two are per-pair / per-vote averages and the last two depend on how stable the plurality is.

## (d) Tree vs bagging, clean and masked
| set | tree acc | bag acc | tree conf | bag conf | tree conf (correct / wrong) | bag conf (correct / wrong) |
|---|---|---|---|---|---|---|
| clean | 0.600 | **0.875** | 0.995 | 0.374 | 0.997 / 0.991 | 0.394 / 0.230 |
| masked | 0.163 | **0.625** | 1.000 | 0.273 | 1.000 / 1.000 | 0.320 / 0.193 |

- Confidence: tree = max predict_proba; bagging = winning vote share.
- Clean: bagging fixes 22 tree errors, breaks 0, and 10 faces are wrong for both.
- **The tree is always about 100% confident**, even when wrong and even on masked faces, because its leaves are pure. Its confidence carries no information. Bagging's vote share is lower but meaningful: 0.39 when right vs 0.23 when wrong.
- Mask: the tree drops 0.600 → 0.163 (35 clean-correct faces flip). Bagging drops 0.875 → 0.625 (20 flip). The single tree relies on a few split pixels; if one of them lies in the zeroed block, the face goes down a completely different branch. Bagging's 200 trees use different split pixels, so many trees still route the face correctly.

Faces shown (`results/figures/q1d_cases.png`). The selection rule was fixed in the code before looking: corrected = largest vote margin; both wrong = fewest votes for the true id.

| test face | true | tree | bagging votes (top) | true-id votes | trees correct |
|---|---|---|---|---|---|
| #36 | id18 | id3 | id18:119, id3:11, id31:11 | 119/200 | 59.5% |
| #35 | id17 | id22 | id17:112, id7:18, id39:15 (id22 only 2) | 112/200 | 56% |
| #19 | id9 | id34 | id24:25, id39:17, id17:14 … | 2/200 | 1% |
| #1 | id0 | id25 | id30:33, id1:27, id15:21 … | 7/200 | 3.5% |

**What bagging can and cannot fix:**
- **Corrected faces = variance errors.** Most trees still get the face wrong (only 13–60% of trees are right across all 22 fixed faces), but the wrong votes are *scattered* over many identities while the right votes all land on one. The tree's mistake was a fluke of its particular training sample. Example: on #35 only 2 of 200 bootstrap trees repeat the tree's answer id22. Averaging cancels these unsystematic errors.
- **Both-wrong faces = bias / systematic errors.** Almost no tree votes for the true identity (1–5%). Every bootstrap tree is built from the same kind of pixel thresholds on the same 6 training faces of that person, so a test face that looks different in raw pixels (both shown faces have their head strongly turned) is misrouted by every tree. Averaging many copies of the same mistake does not remove it. The plurality is weak (25–33 votes) but still wrong. Another systematic case appears in the log: both test faces of id15 go to id35.
- Bagging reduces variance. It cannot reduce bias, because every member has the same inductive bias (axis-aligned pixel thresholds on raw pixels).
