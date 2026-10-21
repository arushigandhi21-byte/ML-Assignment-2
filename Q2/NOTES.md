# Q2 — results and interpretation

Numbers come from `results/q2a.json`, `q2b.json`, `q2c.json`, `q2d.json` and `results/logs/q2*.log`.

## How to reproduce
```
.venv/bin/python -m Q2.q2a_toy_round          # instant
.venv/bin/python -m Q2.q2b_boosting_grid      # ~45 s
.venv/bin/python -m Q2.q2c_boost_vs_bag       # ~7 min (single-core timing, 3 repeats)
.venv/bin/python -m Q2.q2d_masked_comparison  # ~30 s
.venv/bin/python -m Q2.check_against_sklearn  # optional sanity check, ~1 min
```

## Implementation (`common/samme.py`)
- α_t = log((1−ε_t)/ε_t) + log(K−1), K = 40 so log(K−1) = 3.664. Scores S_k(x) = Σ α_t 1[h_t(x)=k], confidence q_k = S_k / Σα_t.
- Weight update: w_i ← w_i · exp(α_t · 1[h_t(x_i) ≠ y_i]), then normalise. Weak learner = `DecisionTreeClassifier(max_depth=d)` fit with `sample_weight=w`.
- **Learners that are not better than random:** random guessing over 40 classes has error 1 − 1/K = 0.975. If ε_t ≥ 0.975, then α_t ≤ 0, so the learner is **rejected** (not added, weights unchanged), refit once with a new random_state, and boosting **stops** if that also fails. If ε_t = 0, ε is clipped to 1e-10 (α finite) and boosting stops, because the weights would no longer change. In the face runs no learner was rejected: ε ranged 0.36–0.95, always below 0.975.
- **Verification:** replaying sklearn's `AdaBoostClassifier(algorithm="SAMME")` trees through my `samme_update` reproduces all of its α_t to within 1e-14 (`Q2/check_against_sklearn.py`). Fitted models still differ from sklearn's, because the tree random_state breaks ties between equally good pixels. That sensitivity matters again in Q3.

## (a) One SAMME round on a toy problem (K=3, n=9, depth-1 stump)
- Data: 3 classes × 3 points in 2D, separated along x1. Initial weights 1/9 = 0.1111 each.
- Stump splits x1 ≤ 2.00 and predicts class 0 on the left and class 1 on the right, so all three class-2 points are wrong.
- ε = 3 × 1/9 = **0.3333**. α = log(0.6667/0.3333) + log 2 = log 2 + log 2 = **1.3863**, so e^α = 4.
- Unnormalised: correct 0.1111, wrong 0.4444. Z = 2. New weights: **correct 0.0556, wrong 0.2222**.
- The wrong points now carry 0.6667 = (K−1)/K of the weight. This holds for every SAMME round: ε·e^α = (1−ε)(K−1) and Z = (1−ε)K. On the new weights the last learner sits exactly at random-level error, so the next learner must do something different. In round 2 the stump moves to x1 ≤ 3.35 and gets class 2 right (ε = 0.167).
- Rejection is demonstrated too: an always-wrong learner (ε=1) and a borderline one (ε=2/3 = 1−1/K) both get status `rejected` with weights unchanged.
- The `SAMME` class reproduces this round exactly, because it calls the same `samme_update`.

## (b) Grid on faces → selected **depth 3, 50 rounds**
| depth \ rounds | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| 1 | 0.050 | 0.037 | 0.075 | 0.087 |
| 2 | 0.100 | 0.138 | 0.150 | 0.250 |
| 3 | 0.125 | 0.188 | **0.325** | 0.325 |
| 4 | 0.113 | 0.100 | 0.237 | 0.237 |
(clean validation accuracy; depth 3 ties at 50 and 100 rounds → fewer rounds kept)

Selected model: train 0.829, val 0.325, test 0.250.

Why boosting is weak here: a depth-d tree has at most 2^d leaves, so it can name at most 2^d of the 40 identities. Stumps name 2, depth 3 names ≤ 8 (5.9 leaves on average). Each round can only "help" a handful of people, and SAMME must accumulate many such small votes. Depth 4 is not better than 3 on validation: 0.237, with train 0.758 at 100 rounds.

Dynamics of the selected model (`results/figures/q2b_dynamics.png`):
- **ε_t** starts at 0.90, falls to 0.58 at round 10, then jumps back and settles at about 0.70–0.80. It always stays far below the 0.975 random line. Early on the weights are spread out and a tree easily isolates a few identities. Once the easy identities are learned, the weight sits on the hard faces and every new learner finds it harder.
- **α_t** mirrors ε: 1.47 → 3.35 at round 10, then about 2.3–2.9. The log(K−1) = 3.66 term keeps α positive even with ε = 0.75. With 40 classes, an error of 0.75 is still far better than chance.
- **Max sample weight** grows from 1/240 = 0.0042 to 0.065 at round 19 (15× uniform). It then oscillates between 0.02 and 0.06: the heaviest face is repeatedly fit, drops, and is replaced.
- **Weight of the 10 heaviest images** grows from 0.042 (uniform) to about 0.20–0.25. 4% of the training set carries a fifth of the attention. At round 50 five of the ten heaviest are faces of identity 12, so a whole identity can become "hard".

## (c) Boosting vs bagging with 25 / 50 / 100 learners
| model | M | train | val | test | fit time (s, 1 core) | s / learner | leaves / learner |
|---|---|---|---|---|---|---|---|
| boosting (d=3) | 25 | 0.350 | 0.188 | 0.175 | 3.3 | 0.13 | 5.9 |
| bagging | 25 | 1.000 | 0.775 | 0.825 | 14.0 | 0.56 | 40.9 |
| boosting | 50 | 0.829 | 0.325 | 0.250 | 6.6 | 0.13 | 5.9 |
| bagging | 50 | 1.000 | 0.787 | 0.838 | 28.3 | 0.57 | 40.8 |
| boosting | 100 | 0.883 | 0.325 | 0.287 | 13.3 | 0.13 | 5.9 |
| bagging | 100 | 1.000 | 0.800 | 0.838 | 56.3 | 0.56 | 40.6 |

Times are wall-clock medians of 3 fits on one core and vary by a few % between runs; one run measured 5.3 s for boosting-25, a one-off slow measurement. Accuracies are deterministic. Take the final numbers from `results/q2c.json`.

- Bagging is far more accurate at every size (test 0.83 vs 0.18–0.29). Its members are full-depth trees (~41 leaves) with low bias; averaging removes their variance. Boosting's members are high-bias depth-3 trees, and 100 rounds are not enough to build up 40-way discrimination from ~6-leaf trees on raw pixels.
- Boosting is about 4× cheaper per learner (0.13 s vs 0.56 s), because a depth-3 tree is much smaller than a full tree. But boosting is **sequential**: learner t needs the weights from t−1. Bagging's trees are independent; with 10 cores the 200-tree fit takes ~20 s instead of ~110 s.
- Boosting's train−val gap (0.83 vs 0.33 at 50 rounds) shows that it also overfits the 240 faces, by concentrating weight on a few hard images.

**Heavy face: train #219, identity 36, peak weight 15.5× uniform (round 20).** (`results/figures/q2c_heavy_face.png`)
- Only 5 of 50 learners classify it correctly, at rounds **5, 20, 28, 38, 47**. Each time, the learner that gets it right predicts id36, i.e. a leaf was spent on this person.
- Between those rounds the weight rises steadily (about ×1.3 per round, on a log scale a straight line): 1.45 → 1.9 → 2.5 → … → 10.7 → 15.5×. After a correct round it collapses to about 1×. This **sawtooth with a period of ~9–10 rounds** is the answer to "why boosting keeps returning":
  1. w_i(t) ∝ exp(Σ_{s<t} α_s · 1[h_s(x_i) ≠ y_i]). The weight is a running, α-weighted count of how often this face has been missed, relative to the other faces.
  2. A depth-3 tree covers at most 8 identities per round and spends its leaves on the identities that currently carry the most weight. While this face is not among them it is missed, and each miss multiplies its weight by e^α_t (4.3–28.5) before renormalising.
  3. When its weight is large enough (~10× uniform), a learner is forced to fit it (rounds 20, 28, 38, 47). By the (K−1)/K property, all the faces that learner missed now share 39/40 of the weight, so this face's relative weight drops, and the cycle starts again.
- The face is never "solved": the ensemble is right only after round 5 and wrong from round 6 onward (final prediction id4). The wrong learners do not scatter randomly; they keep predicting the same few identities (id3 ×5, id4 ×5, id14/22/2 ×3). Those votes accumulate to more than the 5 correct ones. Its 5 other training faces of id36 peak at only 1.5×, so this particular image (beard, glasses, frontal) is an outlier within its own identity.

## (d) All three models, clean vs masked (frozen)
| model | clean acc | masked acc | clean mean winning vote / prob | masked mean winning vote / prob |
|---|---|---|---|---|
| single tree | 0.600 | 0.163 | 0.995 | 1.000 |
| bagging (B=200) | **0.875** | **0.625** | 0.374 | 0.273 |
| boosting (d=3, T=50) | 0.250 | 0.275 | 0.103 | 0.097 |

- Single tree: largest drop (−0.44), with a confidence of 1.0 on masked faces. It is confidently wrong.
- Bagging: drops by 0.25, and its confidence also drops (0.37 → 0.27), so it "knows" the input is harder.
- Boosting: no drop (+2 faces, which is noise), but it starts from a very low level. Its q is very flat (max q ≈ 0.10, i.e. only 4× a uniform 0.025), so many faces sit on a knife edge and the mask flips some wrong→right as well as right→wrong.
- Mask reliance (fraction of split nodes testing a pixel in the block, which is 6.25% of the image): tree 7.7%, bagging 6.4%, boosting 7.2% (α-weighted). None of the models over-uses the centre in general. The difference is in **how the damage propagates**, not in how often the block is used.

**Face where they respond differently: test #54, identity 27** (bagging stays right, boosting flips; chosen by the largest difference in true-class share change). (`results/figures/q2d_masked_face.png`)
- **Bagging, through tree disagreement:**
  - 62.5% of the 200 trees route this face through a masked pixel. 82% of those change their prediction, and trees whose path avoids the block **never** change (0%).
  - Votes for id27 fall from 98 to 57 (42 right trees lost, 1 gained). The lost votes **scatter**: disagreement on this face rises from 0.74 to 0.85, and the strongest rival (id33) collects only 38.
  - id27 keeps the plurality because 57 trees never looked at the masked region. Bagging has many independent "witnesses", built on different bootstrap samples and different pixels, and the mask silences only some of them.
- **Boosting, through sample weighting:**
  - Only **5 of 50 learners** ever vote id27 (rounds 9, 19, 28, 38, 48: the same ~10-round cycle as in (c)). These were the learners fit when id27's training faces were heavy: 0.062–0.086 of the total weight vs 0.025 uniform (2.5–3.4×).
  - Boosting's whole knowledge of id27 sits in these 5 trees (α sum 13.4). The rest of the ensemble votes for other people.
  - Learner 48 is the only one of the five whose path tests a masked pixel. Masking it removes ~2.5 of α. id27's q falls from 0.108 to 0.088, and id3 (0.094 → 0.104) overtakes it. On the clean face id27 led the runner-up id4 (0.0995) by only 0.008.
  - Bagging lost 21% of its correct votes and survived. Boosting lost 1 of its 5 relevant learners and flipped, because sample weighting concentrates class-specific evidence in a few rounds, and the overall vote is so flat that losing one is decisive.
- Summary: **bagging degrades gracefully because its errors under the mask are scattered across many diverse trees. Boosting's evidence for a given identity is concentrated in the few learners that were fit when that identity was heavily weighted, so losing one of them can flip the decision.** (Across the test set boosting is no worse on masked than on clean, but that reflects its low baseline and flat votes, not robustness.)
