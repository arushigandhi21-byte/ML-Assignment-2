import numpy as np
from sklearn.tree import DecisionTreeClassifier

from common.config import SEED
from common.samme import samme_update, SAMME
from common.utils import Tee, save_json

log = Tee("q2a_toy_round")
K = 3

X = np.array([[1.0, 2.0], [1.3, 1.0], [1.6, 3.0],
              [2.4, 2.5], [2.8, 1.5], [3.1, 3.5],
              [3.6, 1.0], [4.0, 3.0], [4.4, 2.0]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
n = len(y)

w0 = np.full(n, 1.0 / n)
stump = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X, y, sample_weight=w0)
pred = stump.predict(X)
upd = samme_update(w0, y, pred, K)

log("Toy data (9 samples, 3 classes)")
log(f"stump split: feature x{stump.tree_.feature[0] + 1} <= {stump.tree_.threshold[0]:.2f}")
log(f"\n{'i':>2} {'x1':>4} {'x2':>4} {'y':>2} {'w_init':>8} {'h1(x)':>6} {'wrong':>6} {'w*exp(a*wrong)':>15} {'w_new':>8}")
unnorm = w0 * np.exp(upd["alpha"] * upd["miss"])
for i in range(n):
    log(f"{i:>2} {X[i, 0]:4.1f} {X[i, 1]:4.1f} {y[i]:>2} {w0[i]:8.4f} {pred[i]:>6} {int(upd['miss'][i]):>6} "
        f"{unnorm[i]:15.4f} {upd['w_new'][i]:8.4f}")
log(f"\nweighted error eps = sum of w over wrong samples = {upd['eps']:.4f}")
log(f"alpha = log((1-eps)/eps) + log(K-1) = log({(1 - upd['eps']) / upd['eps']:.4f}) + log({K - 1}) = {upd['alpha']:.4f}")
log(f"exp(alpha) = {np.exp(upd['alpha']):.4f}  -> each wrong sample's weight is multiplied by this")
log(f"normaliser Z = {unnorm.sum():.4f}; new weights: correct = {upd['w_new'][~upd['miss']][0]:.4f}, "
    f"wrong = {upd['w_new'][upd['miss']][0]:.4f}")
log(f"after the update the wrong samples carry {upd['w_new'][upd['miss']].sum():.4f} of the total weight "
    f"= (K-1)/K = {(K - 1) / K:.4f} (holds for every SAMME round)")

eps_hand = upd["miss"].sum() / n
alpha_hand = np.log((1 - eps_hand) / eps_hand) + np.log(K - 1)
assert np.isclose(eps_hand, upd["eps"]) and np.isclose(alpha_hand, upd["alpha"])

stump2 = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(X, y, sample_weight=upd["w_new"])
pred2 = stump2.predict(X)
upd2 = samme_update(upd["w_new"], y, pred2, K)
log(f"\n(round 2 for context) split: x{stump2.tree_.feature[0] + 1} <= {stump2.tree_.threshold[0]:.2f}, "
    f"predictions {pred2.tolist()}, eps = {upd2['eps']:.4f}, alpha = {upd2['alpha']:.4f}")

bad = samme_update(w0, y, (y + 1) % K, K)
log(f"\nnot-better-than-random check: a learner with eps = {bad['eps']:.4f} "
    f">= 1 - 1/K = {1 - 1 / K:.4f} -> status '{bad['status']}', weights unchanged = "
    f"{np.allclose(bad['w_new'], w0)}")
border_pred = y.copy(); border_pred[:6] = (y[:6] + 1) % K
log(f"borderline learner with eps = {samme_update(w0, y, border_pred, K)['eps']:.4f} -> "
    f"status '{samme_update(w0, y, border_pred, K)['status']}'")

m = SAMME(n_rounds=1, max_depth=1, seed=SEED, n_classes=K).fit(X, y)
assert np.isclose(m.alphas[0], upd["alpha"]) and np.allclose(m.w_history[1], upd["w_new"])
log("\nSAMME class reproduces this round exactly (same samme_update function).")

save_json({"X": X, "y": y, "w_init": w0, "pred": pred, "miss": upd["miss"], "eps": upd["eps"],
           "alpha": upd["alpha"], "w_unnormalised": unnorm, "w_new": upd["w_new"],
           "round2": {"pred": pred2, "eps": upd2["eps"], "alpha": upd2["alpha"]},
           "rejected_example_eps": bad["eps"]}, "q2a")
