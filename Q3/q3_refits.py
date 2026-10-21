import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from common.bagging import Bagging, pairwise_disagreement
from common.config import SEED, N_CLASSES, RESULTS_DIR
from common.data import load_data
from common.samme import SAMME
from common.utils import Tee, save_json, load_json, leaf_stats

R = 30
DEPTHS = [2, 4, 8, 16, None]
B_DEPTH = 100

log = Tee("q3_refits")
d = load_data()
tree_params = load_json("q1a")["selected"]
B_FINAL = load_json("q1bc")["selected_B"]
bsel = load_json("q2b")["selected"]
msl = tree_params["min_samples_leaf"]
log(f"frozen: tree {tree_params}, bagging B={B_FINAL}, boosting depth={bsel['max_depth']} rounds={bsel['rounds']}")
log(f"depth study: depths {DEPTHS}, min_samples_leaf={msl}, bagging with {B_DEPTH} trees")

rng = np.random.default_rng(SEED)
y_tr = d["y_train"]
by_id = [np.flatnonzero(y_tr == c) for c in range(N_CLASSES)]
resamples, model_seeds = [], []
for r in range(R):
    idx = np.concatenate([rng.choice(rows, size=len(rows), replace=True) for rows in by_id])
    resamples.append(idx)
    model_seeds.append(int(rng.integers(2**31 - 1)))
resamples = np.array(resamples)
distinct = np.array([len(np.unique(ix)) for ix in resamples])
log(f"distinct original training faces per resampled set: mean {distinct.mean():.1f} of 240 "
    f"(theory 240*(1-(5/6)^6) = {240 * (1 - (5 / 6) ** 6):.1f})")

SETS = {"clean": d["X_test"], "masked": d["X_test_masked"]}
probs = {}
struct = {"tree": {str(dp): [] for dp in DEPTHS}, "bag": {str(dp): [] for dp in DEPTHS}}


def put(key, r, p):
    if key not in probs:
        probs[key] = np.zeros((R, p.shape[0], N_CLASSES))
    probs[key][r] = p


def tree_proba(tree, X):
    p = np.zeros((len(X), N_CLASSES))
    p[:, tree.classes_] = tree.predict_proba(X)
    return p


t_start = time.perf_counter()
for r in range(R):
    t0 = time.perf_counter()
    X, y = d["X_train"][resamples[r]], y_tr[resamples[r]]
    seed = model_seeds[r]

    for dp in DEPTHS:
        tree = DecisionTreeClassifier(max_depth=dp, min_samples_leaf=msl, random_state=seed).fit(X, y)
        for s, Xs in SETS.items():
            put(f"tree_d{dp}_{s}", r, tree_proba(tree, Xs))
        struct["tree"][str(dp)].append(leaf_stats(tree))

    bag = Bagging(B_FINAL, tree_params, seed).fit(X, y)
    for s, Xs in SETS.items():
        P = bag.tree_predictions(Xs)
        put(f"bag{B_FINAL}_{s}", r, Bagging.votes_from_predictions(P) / B_FINAL)
        put(f"bag{B_DEPTH}_dNone_{s}", r, Bagging.votes_from_predictions(P[:B_DEPTH]) / B_DEPTH)
        if s == "clean":
            dis_none = pairwise_disagreement(P[:B_DEPTH])
    ls = [leaf_stats(t) for t in bag.trees[:B_DEPTH]]
    struct["bag"]["None"].append({"disagreement_test": dis_none,
                                  **{k: float(np.mean([l[k] for l in ls])) for k in ls[0]}})

    for dp in DEPTHS[:-1]:
        params = {"max_depth": dp, "min_samples_leaf": msl}
        bag_d = Bagging(B_DEPTH, params, seed).fit(X, y)
        for s, Xs in SETS.items():
            P = bag_d.tree_predictions(Xs)
            put(f"bag{B_DEPTH}_d{dp}_{s}", r, Bagging.votes_from_predictions(P) / B_DEPTH)
            if s == "clean":
                dis = pairwise_disagreement(P)
        ls = [leaf_stats(t) for t in bag_d.trees]
        struct["bag"][str(dp)].append({"disagreement_test": dis,
                                       **{k: float(np.mean([l[k] for l in ls])) for k in ls[0]}})

    boost = SAMME(bsel["rounds"], bsel["max_depth"], seed, N_CLASSES).fit(X, y, track=False)
    for s, Xs in SETS.items():
        put(f"boost_{s}", r, boost.predict_proba(Xs))

    acc = {k: float(np.mean(probs[k][r].argmax(1) == d["y_test"]))
           for k in ("tree_dNone_clean", f"bag{B_FINAL}_clean", "boost_clean")}
    log(f"rep {r + 1:2d}/{R}: {time.perf_counter() - t0:5.1f}s | clean acc tree {acc['tree_dNone_clean']:.3f} "
        f"bag {acc[f'bag{B_FINAL}_clean']:.3f} boost {acc['boost_clean']:.3f} | rounds fitted {len(boost.learners)}"
        + (f" | events {boost.events}" if boost.events else ""))

log(f"total {time.perf_counter() - t_start:.0f}s")
np.savez_compressed(RESULTS_DIR / "q3_probs.npz", y_test=d["y_test"], resamples=resamples,
                    model_seeds=np.array(model_seeds), **probs)
save_json({"R": R, "depths": [str(x) for x in DEPTHS], "B_depth": B_DEPTH, "B_final": B_FINAL,
           "distinct_per_resample": distinct, "struct": struct}, "q3_struct")
log("saved results/q3_probs.npz and results/q3_struct.json")
