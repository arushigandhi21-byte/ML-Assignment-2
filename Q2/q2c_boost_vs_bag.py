import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.bagging import Bagging
from common.config import SEED, FIG_DIR, N_CLASSES
from common.data import load_data
from common.samme import SAMME
from common.utils import Tee, save_json, load_json

SIZES = [25, 50, 100]
REPEATS = 3

log = Tee("q2c_boost_vs_bag")
d = load_data()
tree_params = load_json("q1a")["selected"]
boost_depth = load_json("q2b")["selected"]["max_depth"]
log(f"bagging tree: {tree_params}; boosting weak learner depth: {boost_depth}")


def timed_fit(make):
    times, model = [], None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        model = make()
        times.append(time.perf_counter() - t0)
    return model, float(np.median(times))


rows = []
for M in SIZES:
    boost, t_boost = timed_fit(lambda: SAMME(M, boost_depth, SEED, N_CLASSES).fit(
        d["X_train"], d["y_train"], track=False))
    bag, t_bag = timed_fit(lambda: Bagging(M, tree_params, SEED).fit(d["X_train"], d["y_train"], n_jobs=1))
    for name, model, t in [("boosting", boost, t_boost), ("bagging", bag, t_bag)]:
        r = {"model": name, "learners": M, "fit_seconds": t}
        for s in ("train", "val", "test"):
            r[f"{s}_acc"] = float(np.mean(model.predict(d[f"X_{s}"]) == d[f"y_{s}"]))
        if name == "bagging":
            leaves = [tr.get_n_leaves() for tr in model.trees]
        else:
            leaves = [h.get_n_leaves() for h in model.learners]
        r["mean_leaves_per_learner"] = float(np.mean(leaves))
        rows.append(r)

log(f"\n{'model':9s} {'M':>4} {'train':>6} {'val':>6} {'test':>6} {'fit s':>7} {'s/learner':>9} {'leaves':>6}")
for r in rows:
    log(f"{r['model']:9s} {r['learners']:>4} {r['train_acc']:6.3f} {r['val_acc']:6.3f} {r['test_acc']:6.3f} "
        f"{r['fit_seconds']:7.2f} {r['fit_seconds'] / r['learners']:9.4f} {r['mean_leaves_per_learner']:6.1f}")

T = load_json("q2b")["selected"]["rounds"]
m = SAMME(T, boost_depth, SEED, N_CLASSES).fit(d["X_train"], d["y_train"])
W = m.w_history
H = m.train_pred_history
y = d["y_train"]
correct = H == y

peak = W.max(axis=0)
i = int(np.argmax(peak))
log(f"\nheaviest face: train index {i}, identity {y[i]}, dataset index {d['idx_train'][i]}, "
    f"peak weight {peak[i]:.4f} = {peak[i] * 240:.1f}x uniform (round {int(W[:, i].argmax())})")
log(f"learners correct on it: {correct[:, i].sum()}/{T}; "
    f"average face: {correct.mean(axis=0).mean() * T:.1f}/{T}; "
    f"rank of this face by #correct learners: {int((correct.sum(axis=0) < correct[:, i].sum()).sum()) + 1} of 240 (1 = fewest)")

alphas = m.alphas
S = np.zeros(N_CLASSES)
ens_pred, true_share = [], []
for t in range(T):
    S[H[t, i]] += alphas[t]
    ens_pred.append(int(S.argmax()))
    true_share.append(S[y[i]] / S.sum())
ens_pred = np.array(ens_pred)

log("\nround | weight before (x uniform) | learner prediction | right? | ensemble prediction after round")
for t in range(T):
    log(f"{t + 1:5d} | {W[t, i] * 240:8.2f} | id{H[t, i]:<3d} | {'yes' if correct[t, i] else 'no ':3s} | id{ens_pred[t]}")

wrong_preds = H[~correct[:, i], i]
vals, cnts = np.unique(wrong_preds, return_counts=True)
order = np.argsort(-cnts)
log("when wrong, learners predict:", [(int(vals[k]), int(cnts[k])) for k in order[:5]])
log(f"exp(alpha_t) range over rounds: {np.exp(alphas).min():.1f} - {np.exp(alphas).max():.1f} "
    f"(weight of a missed face is multiplied by this before renormalising)")
same_id = np.flatnonzero(y == y[i])
log(f"same identity (train idx {same_id.tolist()}): peak weights x uniform "
    f"{[round(float(peak[j] * 240), 1) for j in same_id]}, learners correct "
    f"{[int(correct[:, j].sum()) for j in same_id]}")
streaks = np.diff(np.flatnonzero(np.diff(np.r_[0, correct[:, i].astype(int), 0]) != 0))
log(f"rounds where a learner gets it right: {(np.flatnonzero(correct[:, i]) + 1).tolist()}")

save_json({"rows": rows, "boost_depth": boost_depth, "tree_params": tree_params,
           "heavy_face": {"train_index": i, "identity": int(y[i]), "dataset_index": int(d["idx_train"][i]),
                          "peak_weight": float(peak[i]), "weight_history": W[:, i],
                          "learner_pred": H[:, i], "learner_correct": correct[:, i],
                          "ensemble_pred": ens_pred, "ensemble_true_share": true_share,
                          "alphas": alphas}}, "q2c")

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
for name, c in [("bagging", "C0"), ("boosting", "C1")]:
    sub = [r for r in rows if r["model"] == name]
    for s, ls in [("train", ":"), ("val", "--"), ("test", "-")]:
        axes[0].plot(SIZES, [r[f"{s}_acc"] for r in sub], "o" + ls, color=c, label=f"{name} {s}")
    axes[1].plot(SIZES, [r["fit_seconds"] for r in sub], "o-", color=c, label=name)
axes[0].set_xlabel("number of learners"); axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=6, ncol=2)
axes[1].set_xlabel("number of learners"); axes[1].set_ylabel("fit time (s, 1 core)"); axes[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "q2c_boost_vs_bag.png", dpi=150)

fig = plt.figure(figsize=(10, 3))
ax0 = fig.add_axes([0.0, 0.12, 0.15, 0.78])
ax0.imshow(d["X_train"][i].reshape(64, 64), cmap="gray"); ax0.axis("off")
ax0.set_title(f"train face {i}, id{y[i]}", fontsize=8)
ax = fig.add_axes([0.24, 0.17, 0.64, 0.72])
rounds = np.arange(1, T + 1)
ax.semilogy(rounds, W[:T, i] * 240, "-", color="gray", lw=1, label="weight (x uniform) when learner t is fit")
ax.scatter(rounds[correct[:, i]], W[:T, i][correct[:, i]] * 240, color="C2", zorder=3, label="learner t right")
ax.scatter(rounds[~correct[:, i]], W[:T, i][~correct[:, i]] * 240, color="C3", marker="x", zorder=3, label="learner t wrong")
ax2 = ax.twinx()
ax2.step(rounds, (ens_pred == y[i]).astype(int), where="post", color="C0", alpha=.5)
ax2.set_yticks([0, 1]); ax2.set_yticklabels(["ensemble\nwrong", "ensemble\nright"], fontsize=7)
ax.set_xlabel("round t"); ax.set_ylabel("sample weight / (1/240)"); ax.legend(fontsize=7, loc="upper left")
fig.savefig(FIG_DIR / "q2c_heavy_face.png", dpi=150)
log("figures -> q2c_boost_vs_bag.png, q2c_heavy_face.png")
