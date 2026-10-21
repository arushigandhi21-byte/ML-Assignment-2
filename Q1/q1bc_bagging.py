import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.bagging import Bagging, vote_margin, pairwise_disagreement, oob_accuracy
from common.config import SEED, FIG_DIR
from common.data import load_data
from common.utils import Tee, save_json, load_json

B_LIST = [1, 5, 10, 25, 50, 100, 200]

log = Tee("q1bc_bagging")
d = load_data()
tree_params = load_json("q1a")["selected"]
log("frozen tree params from Q1(a):", tree_params)

model = Bagging(max(B_LIST), tree_params, seed=SEED).fit(d["X_train"], d["y_train"])

P = {s: model.tree_predictions(d[f"X_{s}"]) for s in ("train", "val", "test")}

n = len(d["y_train"])
theory = n * (1 - (1 - 1 / n) ** n)
p1 = (1 - 1 / n) ** n
p2 = (1 - 2 / n) ** n
theory_sd = np.sqrt(n * p1 * (1 - p1) + n * (n - 1) * (p2 - p1 ** 2))
distinct_all = np.array([len(np.unique(idx)) for idx in model.boot_idx])
log(f"theory: E[distinct] = n(1-(1-1/n)^n) = {theory:.2f}  ({theory / n:.4f} n), sd = {theory_sd:.2f}")

rows = []
for B in B_LIST:
    r = {"B": B}
    for s in ("train", "val", "test"):
        votes = Bagging.votes_from_predictions(P[s][:B])
        r[f"{s}_acc"] = float(np.mean(votes.argmax(axis=1) == d[f"y_{s}"]))
        if s == "val":
            top2 = np.sort(votes, axis=1)[:, -2:]
            r["val_ties"] = int(np.sum(top2[:, 0] == top2[:, 1]))
    r["distinct_mean"] = float(distinct_all[:B].mean())
    r["distinct_sd"] = float(distinct_all[:B].std()) if B > 1 else 0.0
    r["distinct_theory"] = theory
    r["oob_acc"], r["oob_coverage"] = oob_accuracy(model, d["X_train"], d["y_train"], B)
    r["disagreement_val"] = pairwise_disagreement(P["val"][:B])
    r["disagreement_test"] = pairwise_disagreement(P["test"][:B])
    r["margin_val"] = float(vote_margin(Bagging.votes_from_predictions(P["val"][:B])).mean())
    rows.append(r)

single_val = (P["val"] == d["y_val"]).mean(axis=1)
single_test = (P["test"] == d["y_test"]).mean(axis=1)
log(f"individual bootstrap trees: val acc mean {single_val.mean():.3f} (min {single_val.min():.3f}, max {single_val.max():.3f}); "
    f"test acc mean {single_test.mean():.3f}")

log(f"\n{'B':>4} {'train':>6} {'val':>6} {'test':>6} {'distinct':>13} {'theory':>7} "
    f"{'OOB':>6} {'cover':>6} {'disagr':>7} {'margin':>7} {'ties':>5}")
for r in rows:
    log(f"{r['B']:>4} {r['train_acc']:6.3f} {r['val_acc']:6.3f} {r['test_acc']:6.3f} "
        f"{r['distinct_mean']:7.2f}±{r['distinct_sd']:5.2f} {r['distinct_theory']:7.2f} "
        f"{r['oob_acc']:6.3f} {r['oob_coverage']:6.3f} {r['disagreement_val']:7.3f} "
        f"{r['margin_val']:7.3f} {r['val_ties']:5d}")

best = max(rows, key=lambda r: (r["val_acc"], -r["B"]))
log("\nval acc ties at best:", [r["B"] for r in rows if r["val_acc"] == best["val_acc"]])
log(f"SELECTED final B = {best['B']} (val acc {best['val_acc']:.4f})")

save_json({"tree_params": tree_params, "rows": rows, "selected_B": best["B"],
           "selection_rule": "max val_acc, ties -> smaller B",
           "theory_distinct": theory, "theory_distinct_sd": theory_sd,
           "all_bootstrap_distinct": distinct_all,
           "single_tree_val_acc_mean": single_val.mean(),
           "single_tree_test_acc_mean": single_test.mean()}, "q1bc")

Bs = [r["B"] for r in rows]

fig, ax = plt.subplots(figsize=(4.6, 3.2))
for s, m in [("train", "o"), ("val", "s"), ("test", "^")]:
    ax.plot(Bs, [r[f"{s}_acc"] for r in rows], m + "-", label=s)
ax.axhline(load_json("q1a")["selected_row"]["val_acc"], color="gray", ls=":", label="single tree (val)")
ax.set_xscale("log"); ax.set_xticks(Bs); ax.set_xticklabels(Bs)
ax.set_xlabel("number of trees B"); ax.set_ylabel("accuracy"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "q1b_accuracy_vs_B.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
Bs2 = [b for b in Bs if b > 1]
axes[0].plot(Bs, [r["oob_acc"] for r in rows], "o-", color="C0", label="OOB acc")
axes[0].plot(Bs, [r["val_acc"] for r in rows], "s--", color="C1", alpha=.6, label="val acc")
axes[0].set_title("out-of-bag accuracy"); axes[0].legend(fontsize=8)
axes[1].plot(Bs2, [r["disagreement_val"] for r in rows if r["B"] > 1], "o-", color="C2")
axes[1].set_title("avg pairwise disagreement (val)")
axes[1].set_ylim(0, 1)
axes[2].plot(Bs, [r["margin_val"] for r in rows], "o-", color="C3")
axes[2].set_title("mean vote margin (val)")
for ax in axes:
    ax.set_xscale("log"); ax.set_xticks(Bs); ax.set_xticklabels(Bs); ax.set_xlabel("B")
fig.tight_layout(); fig.savefig(FIG_DIR / "q1c_oob_disagreement_margin.png", dpi=150)

fig, ax = plt.subplots(figsize=(4.2, 3))
ax.hist(distinct_all, bins=15, color="C0", alpha=.8)
ax.axvline(theory, color="k", ls="--", label=f"theory {theory:.1f}")
ax.axvline(distinct_all.mean(), color="C3", label=f"observed {distinct_all.mean():.1f}")
ax.set_xlabel("distinct training images in a bootstrap sample"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_DIR / "q1b_distinct_hist.png", dpi=150)
log("figures -> q1b_accuracy_vs_B.png, q1c_oob_disagreement_margin.png, q1b_distinct_hist.png")
