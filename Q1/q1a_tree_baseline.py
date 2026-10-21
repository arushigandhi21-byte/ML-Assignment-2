import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from common.config import SEED, FIG_DIR
from common.data import load_data
from common.utils import Tee, save_json, leaf_stats

DEPTHS = [2, 4, 8, 16, None]
MIN_LEAF = [1, 2, 4]

log = Tee("q1a_tree_baseline")
d = load_data()

rows = []
for depth in DEPTHS:
    for msl in MIN_LEAF:
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=SEED)
        tree.fit(d["X_train"], d["y_train"])
        row = {
            "max_depth": depth,
            "min_samples_leaf": msl,
            "train_acc": tree.score(d["X_train"], d["y_train"]),
            "val_acc": tree.score(d["X_val"], d["y_val"]),
            "test_acc": tree.score(d["X_test"], d["y_test"]),
            **leaf_stats(tree),
        }
        rows.append(row)

log(f"{'depth':>6} {'msl':>4} {'train':>6} {'val':>6} {'test':>6} {'leaves':>7} {'mean/leaf':>9} {'frac1':>6} {'real_depth':>10}")
for r in rows:
    log(f"{str(r['max_depth']):>6} {r['min_samples_leaf']:>4} {r['train_acc']:6.3f} {r['val_acc']:6.3f} "
        f"{r['test_acc']:6.3f} {r['n_leaves']:7d} {r['mean_samples_per_leaf']:9.2f} "
        f"{r['frac_single_sample_leaves']:6.3f} {r['depth']:10d}")

best = max(rows, key=lambda r: (r["val_acc"], -r["n_leaves"]))
log("\nties at the best val acc:",
    [(r["max_depth"], r["min_samples_leaf"]) for r in rows if r["val_acc"] == best["val_acc"]])
log("SELECTED (frozen):", {k: best[k] for k in ("max_depth", "min_samples_leaf")})
log({k: best[k] for k in ("train_acc", "val_acc", "test_acc", "n_leaves",
                           "mean_samples_per_leaf", "frac_single_sample_leaves", "depth")})

save_json({"grid": rows,
           "selected": {"max_depth": best["max_depth"], "min_samples_leaf": best["min_samples_leaf"]},
           "selected_row": best,
           "selection_rule": "max val_acc, ties -> fewer leaves"}, "q1a")

depth_labels = [str(x) for x in DEPTHS]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for msl, ls in [(1, "-"), (2, ":"), (4, "--")]:
    sub = [r for r in rows if r["min_samples_leaf"] == msl]
    axes[0].plot(depth_labels, [r["train_acc"] for r in sub], "o" + ls, color="C0", label=f"train, msl={msl}")
    axes[0].plot(depth_labels, [r["val_acc"] for r in sub], "s" + ls, color="C1", label=f"val, msl={msl}")
    axes[1].plot(depth_labels, [r["n_leaves"] for r in sub], "o" + ls, color="C2", label=f"leaves, msl={msl}")
ax2 = axes[1].twinx()
for msl, ls in [(1, "-"), (2, ":"), (4, "--")]:
    sub = [r for r in rows if r["min_samples_leaf"] == msl]
    ax2.plot(depth_labels, [r["frac_single_sample_leaves"] for r in sub], "^" + ls, color="C3")
axes[0].set_xlabel("max_depth"); axes[0].set_ylabel("accuracy"); axes[0].legend(fontsize=7)
axes[1].set_xlabel("max_depth"); axes[1].set_ylabel("number of leaves", color="C2")
ax2.set_ylabel("fraction of 1-sample leaves", color="C3")
fig.tight_layout()
fig.savefig(FIG_DIR / "q1a_depth.png", dpi=150)
log("figure -> results/figures/q1a_depth.png")
