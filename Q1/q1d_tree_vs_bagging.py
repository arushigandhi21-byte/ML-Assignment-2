import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from common.bagging import Bagging
from common.config import SEED, FIG_DIR
from common.data import load_data
from common.utils import Tee, save_json, load_json

log = Tee("q1d_tree_vs_bagging")
d = load_data()
tree_params = load_json("q1a")["selected"]
B = load_json("q1bc")["selected_B"]
log(f"frozen tree: {tree_params}, final bagging B = {B}")

tree = DecisionTreeClassifier(**tree_params, random_state=SEED).fit(d["X_train"], d["y_train"])
bag = Bagging(B, tree_params, seed=SEED).fit(d["X_train"], d["y_train"])
y = d["y_test"]

res = {}
summary = {}
for name, X in [("clean", d["X_test"]), ("masked", d["X_test_masked"])]:
    t_proba = tree.predict_proba(X)
    t_pred = t_proba.argmax(axis=1)
    P = bag.tree_predictions(X)
    votes = Bagging.votes_from_predictions(P)
    b_pred = votes.argmax(axis=1)
    b_conf = votes.max(axis=1) / B
    t_conf = t_proba.max(axis=1)
    per_tree_correct = (P == y).mean(axis=0)
    res[name] = dict(t_pred=t_pred, t_conf=t_conf, votes=votes, b_pred=b_pred, b_conf=b_conf,
                     per_tree_correct=per_tree_correct)
    t_ok, b_ok = t_pred == y, b_pred == y
    summary[name] = {
        "tree_acc": t_ok.mean(), "bag_acc": b_ok.mean(),
        "tree_conf_mean": t_conf.mean(), "bag_conf_mean": b_conf.mean(),
        "tree_conf_correct": t_conf[t_ok].mean(), "tree_conf_wrong": t_conf[~t_ok].mean(),
        "bag_conf_correct": b_conf[b_ok].mean(), "bag_conf_wrong": b_conf[~b_ok].mean(),
        "mean_individual_tree_acc": per_tree_correct.mean(),
        "n_bag_fixes_tree": int((~t_ok & b_ok).sum()),
        "n_bag_breaks_tree": int((t_ok & ~b_ok).sum()),
        "n_both_wrong": int((~t_ok & ~b_ok).sum()),
    }

log(f"\n{'set':7s} {'tree acc':>8} {'bag acc':>8} {'tree conf':>9} {'bag conf':>9} "
    f"{'tree conf ok/wrong':>19} {'bag conf ok/wrong':>18} {'fix':>4} {'break':>5} {'both':>5}")
for name, s in summary.items():
    log(f"{name:7s} {s['tree_acc']:8.3f} {s['bag_acc']:8.3f} {s['tree_conf_mean']:9.3f} {s['bag_conf_mean']:9.3f} "
        f"{s['tree_conf_correct']:9.3f}/{s['tree_conf_wrong']:.3f} {s['bag_conf_correct']:9.3f}/{s['bag_conf_wrong']:.3f} "
        f"{s['n_bag_fixes_tree']:4d} {s['n_bag_breaks_tree']:5d} {s['n_both_wrong']:5d}")
    log(f"        mean accuracy of an individual bootstrap tree: {s['mean_individual_tree_acc']:.3f}")


def describe(i, r):
    v = r["votes"][i]
    order = np.argsort(-v)[:4]
    top = ", ".join(f"id{k}:{v[k]}" for k in order if v[k] > 0)
    return (f"test#{i} (dataset idx {d['idx_test'][i]}) true id{y[i]} | tree -> id{r['t_pred'][i]} "
            f"(conf {r['t_conf'][i]:.2f}) | bagging -> id{r['b_pred'][i]} with votes [{top}] "
            f"| votes for true id {v[y[i]]}/{B} | fraction of trees correct {r['per_tree_correct'][i]:.3f}")


r = res["clean"]
fixed = np.flatnonzero((r["t_pred"] != y) & (r["b_pred"] == y))
both = np.flatnonzero((r["t_pred"] != y) & (r["b_pred"] != y))
log("\nALL clean faces where bagging fixes the tree:")
for i in fixed:
    log("  " + describe(i, r))
log("ALL clean faces where both are wrong:")
for i in both:
    log("  " + describe(i, r))

def margin_votes(v):
    s = np.sort(v)
    return s[-1] - s[-2]

show_fixed = sorted(fixed, key=lambda i: -margin_votes(r["votes"][i]))[:2]
show_both = sorted(both, key=lambda i: r["votes"][i][y[i]])[:2]
log("\nshown in report  corrected:", list(show_fixed), " both wrong:", list(show_both))

for model in ("t", "b"):
    flip = (res["clean"][f"{model}_pred"] == y) & (res["masked"][f"{model}_pred"] != y)
    log(f"{'tree' if model == 't' else 'bagging'}: correct on clean but wrong on masked = {flip.sum()}")

save_json({"tree_params": tree_params, "B": B, "summary": summary,
           "fixed_clean": fixed, "both_wrong_clean": both,
           "shown_fixed": show_fixed, "shown_both_wrong": show_both,
           "cases": {int(i): {"true": y[i], "tree_pred": r["t_pred"][i], "tree_conf": r["t_conf"][i],
                              "bag_pred": r["b_pred"][i], "votes": r["votes"][i],
                              "per_tree_correct": r["per_tree_correct"][i]}
                     for i in list(show_fixed) + list(show_both)}}, "q1d")

train_imgs = d["X_train"].reshape(-1, 64, 64)


def example_of(identity):
    return train_imgs[np.flatnonzero(d["y_train"] == identity)[0]]


cases = [(i, "bagging corrects tree") for i in show_fixed] + [(i, "both wrong") for i in show_both]
fig, grid = plt.subplots(2, 9, figsize=(16, 4.4),
                         gridspec_kw={"width_ratios": [1, 1, 1, 2.2, 0.35, 1, 1, 1, 2.2]})
for ax in grid[:, 4]:
    ax.axis("off")
rows = [grid[0, :4], grid[0, 5:], grid[1, :4], grid[1, 5:]]
for row, (i, label) in zip(rows, cases):
    v = r["votes"][i]
    row[0].imshow(d["X_test"][i].reshape(64, 64), cmap="gray")
    row[0].set_title(f"test face, true id{y[i]}\n({label})", fontsize=7)
    row[1].imshow(example_of(r["t_pred"][i]), cmap="gray")
    row[1].set_title(f"tree says id{r['t_pred'][i]}", fontsize=7)
    row[2].imshow(example_of(r["b_pred"][i]), cmap="gray")
    row[2].set_title(f"bagging says id{r['b_pred'][i]}", fontsize=7)
    order = list(np.argsort(-v)[:5])
    for k in (r["t_pred"][i], y[i]):
        if k not in order:
            order.append(k)
    order = np.array(sorted(order, key=lambda k: -v[k]))
    colors = ["C2" if k == y[i] else ("C3" if k == r["t_pred"][i] else "C0") for k in order]
    row[3].barh([f"id{k}" for k in order][::-1], v[order][::-1], color=colors[::-1])
    row[3].set_xlabel(f"votes out of {B} (green = true, red = tree's choice)", fontsize=7)
    row[3].tick_params(labelsize=7)
    for ax in row[:3]:
        ax.axis("off")
fig.tight_layout()
fig.savefig(FIG_DIR / "q1d_cases.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)
for ax, name in zip(axes, ("clean", "masked")):
    ax.hist(res[name]["t_conf"], bins=np.linspace(0, 1, 21), alpha=.6, label="tree max proba")
    ax.hist(res[name]["b_conf"], bins=np.linspace(0, 1, 21), alpha=.6, label="bagging vote share")
    ax.set_title(f"{name} test", fontsize=9); ax.set_xlabel("confidence")
axes[0].legend(fontsize=7); axes[0].set_ylabel("faces")
fig.tight_layout()
fig.savefig(FIG_DIR / "q1d_confidence.png", dpi=150)
log("figures -> q1d_cases.png, q1d_confidence.png")
