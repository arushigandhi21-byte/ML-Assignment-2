import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from common.bagging import Bagging
from common.config import SEED, FIG_DIR, N_CLASSES
from common.data import load_data
from common.samme import SAMME
from common.utils import Tee, save_json, load_json

log = Tee("q2d_masked_comparison")
d = load_data()
tree_params = load_json("q1a")["selected"]
B = load_json("q1bc")["selected_B"]
bsel = load_json("q2b")["selected"]
log(f"tree {tree_params} | bagging B={B} | boosting depth={bsel['max_depth']} rounds={bsel['rounds']}")

tree = DecisionTreeClassifier(**tree_params, random_state=SEED).fit(d["X_train"], d["y_train"])
bag = Bagging(B, tree_params, SEED).fit(d["X_train"], d["y_train"])
boost = SAMME(bsel["rounds"], bsel["max_depth"], SEED, N_CLASSES).fit(d["X_train"], d["y_train"])
y = d["y_test"]

out = {}
for s, X in [("clean", d["X_test"]), ("masked", d["X_test_masked"])]:
    Pbag = bag.tree_predictions(X)
    Pboost = boost.learner_predictions(X)
    out[s] = {
        "tree_proba": tree.predict_proba(X),
        "bag_P": Pbag, "bag_proba": Bagging.votes_from_predictions(Pbag) / B,
        "boost_P": Pboost, "boost_proba": boost.predict_proba(None, P=Pboost),
    }

summary = {}
log(f"\n{'model':9s} {'clean acc':>9} {'masked acc':>10} {'clean conf':>10} {'masked conf':>11} {'drop':>6}")
for name in ("tree", "bag", "boost"):
    row = {}
    for s in ("clean", "masked"):
        p = out[s][f"{name}_proba"]
        row[f"{s}_acc"] = float(np.mean(p.argmax(axis=1) == y))
        row[f"{s}_conf"] = float(p.max(axis=1).mean())
    summary[name] = row
    log(f"{name:9s} {row['clean_acc']:9.3f} {row['masked_acc']:10.3f} {row['clean_conf']:10.3f} "
        f"{row['masked_conf']:11.3f} {row['clean_acc'] - row['masked_acc']:6.3f}")

ok = {(m, s): out[s][f"{m}_proba"].argmax(axis=1) == y for m in ("bag", "boost") for s in ("clean", "masked")}
bag_keeps = ok["bag", "clean"] & ok["bag", "masked"]
boost_keeps = ok["boost", "clean"] & ok["boost", "masked"]
bag_loses = ok["bag", "clean"] & ~ok["bag", "masked"]
boost_loses = ok["boost", "clean"] & ~ok["boost", "masked"]
typeA = np.flatnonzero(bag_keeps & boost_loses)
typeB = np.flatnonzero(boost_keeps & bag_loses)
log(f"\nboth right on clean: bagging survives mask but boosting breaks: {typeA.tolist()}")
log(f"both right on clean: boosting survives mask but bagging breaks: {typeB.tolist()}")


def true_share(m, s, i):
    return out[s][f"{m}_proba"][i, y[i]]


def mask_pixel(feature):
    r, c = divmod(int(feature), 64)
    return 24 <= r < 40 and 24 <= c < 40


def path_uses_mask(estimator, x):
    nodes = estimator.decision_path(x[None, :]).indices
    feats = estimator.tree_.feature[nodes]
    return any(mask_pixel(f) for f in feats if f >= 0)


def face_disagreement(P_col):
    c = np.bincount(P_col, minlength=N_CLASSES)
    n = len(P_col)
    return 1 - (c * (c - 1)).sum() / (n * (n - 1))


cands = list(typeA) + list(typeB)
for i in cands:
    log(f"  test#{i} id{y[i]}: bag true share {true_share('bag', 'clean', i):.3f} -> {true_share('bag', 'masked', i):.3f}; "
        f"boost true q {true_share('boost', 'clean', i):.3f} -> {true_share('boost', 'masked', i):.3f}")

i = max(cands, key=lambda j: abs((true_share("bag", "masked", j) - true_share("bag", "clean", j))
                                 - (true_share("boost", "masked", j) - true_share("boost", "clean", j))))
log(f"\nCHOSEN face: test#{i}, identity {y[i]}, dataset index {d['idx_test'][i]}")
xc, xm = d["X_test"][i], d["X_test_masked"][i]

Pc, Pm = out["clean"]["bag_P"][:, i], out["masked"]["bag_P"][:, i]
bag_uses = np.array([path_uses_mask(t, xc) for t in bag.trees])
changed = Pc != Pm
vc = np.bincount(Pc, minlength=N_CLASSES); vm = np.bincount(Pm, minlength=N_CLASSES)
info = {"bagging": {
    "votes_true_clean": int(vc[y[i]]), "votes_true_masked": int(vm[y[i]]),
    "pred_clean": int(vc.argmax()), "pred_masked": int(vm.argmax()),
    "top_masked": [(int(k), int(vm[k])) for k in np.argsort(-vm)[:4]],
    "frac_trees_path_uses_mask": float(bag_uses.mean()),
    "frac_trees_changed": float(changed.mean()),
    "frac_changed_given_uses_mask": float(changed[bag_uses].mean()) if bag_uses.any() else None,
    "frac_changed_given_not": float(changed[~bag_uses].mean()) if (~bag_uses).any() else None,
    "trees_right_clean": int((Pc == y[i]).sum()), "trees_right_masked": int((Pm == y[i]).sum()),
    "right_trees_lost": int(((Pc == y[i]) & (Pm != y[i])).sum()),
    "right_trees_gained": int(((Pc != y[i]) & (Pm == y[i])).sum()),
    "disagreement_clean": face_disagreement(Pc), "disagreement_masked": face_disagreement(Pm),
}}
log("bagging:", info["bagging"])

Qc, Qm = out["clean"]["boost_P"][:, i], out["masked"]["boost_P"][:, i]
a = boost.alphas
same_id = np.flatnonzero(d["y_train"] == y[i])
id_weight = boost.w_history[:-1][:, same_id].sum(axis=1)
boost_uses = np.array([path_uses_mask(h, xc) for h in boost.learners])
voted_true_c, voted_true_m = Qc == y[i], Qm == y[i]
Sc = np.bincount(Qc, weights=a, minlength=N_CLASSES); Sm = np.bincount(Qm, weights=a, minlength=N_CLASSES)
info["boosting"] = {
    "q_true_clean": float(Sc[y[i]] / a.sum()), "q_true_masked": float(Sm[y[i]] / a.sum()),
    "pred_clean": int(Sc.argmax()), "pred_masked": int(Sm.argmax()),
    "top_masked": [(int(k), round(float(Sm[k] / a.sum()), 3)) for k in np.argsort(-Sm)[:4]],
    "n_learners_true_clean": int(voted_true_c.sum()), "n_learners_true_masked": int(voted_true_m.sum()),
    "alpha_true_clean": float(a[voted_true_c].sum()), "alpha_true_masked": float(a[voted_true_m].sum()),
    "learners_for_true_clean": (np.flatnonzero(voted_true_c) + 1).tolist(),
    "learners_for_true_masked": (np.flatnonzero(voted_true_m) + 1).tolist(),
    "frac_learners_path_uses_mask": float(boost_uses.mean()),
    "frac_learners_changed": float((Qc != Qm).mean()),
    "alpha_share_of_changed_learners": float(a[Qc != Qm].sum() / a.sum()),
    "uniform_identity_weight": 6 / 240,
    "identity_weight_when_true_voters_fit": [round(float(id_weight[t]), 3) for t in np.flatnonzero(voted_true_c)],
    "true_voters_path_uses_mask": [bool(boost_uses[t]) for t in np.flatnonzero(voted_true_c)],
    "mean_identity_weight_all_rounds": float(id_weight.mean()),
}
log("boosting:", info["boosting"])

def frac_mask_splits(estimators, weights=None):
    fr = []
    for e in estimators:
        f = e.tree_.feature[e.tree_.feature >= 0]
        fr.append(np.mean([mask_pixel(v) for v in f]) if len(f) else 0.0)
    return float(np.average(fr, weights=weights))

info["model_mask_reliance"] = {
    "mask_area_fraction": 256 / 4096,
    "tree": frac_mask_splits([tree]),
    "bagging": frac_mask_splits(bag.trees),
    "boosting_alpha_weighted": frac_mask_splits(boost.learners, a),
}
log("fraction of split nodes testing a masked pixel:", info["model_mask_reliance"])

save_json({"summary": summary, "typeA_bag_survives": typeA, "typeB_boost_survives": typeB,
           "chosen_face": int(i), "true_id": int(y[i]), "details": info}, "q2d")

fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.6, 1.6]})
axes[0].imshow(xc.reshape(64, 64), cmap="gray"); axes[0].set_title(f"test#{i} clean (id{y[i]})", fontsize=8)
axes[1].imshow(xm.reshape(64, 64), cmap="gray"); axes[1].set_title("masked", fontsize=8)
for ax in axes[:2]:
    ax.axis("off")
ks = sorted(set(np.argsort(-vc)[:4]) | set(np.argsort(-vm)[:4]) | {y[i]}, key=lambda k: -max(vc[k], vm[k]))[:6]
xs = np.arange(len(ks))
axes[2].bar(xs - .2, vc[ks] / B, .4, label="clean"); axes[2].bar(xs + .2, vm[ks] / B, .4, label="masked")
axes[2].set_xticks(xs); axes[2].set_xticklabels([f"id{k}" + ("*" if k == y[i] else "") for k in ks], fontsize=7)
axes[2].set_title("bagging vote share", fontsize=8); axes[2].legend(fontsize=7)
ks = sorted(set(np.argsort(-Sc)[:4]) | set(np.argsort(-Sm)[:4]) | {y[i]}, key=lambda k: -max(Sc[k], Sm[k]))[:6]
xs = np.arange(len(ks))
axes[3].bar(xs - .2, Sc[ks] / a.sum(), .4, label="clean"); axes[3].bar(xs + .2, Sm[ks] / a.sum(), .4, label="masked")
axes[3].set_xticks(xs); axes[3].set_xticklabels([f"id{k}" + ("*" if k == y[i] else "") for k in ks], fontsize=7)
axes[3].set_title("boosting normalised vote q (* = true id)", fontsize=8); axes[3].legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG_DIR / "q2d_masked_face.png", dpi=150)
log("figure -> q2d_masked_face.png")
