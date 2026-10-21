import numpy as np

from common.bagging import Bagging
from common.config import SEED
from common.data import load_data
from common.utils import Tee, save_json, load_json

log = Tee("q3c_coverage")
d = load_data()
msl = load_json("q1a")["selected"]["min_samples_leaf"]
y = d["y_test"]
rows = []
log(f"{'depth':>5} {'ids/tree':>8} {'true id nameable':>16} {'member acc':>10} {'ensemble ids':>12} {'ensemble acc':>12}")
for dp in [2, 4, 8, 16, None]:
    bag = Bagging(100, {"max_depth": dp, "min_samples_leaf": msl}, SEED).fit(d["X_train"], d["y_train"])
    ids_per_tree, nameable = [], []
    for t in bag.trees:
        leaves = t.tree_.children_left == -1
        labels = t.classes_[t.tree_.value[leaves, 0].argmax(axis=1)]
        ids = set(labels.tolist())
        ids_per_tree.append(len(ids))
        nameable.append(np.mean([c in ids for c in y]))
    P = bag.tree_predictions(d["X_test"])
    ens = bag.predict(d["X_test"])
    r = {"depth": str(dp), "ids_per_tree": float(np.mean(ids_per_tree)),
         "true_id_nameable": float(np.mean(nameable)),
         "member_acc": float((P == y).mean()), "ensemble_ids": int(len(np.unique(ens))),
         "ensemble_acc": float((ens == y).mean())}
    rows.append(r)
    log(f"{r['depth']:>5} {r['ids_per_tree']:8.1f} {r['true_id_nameable']:16.3f} {r['member_acc']:10.3f} "
        f"{r['ensemble_ids']:12d} {r['ensemble_acc']:12.3f}")
save_json(rows, "q3c_coverage")
