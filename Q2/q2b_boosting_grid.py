import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.config import SEED, FIG_DIR, N_CLASSES
from common.data import load_data
from common.samme import SAMME
from common.utils import Tee, save_json

DEPTHS = [1, 2, 3, 4]
ROUNDS = [10, 25, 50, 100]

log = Tee("q2b_boosting_grid")
d = load_data()
K = N_CLASSES
log(f"K = {K}, random-guess error = 1 - 1/K = {1 - 1 / K:.4f}, log(K-1) = {np.log(K - 1):.4f}")

rows, models = [], {}
for depth in DEPTHS:
    m = SAMME(max(ROUNDS), depth, seed=SEED, n_classes=K).fit(d["X_train"], d["y_train"])
    models[depth] = m
    P = {s: m.learner_predictions(d[f"X_{s}"]) for s in ("train", "val", "test")}
    log(f"depth {depth}: fitted {len(m.learners)} rounds, eps range {m.eps.min():.3f}-{m.eps.max():.3f}, "
        f"alpha range {m.alphas.min():.3f}-{m.alphas.max():.3f}, events: {m.events or 'none'}")
    for T in ROUNDS:
        r = {"depth": depth, "rounds": T, "rounds_fitted": min(T, len(m.learners))}
        for s in ("train", "val", "test"):
            r[f"{s}_acc"] = float(np.mean(m.predict(None, T, P=P[s]) == d[f"y_{s}"]))
        rows.append(r)

log(f"\n{'depth':>5} {'rounds':>6} {'train':>6} {'val':>6} {'test':>6}")
for r in rows:
    log(f"{r['depth']:>5} {r['rounds']:>6} {r['train_acc']:6.3f} {r['val_acc']:6.3f} {r['test_acc']:6.3f}")

best = max(rows, key=lambda r: (r["val_acc"], -r["depth"], -r["rounds"]))
log("\nval ties at best:", [(r["depth"], r["rounds"]) for r in rows if r["val_acc"] == best["val_acc"]])
log(f"SELECTED: depth = {best['depth']}, rounds = {best['rounds']} (val {best['val_acc']:.4f}, "
    f"train {best['train_acc']:.4f}, test {best['test_acc']:.4f})")

sel = models[best["depth"]]
T = best["rounds"]
W = sel.w_history[:T + 1]
top10 = np.sort(W, axis=1)[:, -10:].sum(axis=1)
dyn = {"eps": sel.eps[:T], "alpha": sel.alphas[:T], "max_w": W.max(axis=1), "top10_w": top10}
log(f"selected model dynamics: eps first/last {dyn['eps'][0]:.3f}/{dyn['eps'][-1]:.3f}, "
    f"alpha first/last {dyn['alpha'][0]:.3f}/{dyn['alpha'][-1]:.3f}, "
    f"max weight start/end {W[0].max():.4f}/{W[-1].max():.4f} (uniform = {1 / 240:.4f}), "
    f"top-10 weight start/end {top10[0]:.3f}/{top10[-1]:.3f}")
heaviest = np.argsort(-W[-1])[:10]
log("10 heaviest training images at the end (train index, identity, weight):",
    [(int(i), int(d["y_train"][i]), round(float(W[-1][i]), 4)) for i in heaviest])

save_json({"grid": rows, "selected": {"max_depth": best["depth"], "rounds": best["rounds"]},
           "selected_row": best, "selection_rule": "max val_acc, ties -> smaller depth, fewer rounds",
           "events": {str(k): v.events for k, v in models.items()},
           "dynamics": dyn}, "q2b")

fig, ax = plt.subplots(figsize=(4, 3))
grid = np.array([[next(r["val_acc"] for r in rows if r["depth"] == dp and r["rounds"] == t) for t in ROUNDS]
                 for dp in DEPTHS])
im = ax.imshow(grid, cmap="viridis", aspect="auto")
for a in range(len(DEPTHS)):
    for b in range(len(ROUNDS)):
        ax.text(b, a, f"{grid[a, b]:.2f}", ha="center", va="center",
                color="w" if grid[a, b] < grid.max() * .7 else "k", fontsize=8)
ax.set_xticks(range(len(ROUNDS))); ax.set_xticklabels(ROUNDS); ax.set_xlabel("rounds")
ax.set_yticks(range(len(DEPTHS))); ax.set_yticklabels(DEPTHS); ax.set_ylabel("weak learner depth")
ax.set_title("SAMME clean validation accuracy", fontsize=9)
fig.colorbar(im); fig.tight_layout(); fig.savefig(FIG_DIR / "q2b_grid.png", dpi=150)

t = np.arange(1, T + 1)
fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
axes[0].plot(t, dyn["eps"], color="C0"); axes[0].axhline(1 - 1 / K, color="gray", ls=":", label="random (1-1/K)")
axes[0].set_title("weighted training error $\\epsilon_t$"); axes[0].legend(fontsize=7)
axes[1].plot(t, dyn["alpha"], color="C1"); axes[1].set_title("learner weight $\\alpha_t$")
axes[2].plot(np.arange(0, T + 1), dyn["max_w"], color="C2"); axes[2].axhline(1 / 240, color="gray", ls=":", label="uniform 1/240")
axes[2].set_title("maximum sample weight"); axes[2].legend(fontsize=7)
axes[3].plot(np.arange(0, T + 1), dyn["top10_w"], color="C3"); axes[3].axhline(10 / 240, color="gray", ls=":", label="uniform 10/240")
axes[3].set_title("weight of 10 heaviest images"); axes[3].legend(fontsize=7)
for ax in axes:
    ax.set_xlabel("round t")
fig.suptitle(f"SAMME, depth {best['depth']}, {T} rounds", fontsize=9)
fig.tight_layout(); fig.savefig(FIG_DIR / "q2b_dynamics.png", dpi=150)
log("figures -> q2b_grid.png, q2b_dynamics.png")
