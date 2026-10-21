import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.biasvar import bias_variance
from common.config import FIG_DIR, RESULTS_DIR
from common.utils import Tee, save_json, load_json

log = Tee("q3_analysis")
Z = np.load(RESULTS_DIR / "q3_probs.npz")
S = load_json("q3_struct")
y = Z["y_test"]
B_FINAL, B_DEPTH, R = S["B_final"], S["B_depth"], S["R"]
DEPTHS = S["depths"]
SHOWN = ["4", "16", "None"]
log(f"R = {R} refits, N = {len(y)} clean test images; distinct originals per resample: "
    f"{np.mean(S['distinct_per_resample']):.1f}/240")

MODELS = {"single tree": "tree_dNone", f"bagging (B={B_FINAL})": f"bag{B_FINAL}", "boosting (d3, T50)": "boost"}
res_a = {}
log(f"\n(a) {'model':20s} {'bias^2':>7} {'variance':>8} {'MSE':>6} {'acc mean':>8} {'acc std':>7} "
    f"{'flip rate':>9} {'mean max p':>10}")
for name, key in MODELS.items():
    bv = bias_variance(Z[f"{key}_clean"], y)
    res_a[name] = bv
    log(f"    {name:20s} {bv['bias2']:7.3f} {bv['variance']:8.3f} {bv['mse']:6.3f} {bv['acc_mean']:8.3f} "
        f"{bv['acc_std']:7.3f} {bv['label_flip_rate']:9.3f} {bv['mean_max_prob']:10.3f}")
log("    (MSE = bias^2 + variance checked exactly; flip rate = fraction of (face, refit) whose label "
    "differs from that face's most common label)")

res_b = {}
log(f"\n(b) {'depth':>5} | {'tree bias2':>10} {'tree var':>8} {'tree acc':>13} | {'bag bias2':>9} {'bag var':>7} "
    f"{'bag acc':>13} | {'var removed':>11} {'% removed':>9} {'acc gain':>13}")
for dp in DEPTHS:
    t = bias_variance(Z[f"tree_d{dp}_clean"], y)
    b = bias_variance(Z[f"bag{B_DEPTH}_d{dp}_clean"], y)
    gain = b["acc_per_rep"] - t["acc_per_rep"]
    res_b[dp] = {"tree": t, "bag": b, "var_removed": t["variance"] - b["variance"],
                 "var_removed_pct": 100 * (t["variance"] - b["variance"]) / t["variance"],
                 "bias2_change": b["bias2"] - t["bias2"],
                 "acc_gain_mean": float(gain.mean()), "acc_gain_std": float(gain.std(ddof=1))}
    rb = res_b[dp]
    log(f"    {dp:>5} | {t['bias2']:10.3f} {t['variance']:8.3f} {t['acc_mean']:6.3f}±{t['acc_std']:.3f} | "
        f"{b['bias2']:9.3f} {b['variance']:7.3f} {b['acc_mean']:6.3f}±{b['acc_std']:.3f} | "
        f"{rb['var_removed']:11.3f} {rb['var_removed_pct']:8.1f}% {rb['acc_gain_mean']:+.3f}±{rb['acc_gain_std']:.3f}")

res_c = {}
log(f"\n(c) {'depth':>5} | single tree: {'leaves':>6} {'depth':>5} {'mean/leaf':>9} {'1-leaf':>6} | "
    f"bag member: {'leaves':>6} {'mean/leaf':>9} {'1-leaf':>6} {'disagree':>8}")
for dp in DEPTHS:
    tr = S["struct"]["tree"][dp]
    bg = S["struct"]["bag"][dp]
    m = lambda rows, k: float(np.mean([x[k] for x in rows]))
    res_c[dp] = {
        "tree_leaves": m(tr, "n_leaves"), "tree_depth": m(tr, "depth"),
        "tree_mean_per_leaf": m(tr, "mean_samples_per_leaf"), "tree_frac_single": m(tr, "frac_single_sample_leaves"),
        "bag_leaves": m(bg, "n_leaves"), "bag_mean_per_leaf": m(bg, "mean_samples_per_leaf"),
        "bag_frac_single": m(bg, "frac_single_sample_leaves"), "disagreement": m(bg, "disagreement_test"),
    }
    c = res_c[dp]
    log(f"    {dp:>5} | {'':12s} {c['tree_leaves']:6.1f} {c['tree_depth']:5.1f} {c['tree_mean_per_leaf']:9.2f} "
        f"{c['tree_frac_single']:6.3f} | {'':10s} {c['bag_leaves']:6.1f} {c['bag_mean_per_leaf']:9.2f} "
        f"{c['bag_frac_single']:6.3f} {c['disagreement']:8.3f}")
best_gain = max(DEPTHS, key=lambda dp: res_b[dp]["acc_gain_mean"])
log(f"    largest mean bagging accuracy gain at depth {best_gain} "
    f"({res_b[best_gain]['acc_gain_mean']:+.3f}); deepest (None): {res_b['None']['acc_gain_mean']:+.3f}")
gains = np.array([res_b[dp]["bag"]["acc_per_rep"] - res_b[dp]["tree"]["acc_per_rep"] for dp in DEPTHS])
winner_counts = {dp: int(np.sum(gains.argmax(axis=0) == j)) for j, dp in enumerate(DEPTHS)}
log(f"    per-refit winner of the largest gain (ties -> shallower): {winner_counts}")
from scipy import stats
g_best = gains[DEPTHS.index(best_gain)]
g_deep = gains[DEPTHS.index("None")]
diff = g_best - g_deep
gain_test = {"mean_diff": float(diff.mean()), "sd_diff": float(diff.std(ddof=1)),
             "n_best_wins": int((diff > 0).sum()), "n_ties": int((diff == 0).sum()),
             "t_test_p": float(stats.ttest_rel(g_best, g_deep).pvalue),
             "wilcoxon_p": float(stats.wilcoxon(g_best, g_deep).pvalue)}
log(f"    gain(depth {best_gain}) - gain(None): {gain_test['mean_diff']:+.3f} ± {gain_test['sd_diff']:.3f}, "
    f"larger in {gain_test['n_best_wins']}/{R} refits ({gain_test['n_ties']} ties), "
    f"paired t-test p = {gain_test['t_test_p']:.2g}, Wilcoxon p = {gain_test['wilcoxon_p']:.2g}")

frozen = load_json("q2d")["summary"]
frozen_key = {"single tree": "tree", f"bagging (B={B_FINAL})": "bag", "boosting (d3, T50)": "boost"}
res_d = {}
log(f"\n(d) {'model':20s} {'clean acc':>13} {'masked acc':>13} {'drop':>6} {'bias^2':>7} {'variance':>8} "
    f"| masked: {'bias^2':>7} {'variance':>8} | frozen model clean/masked")
for name, key in MODELS.items():
    c = res_a[name]
    m = bias_variance(Z[f"{key}_masked"], y)
    f = frozen[frozen_key[name]]
    res_d[name] = {"clean_acc_mean": c["acc_mean"], "clean_acc_std": c["acc_std"],
                   "masked_acc_mean": m["acc_mean"], "masked_acc_std": m["acc_std"],
                   "drop": c["acc_mean"] - m["acc_mean"], "bias2": c["bias2"], "variance": c["variance"],
                   "masked_bias2": m["bias2"], "masked_variance": m["variance"],
                   "frozen_clean_acc": f["clean_acc"], "frozen_masked_acc": f["masked_acc"]}
    r = res_d[name]
    log(f"    {name:20s} {r['clean_acc_mean']:6.3f}±{r['clean_acc_std']:.3f} {r['masked_acc_mean']:6.3f}±{r['masked_acc_std']:.3f} "
        f"{r['drop']:6.3f} {r['bias2']:7.3f} {r['variance']:8.3f} | {'':7s} {r['masked_bias2']:7.3f} "
        f"{r['masked_variance']:8.3f} | {f['clean_acc']:.3f}/{f['masked_acc']:.3f}")

save_json({"a": res_a, "b": res_b, "c": res_c, "d": res_d, "best_gain_depth": best_gain,
           "gain_winner_counts": winner_counts, "gain_test": gain_test, "shown_depths": SHOWN}, "q3")

lbl = lambda dp: "None" if dp == "None" else dp
xs = np.arange(len(DEPTHS))
fig, axes = plt.subplots(1, 3, figsize=(11, 3))
axes[0].plot(xs, [res_b[dp]["tree"]["bias2"] for dp in DEPTHS], "o-", color="C0", label="tree bias$^2$")
axes[0].plot(xs, [res_b[dp]["bag"]["bias2"] for dp in DEPTHS], "o--", color="C0", alpha=.6, label="bagging bias$^2$")
axes[0].plot(xs, [res_b[dp]["tree"]["variance"] for dp in DEPTHS], "s-", color="C3", label="tree variance")
axes[0].plot(xs, [res_b[dp]["bag"]["variance"] for dp in DEPTHS], "s--", color="C3", alpha=.6, label="bagging variance")
axes[0].set_title("bias$^2$ and variance vs depth", fontsize=9); axes[0].legend(fontsize=7)
axes[1].errorbar(xs, [res_b[dp]["tree"]["acc_mean"] for dp in DEPTHS], [res_b[dp]["tree"]["acc_std"] for dp in DEPTHS],
                 fmt="o-", capsize=3, label="single tree")
axes[1].errorbar(xs, [res_b[dp]["bag"]["acc_mean"] for dp in DEPTHS], [res_b[dp]["bag"]["acc_std"] for dp in DEPTHS],
                 fmt="s-", capsize=3, label=f"bagging ({B_DEPTH})")
axes[1].bar(xs, [res_b[dp]["acc_gain_mean"] for dp in DEPTHS], color="gray", alpha=.3, label="gain (bag - tree)")
axes[1].set_title("clean test accuracy over 30 refits", fontsize=9); axes[1].legend(fontsize=7)
ax = axes[2]
ax.plot(xs, [res_c[dp]["disagreement"] for dp in DEPTHS], "o-", color="C2")
ax.set_ylim(0.8, 1.0); ax.set_ylabel("pairwise disagreement (test)", color="C2", fontsize=8)
ax2 = ax.twinx()
ax2.plot(xs, [res_c[dp]["bag_mean_per_leaf"] for dp in DEPTHS], "s--", color="C1")
ax2.set_ylabel("mean samples per leaf", color="C1", fontsize=8)
ax.set_title("bagged-tree structure (100 trees)", fontsize=9)
for a in axes:
    a.set_xticks(xs); a.set_xticklabels([lbl(dp) for dp in DEPTHS]); a.set_xlabel("max_depth")
fig.tight_layout(); fig.savefig(FIG_DIR / "q3bc_depth.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
names = list(MODELS)
short = ["tree", "bagging", "boosting"]
xs = np.arange(3)
axes[0].bar(xs - .2, [res_d[n]["clean_acc_mean"] for n in names], .4, yerr=[res_d[n]["clean_acc_std"] for n in names],
            capsize=3, label="clean")
axes[0].bar(xs + .2, [res_d[n]["masked_acc_mean"] for n in names], .4, yerr=[res_d[n]["masked_acc_std"] for n in names],
            capsize=3, label="masked")
axes[0].set_title("test accuracy, mean ± sd over 30 refits", fontsize=9); axes[0].legend(fontsize=7)
axes[1].bar(xs - .2, [res_d[n]["bias2"] for n in names], .4, label="bias$^2$")
axes[1].bar(xs + .2, [res_d[n]["variance"] for n in names], .4, label="variance")
axes[1].set_title("clean bias$^2$ and variance", fontsize=9); axes[1].legend(fontsize=7)
for a in axes:
    a.set_xticks(xs); a.set_xticklabels(short)
fig.tight_layout(); fig.savefig(FIG_DIR / "q3ad_models.png", dpi=150)
log("figures -> q3bc_depth.png, q3ad_models.png")
