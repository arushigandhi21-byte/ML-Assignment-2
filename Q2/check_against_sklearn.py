import warnings

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from common.config import SEED, N_CLASSES
from common.data import load_data
from common.samme import samme_update
from common.utils import Tee

warnings.filterwarnings("ignore")
log = Tee("q2_check_against_sklearn")
d = load_data()
X, y = d["X_train"], d["y_train"]
for depth in [1, 2, 3, 4]:
    sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), n_estimators=100,
                            algorithm="SAMME", random_state=SEED).fit(X, y)
    w = np.full(len(y), 1 / len(y))
    alphas = []
    for est in sk.estimators_:
        u = samme_update(w, y, est.predict(X), N_CLASSES)
        alphas.append(u["alpha"])
        w = u["w_new"]
    diff = np.abs(np.array(alphas) - sk.estimator_weights_).max()
    log(f"depth {depth}: max |alpha_mine - alpha_sklearn| over {len(alphas)} rounds = {diff:.2e}")
