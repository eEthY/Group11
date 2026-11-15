import warnings
warnings.filterwarnings('ignore')

import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.impute import SimpleImputer

# =====================================================================
# Module 1: Load Original UCI Datasets (Table 1 in Breiman 2001)
# =====================================================================
def load_paper_datasets():
    """
    Fetch the exact UCI datasets used in Table 1 of Breiman (2001):
    1. Breast Cancer Wisconsin (Original): N = 699, M = 9, Classes = 2
    2. Sonar: N = 208, M = 60, Classes = 2
    """
    print("Fetching original UCI datasets (Breast Cancer Original & Sonar)...")
    
    # --- (1) Load Breast Cancer Wisconsin (Original): 699 instances, 9 inputs ---
    try:
        url_bc = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
        df_bc = pd.read_csv(url_bc, header=None, na_values='?')
        # Column 0 is Sample ID (drop); Columns 1-9 are the 9 features; Column 10 is Class (2 or 4)
        X_bc = df_bc.iloc[:, 1:10].values.astype(float)
        y_bc = df_bc.iloc[:, 10].values
    except Exception:
        # Fallback: OpenML exact ID 15 ('breast-w')
        bc = fetch_openml(data_id=15, as_frame=True, parser='auto')
        X_bc = bc.data.values.astype(float)
        y_bc = bc.target.values

    # Impute the 16 missing values ('?') using median to preserve all 699 instances
    X_bc = SimpleImputer(strategy='median').fit_transform(X_bc)
    _, y_bc = np.unique(y_bc, return_inverse=True)

    # --- (2) Load Sonar: 208 instances, 60 inputs ---
    try:
        url_sonar = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
        df_sonar = pd.read_csv(url_sonar, header=None)
        X_sonar = df_sonar.iloc[:, :60].values.astype(float)
        y_sonar = df_sonar.iloc[:, 60].values
    except Exception:
        # Fallback: OpenML exact ID 40 ('sonar')
        sonar = fetch_openml(data_id=40, as_frame=True, parser='auto')
        X_sonar = sonar.data.values.astype(float)
        y_sonar = sonar.target.values

    _, y_sonar = np.unique(y_sonar, return_inverse=True)
    
    print(f"-> Breast Cancer loaded: N={X_bc.shape[0]}, M={X_bc.shape[1]} (Paper Table 1: N=699, M=9)")
    print(f"-> Sonar loaded        : N={X_sonar.shape[0]}, M={X_sonar.shape[1]} (Paper Table 1: N=208, M=60)\n")
    return (X_bc, y_bc), (X_sonar, y_sonar)


# =====================================================================
# Module 2: Build Forest-RI and Appendix II Internal Estimates from Scratch
# =====================================================================
class BreimanRandomForestRI:
    """
    Implementation of Forest-RI (Random Input selection) following Breiman (2001).
    Uses fresh OS entropy (random_state=None by default) for genuine randomness
    on every run, computing OOB error, strength (s), and correlation (rho_bar).
    """
    def __init__(self, n_trees=100, F='log2', random_state=None):
        self.n_trees = n_trees
        self.F = F
        self.random_state = random_state

    def _resolve_F(self, M):
        # Section 4: F = 1 or int(log2(M) + 1)
        if self.F == 'log2':
            return int(np.log2(M) + 1)
        return min(int(self.F), M)

    def fit(self, X, y):
        # Initialize RNG from OS hardware entropy when random_state is None
        rng = np.random.default_rng(self.random_state)
        N, M = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_mapped = np.searchsorted(self.classes_, y)
        actual_F = self._resolve_F(M)

        self.trees_ = []
        oob_votes = np.zeros((N, n_classes), dtype=float)
        oob_indices_list = []
        all_tree_preds = np.zeros((self.n_trees, N), dtype=int)
        single_tree_oob_errors = []

        for k in range(self.n_trees):
            # Step A: Bootstrap sampling with replacement to form training set T_k
            boot_idx = rng.choice(N, size=N, replace=True)
            
            # Identify Out-of-Bag (OOB) instances (~1/3 of the data left out)
            oob_mask = np.ones(N, dtype=bool)
            oob_mask[boot_idx] = False
            oob_idx = np.where(oob_mask)[0]
            oob_indices_list.append(oob_idx)

            # Step B: Grow an unpruned CART tree splitting on actual_F random features at each node
            tree = DecisionTreeClassifier(
                criterion='gini',
                max_features=actual_F,
                max_depth=None,      # Do not prune
                random_state=None    # True unseeded randomness for node splits
            )
            tree.fit(X[boot_idx], y_mapped[boot_idx])
            self.trees_.append(tree)

            # Step C: Aggregate OOB votes and record single-tree OOB error
            if len(oob_idx) > 0:
                preds_oob = tree.predict(X[oob_idx])
                all_tree_preds[k, oob_idx] = preds_oob
                single_tree_oob_errors.append(np.mean(preds_oob != y_mapped[oob_idx]))
                for idx, c in zip(oob_idx, preds_oob):
                    oob_votes[idx, c] += 1.0

        # Step D: Compute OOB Error, Strength (s), and Correlation (rho_bar) per Appendix II
        valid = oob_votes.sum(axis=1) > 0
        # Q(x, j): OOB proportion of votes cast at x for class j
        Q = np.zeros_like(oob_votes)
        Q[valid] = oob_votes[valid] / oob_votes[valid].sum(axis=1, keepdims=True)

        # 1. Forest OOB Error & Average Single-Tree OOB Error ("One tree" column in Table 2)
        oob_pred = np.argmax(Q[valid], axis=1)
        self.oob_error_ = np.mean(oob_pred != y_mapped[valid])
        self.one_tree_oob_error_ = np.mean(single_tree_oob_errors)

        # 2. Strength (s): E[Q(x, y) - max_{j != y} Q(x, j)]
        Q_true = Q[valid, y_mapped[valid]]
        Q_other = Q[valid].copy()
        Q_other[np.arange(np.sum(valid)), y_mapped[valid]] = -1.0
        j_hat_valid = np.argmax(Q_other, axis=1)
        Q_max_other = Q_other[np.arange(np.sum(valid)), j_hat_valid]

        j_hat_all = np.zeros(N, dtype=int)
        j_hat_all[valid] = j_hat_valid

        margins = Q_true - Q_max_other
        self.strength_ = np.mean(margins)
        var_mr = np.mean(margins ** 2) - (self.strength_ ** 2)  # Appendix II Eq. (A1)

        # 3. Correlation (rho_bar): var(mr) / (E[sd(Theta)])^2
        sd_list = []
        for k, oob_idx in enumerate(oob_indices_list):
            if len(oob_idx) == 0:
                continue
            preds_k = all_tree_preds[k, oob_idx]
            p1 = np.mean(preds_k == y_mapped[oob_idx])
            p2 = np.mean(preds_k == j_hat_all[oob_idx])
            # Corrected variance of raw margin: E[rmg^2] - (E[rmg])^2 = (p1 + p2) - (p1 - p2)^2
            var_k = max(0.0, p1 + p2 - (p1 - p2) ** 2)
            sd_list.append(np.sqrt(var_k))

        E_sd = np.mean(sd_list)
        self.correlation_ = var_mr / (E_sd ** 2) if E_sd > 0 else 0.0
        return self

    def score_error(self, X_test, y_test):
        """Compute Test Set Misclassification Error Rate"""
        tree_preds = np.array([tree.predict(X_test) for tree in self.trees_])
        y_mapped = np.searchsorted(self.classes_, y_test)
        maj_vote = [np.argmax(np.bincount(tree_preds[:, i], minlength=len(self.classes_)))
                    for i in range(X_test.shape[0])]
        return np.mean(np.array(maj_vote) != y_mapped)


# =====================================================================
# Module 3: Reproduce Section 4 (Table 2) Experiments (True Random)
# =====================================================================
def run_table2_comparison(X, y, name, paper_vals, n_repeats=30):
    """
    Follow Section 4 protocol with unseeded random splits (random_state=None).
    """
    rs = ShuffleSplit(n_splits=n_repeats, test_size=0.10, random_state=None)
    err_ada, err_sel, err_f1, err_tree = [], [], [], []
    s_list, rho_list = [], []

    for train_idx, test_idx in rs.split(X):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # Run 1: Forest-RI with F = 1 (Single Input)
        rf_1 = BreimanRandomForestRI(n_trees=100, F=1, random_state=None).fit(X_tr, y_tr)
        te_err_1 = rf_1.score_error(X_te, y_te)

        # Run 2: Forest-RI with F = int(log2(M) + 1)
        rf_log = BreimanRandomForestRI(n_trees=100, F='log2', random_state=None).fit(X_tr, y_tr)
        te_err_log = rf_log.score_error(X_te, y_te)

        # Selection rule: pick the forest with the lower Out-of-Bag (OOB) error estimate
        if rf_1.oob_error_ < rf_log.oob_error_:
            err_sel.append(te_err_1)
            err_tree.append(rf_1.one_tree_oob_error_)
        else:
            err_sel.append(te_err_log)
            err_tree.append(rf_log.one_tree_oob_error_)

        err_f1.append(te_err_1)
        s_list.append(rf_log.strength_)
        rho_list.append(rf_log.correlation_)

        # Run 3: AdaBoost baseline combining 50 trees
        ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=50, random_state=None)
        ada.fit(X_tr, y_tr)
        err_ada.append(1.0 - ada.score(X_te, y_te))

    print(f"[{name} Dataset - Table 2] (Averaged over {n_repeats} unseeded random 90/10 splits)")
    print(f"  Internal Estimates (F=int(log2M+1)) -> Mean Strength s = {np.mean(s_list):.3f}, Mean Correlation rho = {np.mean(rho_list):.3f}")
    print(f"  {'Metric Column':<22} | {'Our Reproduction':<16} | {'Paper Table 2 Reported':<22}")
    print(f"  {'-'*66}")
    print(f"  {'Adaboost':<22} | {np.mean(err_ada)*100:14.1f}% | {paper_vals[0]:20.1f}%")
    print(f"  {'Forest-RI Selection':<22} | {np.mean(err_sel)*100:14.1f}% | {paper_vals[1]:20.1f}%")
    print(f"  {'Forest-RI (F=1)':<22} | {np.mean(err_f1)*100:14.1f}% | {paper_vals[2]:20.1f}%")
    print(f"  {'One tree (OOB)':<22} | {np.mean(err_tree)*100:14.1f}% | {paper_vals[3]:20.1f}%\n")


# =====================================================================
# Module 4: Reproduce Section 6 (Figure 1) on Sonar Data (F from 1 to 50)
# =====================================================================
def plot_sonar_figure1(X_sonar, y_sonar, n_iter=15):
    print("Running Sonar feature-size sweep F=1..50 (Reproducing Figure 1 on Page 17, please wait ~10s)...")
    F_grid = [1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    s_curve, rho_curve, test_err_curve, oob_err_curve = [], [], [], []
    rs = ShuffleSplit(n_splits=n_iter, test_size=0.10, random_state=None)
    splits = list(rs.split(X_sonar))

    for F in F_grid:
        s_tmp, rho_tmp, te_tmp, oob_tmp = [], [], [], []
        for tr_idx, te_idx in splits:
            rf = BreimanRandomForestRI(n_trees=100, F=F, random_state=None)
            rf.fit(X_sonar[tr_idx], y_sonar[tr_idx])
            
            s_tmp.append(rf.strength_)
            rho_tmp.append(rf.correlation_)
            oob_tmp.append(rf.oob_error_ * 100)
            te_tmp.append(rf.score_error(X_sonar[te_idx], y_sonar[te_idx]) * 100)
            
        s_curve.append(np.mean(s_tmp))
        rho_curve.append(np.mean(rho_tmp))
        oob_err_curve.append(np.mean(oob_tmp))
        test_err_curve.append(np.mean(te_tmp))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    ax1.plot(F_grid, s_curve, 'o-', label='strength', color='tab:blue')
    ax1.plot(F_grid, rho_curve, 's-', label='correlation', color='tab:orange')
    ax1.set_title('CORRELATION AND STRENGTH (Sonar Data - Fig. 1 Reproduction)')
    ax1.set_xlabel('number of inputs (F)')
    ax1.set_ylabel('Y Variables')
    ax1.set_ylim(0, 0.6)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    ax2.plot(F_grid, test_err_curve, 'o-', label='test set error', color='tab:blue')
    ax2.plot(F_grid, oob_err_curve, 's-', label='OB error', color='tab:orange')
    ax2.set_title('TEST SET AND OB ERROR (Sonar Data - Fig. 1 Reproduction)')
    ax2.set_xlabel('number of features (F)')
    ax2.set_ylabel('percent (%)')
    ax2.set_ylim(0, 25)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()


# =====================================================================
# Module 5: Reproduce Section 8 (Table 4) - The Effects of 5% Output Noise
# =====================================================================
def run_table4_noise_experiment(X, y, name, best_F, paper_vals, n_repeats=30):
    """
    Follow Section 8 protocol with unseeded randomness:
    Split 10% clean test set. Run 1 on clean 90% training set.
    Run 2 on noisy 90% training set (5% of class labels randomly flipped).
    Compute the relative percentage increase in test set error due to noise.
    """
    rs = ShuffleSplit(n_splits=n_repeats, test_size=0.10, random_state=None)
    rng = np.random.default_rng(None)  # Fresh OS entropy

    ada_clean_err, ada_noisy_err = [], []
    rf_clean_err, rf_noisy_err = [], []

    for train_idx, test_idx in rs.split(X):
        X_tr, y_tr_clean = X[train_idx], y[train_idx].copy()
        X_te, y_te = X[test_idx], y[test_idx]

        # Inject 5% output noise into a copy of the training labels
        y_tr_noisy = y_tr_clean.copy()
        n_noisy = max(1, int(0.05 * len(y_tr_noisy)))
        noisy_idx = rng.choice(len(y_tr_noisy), size=n_noisy, replace=False)
        y_tr_noisy[noisy_idx] = 1 - y_tr_noisy[noisy_idx]

        # 1. Forest-RI (Clean vs. Noisy)
        rf_clean = BreimanRandomForestRI(n_trees=100, F=best_F, random_state=None).fit(X_tr, y_tr_clean)
        rf_noisy = BreimanRandomForestRI(n_trees=100, F=best_F, random_state=None).fit(X_tr, y_tr_noisy)
        rf_clean_err.append(rf_clean.score_error(X_te, y_te))
        rf_noisy_err.append(rf_noisy.score_error(X_te, y_te))

        # 2. AdaBoost (Clean vs. Noisy)
        ada_clean = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=None)
        ada_noisy = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, random_state=None)
        ada_clean.fit(X_tr, y_tr_clean)
        ada_noisy.fit(X_tr, y_tr_noisy)
        ada_clean_err.append(1.0 - ada_clean.score(X_te, y_te))
        ada_noisy.fit(X_tr, y_tr_noisy)
        ada_noisy_err.append(1.0 - ada_noisy.score(X_te, y_te))

    ada_inc = ((np.mean(ada_noisy_err) - np.mean(ada_clean_err)) / np.mean(ada_clean_err)) * 100.0
    rf_inc = ((np.mean(rf_noisy_err) - np.mean(rf_clean_err)) / np.mean(rf_clean_err)) * 100.0

    print(f"[{name} Dataset - Table 4 (5% Output Noise)] ({n_repeats} unseeded runs)")
    print(f"  AdaBoost Test Error : Clean = {np.mean(ada_clean_err)*100:.2f}% -> Noisy = {np.mean(ada_noisy_err)*100:.2f}%")
    print(f"  Forest-RI Test Error: Clean = {np.mean(rf_clean_err)*100:.2f}% -> Noisy = {np.mean(rf_noisy_err)*100:.2f}%")
    print(f"  {'Algorithm':<15} | {'Our Error Increase (%)':<24} | {'Paper Table 4 Reported (%)':<26}")
    print(f"  {'-'*70}")
    print(f"  {'Adaboost':<15} | {ada_inc:+22.1f}% | {paper_vals[0]:+24.1f}%")
    print(f"  {'Forest-RI':<15} | {rf_inc:+22.1f}% | {paper_vals[1]:+24.1f}%\n")

# =====================================================================
# Module 6: Reproduce Section 10 (Figures 4 & 6 + Text Verification Reruns)
# =====================================================================
def load_section10_datasets():
    """
    Fetch the exact two datasets used in Section 10 (Pages 24-25):
    1. Pima Indians Diabetes: N = 768, M = 8, Classes = 2 (Figure 4)
    2. Congressional Voting Records: N = 435, M = 16, Classes = 2 (Figure 6)
    """
    print("Fetching Section 10 datasets (Diabetes & Congressional Votes)...")
    
    # (1) Diabetes (OpenML data_id=37): 768 instances, 8 inputs
    diab = fetch_openml(data_id=37, as_frame=True, parser='auto')
    X_diab = diab.data.values.astype(float)
    _, y_diab = np.unique(diab.target.values, return_inverse=True)

    # (2) Votes (UCI direct or OpenML data_id=56): 435 instances, 16 inputs
    try:
        url_votes = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
        df_v = pd.read_csv(url_votes, header=None)
        y_votes = np.where(df_v.iloc[:, 0].values == 'republican', 1, 0)
        # Map 'y' -> 1.0, 'n' -> 0.0, '?' -> 0.5 (or median impute)
        X_votes = df_v.iloc[:, 1:17].replace({'y': 1.0, 'n': 0.0, '?': np.nan}).values.astype(float)
        X_votes = SimpleImputer(strategy='most_frequent').fit_transform(X_votes)
    except Exception:
        votes = fetch_openml(data_id=56, as_frame=True, parser='auto')
        X_votes = pd.get_dummies(votes.data, drop_first=True).values.astype(float)
        _, y_votes = np.unique(votes.target.values, return_inverse=True)

    print(f"-> Diabetes loaded: N={X_diab.shape[0]}, M={X_diab.shape[1]} (Paper Table 1: N=768, M=8)")
    print(f"-> Votes loaded   : N={X_votes.shape[0]}, M={X_votes.shape[1]} (Paper Table 1: N=435, M=16)\n")
    return (X_diab, y_diab), (X_votes, y_votes)


def compute_oob_importance(X, y, F=1, n_trees=1000):
    """Run 1,000 trees and compute OOB permutation percent increase for each variable."""
    rng = np.random.default_rng(None)
    N, M = X.shape
    n_classes = len(np.unique(y))
    actual_F = int(np.log2(M) + 1) if F == 'log2' else min(int(F), M)

    trees, oob_indices_list = [], []
    oob_votes_base = np.zeros((N, n_classes), dtype=float)

    for _ in range(n_trees):
        boot_idx = rng.choice(N, size=N, replace=True)
        oob_mask = np.ones(N, dtype=bool)
        oob_mask[boot_idx] = False
        oob_idx = np.where(oob_mask)[0]
        oob_indices_list.append(oob_idx)

        tree = DecisionTreeClassifier(criterion='gini', max_features=actual_F, max_depth=None, random_state=None)
        tree.fit(X[boot_idx], y[boot_idx])
        trees.append(tree)

        if len(oob_idx) > 0:
            preds = tree.predict(X[oob_idx])
            for idx, c in zip(oob_idx, preds):
                oob_votes_base[idx, c] += 1.0

    valid_base = oob_votes_base.sum(axis=1) > 0
    base_oob_err = np.mean(np.argmax(oob_votes_base[valid_base], axis=1) != y[valid_base])

    percent_increases = []
    for m in range(M):
        oob_votes_noised = np.zeros((N, n_classes), dtype=float)
        for k, tree in enumerate(trees):
            oob_idx = oob_indices_list[k]
            if len(oob_idx) == 0:
                continue
            X_oob_noised = X[oob_idx].copy()
            X_oob_noised[:, m] = rng.permutation(X_oob_noised[:, m])
            preds_noised = tree.predict(X_oob_noised)
            for idx, c in zip(oob_idx, preds_noised):
                oob_votes_noised[idx, c] += 1.0

        valid_m = oob_votes_noised.sum(axis=1) > 0
        noised_oob_err = np.mean(np.argmax(oob_votes_noised[valid_m], axis=1) != y[valid_m])
        pct_inc = ((noised_oob_err - base_oob_err) / base_oob_err) * 100.0
        percent_increases.append(pct_inc)

    return percent_increases


def eval_subset_test_error(X_sub, y, F=1, n_repeats=30):
    """Helper to evaluate 90/10 test error using only a selected subset of variables."""
    rs = ShuffleSplit(n_splits=n_repeats, test_size=0.10, random_state=None)
    errs = []
    for tr_idx, te_idx in rs.split(X_sub):
        rf = BreimanRandomForestRI(n_trees=100, F=F, random_state=None).fit(X_sub[tr_idx], y[tr_idx])
        errs.append(rf.score_error(X_sub[te_idx], y[te_idx]))
    return np.mean(errs) * 100.0


def run_section10_experiments():
    (X_diab, y_diab), (X_votes, y_votes) = load_section10_datasets()

    # --- Part A: Diabetes Dataset (Figure 4 + Subset Reruns on Page 24) ---
    print("Running Section 10 Experiment on Diabetes Data (1,000 trees, F=1)...")
    diab_inc = compute_oob_importance(X_diab, y_diab, F=1, n_trees=1000)
    # Approximate values read directly from Breiman's Figure 4 (Page 24)
    paper_fig4_approx = [1.5, 34.0, 5.0, 0.0, 4.0, 10.0, 2.0, 14.0]

    print(f"  [Diabetes Variable Importance (% Error Increase when Noised)]")
    print(f"  {'Variable':<12} | {'Our Reproduction':<18} | {'Paper Fig. 4 Approx':<20}")
    print(f"  {'-'*56}")
    for m in range(8):
        print(f"  {'Var ' + str(m+1):<12} | {diab_inc[m]:+16.1f}% | {paper_fig4_approx[m]:+18.1f}%")

    # Reproduce Page 24 verification reruns using selected variables (Index 1 is Var 2, Index 7 is Var 8, Index 5 is Var 6)
    print("\n  Running Diabetes Variable-Subset Reruns (Page 24 text comparison, 30 splits)...")
    err_all = eval_subset_test_error(X_diab, y_diab, F='log2', n_repeats=30)
    err_v2 = eval_subset_test_error(X_diab[:, [1]], y_diab, F=1, n_repeats=30)
    err_v2_v8 = eval_subset_test_error(X_diab[:, [1, 7]], y_diab, F=1, n_repeats=30)
    err_v2_v6 = eval_subset_test_error(X_diab[:, [1, 5]], y_diab, F=1, n_repeats=30)

    print(f"  {'Variable Subset Used':<26} | {'Our Test Error':<16} | {'Paper Page 24 Reported':<22}")
    print(f"  {'-'*70}")
    print(f"  {'All 8 Variables':<26} | {err_all:14.1f}% | {'23.1% (or 24.2% RI)':>22}")
    print(f"  {'Only Variable 2':<26} | {err_v2:14.1f}% | {'29.7%':>22}")
    print(f"  {'Variable 2 + Variable 8':<26} | {err_v2_v8:14.1f}% | {'29.4%':>22}")
    print(f"  {'Variable 2 + Variable 6':<26} | {err_v2_v6:14.1f}% | {'26.4%':>22}\n")

    # --- Part B: Votes Dataset (Figure 6 + Subset Rerun on Page 24-25) ---
    print("Running Section 10 Experiment on Congressional Votes Data (1,000 trees, F=5)...")
    votes_inc = compute_oob_importance(X_votes, y_votes, F=5, n_trees=1000)
    # Approximate values read from Breiman's Figure 6 (Page 25): Variable 4 spikes past +300%
    paper_fig6_approx = [10, 10, 25, 325, 10, 5, 10, -5, 15, 5, 5, 5, 5, 0, 5, 15]

    err_votes_all = eval_subset_test_error(X_votes, y_votes, F=5, n_repeats=30)
    err_votes_v4 = eval_subset_test_error(X_votes[:, [3]], y_votes, F=1, n_repeats=30)

    print(f"  [Votes Variable-Subset Reruns (Page 24-25 text comparison)]")
    print(f"  {'Variable Subset Used':<26} | {'Our Test Error':<16} | {'Paper Page 24 Reported':<22}")
    print(f"  {'-'*70}")
    print(f"  {'All 16 Variables':<26} | {err_votes_all:14.1f}% | {'4.1%':>22}")
    print(f"  {'Only Variable 4':<26} | {err_votes_v4:14.1f}% | {'4.3%':>22}\n")

    # --- Plot Figure 4 and Figure 6 Side-by-Side with Paper Overlay ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    v_diab = np.arange(1, 9)
    ax1.scatter(v_diab, diab_inc, marker='D', s=70, color='gray', edgecolor='black', label='Our Reproduction')
    ax1.plot(v_diab, paper_fig4_approx, 'ro--', alpha=0.6, label='Paper Fig. 4 Approx')
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax1.set_title('Figure 4 Reproduction: Variable Importance - Diabetes Data')
    ax1.set_xlabel('variable')
    ax1.set_ylabel('percent increase')
    ax1.set_xticks(v_diab)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend()

    v_votes = np.arange(1, 17)
    ax2.scatter(v_votes, votes_inc, marker='D', s=70, color='gray', edgecolor='black', label='Our Reproduction')
    ax2.plot(v_votes, paper_fig6_approx, 'ro--', alpha=0.6, label='Paper Fig. 6 Approx')
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_title('Figure 6 Reproduction: Variable Importance - Votes Data')
    ax2.set_xlabel('variables')
    ax2.set_ylabel('percent increase')
    ax2.set_xticks(v_votes)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    (X_bc, y_bc), (X_sonar, y_sonar) = load_paper_datasets()
    
    # 1. Reproduce Table 2 experiments (Page 13 Table 2)
    run_table2_comparison(X_bc, y_bc, "Breast cancer", paper_vals=[3.2, 2.9, 2.7, 6.3], n_repeats=30)
    run_table2_comparison(X_sonar, y_sonar, "Sonar", paper_vals=[15.6, 15.9, 18.0, 31.7], n_repeats=30)
    
    # 2. Reproduce Table 4 output noise experiments (Page 21 Table 4)
    run_table4_noise_experiment(X_bc, y_bc, "Breast cancer", best_F=1, paper_vals=[43.2, 1.8], n_repeats=30)
    run_table4_noise_experiment(X_sonar, y_sonar, "Sonar", best_F='log2', paper_vals=[15.1, -6.6], n_repeats=30)

    # 3. Plot Figure 1 reproduction on the Sonar dataset (Page 17 Figure 1)
    plot_sonar_figure1(X_sonar, y_sonar, n_iter=15)

    # 4. Plot Variable Importance reproduction on Breast Cancer (Page 24-25 Figures 4-6)
    run_section10_experiments()
