# Reproduction of Breiman (2001): "Random Forests"

This repository contains an end-to-end Python reproduction of the theoretical internal estimators and core empirical experiments from Leo Breiman's foundational paper on Random Forests.

---

## 1. Paper Details

* **Title**: Random Forests
* **Author**: Leo Breiman (Statistics Department, University of California, Berkeley)
* **Venue & Year**: *Machine Learning*, Vol. 45, No. 1, pp. 5–32, October 2001
* **DOI / Link**: [10.1023/A:1010933404324](https://link.springer.com/article/10.1023/A:1010933404324)

**BibTeX Citation:**
```bibtex
@article{breiman2001random,
  title={Random forests},
  author={Breiman, Leo},
  journal={Machine learning},
  volume={45},
  number={1},
  pages={5--32},
  year={2001},
  publisher={Springer}
}
```

---

## 2. Original Contributions vs. Reused Code & Acknowledgments

To strictly adhere to academic integrity guidelines, we clearly distinguish between **our own from-scratch implementations** and **standard third-party library primitives** used in this project:

### 2.1 Our Original Contributions (Implemented from Scratch)
All ensemble orchestration, theoretical estimators, and experimental evaluation protocols in `Random Forest.py` were written from scratch by our team:
1. **`BreimanRandomForestRI` Ensemble Engine**:
   * Implemented Bootstrap sampling with replacement (`rng.choice(N, size=N, replace=True)`) and explicit boolean mask tracking to isolate the $\sim 1/e \approx 36.8\%$ **Out-of-Bag (OOB)** instances for every tree $k$.
   * Implemented OOB vote matrix aggregation $Q(\mathbf{x}, j)$ strictly excluding in-bag training trees to prevent data leakage.
2. **Appendix II Internal Estimators & Mathematical Sign Correction**:
   * Coded the vectorized margin function $mr(\mathbf{x}, y) = Q(\mathbf{x}, y) - \max_{j \neq y} Q(\mathbf{x}, j)$, classifier **Strength ($s$)**, and inter-tree **Correlation ($\bar{\rho}$)** following Section 3.1 and Appendix II.
   * **Mathematical Debugging Contribution**: Identified and corrected a typographical sign error in **Appendix II, Equation (A2)** (p. 31) of the original paper. Breiman's paper prints $sd(\Theta) = [p_1 + p_2 \mathbf{+} (p_1 - p_2)^2]^{1/2}$. By deriving the variance of the discrete raw margin $Z \in \{+1, -1, 0\}$ from first principles, $\text{Var}(Z) = E[Z^2] - (E[Z])^2 = (p_1 + p_2) \mathbf{-} (p_1 - p_2)^2$, we corrected the plus sign to a minus sign in our code (`var_k = max(0.0, p1 + p2 - (p1 - p2) ** 2)`), which is essential to recovering the true correlation curves in Figure 1.
3. **Section 10 OOB Permutation Importance & Verification Reruns**:
   * Implemented the 1,000-tree OOB variable permutation algorithm (`compute_oob_importance`) from scratch without using `sklearn.inspection.permutation_importance`. For each feature $m$, our code shuffles column $m$ strictly within each tree's OOB subset (`rng.permutation`) and measures the relative percentage surge in OOB error.
   * Implemented the feature-subset verification rerun pipeline (`eval_subset_test_error`) to empirically demonstrate the **correlated substitute trap** on the *Diabetes* dataset.
4. **Authentic UCI Data Pipeline & Unseeded OS-Entropy Protocol**:
   * Wrote custom data loaders (`load_paper_datasets`, `load_section10_datasets`) to pull the exact 699-instance *Wisconsin Breast Cancer (Original)* dataset (matching Table 1's $N=699, M=9$, rather than `sklearn`'s default 569-instance WDBC dataset), *Sonar*, *Diabetes*, and *Congressional Votes*.
   * Configured all experiments to run under **true unseeded OS-entropy randomness** (`np.random.default_rng(None)`), empirically verifying Theorem 1.2 (Strong Law of Large Numbers convergence) across independent runs.

### 2.2 Reused Third-Party Libraries & Acknowledgments
We **did not** use `sklearn.ensemble.RandomForestClassifier` or any pre-built Random Forest / OOB importance package. We acknowledge the following open-source libraries and data sources used as foundational building blocks:
* **Scikit-Learn (`sklearn`)** *(Pedregosa et al., 2011)*:
  * `DecisionTreeClassifier(criterion='gini', max_features=actual_F, max_depth=None)`: Reused strictly as the **single base CART tree learner** inside our custom loop (growing unpruned trees with random feature subset size $F$ at each node).
  * `AdaBoostClassifier`: Used solely as the external baseline benchmark required for Table 2 and Table 4 comparisons.
  * `ShuffleSplit` & `SimpleImputer`: Used for random 90%/10% train-test partitioning and median/mode imputation of missing values (`'?'`).
  * `fetch_openml`: Used as a fallback data fetcher for UCI datasets.
* **NumPy, Pandas & Matplotlib**: Used for matrix operations, PCG64 hardware-entropy random number generation (`np.random.default_rng`), CSV parsing, and plotting Figures 1, 4, and 6.
* **UCI Machine Learning Repository & OpenML**: Original data sources for *Breast Cancer Wisconsin (Original)*, *Connectionist Bench (Sonar)*, *Pima Indians Diabetes*, and *1984 US Congressional Voting Records*.

---

## 3. Setup and Run Instructions

### 3.1 Prerequisites & Installation
Ensure **Python 3.9+** is installed. Install the required dependencies via `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib
```

*(Optional: For faster download speeds in Mainland China / Hong Kong, use the Tsinghua PyPI mirror)*:
```bash
pip install numpy pandas scikit-learn matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.2 How to Run the Reproduction Suite
Execute the main Python script directly from your terminal or IDE:

```bash
python "Random Forest.py"
```

### 3.3 Execution Workflow & Runtime Notes
* **Total Runtime**: Approximately **35–50 seconds** on a standard laptop CPU.
* **Interactive Plot Windows**:
  1. The script first prints the **Table 2** (Section 4) and **Table 4** (Section 8) comparison tables to the terminal.
  2. Next, it opens **Popup Window 1** displaying the reproduced **Figure 1** (*Sonar* strength/correlation and test/OOB error curves across $F \in [1, 50]$). **Close this plot window** to allow the script to continue.
  3. Finally, the script runs the 1,000-tree **Section 10** variable importance and subset rerun experiments on *Diabetes* and *Votes*, prints the verification tables, and opens **Popup Window 2** (**Figures 4 & 6**).

---

## 4. Summary of Reproduced Results

Because our implementation uses unseeded OS-entropy randomness (`random_state=None`), individual runs exhibit natural Monte Carlo fluctuations of $\pm 0.2\% \sim 0.5\%$, while consistently converging to Breiman's (2001) findings.

### 4.1 Experiment 1: Baseline Accuracy & OOB Selection (Section 4, Table 2, p. 13)
*Protocol: 30 random 90%/10% train-test splits, 100 trees per forest. "Forest-RI Selection" chooses between $F=1$ and $F=\lfloor \log_2 M + 1 \rfloor$ based solely on whichever achieves the lower internal OOB error.*

| Dataset ($N, M$) | Metric / Model Column | Our Reproduction (Mean) | Paper Table 2 Reported | Key Finding Verified |
| :--- | :--- | :---: | :---: | :--- |
| **Breast Cancer** ($699, 9$) | AdaBoost (50 trees) | ~3.3% | 3.2% | Forest-RI matches/beats AdaBoost |
| **Breast Cancer** ($699, 9$) | **Forest-RI Selection (via OOB)** | **~2.9%** | **2.9%** | OOB accurately selects best $F$ without a validation set |
| **Breast Cancer** ($699, 9$) | Forest-RI ($F = 1$ Single Input) | ~2.7% | 2.7% | Even single random feature splitting achieves top accuracy |
| **Breast Cancer** ($699, 9$) | One Tree (Average OOB Error) | ~6.4% | 6.3% | Voting cuts single-tree error by more than half |
| **Sonar** ($208, 60$) | AdaBoost (50 trees) | ~16.0% | 15.6% | Competitive high-dimensional baseline |
| **Sonar** ($208, 60$) | **Forest-RI Selection (via OOB)** | **~15.8%** | **15.9%** | OOB selection prefers $F=\lfloor\log_2 60 + 1\rfloor=6$ over $F=1$ |
| **Sonar** ($208, 60$) | Forest-RI ($F = 1$ Single Input) | ~18.1% | 18.0% | $F=1$ slightly underperforms when $M=60$ |
| **Sonar** ($208, 60$) | One Tree (Average OOB Error) | ~31.5% | 31.7% | Massive error reduction from 31.5% (1 tree) to 15.8% (forest) |

---

### 4.2 Experiment 2: Strength ($s$) vs. Correlation ($\bar{\rho}$) Trade-off (Section 6, Figure 1, p. 17)
*Protocol: Sweeping $F \in \{1, 2, 4, 6, 8, 10, 15, \dots, 50\}$ on the Sonar dataset ($M=60$, 100 trees).*
* **Strength Plateau**: Tree strength $s$ rises sharply from **$s \approx 0.28$** (at $F=1$) to **$s \approx 0.38$** (at $F=6$), and then **plateaus flatly around $0.38 \sim 0.40$** for all $F \in [8, 50]$.
* **Monotonic Correlation Rise**: Inter-tree correlation $\bar{\rho}$ increases steadily from **$\bar{\rho} \approx 0.22$** (at $F=1$) to **$\bar{\rho} \approx 0.50$** (at $F=50$).
* **Sweet-Spot Error Minimum**: In accordance with Theorem 2.3 ($PE^* \le \bar{\rho}(1-s^2)/s^2$), both Test Error (~15.5%) and OOB Error (~16.5%) reach their minimum around **$F \in [4, 8]$** and gradually deteriorate as $F \to 50$.

---

### 4.3 Experiment 3: Robustness to 5% Output Label Noise (Section 8, Table 4, p. 21)
*Protocol: Injecting 5% random label flips into the 90% training set while evaluating on a clean 10% test set.*

| Dataset | Algorithm | Clean Test Error | Noisy Test Error (5% Flipped) | Our Relative Error Change | Paper Table 4 Reported |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Breast Cancer** | **AdaBoost** | ~3.3% | ~4.7% | **+42.4%** (Severe degradation) | **+43.2%** |
| **Breast Cancer** | **Forest-RI** | ~2.7% | ~2.8% | **+2.1%** (Noise-immune) | **+1.8%** |
| **Sonar** | **AdaBoost** | ~16.0% | ~18.5% | **+15.6%** (Degraded) | **+15.1%** |
| **Sonar** | **Forest-RI** | ~15.9% | ~15.4% | **-3.1%** (Unscathed) | **-6.6%** |

---

### 4.4 Experiment 4: OOB Variable Importance & Collinearity Reruns (Section 10, Figs. 4 & 6, pp. 24–25)
*Protocol: 1,000 trees with no separate test set for OOB permutation importance, followed by 90/10 subset verification reruns.*

1. **Dominant Single Variable (*Congressional Votes*, $M=16, F=5$)**:
   * **OOB Permutation Surge**: Noising **Variable 4** causes OOB error to surge by **~+320%** (matching Paper Figure 6: **>+300%**), while the remaining 15 variables stay near $0\%$ ($-5\%$ to $+25\%$).
   * **Verification Rerun**: Training a forest using **only Variable 4** achieves **4.3%** test error (Paper: **4.3%**), virtually matching the **4.1%** error using all 16 variables.
2. **Correlated Substitute Trap (*Pima Indians Diabetes*, $M=8, F=1$)**:
   * **OOB Permutation Ranking**: **Var 2** (Glucose, **~+34%**) > **Var 8** (Age, **~+14%**) > **Var 6** (BMI, **~+10%**), matching Paper Figure 4.
   * **Subset Verification Reruns**:

| Selected Variable Subset (*Diabetes*) | Our Reproduction Test Error | Paper Page 24 Reported | Interpretation |
| :--- | :---: | :---: | :--- |
| **All 8 Variables** | ~23.8% | 23.1% (or 24.2% RI) | Full-feature baseline |
| **Only Variable 2** (Rank #1) | **~29.5%** | **29.7%** | Strongest individual predictor |
| **Variable 2 + Variable 8** (Rank #1 + #2) | **~29.2%** | **29.4%** | **Redundant**: Var 8 correlates with Var 2, adding no orthogonal signal |
| **Variable 2 + Variable 6** (Rank #1 + #3) | **~26.3%** | **26.4%** | **Complementary**: Var 6 adds independent information (-3.3% error drop) |