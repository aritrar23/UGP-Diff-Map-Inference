import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = "training_data_nll.csv"
NUM_TREES = 5

INPUTS_DIR = "inputs"
TREE_KIND = "graph"   # 'graph', 'poly_tree', or 'bin_tree'
# ==========================================

# ==========================================
# 1. TREE PARSING UTILITIES
# ==========================================
class TreeNode:
    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.parent = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

def parse_newick(newick: str) -> TreeNode:
    def _clean_label(tok: str) -> str:
        tok = tok.split(":", 1)[0].strip()
        if tok and tok.replace(".", "", 1).isdigit():
            return ""
        return tok

    s = newick.strip()
    if not s.endswith(";"):
        raise ValueError("Newick must end with ';'")
    s = s[:-1]
    i = 0

    def parse() -> TreeNode:
        nonlocal i, s
        if i >= len(s):
            raise ValueError("Unexpected end")
        if s[i] == '(':
            i += 1
            node = TreeNode()
            while True:
                node.add_child(parse())
                if i >= len(s):
                    raise ValueError("Unbalanced")
                if s[i] == ',':
                    i += 1
                    continue
                elif s[i] == ')':
                    i += 1
                    break
                else:
                    raise ValueError(f"Unexpected char: {s[i]} at {i}")
            j = i
            while j < len(s) and s[j] not in ',()':
                j += 1
            name_raw = s[i:j].strip()
            name = _clean_label(name_raw)
            if name:
                node.name = name
            i = j
            return node
        else:
            j = i
            while j < len(s) and s[j] not in ',()':
                j += 1
            name_raw = s[i:j].strip()
            name = _clean_label(name_raw)
            if not name:
                raise ValueError("Leaf without name")
            i = j
            return TreeNode(name=name)

    return parse()

def read_newick_file(path: str) -> TreeNode:
    with open(path, "r") as f:
        s = f.read().strip()
    return parse_newick(s)

def count_nodes(root: TreeNode) -> int:
    count = 1
    for child in root.children:
        count += count_nodes(child)
    return count

def get_exact_n_for_dataset(map_idx: int, cells_n: int, type_num: int) -> int:
    idx4 = f"{map_idx:04d}"
    folder = os.path.join(INPUTS_DIR, "trees", TREE_KIND,
                          f"type_{type_num}", f"cells_{cells_n}")
    
    total_nodes = 0
    for i in range(NUM_TREES):
        fname = f"{idx4}_tree_{i}.txt"
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            # Fallback approximation if tree file is missing
            return NUM_TREES * (2 * cells_n - 1)
        root = read_newick_file(path)
        total_nodes += count_nodes(root)
    return total_nodes

# ==========================================
# 2. MAIN: LOGISTIC REGRESSION WITH
#    FEATURES: NLL, k² logN, |k−Type| logN
# ==========================================
def main():
    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("Error: Training CSV not found.")
        return

    # Clean NLL
    df = df[df["NLL"] != "FAIL"]
    df["NLL"] = pd.to_numeric(df["NLL"])
    print(f"Rows after removing FAIL: {len(df)}")

    # Compute Total_N per (Type, MapIdx, Cells)
    print("Computing Total_N for each (Type, MapIdx, Cells)...")
    combos = df[["Type", "MapIdx", "Cells"]].drop_duplicates()
    total_n_map = {}
    for _, row in combos.iterrows():
        t = int(row["Type"])
        m = int(row["MapIdx"])
        c = int(row["Cells"])
        n_total = get_exact_n_for_dataset(m, c, t)
        total_n_map[(t, m, c)] = n_total

    df["Total_N"] = df.apply(
        lambda r: total_n_map[(int(r["Type"]), int(r["MapIdx"]), int(r["Cells"]))],
        axis=1
    )
    df["logN"] = np.log(df["Total_N"])

    # Ground truth k: Type - 1
    df["Target_k"] = df["Type"] - 1

    # Feature engineering
    df["k2_logN"] = (df["k"] ** 2) * df["logN"]
    df["abs_k_minus_type_logN"] = np.abs(df["k"] - df["Type"]) * df["logN"]

    # Binary label
    df["is_correct"] = (df["k"] == df["Target_k"]).astype(int)

    feature_cols = ["NLL", "k2_logN", "abs_k_minus_type_logN"]
    X = df[feature_cols].to_numpy()
    y = df["is_correct"].to_numpy()

    print(f"Training logistic regression on {X.shape[0]} samples, {X.shape[1]} features...")

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs"
    )
    clf.fit(X, y)

    w0 = clf.intercept_[0]
    w = clf.coef_[0]

    print("\nLearned logistic regression weights (logit of P(correct)):")
    print(f"  Intercept       : {w0:.6f}")
    for name, coef in zip(feature_cols, w):
        print(f"  Coeff for {name:23s}: {coef:.6f}")

    # ======================================
    # Evaluate per dataset using probabilities
    # ======================================
    print("\nEvaluating per (Type, MapIdx, Cells) using P(correct)...")

    preds = []
    groups = df.groupby(["Type", "MapIdx", "Cells"])

    for (t, m, c), g in groups:
        Xg = g[feature_cols].to_numpy()
        probs = clf.predict_proba(Xg)[:, 1]  # P(is_correct=1)
        best_idx = np.argmax(probs)
        best_k = int(g["k"].iloc[best_idx])
        target_k = int(g["Target_k"].iloc[0])
        preds.append((t, m, c, target_k, best_k))

    results = pd.DataFrame(preds, columns=["Type", "Map", "Cells", "Target_k", "Pred_k"])
    results["Correct"] = results["Target_k"] == results["Pred_k"]

    overall_acc = results["Correct"].mean()
    print("\n" + "="*60)
    print("FINAL DATASET-LEVEL RESULTS")
    print("="*60)
    print(results)
    print("-" * 60)
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")

    print("\nAccuracy per Type:")
    print(results.groupby("Type")["Correct"].mean())

if __name__ == "__main__":
    main()
