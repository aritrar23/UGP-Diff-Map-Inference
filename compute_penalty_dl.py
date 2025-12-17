import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn

# =====================================
# CONFIGURATION
# =====================================

# Choose mode: "multi" for full CSV with many datasets, "single" for one dataset
MODE = "single"   # "multi" or "single"

# Files
MODEL_PATH = "trained_nn_model.pth"

# MULTI-DATASET MODE INPUT (Type, MapIdx, Cells, k, NLL, etc.)
MULTI_INPUT_CSV = "testing_data_nll_all.csv"

# SINGLE-DATASET MODE INPUT (just columns: k, logN, NLL)
SINGLE_INPUT_CSV = "tls_k_sweep.csv"

# Tree configuration (used only in multi mode to compute Total_N)
NUM_TREES = 5
INPUTS_DIR = "inputs"
TREE_KIND = "graph"   # 'graph', 'poly_tree', or 'bin_tree'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLOT_DIR_MULTI = "nn_k_plots_multi"
PLOT_DIR_SINGLE = "nn_k_plots_single"

# =====================================
# MODEL + TREE UTILITIES
# =====================================

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
            # fallback if tree file missing
            return NUM_TREES * (2 * cells_n - 1)
        root = read_newick_file(path)
        total_nodes += count_nodes(root)
    return total_nodes

class SimpleNN(nn.Module):
    def __init__(self, in_dim=3, h1=32, h2=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1)  # logit
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# =====================================
# LOAD TRAINED MODEL + NORMALIZATION
# =====================================

def load_model_and_norm():
    # IMPORTANT: weights_only=False because checkpoint has numpy arrays (mean, std)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    mean = checkpoint["mean"]
    std = checkpoint["std"]

    model = SimpleNN(in_dim=3, h1=32, h2=16).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("✅ Loaded trained model and normalization from", MODEL_PATH)
    return model, mean, std

# =====================================
# MULTI-DATASET MODE
# =====================================

def run_multi_mode():
    os.makedirs(PLOT_DIR_MULTI, exist_ok=True)
    model, mean, std = load_model_and_norm()

    print(f"Loading {MULTI_INPUT_CSV}...")
    df = pd.read_csv(MULTI_INPUT_CSV)
    df = df[df["NLL"] != "FAIL"]
    df["NLL"] = pd.to_numeric(df["NLL"])
    print(f"Rows after removing FAIL: {len(df)}")

    # Compute Total_N & logN per (Type, MapIdx, Cells)
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

    # Ground truth
    df["Target_k"] = df["Type"] - 1

    # Features
    feature_cols = ["NLL", "k", "logN"]
    X_np = df[feature_cols].to_numpy(dtype=np.float32)
    X_std = (X_np - mean) / std
    X = torch.from_numpy(X_std).to(DEVICE)

    with torch.no_grad():
        logits = model(X).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    df["prob_correct"] = probs

    # Pick best k per dataset
    preds = []
    for (t, m, c), g in df.groupby(["Type", "MapIdx", "Cells"]):
        idx_best = g["prob_correct"].idxmax()
        best_k = int(df.loc[idx_best, "k"])
        target_k = int(g["Target_k"].iloc[0])
        preds.append((t, m, c, target_k, best_k))

        # Plot P(correct) vs k for this dataset
        g_sorted = g.sort_values("k")
        ks = g_sorted["k"].to_numpy()
        ps = g_sorted["prob_correct"].to_numpy()

        plt.figure()
        plt.plot(ks, ps, marker="o", label="P(correct)")
        plt.axvline(target_k, color="green", linestyle="--", label=f"True k={target_k}")
        plt.axvline(best_k, color="red", linestyle=":", label=f"Pred k={best_k}")
        plt.xlabel("k")
        plt.ylabel("P(correct)")
        plt.ylim(0, 1)
        plt.title(f"Type {t}, Map {m}, Cells {c}")
        plt.legend()
        fname = f"type{t}_map{m}_cells{c}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_MULTI, fname))
        plt.close()

    results = pd.DataFrame(preds, columns=["Type", "Map", "Cells", "Target_k", "Pred_k"])
    results["Correct"] = results["Target_k"] == results["Pred_k"]
    overall_acc = results["Correct"].mean()

    print("\n" + "="*60)
    print("FINAL PREDICTIONS (MULTI MODE)")
    print("="*60)
    print(results)
    print("-" * 60)
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print("\nAccuracy per Type:")
    print(results.groupby("Type")["Correct"].mean())
    print(f"\nPlots saved to: {PLOT_DIR_MULTI}/")

# =====================================
# SINGLE-DATASET MODE
# =====================================

def run_single_mode():
    """
    SINGLE_INPUT_CSV should have columns:
        k, logN, NLL
    representing ONE dataset with several candidate k values.
    """
    os.makedirs(PLOT_DIR_SINGLE, exist_ok=True)
    model, mean, std = load_model_and_norm()

    print(f"Loading single-dataset file {SINGLE_INPUT_CSV}...")
    df = pd.read_csv(SINGLE_INPUT_CSV)

    # Expect columns: k, logN, NLL
    if not {"k", "logN", "NLL"}.issubset(df.columns):
        raise ValueError("SINGLE_INPUT_CSV must contain columns: 'k', 'logN', 'NLL'")

    df = df[df["NLL"] != "FAIL"] if df["NLL"].dtype == object else df
    df["NLL"] = pd.to_numeric(df["NLL"])

    # Features
    feature_cols = ["NLL", "k", "logN"]
    X_np = df[feature_cols].to_numpy(dtype=np.float32)
    X_std = (X_np - mean) / std
    X = torch.from_numpy(X_std).to(DEVICE)

    with torch.no_grad():
        logits = model(X).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    df["prob_correct"] = probs

    # Pick best k
    idx_best = df["prob_correct"].idxmax()
    best_k = int(df.loc[idx_best, "k"])

    print("\n" + "="*60)
    print("SINGLE-DATASET PREDICTION (DL MODE)")
    print("="*60)
    print(df[["k", "NLL", "logN", "prob_correct"]])
    print("-" * 60)
    print(f"Predicted optimal k: {best_k}")

    # Plot P(correct) vs k
    df_sorted = df.sort_values("k")
    ks = df_sorted["k"].to_numpy()
    ps = df_sorted["prob_correct"].to_numpy()

    plt.figure()
    plt.plot(ks, ps, marker="o", label="P(correct)")
    plt.axvline(best_k, color="red", linestyle=":", label=f"Pred k={best_k}")
    plt.xlabel("k")
    plt.ylabel("P(correct)")
    plt.ylim(0, 1)
    plt.title("Single dataset: NN probability of correctness vs k")
    plt.legend()
    fname = "single_dataset_prob_vs_k.png"
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR_SINGLE, fname))
    plt.close()

    print(f"\nPlot saved to: {os.path.join(PLOT_DIR_SINGLE, fname)}")

# =====================================
# MAIN
# =====================================

def main():
    if MODE == "multi":
        run_multi_mode()
    elif MODE == "single":
        run_single_mode()
    else:
        raise ValueError("MODE must be 'multi' or 'single'")

if __name__ == "__main__":
    main()
