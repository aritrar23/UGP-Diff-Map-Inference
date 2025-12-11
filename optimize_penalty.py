import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = "training_data_nll.csv"
TARGET_K = 5
NUM_TREES = 5

# Path Configuration to find the trees
INPUTS_DIR = "inputs" 
TREE_KIND = "graph"   # 'graph', 'poly_tree', or 'bin_tree'
TYPE_NUM = 6          # Change this to match the type in your CSV

# Search Space
LAMBDA_RANGE = np.linspace(0.005, 0.015, 100)
ALPHA_RANGE = np.linspace(0.9, 1.4, 100)
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
        self.children.append(child); child.parent = self

def parse_newick(newick: str) -> TreeNode:
    def _clean_label(tok: str) -> str:
        tok = tok.split(":", 1)[0].strip()
        if tok and tok.replace(".", "", 1).isdigit(): return ""
        return tok

    s = newick.strip()
    if not s.endswith(";"): raise ValueError("Newick must end with ';'")
    s = s[:-1]; i = 0

    def parse() -> TreeNode:
        nonlocal i, s
        if i >= len(s): raise ValueError("Unexpected end")
        if s[i] == '(':
            i += 1
            node = TreeNode()
            while True:
                node.add_child(parse())
                if i >= len(s): raise ValueError("Unbalanced")
                if s[i] == ',':
                    i += 1; continue
                elif s[i] == ')':
                    i += 1; break
                else:
                    raise ValueError(f"Unexpected char: {s[i]} at {i}")
            j = i
            while j < len(s) and s[j] not in ',()': j += 1
            name_raw = s[i:j].strip()
            name = _clean_label(name_raw)
            if name: node.name = name
            i = j
            return node
        else:
            j = i
            while j < len(s) and s[j] not in ',()': j += 1
            name_raw = s[i:j].strip()
            name = _clean_label(name_raw)
            if not name: raise ValueError("Leaf without name")
            i = j
            return TreeNode(name=name)

    root = parse()
    return root

def read_newick_file(path: str) -> TreeNode:
    with open(path, "r") as f: s = f.read().strip()
    return parse_newick(s)

def count_nodes(root: TreeNode) -> int:
    """Recursively count all nodes (Internal + Leaves)"""
    count = 1 
    for child in root.children:
        count += count_nodes(child)
    return count

def get_exact_n_for_dataset(map_idx: int, cells_n: int, type_num: int, tree_kind: str) -> int:
    idx4 = f"{map_idx:04d}"
    folder = os.path.join(INPUTS_DIR, "trees", tree_kind, f"type_{type_num}", f"cells_{cells_n}")
    
    total_nodes = 0
    for i in range(NUM_TREES):
        fname = f"{idx4}_tree_{i}.txt"
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            print(f"[WARN] Tree file missing: {path}. Using 2*Leaves-1 approx.")
            return NUM_TREES * (2 * cells_n - 1)
        root = read_newick_file(path)
        total_nodes += count_nodes(root)
    return total_nodes

# ==========================================
# 2. MAIN OPTIMIZATION LOGIC
# ==========================================
def main():
    # Configure pandas to print everything
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print("Error: Training CSV not found.")
        return

    # Filter
    df = df[df['NLL'] != 'FAIL']
    df['NLL'] = pd.to_numeric(df['NLL'])
    
    pivot = df.pivot_table(index=['MapIdx', 'Cells'], columns='k', values='NLL')
    pivot = pivot.dropna()
    valid_samples = len(pivot)
    
    print(f"Optimization running on {valid_samples} samples.")
    if valid_samples == 0: return

    # Arrays
    nll_matrix = pivot.to_numpy()
    k_vals = pivot.columns.to_numpy()
    
    # Exact N Calculation
    print(f"Calculating exact node counts (Total N) for all datasets...")
    exact_n_list = []
    for map_idx, cells_n in pivot.index:
        n_total = get_exact_n_for_dataset(int(map_idx), int(cells_n), TYPE_NUM, TREE_KIND)
        exact_n_list.append(n_total)

    n_vector = np.array(exact_n_list)
    log_n_vector = np.log(n_vector).reshape(-1, 1)

    print("Running Grid Search...")
    
    accuracy_grid = np.zeros((len(LAMBDA_RANGE), len(ALPHA_RANGE)))
    
    for i, lam in enumerate(LAMBDA_RANGE):
        for j, alp in enumerate(ALPHA_RANGE):
            exp_term = np.exp(alp * k_vals).reshape(1, -1)
            penalties = (lam * log_n_vector) @ exp_term
            scores = nll_matrix + penalties
            best_indices = np.argmin(scores, axis=1)
            predicted_ks = k_vals[best_indices]
            correct = np.sum(predicted_ks == TARGET_K)
            accuracy_grid[i, j] = correct / valid_samples

    # Results
    max_acc = np.max(accuracy_grid)
    best_indices = np.argwhere(accuracy_grid == max_acc)

    print("\n" + "="*60)
    print(f"OPTIMIZATION RESULTS (Exact N)")
    print("="*60)
    print(f"Max Accuracy: {max_acc*100:.2f}%")
    print(f"Number of optimal pairs: {len(best_indices)}")
    
    print("\n--- All Optimal Pairs (Lambda, Alpha) ---")
    for idx in best_indices:
        lam = LAMBDA_RANGE[idx[0]]
        alp = ALPHA_RANGE[idx[1]]
        print(f"  λ = {lam:.5f},  α = {alp:.5f}")

    # Center Calculation
    center = best_indices.mean(axis=0)
    distances = np.linalg.norm(best_indices - center, axis=1)
    closest_idx_in_best = np.argmin(distances)
    final_idx = best_indices[closest_idx_in_best]
    
    best_lambda = LAMBDA_RANGE[final_idx[0]]
    best_alpha = ALPHA_RANGE[final_idx[1]]
    center_acc = accuracy_grid[final_idx[0], final_idx[1]]

    print("\n" + "-"*60)
    print(f"REPRESENTATIVE CENTER PAIR:")
    print(f"  Lambda = {best_lambda:.5f}")
    print(f"  Alpha  = {best_alpha:.5f}")
    print(f"  Accuracy: {center_acc*100:.2f}%")
    print("-" * 60)

    # ---------------------------------------------------------
    # PRINT DETAILED TABLE USING CENTER PARAMETERS
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"DETAILED RESULTS (Using Center Parameters)")
    print("="*60)

    # Recalculate scores specifically for the "Center" pair
    best_exp_term = np.exp(best_alpha * k_vals).reshape(1, -1)
    best_penalties = (best_lambda * log_n_vector) @ best_exp_term
    final_scores = nll_matrix + best_penalties
    final_winners = k_vals[np.argmin(final_scores, axis=1)]
    
    # Build Table
    map_indices = pivot.index.get_level_values('MapIdx')
    cell_counts = pivot.index.get_level_values('Cells')
    is_correct = (final_winners == TARGET_K)
    
    results_df = pd.DataFrame({
        'Map': map_indices,
        'Cells': cell_counts,
        'Total_N': n_vector,        # <--- Exact N Included
        'Pred_k': final_winners,
        'Correct': is_correct
    })
    
    print(results_df)
    print("-" * 60)

    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(accuracy_grid, 
                     xticklabels=np.round(ALPHA_RANGE, 2)[::5], 
                     yticklabels=np.round(LAMBDA_RANGE, 2)[::5],
                     cmap="viridis")
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, len(ALPHA_RANGE), 5))
    ax.set_yticks(np.arange(0, len(LAMBDA_RANGE), 5))
    plt.xlabel("Alpha (Slope)")
    plt.ylabel("Lambda (Base)")
    plt.title(f"Accuracy with Exact N\nMax: {max_acc*100:.1f}%")

    rows, cols = zip(*best_indices)
    plt.scatter(np.array(cols) + 0.5, np.array(rows) + 0.5, 
                color='red', marker='*', s=80, label='All Optimal Pairs')

    plt.scatter(final_idx[1] + 0.5, final_idx[0] + 0.5, 
                color='white', edgecolors='black', marker='*', s=250, 
                label='Representative Center')

    plt.legend()
    plt.savefig("optimization_heatmap_exact.png")
    print("\nHeatmap saved to 'optimization_heatmap_exact.png'")

if __name__ == "__main__":
    main()