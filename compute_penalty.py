import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = "testing_data_nll.csv"
TARGET_K = 5
NUM_TREES = 5

# Inputs configuration for Exact N calculation
INPUTS_DIR = "inputs"
TREE_KIND = "graph"
TYPE_NUM = 6

# Single pair to evaluate
# LAMBDA = 0.007216
# ALPHA = 1.303

# LAMBDA = 0.00908
# ALPHA = 1.38485

LAMBDA = 0.01500  
ALPHA = 1.28889
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
                if s[i] == ',': i += 1; continue
                elif s[i] == ')': i += 1; break
                else: raise ValueError(f"Unexpected char: {s[i]} at {i}")
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
    return parse()

def read_newick_file(path: str) -> TreeNode:
    with open(path, "r") as f: s = f.read().strip()
    return parse_newick(s)

def count_nodes(root: TreeNode) -> int:
    count = 1
    for child in root.children: count += count_nodes(child)
    return count

def get_exact_n_for_dataset(map_idx: int, cells_n: int) -> int:
    idx4 = f"{map_idx:04d}"
    folder = os.path.join(INPUTS_DIR, "trees", TREE_KIND, f"type_{TYPE_NUM}", f"cells_{cells_n}")
    total_nodes = 0
    for i in range(NUM_TREES):
        fname = f"{idx4}_tree_{i}.txt"
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            return NUM_TREES * (2 * cells_n - 1)
        root = read_newick_file(path)
        total_nodes += count_nodes(root)
    return total_nodes

# ==========================================
# 2. MAIN EVALUATION
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
        print("Error: Testing CSV not found.")
        return

    # 1. Filter and Clean
    df = df[df['NLL'] != 'FAIL']
    df['NLL'] = pd.to_numeric(df['NLL'])
    
    pivot = df.pivot_table(index=['MapIdx', 'Cells'], columns='k', values='NLL')
    pivot = pivot.dropna()
    valid_samples = len(pivot)
    
    if valid_samples == 0:
        return

    nll_matrix = pivot.to_numpy()
    k_vals = pivot.columns.to_numpy()
    
    # 2. Calculate Exact N
    print("Calculating exact node counts (Total N) for all datasets...")
    exact_n_list = []
    for map_idx, cells_n in pivot.index:
        n_total = get_exact_n_for_dataset(int(map_idx), int(cells_n))
        exact_n_list.append(n_total)
    
    n_vector = np.array(exact_n_list)
    log_n_vector = np.log(n_vector).reshape(-1, 1)

    # 3. Compute Penalties
    print(f"\nEvaluating: λ = {LAMBDA}, α = {ALPHA}")
    exp_term = np.exp(ALPHA * k_vals).reshape(1, -1)
    penalties = (LAMBDA * log_n_vector) @ exp_term
    scores = nll_matrix + penalties

    best_indices = np.argmin(scores, axis=1)
    predicted_ks = k_vals[best_indices]
    
    # 4. Build Full Results Table
    map_indices = pivot.index.get_level_values('MapIdx')
    cell_counts = pivot.index.get_level_values('Cells')
    is_correct = (predicted_ks == TARGET_K)
    
    results_df = pd.DataFrame({
        'Map': map_indices,
        'Cells': cell_counts,
        'Total_N': n_vector,        # <--- Added Total N
        'Pred_k': predicted_ks,
        'Correct': is_correct
    })
    
    accuracy = np.mean(is_correct)

    print("\n" + "="*60)
    print(f"FULL RESULTS FOR ALL {valid_samples} DATASETS")
    print("="*60)
    print(results_df)
    print("-" * 60)
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Average N:        {np.mean(n_vector):.1f}")
    print("-" * 60)

if __name__ == "__main__":
    main()