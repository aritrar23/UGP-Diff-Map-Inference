# probabilistic_carta_miec_fixed_leaves.py
# Requires: pip install cvxpy mosek
import cvxpy as cp
import numpy as np

def solve_prob_carta_miec(
    k: int,
    type_names: list,                 # len = |S|
    b_matrix: np.ndarray,             # shape [|V|, |S|], 0/1 for internal nodes; leaves can be arbitrary (ignored)
    edges: list,                      # list of (u, v) edges (parent u -> child v)
    leaf_type_pairs: list,            # list of (v, t_obs) for leaves
    eps: float = 1e-6,
    alpha_l0_on_x: float = 0.0,       # optional sparsity penalty on x
    weights: np.ndarray | None = None # same shape as b_matrix; None => ones
):
    """
    Returns:
        result: dict with keys:
          'status', 'obj', 'x', 'y', 'pi', 'z', 'potencies_idx', 'potencies_named'
    """
    # ---- sets & basic checks
    P = range(k)
    S = range(len(type_names))
    V = range(b_matrix.shape[0])
    L = set(v for (v, _) in leaf_type_pairs)
    tau = {int(v): int(t) for (v, t) in leaf_type_pairs}
    assert b_matrix.shape[1] == len(type_names), "b_matrix second dim must equal number of types"

    if weights is None:
        weights = np.ones_like(b_matrix, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        assert weights.shape == b_matrix.shape

    # ---- decision variables
    x  = cp.Variable((k, len(S)), boolean=True)          # progenitor potencies
    y  = cp.Variable((len(V), k))                        # node->progenitor probabilities
    s  = cp.Variable((len(V), k, len(S)))                # product linearization: y * x
    pi = cp.Variable((len(V), len(S)))                   # permission probabilities
    z  = cp.Variable((len(V), len(S)))                   # epigraph for -log terms

    cons = []

    # Label probabilities
    for v in V:
        cons += [cp.sum(y[v, :]) == 1, y[v, :] >= 0]

    # McCormick envelopes for s = y * x  (x binary)
    for v in V:
        for p in P:
            for t in S:
                cons += [s[v, p, t] >= 0]
                cons += [s[v, p, t] <= y[v, p]]
                cons += [s[v, p, t] <= x[p, t]]
                cons += [s[v, p, t] >= y[v, p] - (1 - x[p, t])]

    # Define pi from s
    for v in V:
        for t in S:
            cons += [pi[v, t] == cp.sum(s[v, :, t])]

    # Bounds / fixed leaves
    for v in V:
        if v in L:
            t_obs = tau[v]
            for t in S:
                cons += [pi[v, t] == (1.0 if t == t_obs else 0.0)]
        else:
            for t in S:
                cons += [pi[v, t] >= eps, pi[v, t] <= 1 - eps]

    # Inheritance: parent >= child
    for (u, v) in edges:
        for t in S:
            cons += [pi[u, t] >= pi[v, t]]

    # Exponential-cone epigraphs for negative log-likelihood (internal nodes only)
    b = np.asarray(b_matrix, dtype=float)
    for v in V:
        if v in L:
            for t in S:
                cons += [z[v, t] == 0]   # leaves contribute nothing
            continue
        for t in S:
            if b[v, t] >= 0.5:
                cons += [cp.ExpCone(1.0, z[v, t],  pi[v, t])]      # z >= -log(pi)
            else:
                cons += [cp.ExpCone(1.0, z[v, t], 1 - pi[v, t])]   # z >= -log(1-pi)

    # Objective
    reg = alpha_l0_on_x * cp.sum(x)
    obj = cp.Minimize(cp.sum(cp.multiply(weights, z)) + reg)

    # Solve
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.MOSEK, verbose=True, mosek_params={"mioTolRelGap": 1e-3})
    except cp.SolverError as e:
        raise RuntimeError(
            "MOSEK is required for mixed-integer exponential-cone optimization. "
            "Install and license MOSEK, then retry."
        ) from e

    # ---- postprocess results
    BIN_TOL = 1e-5
    x_val = np.array(x.value)
    y_val = np.array(y.value)
    pi_val = np.array(pi.value)
    z_val  = np.array(z.value)

    x_bin = (x_val >= 1 - BIN_TOL).astype(int)  # safe rounding
    potencies_idx = {p: np.where(x_bin[p] == 1)[0].tolist() for p in range(k)}
    potencies_named = {p: [type_names[t] for t in potencies_idx[p]] for p in range(k)}

    return {
        "status": prob.status,
        "obj": prob.value,
        "x": x_val,
        "y": y_val,
        "pi": pi_val,
        "z": z_val,
        "x_binary": x_bin,
        "potencies_idx": potencies_idx,
        "potencies_named": potencies_named,
    }

# ---------------------------
# Example usage (fill inputs)
# ---------------------------
if __name__ == "__main__":
    # Example placeholders — replace with your data
    k = 3
    type_names = ["A", "B", "C", "D"]       # |S| = 4
    # toy tree: V = {0,1,2,3}; edges 0->1, 0->2, 2->3; leaves {1,3}
    edges = [(0,1), (0,2), (2,3)]
    leaf_type_pairs = [(1, 2), (3, 0)]      # leaf 1 is type "C", leaf 3 is type "A"

    # Build b_matrix for all nodes; internal nodes can have 0/1 indicating presence in subtree
    # Here we just put some dummy observations for internal nodes 0 and 2
    V = 4; S = len(type_names)
    b_matrix = np.zeros((V, S))
    b_matrix[0, :] = [1, 0, 1, 0]           # internal node 0 observed types in its subtree
    b_matrix[2, :] = [1, 1, 0, 0]           # internal node 2 observed types

    result = solve_prob_carta_miec(
        k=k,
        type_names=type_names,
        b_matrix=b_matrix,
        edges=edges,
        leaf_type_pairs=leaf_type_pairs,
        eps=1e-6,
        alpha_l0_on_x=0.0
    )

    print("\nStatus:", result["status"])
    print("Objective (neg log-likelihood):", result["obj"])
    print("\nFinal progenitor potencies:")
    for p, names in result["potencies_named"].items():
        print(f"  Progenitor {p}: {names if names else '∅'}")
