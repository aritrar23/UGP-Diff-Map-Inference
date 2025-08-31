from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import csv
from typing import Iterable, Tuple, List, Optional, Dict, Set, FrozenSet
from collections import Counter, defaultdict
from tqdm import trange
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MAP structure search for Carta-CDMIP model (debug-instrumented, with tqdm bars in parallel).

- Newick parser (no external deps)
- DP over labelings with (O,D) sparse tables
- Priors: fixed-k (uniform over potency sets) OR Bernoulli(pi_P); edges Bernoulli(rho)
- Stochastic hill-climb + simulated annealing over F=(Z,A)
- Parallel restarts with visible per-worker progress bars (tqdm position)
- Verbose debug prints controlled by env var CDMIP_DEBUG (1=on, 0=off)
"""

import math
import itertools
import json

# ----------------------------
# Tiny debug logger
# ----------------------------
DEBUG = bool(int(os.environ.get("CDMIP_DEBUG", "1")))  # export CDMIP_DEBUG=0 to silence

def debug(*args, **kwargs):
    if DEBUG:
        print("[debug]", *args, **kwargs)

# ----------------------------
# Tree structures and Newick
# ----------------------------

class TreeNode:
    def __init__(self, name: Optional[str] = None):
        self.name: Optional[str] = name
        self.children: List["TreeNode"] = []
        self.parent: Optional["TreeNode"] = None
        self.potency = None

    def is_leaf(self): return len(self.children) == 0

    def add_child(self, child: "TreeNode"):
        self.children.append(child); child.parent = self

    def __repr__(self):
        return f"Leaf({self.name})" if self.is_leaf() else f"Node({self.name}, k={len(self.children)})"


def iter_edges(root: TreeNode) -> Iterable[Tuple[TreeNode, TreeNode]]:
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            yield (node, child)
            stack.append(child)


def count_edges(root: TreeNode) -> int:
    return sum(1 for _ in iter_edges(root))


# -------------------------
# Union-only Fitch labeling
# -------------------------
def assign_union_potency(root: TreeNode, leaf_type_map: Dict[str, str]) -> Set[str]:
    if root.is_leaf():
        if root.name is None:
            raise KeyError("Leaf has no .name; cannot map to leaf_type_map")
        if root.name not in leaf_type_map:
            raise KeyError(f"Leaf name '{root.name}' not found in leaf_type_map")
        root.potency = {leaf_type_map[root.name]}
        return root.potency

    union_set: Set[str] = set()
    for child in root.children:
        child_set = assign_union_potency(child, leaf_type_map)
        union_set |= child_set
    root.potency = union_set
    return root.potency


# -------------------------
# Per-tree transition counts
# -------------------------
def per_tree_transition_counts(root: TreeNode) -> Counter:
    C = Counter()
    for (u, v) in iter_edges(root):
        su = frozenset(u.potency if u.potency is not None else set())
        sv = frozenset(v.potency if v.potency is not None else set())
        if su != sv:
            C[(su, sv)] += 1
    return C


# -------------------------
# Aggregation + top-k picking
# -------------------------
def init_progenitors_union_fitch(
    S: List[str],
    trees: List["TreeNode"],
    leaf_type_maps: List[Dict[str, str]],
    k: int,
) -> Tuple[Dict[Tuple[frozenset, frozenset], float], Set[frozenset]]:
    debug("Init progenitors via union-Fitch; |S|=", len(S), "k=", k, "trees=", len(trees))
    if len(trees) != len(leaf_type_maps):
        raise ValueError("Provide exactly one leaf_type_map per tree (same order).")

    ROOT = frozenset(S)
    aggregated_transitions: Dict[Tuple[frozenset, frozenset], float] = defaultdict(float)
    row_sum: Dict[frozenset, float] = defaultdict(float)

    for ti, (tree, ltm) in enumerate(zip(trees, leaf_type_maps), 1):
        debug(f"  Fitch on tree {ti}")
        assign_union_potency(tree, ltm)
        C_T = per_tree_transition_counts(tree)
        T = sum(C_T.values())
        debug(f"    real transitions={T}")
        if T == 0:
            continue
        for (i_set, j_set), cnt in C_T.items():
            incr = cnt / T
            aggregated_transitions[(i_set, j_set)] += incr
            row_sum[i_set] += incr

    Z_init: Set[frozenset] = {ROOT} | {frozenset([cell]) for cell in S}
    candidates = [ps for ps in row_sum.keys() if ps != ROOT and len(ps) >= 2]
    candidates.sort(key=lambda ps: (-row_sum[ps], -len(ps), tuple(sorted(ps))))
    top_progenitors = candidates[:max(0, k - 1)]
    Z_init |= set(top_progenitors)

    debug("  Aggregated transitions:", len(aggregated_transitions))
    debug("  Candidates (size>=2):", len([ps for ps in row_sum.keys() if len(ps) >= 2]))
    debug("  Z_init size:", len(Z_init))
    return dict(aggregated_transitions), Z_init


def parse_newick(newick: str) -> TreeNode:
    debug("Parsing Newick (len):", len(newick))

    def _clean_label(tok: str) -> str:
        tok = tok.split(":", 1)[0].strip()
        if tok and tok.replace(".", "", 1).isdigit():
            return ""
        return tok

    s = newick.strip()
    if not s.endswith(";"):
        raise ValueError("Newick must end with ';'")
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
            if name:
                node.name = name
            i = j
            return node
        else:
            j = i
            while j < len(s) and s[j] not in ',()': j += 1
            name_raw = s[i:j].strip()
            name = _clean_label(name_raw)
            if not name:
                raise ValueError("Leaf without name")
            i = j
            return TreeNode(name=name)

    root = parse()
    if i != len(s):
        raise ValueError(f"Trailing characters: '{s[i:]}'")
    debug("Newick parsed OK")
    return root


def to_newick(root: TreeNode) -> str:
    def rec(n: TreeNode) -> str:
        if n.is_leaf(): return n.name or ""
        return f"({','.join(rec(c) for c in n.children)}){n.name or ''}"
    return rec(root) + ";"


def read_newick_file(path: str) -> TreeNode:
    debug("Reading NEWICK:", path)
    with open(path, "r") as f:
        s = f.read().strip()
    debug("Newick size:", len(s), "preview:", s[:80].replace("\n", " ") + ("..." if len(s) > 80 else ""))
    root = parse_newick(s)
    debug("Parsed tree; edges:", count_edges(root), "leaves:", len(collect_leaf_names(root)))
    return root


def write_newick_file(path: str, root: TreeNode):
    debug("Writing NEWICK:", path)
    with open(path, "w") as f:
        f.write(to_newick(root) + "\n")


def random_tree_newick(n_leaves: int, leaf_prefix="L") -> Tuple[TreeNode, List[str]]:
    leaves = [TreeNode(f"{leaf_prefix}{i+1}") for i in range(n_leaves)]
    nodes = leaves[:]
    while len(nodes) > 1:
        k = 2 if len(nodes) < 4 else random.choice([2, 2, 2, 3])
        k = min(k, len(nodes))
        picks = random.sample(nodes, k)
        for p in picks: nodes.remove(p)
        parent = TreeNode()
        for p in picks: parent.add_child(p)
        nodes.append(parent)
    return nodes[0], [l.name for l in leaves]


def collect_leaf_names(root: TreeNode) -> List[str]:
    out = []
    def dfs(v):
        if v.is_leaf(): out.append(v.name)
        else:
            for c in v.children: dfs(c)
    dfs(root); return out

# ----------------------------
# Potency universe and structure
# ----------------------------

def all_nonempty_subsets(S: List[str], max_size: Optional[int] = None) -> List[FrozenSet[str]]:
    R = len(S); max_k = R if max_size is None else min(max_size, R)
    res = []
    for k in range(1, max_k + 1):
        for comb in itertools.combinations(S, k): res.append(frozenset(comb))
    return res

def singletons(S: List[str]) -> Set[FrozenSet[str]]:
    return {frozenset([t]) for t in S}

def build_Z_active(S: List[str], fixed_k: Optional[int], max_potency_size: Optional[int], seed=0) -> Set[FrozenSet[str]]:
    rng = random.Random(seed)
    P_all = all_nonempty_subsets(S, max_potency_size)
    singles = singletons(S)
    multis = [P for P in P_all if len(P) >= 2]
    Z = set(singles)
    if fixed_k is not None:
        if fixed_k > len(multis):
            raise ValueError("fixed_k too large")
        root = frozenset(S)
        Z.add(root)
        remaining_multis = [P for P in multis if P != root]
        Z.update(rng.sample(remaining_multis, fixed_k - 1))
    else:
        Z.update(multis)
    return Z

def admissible_edge(P: FrozenSet[str], Q: FrozenSet[str], unit_drop: bool) -> bool:
    if Q == P: return False
    if not Q.issubset(P): return False
    if len(Q) >= len(P): return False
    if unit_drop and len(P - Q) != 1: return False
    return True

def build_edges(Z_active: Set[FrozenSet[str]], forbid_fn=None, unit_drop=True) -> Dict[Tuple[FrozenSet[str], FrozenSet[str]], int]:
    A = {}
    for P in Z_active:
        for Q in Z_active:
            if not admissible_edge(P, Q, unit_drop): continue
            if forbid_fn and forbid_fn(P, Q): continue
            A[(P, Q)] = 1
    debug("build_edges: |A|=", len(A))
    return A

def build_mid_sized_connected_dag(Z_active, keep_prob=0.3, unit_drop=False, rng=None):
    debug("Building mid-density DAG: nodes=", len(Z_active), "keep_prob=", keep_prob, "unit_drop=", unit_drop)
    if rng is None:
        rng = random.Random()

    root = frozenset().union(*Z_active)
    if root not in Z_active:
        raise ValueError("Root potency (all singletons) not present in Z_active.")

    nodes = list(Z_active)
    full_edges = {
        (P, Q): 1
        for P in Z_active
        for Q in Z_active
        if P != Q and admissible_edge(P, Q, unit_drop)
    }

    A = {}
    visited = {root}
    to_visit = set(nodes) - {root}

    while to_visit:
        parent = rng.choice(list(visited))
        candidates = [(parent, q) for q in to_visit if (parent, q) in full_edges]
        if not candidates:
            candidates = [
                (p, q) for p in visited for q in to_visit if (p, q) in full_edges
            ]
        edge = rng.choice(candidates)
        A[edge] = 1
        visited.add(edge[1])
        to_visit.remove(edge[1])

    debug("  Spanning edges added:", len(A))

    for edge in full_edges:
        if edge in A:
            continue
        if rng.random() < keep_prob:
            A[edge] = 1

    debug("  Final edges:", len(A))
    return A


def transitive_closure(labels: List[FrozenSet[str]], A: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int]) -> Dict[FrozenSet[str], Set[FrozenSet[str]]]:
    idx = {L: i for i, L in enumerate(labels)}
    n = len(labels)
    M = [[False] * n for _ in range(n)]
    for i in range(n): M[i][i] = True
    for (P, Q), v in A.items():
        if v:
            i, j = idx[P], idx[Q]; M[i][j] = True
    for k in range(n):
        Mk = M[k]
        for i in range(n):
            if M[i][k]:
                Mi = M[i]
                for j in range(n):
                    if Mk[j]: Mi[j] = True
    Reach = {L: set() for L in labels}
    for i, L in enumerate(labels):
        for j, U in enumerate(labels):
            if M[i][j]: Reach[L].add(U)
    return Reach


# ----------------------------
# DP over labelings (integrated Beta)
# ----------------------------

def compute_B_sets(root: TreeNode, leaf_to_type: Dict[str, str]) -> Dict[TreeNode, Set[str]]:
    B = {}
    def post(v: TreeNode) -> Set[str]:
        if v.is_leaf():
            t = leaf_to_type.get(v.name)
            B[v] = {t} if t is not None else set()
            return B[v]
        acc = set()
        for c in v.children: acc |= post(c)
        B[v] = acc; return acc
    post(root)
    debug("compute_B_sets: unique types under root:", len(B[root]))
    return B

def beta_integral(O: int, D: int) -> float:
    return math.exp(math.lgamma(O + 1) + math.lgamma(D + 1) - math.lgamma(O + D + 2))

def sparse_convolve_2d(A: Dict[Tuple[int, int], float], B: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    if not A: return B.copy()
    if not B: return A.copy()
    out = defaultdict(float)
    for (o1, d1), w1 in A.items():
        for (o2, d2), w2 in B.items():
            out[(o1 + o2, d1 + d2)] += w1 * w2
    return dict(out)

def dp_tree_root_table(
    root: TreeNode,
    active_labels: List[FrozenSet[str]],
    Reach: Dict[FrozenSet[str], Set[FrozenSet[str]]],
    B_sets: Dict[TreeNode, Set[str]],
    prune_eps: float = 0.0
) -> Dict[Tuple[int, int], float]:
    debug("DP root table: labels=", len(active_labels), "prune_eps=", prune_eps)
    label_index = {L: i for i, L in enumerate(active_labels)}
    memo: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}

    def nid(v: TreeNode) -> int: return id(v)

    def M(v: TreeNode, P: Optional[FrozenSet[str]]) -> Dict[Tuple[int, int], float]:
        key = (nid(v), -1 if P is None else label_index[P])
        if key in memo: return memo[key]

        if v.is_leaf():
            memo[key] = {(0, 0): 1.0}; return memo[key]

        Bv = B_sets[v]
        out = defaultdict(float)
        parent_reach = active_labels if P is None else list(Reach[P])

        for L in parent_reach:
            if not Bv.issubset(L):
                continue
            o_local = len(L & Bv)
            d_local = len(L - Bv)

            child_tabs = []
            ok = True
            for u in v.children:
                tab = M(u, L)
                if not tab:
                    ok = False; break
                child_tabs.append(tab)
            if not ok:
                continue

            conv = child_tabs[0] if child_tabs else {(0, 0): 1.0}
            for t in child_tabs[1:]:
                conv = sparse_convolve_2d(conv, t)

            for (Oc, Dc), w in conv.items():
                out[(Oc + o_local, Dc + d_local)] += w

        if prune_eps > 0 and out:
            total = sum(out.values()); thresh = prune_eps * total
            out = {k: v for k, v in out.items() if v >= thresh}

        memo[key] = dict(out); return memo[key]

    C = M(root, None)
    debug("DP root table entries:", len(C))
    return C

def tree_marginal_from_root_table(C: Dict[Tuple[int, int], float]) -> float:
    val = sum(w * beta_integral(O, D) for (O, D), w in C.items())
    debug("Tree marginal:", val)
    return val

# ----------------------------
# Priors and scoring
# ----------------------------

class Priors:
    def __init__(self,
                 potency_mode: str = "fixed_k",  # "fixed_k" or "bernoulli"
                 fixed_k: int = 2,
                 pi_P: float = 0.25,
                 rho: float = 0.25):
        self.potency_mode = potency_mode
        self.fixed_k = fixed_k
        self.pi_P = pi_P
        self.rho = rho

    def log_prior_Z(self, S: List[str], Z_active: Set[FrozenSet[str]]) -> float:
        multis = [P for P in Z_active if len(P) >= 2]
        all_multis = [P for P in all_nonempty_subsets(S) if len(P) >= 2]

        if self.potency_mode == "fixed_k":
            k = len(multis)
            if k != self.fixed_k:
                return float("-inf")
            total = math.comb(len(all_multis), k)
            return -math.log(total) if total > 0 else float("-inf")
        else:
            k_log = 0.0
            for P in all_multis:
                if P in Z_active: k_log += math.log(self.pi_P)
                else: k_log += math.log(1 - self.pi_P)
            return k_log

    def log_prior_A(self, Z_active: Set[FrozenSet[str]], A: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int], unit_drop=True) -> float:
        labels = list(Z_active)
        logp = 0.0
        for P in labels:
            for Q in labels:
                if admissible_edge(P, Q, unit_drop):
                    a = 1 if A.get((P, Q), 0) == 1 else 0
                    logp += math.log(self.rho) if a == 1 else math.log(1 - self.rho)
        return logp


class Structure:
    def __init__(self,
                 S: List[str],
                 Z_active: Set[FrozenSet[str]],
                 A: Dict[Tuple[FrozenSet[str], FrozenSet[str]], int],
                 unit_drop: bool = True):
        self.S = S
        self.Z_active = set(Z_active)
        self.A = dict(A)
        self.unit_drop = unit_drop
        self.labels_list = self._sorted_labels()
        self.Reach = transitive_closure(self.labels_list, self.A)
        debug("Structure init: |S|=", len(S), "|Z|=", len(Z_active), "|A|=", sum(self.A.values()))

    def _sorted_labels(self) -> List[FrozenSet[str]]:
        return sorted(list(self.Z_active), key=lambda x: (len(x), tuple(sorted(list(x)))))

    def recompute_reach(self):
        self.labels_list = self._sorted_labels()
        self.Reach = transitive_closure(self.labels_list, self.A)
        debug("Recompute reach: |labels|=", len(self.labels_list), "|A|=", sum(self.A.values()))

    def clone(self) -> "Structure":
        return Structure(self.S, set(self.Z_active), dict(self.A), self.unit_drop)

    # Moves
    def potencies_multi_all(self) -> List[FrozenSet[str]]:
        return [P for P in all_nonempty_subsets(self.S) if len(P) >= 2]

    def propose_add_potency(self, rng: random.Random) -> Optional["Structure"]:
        candidates = [P for P in self.potencies_multi_all() if P not in self.Z_active]
        if not candidates: return None
        P = rng.choice(candidates)
        new = self.clone()
        new.Z_active.add(P)
        new.recompute_reach()
        return new

    def propose_remove_potency(self, rng: random.Random) -> Optional["Structure"]:
        candidates = [P for P in self.Z_active if len(P) >= 2]
        if not candidates: return None
        P = rng.choice(candidates)
        new = self.clone()
        new.Z_active.remove(P)
        new.A = {e: v for e, v in new.A.items() if P not in e}
        new.recompute_reach()
        return new

    def propose_swap_potency(self, rng: random.Random) -> Optional["Structure"]:
        remove_candidates = [P for P in self.Z_active if len(P) >= 2]
        add_candidates = [P for P in self.potencies_multi_all() if P not in self.Z_active]
        if not remove_candidates or not add_candidates: return None
        P_rm = rng.choice(remove_candidates)
        P_add = rng.choice(add_candidates)
        new = self.clone()
        new.Z_active.remove(P_rm)
        new.A = {e: v for e, v in new.A.items() if P_rm not in e}
        new.Z_active.add(P_add)
        new.recompute_reach()
        return new

    def all_edge_pairs(self) -> List[Tuple[FrozenSet[str], FrozenSet[str]]]:
        L = list(self.Z_active)
        pairs = []
        for P in L:
            for Q in L:
                if admissible_edge(P, Q, self.unit_drop):
                    pairs.append((P, Q))
        return pairs

    def propose_add_edge(self, rng: random.Random) -> Optional["Structure"]:
        pairs = [e for e in self.all_edge_pairs() if self.A.get(e, 0) == 0]
        if not pairs: return None
        e = rng.choice(pairs)
        new = self.clone()
        new.A[e] = 1
        new.recompute_reach()
        return new

    def propose_remove_edge(self, rng: random.Random) -> Optional["Structure"]:
        edges = [e for e, v in self.A.items() if v == 1]
        if not edges: return None
        e = rng.choice(edges)
        new = self.clone()
        del new.A[e]
        new.recompute_reach()
        return new


def score_structure(struct: Structure,
                    trees: List[TreeNode],
                    leaf_type_maps: List[Dict[str, str]],
                    priors: Priors,
                    prune_eps: float = 0.0) -> Tuple[float, List[float]]:
    debug("Scoring structure: |Z|=", len(struct.Z_active), "|A|=", sum(struct.A.values()),
          "unit_drop=", struct.unit_drop)
    logp_Z = priors.log_prior_Z(struct.S, struct.Z_active)
    if not math.isfinite(logp_Z):
        debug("  logP(Z) = -inf (violates prior)")
        return float("-inf"), []
    logp_A = priors.log_prior_A(struct.Z_active, struct.A, unit_drop=struct.unit_drop)
    debug("  logP(Z)=", f"{logp_Z:.6f}", "logP(A|Z)=", f"{logp_A:.6f}")

    logLs = []
    for ti, (root, leaf_to_type) in enumerate(zip(trees, leaf_type_maps), 1):
        B_sets = compute_B_sets(root, leaf_to_type)
        if not B_sets.get(root, set()):
            debug(f"  Tree {ti}: neutral (no mapped leaves) -> logL=0")
            logLs.append(0.0)
            continue
        C = dp_tree_root_table(root, struct.labels_list, struct.Reach, B_sets, prune_eps=prune_eps)
        P_T = tree_marginal_from_root_table(C)
        if P_T <= 0 or not math.isfinite(P_T):
            debug(f"  Tree {ti}: P(T|F) invalid (<=0 or NaN)")
            return float("-inf"), []
        lg = math.log(P_T)
        debug(f"  Tree {ti}: log P(T|F) = {lg:.6f}")
        logLs.append(lg)

    total = logp_Z + logp_A + sum(logLs)
    debug("Total log posterior:", f"{total:.6f}")
    return total, logLs


# ----------------------------
# Annealed stochastic search
# ----------------------------

def map_search(
    S: List[str],
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str, str]],
    priors: Priors,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None,
    init_seed: int = 0,
    iters: int = 500,
    restarts: int = 3,
    temp_init: float = 1.0,
    temp_decay: float = 0.995,
    move_probs = (0.25, 0.25, 0.25, 0.25),  # addP, rmP, addE, rmE (swap used when fixed_k)
    prune_eps: float = 0.0,
    progress: bool = True,
    tqdm_position: int = 0,   # NEW: position slot for tqdm (useful in parallel)
):
    debug("map_search: restarts=", restarts, "iters=", iters, "fixed_k=", fixed_k,
          "potency_mode=", priors.potency_mode)
    rng = random.Random(init_seed)

    best_global = None
    best_score = float("-inf")
    best_logs = None

    for rs in range(restarts):
        debug(f"[restart {rs+1}/{restarts}] initializing")
        if priors.potency_mode == "fixed_k":
            _, Z = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
        else:
            base = build_Z_active(S, fixed_k=0, max_potency_size=len(S), seed=rng.randint(0, 10**9))
            Z = base
        A = build_mid_sized_connected_dag(Z, keep_prob=0.3, rng=None)
        current = Structure(S, Z, A, unit_drop=unit_drop_edges)
        curr_score, _ = score_structure(current, trees, leaf_type_maps, priors, prune_eps)

        attempts = 0
        while not math.isfinite(curr_score) and attempts < 720:
            attempts += 1
            debug("  re-init attempt", attempts)
            _, Z = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
            A = build_mid_sized_connected_dag(Z, keep_prob=0.3, rng=None)
            current = Structure(S, Z, A, unit_drop=unit_drop_edges)
            curr_score, _ = score_structure(current, trees, leaf_type_maps, priors, prune_eps)

        if not math.isfinite(curr_score):
            raise RuntimeError("Failed to initialize a valid structure; consider easing settings.")
        debug("  init score:", curr_score)

        local_best = current.clone()
        local_best_score = curr_score
        if best_score < curr_score:
            best_score = curr_score
            best_global = current.clone()

        tau = temp_init
        addP, rmP, addE, rmE = move_probs
        iterator = (
            trange(
                iters,
                desc=f"Restart {rs+1}/{restarts}",
                leave=True,
                position=tqdm_position,  # show at given terminal row
            )
            if progress else range(iters)
        )

        for it in iterator:
            prop = None
            r = rng.random()
            if priors.potency_mode == "fixed_k":
                if r < addE:
                    prop = current.propose_add_edge(rng)
                elif r < addE + rmE:
                    prop = current.propose_remove_edge(rng)
                else:
                    prop = current.propose_swap_potency(rng)
            else:
                if r < addP:
                    prop = current.propose_add_potency(rng)
                elif r < addP + rmP:
                    prop = current.propose_remove_potency(rng)
                elif r < addP + rmP + addE:
                    prop = current.propose_add_edge(rng)
                else:
                    prop = current.propose_remove_edge(rng)

            if prop is None:
                tau *= temp_decay
                if progress and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"Best": f"{best_score:.3f}", "Curr": f"{curr_score:.3f}", "Temp": f"{tau:.3f}"})
                continue

            prop_score, _ = score_structure(prop, trees, leaf_type_maps, priors, prune_eps)
            delta = prop_score - curr_score
            accept = (delta >= 0) or (rng.random() < math.exp(delta / max(tau, 1e-12)))
            debug(f"  it={it} accept? {accept}  Δ={delta:.4f}  curr={curr_score:.4f}  prop={prop_score:.4f}")
            if accept:
                current = prop
                curr_score = prop_score

                if curr_score > local_best_score:
                    local_best = current.clone()
                    local_best_score = curr_score

                if curr_score > best_score:
                    best_global = current.clone()
                    best_score = curr_score
                    best_logs = None

            tau *= temp_decay
            if progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix({"Best": f"{best_score:.3f}", "Curr": f"{curr_score:.3f}", "Temp": f"{tau:.3f}"})

    debug("Best global score:", best_score)
    final_score, logLs = score_structure(best_global, trees, leaf_type_maps, priors, prune_eps)
    debug("Final recomputed score:", final_score)
    return best_global, final_score, logLs


def _map_search_worker(args):
    (S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
     init_seed, iters, restarts, temp_init, temp_decay, move_probs,
     prune_eps, worker_pos) = args  # worker_pos added for tqdm position
    debug("[worker] starting with restarts=", restarts, "iters=", iters, "seed=", init_seed, "pos=", worker_pos)

    return map_search(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=priors,
        unit_drop_edges=unit_drop_edges,
        fixed_k=fixed_k,
        init_seed=init_seed,
        iters=iters,
        restarts=restarts,
        temp_init=temp_init,
        temp_decay=temp_decay,
        move_probs=move_probs,
        prune_eps=prune_eps,
        progress=True,             # show bars in workers
        tqdm_position=worker_pos,  # distinct row per worker
    )


def map_search_parallel(
    S: List[str],
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str, str]],
    priors: Priors,
    unit_drop_edges: bool = True,
    fixed_k: Optional[int] = None,
    init_seed: int = 0,
    iters: int = 500,
    restarts: int = 12,
    temp_init: float = 1.0,
    temp_decay: float = 0.995,
    move_probs = (0.25, 0.25, 0.25, 0.25),
    prune_eps: float = 0.0,
    n_jobs: Optional[int] = None,
):
    """
    Parallelizes restarts across processes and returns the best result.
    NOTE: On Windows/macOS, call this under `if __name__ == "__main__":` to avoid spawn issues.
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    debug("map_search_parallel: restarts=", restarts, "n_jobs=", n_jobs)

    per_job = [restarts // n_jobs] * n_jobs
    for i in range(restarts % n_jobs):
        per_job[i] += 1
    per_job = [r for r in per_job if r > 0]
    n_jobs = len(per_job)
    seeds = [init_seed + 10_000 * i for i in range(n_jobs)]

    tasks = []
    for pos, (r, seed) in enumerate(zip(per_job, seeds)):
        tasks.append((S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
                      seed, iters, r, temp_init, temp_decay, move_probs, prune_eps, pos))  # pos passed

    best_global = None
    best_score = float("-inf")
    best_logs = None

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_map_search_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            bestF, score, logs = fut.result()
            debug(f"[worker {i}] result score={score:.6f}")
            if score > best_score:
                best_global, best_score, best_logs = bestF, score, logs

    debug("Parallel best score:", best_score)
    return best_global, best_score, best_logs


# ----------------------------
# File readers / validation
# ----------------------------

def read_leaf_type_map(path: str) -> Dict[str, str]:
    debug("Reading leaf→type map:", path)
    import csv as _csv

    ext = os.path.splitext(path)[1].lower()
    if ext in (".json",):
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: JSON must be an object mapping leaf->type.")
        debug("Map entries:", len(data))
        return {str(k): str(v) for k, v in data.items()}

    elif ext in (".csv", ".tsv", ".txt"):
        delim = "\t" if ext in (".tsv", ".txt") else ","
        out = {}
        with open(path, "r", newline="") as f:
            reader = _csv.reader(f, delimiter=delim)
            rows = list(reader)
            if not rows:
                raise ValueError(f"{path}: empty file")

            start_idx = 0
            header = [h.strip().lower() for h in rows[0]] if rows and rows[0] else []
            has_header = False
            if len(header) >= 2:
                if ("leaf" in header[0] or "cellbc" in header[0]) and ("type" in header[1] or "cell_state" in header[1]):
                    has_header = True
                if not has_header:
                    left_has = any(x in header for x in ("leaf", "cellbc"))
                    right_has = any(x in header for x in ("type", "cell_state"))
                    has_header = left_has and right_has

            if has_header:
                start_idx = 1

            for i in range(start_idx, len(rows)):
                row = rows[i]
                if len(row) < 2:
                    raise ValueError(f"{path}: line {i+1} needs at least 2 columns (leaf,type)")
                leaf = row[0].strip()
                typ  = row[1].strip()
                if not leaf or not typ:
                    raise ValueError(f"{path}: line {i+1} has empty leaf/type")
                if leaf in out:
                    raise ValueError(f"{path}: duplicate leaf '{leaf}' at line {i+1}")
                out[leaf] = str(typ)
        debug("Map entries:", len(out))
        return out

    else:
        raise ValueError(f"Unsupported mapping file type: {path} (use .csv, .tsv, .txt, or .json)")


def validate_leaf_type_map(root: TreeNode, leaf_map: Dict[str, str], S: List[str]) -> None:
    leaves_in_tree = set(collect_leaf_names(root))
    leaves_in_map  = set(leaf_map.keys())
    missing = leaves_in_tree - leaves_in_map
    extra   = leaves_in_map  - leaves_in_tree
    if missing:
        raise ValueError(f"Leaf map missing leaves: {sorted(missing)}")
    if extra:
        raise ValueError(f"Leaf map has unknown leaves not in tree: {sorted(extra)}")
    allowed = set(S)
    bad_types = {t for t in leaf_map.values() if t not in allowed}
    if bad_types:
        raise ValueError(f"Leaf map contains types not in S={S}: {sorted(bad_types)}")


def filter_leaf_map_to_tree(root: TreeNode, leaf_map: Dict[str, str]) -> Dict[str, str]:
    leaves = set(collect_leaf_names(root))
    filtered = {leaf: str(typ) for leaf, typ in leaf_map.items() if leaf in leaves}
    debug("Filter map to tree: leaves_in_tree=", len(leaves),
          "map_entries=", len(leaf_map), "filtered_entries=", len(filtered))
    missing = leaves - set(filtered.keys())
    if missing:
        debug("  WARNING:", len(missing), "leaves missing from map (ignored), e.g.", list(sorted(missing))[:5])
    return filtered


# ----------------------------
# Helpers for ground-truth map files
# ----------------------------

def _read_json_objects_exact(path: str):
    debug("Reading ground-truth JSON lines:", path)
    objs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(json.loads(line))
    if not objs:
        raise ValueError(f"{path}: no JSON objects found")
    debug("  objects found:", len(objs))
    return objs

def _extract_vertices_edges_from_adj(adj):
    V = set(adj.keys())
    for chs in adj.values():
        if isinstance(chs, list):
            V.update(chs)
    E = []
    for u, chs in adj.items():
        if isinstance(chs, list):
            for v in chs:
                E.append((str(u), str(v)))
    V = sorted(map(str, V), key=lambda x: (len(x), x))
    E = sorted(E, key=lambda e: (e[0], e[1]))
    debug("Adj summary: |V|=", len(V), "|E|=", len(E))
    return V, E

def _normalize_adj_remove_synthetic_root(adj: dict) -> dict:
    adj2 = {str(k): (list(v) if isinstance(v, list) else v) for k, v in adj.items()}
    if "root" in adj2:
        ch = adj2["root"]
        if not isinstance(ch, list) or len(ch) != 1:
            raise ValueError("Synthetic 'root' must have exactly one child")
        del adj2["root"]
        debug("Removed synthetic 'root' from adjacency")
    return adj2

def _resolve_id_to_set(id_str: str, comp_map: dict, memo: dict, visiting: set) -> frozenset:
    id_str = str(id_str)
    if id_str.startswith("-"):
        return frozenset([id_str])
    if id_str in memo:
        return memo[id_str]
    if id_str in visiting:
        raise ValueError(f"Cycle detected while resolving potency '{id_str}'")
    if id_str not in comp_map:
        raise ValueError(f"Positive id '{id_str}' appears but not defined in composition map")
    visiting.add(id_str)
    val = comp_map[id_str]
    acc = set()
    if isinstance(val, list):
        for child in val:
            acc |= _resolve_id_to_set(str(child), comp_map, memo, visiting)
    else:
        acc |= _resolve_id_to_set(str(val), comp_map, memo, visiting)
    visiting.remove(id_str)
    memo[id_str] = frozenset(acc)
    return memo[id_str]

def _build_ZA_from_txt(adj: dict, comp_map: dict, unit_drop_edges: bool):
    adj = _normalize_adj_remove_synthetic_root(adj)
    ids_seen = set(map(str, comp_map.keys()))
    for u, chs in adj.items():
        ids_seen.add(str(u))
        if isinstance(chs, list):
            for v in chs:
                ids_seen.add(str(v))
    memo = {}
    potency_id_to_set = {}
    base_types = set()
    for idv in ids_seen:
        if idv.startswith("-"):
            memo[idv] = frozenset([idv])
        else:
            s = _resolve_id_to_set(idv, comp_map, memo, visiting=set())
            potency_id_to_set[idv] = s
    for s in memo.values():
        for t in s:
            if t.startswith("-"):
                base_types.add(t)
    Z_active = {frozenset([t]) for t in base_types}
    for pid, s in potency_id_to_set.items():
        if len(s) >= 2:
            Z_active.add(s)
    A = {}
    def id_to_set(x: str) -> frozenset:
        x = str(x)
        if x.startswith("-"):
            return frozenset([x])
        return potency_id_to_set[x]
    for u, chs in adj.items():
        Pu = id_to_set(u)
        for v in chs:
            Qv = id_to_set(v)
            if admissible_edge(Pu, Qv, unit_drop_edges):
                A[(Pu, Qv)] = 1
    debug("_build_ZA_from_txt: |Z|=", len(Z_active), "|A|=", len(A), "base_types=", len(base_types))
    return Z_active, A, sorted(base_types), potency_id_to_set

def score_given_map_and_trees(txt_path: str, trees, meta_paths, fixed_k,
                              unit_drop_edges = False):
    objs = _read_json_objects_exact(txt_path)
    if len(objs) < 4:
        raise ValueError("Expected at least 4 JSON lines (adjacency, weights, composition map, root).")

    adj = None
    for o in objs:
        if isinstance(o, dict) and any(isinstance(v, list) for v in o.values()):
            adj = {str(k): [str(x) for x in v] for k, v in o.items() if isinstance(v, list)}
            break
    if adj is None:
        raise ValueError("Could not locate adjacency dict in the file.")

    comp_map = objs[2]
    if not isinstance(comp_map, dict):
        raise ValueError("Third JSON must be the composition map (dict).")

    root_id = objs[3]
    if isinstance(root_id, dict) and "root_id" in root_id:
        root_id = root_id["root_id"]
    root_id = str(root_id)

    V, E = _extract_vertices_edges_from_adj(adj)

    Z_from_map, A_from_map, base_types_map, potency_def = _build_ZA_from_txt(
        adj=adj,
        comp_map=comp_map,
        unit_drop_edges=unit_drop_edges
    )

    raw_maps = [read_leaf_type_map(p) for p in meta_paths]
    leaf_type_maps = [filter_leaf_map_to_tree(root, m) for root, m in zip(trees, raw_maps)]
    base_types_data = sorted({str(t) for m in leaf_type_maps for t in m.values()})

    S_all = sorted(set(base_types_map) | set(base_types_data))
    Z_active = set(Z_from_map) | {frozenset([t]) for t in S_all}
    A = dict(A_from_map)

    struct = Structure(S=S_all, Z_active=Z_active, A=A, unit_drop=unit_drop_edges)
    dummy_priors = Priors(potency_mode="fixed_k", fixed_k=fixed_k, rho=0.2)

    log_post, per_tree_logs = score_structure(
        struct=struct,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=dummy_priors,
        prune_eps=0.0
    )

    total_ll = sum(per_tree_logs)

    print("\n=== Ground Truth Log-likelihoods (given F from map) ===")
    for i, lg in enumerate(per_tree_logs, 1):
        print(f"Tree {i}: log P(T|F) = {lg:.6f}")
    print(f"Total log-likelihood = {total_ll:.6f}")

    potency_sets = {frozenset(members) for members in potency_def.values()}
    return potency_sets, total_ll

def jaccard_distance(set1, set2):
    if not set1 and not set2:
        return 0.0
    return 1 - len(set1 & set2) / len(set1 | set2)

def pot_str(P): return "{" + ",".join(sorted(list(P))) + "}"

def pretty_print_sets(name, sets):
    print(f"\n{name}:")
    for s in sorted(sets, key=lambda x: (len(x), sorted(x))):
        print("  ", sorted(list(s)))

# ----------------------------
# Pipeline for this dataset layout
# ----------------------------

def build_fate_map_path(map_idx: int, type_num: int) -> Tuple[str, str]:
    idx4 = f"{map_idx:04d}"
    fate_map_path = os.path.join(
        "inputs", "differentiation_maps", "graph", f"type_{type_num}",
        f"graph_fate_map{idx4}.txt"
    )
    debug("Looking for fate map:", fate_map_path)
    if not os.path.exists(fate_map_path):
        debug("FATE MAP MISSING!", fate_map_path)
        raise FileNotFoundError(f"Missing fate map: {fate_map_path}")
    debug("Found fate map ✓")
    return fate_map_path, idx4

def build_tree_and_meta_paths(map_idx: int, type_num: int, cells_n: int) -> Tuple[List[str], List[str]]:
    idx4 = f"{map_idx:04d}"
    folder = os.path.join("inputs", "trees", "graph", f"type_{type_num}", f"cells_{cells_n}")
    tree_paths = [os.path.join(folder, f"{idx4}_tree_{i}.txt") for i in range(5)]
    meta_paths = [os.path.join(folder, f"{idx4}_meta_{i}.txt") for i in range(5)]
    debug("Expecting tree+meta in folder:", folder)
    for p in tree_paths + meta_paths:
        debug("  check:", p, "exists?", os.path.exists(p))
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found: {p}")
    debug("All tree/meta files present ✓")
    return tree_paths, meta_paths

def read_trees_and_maps(tree_paths: List[str], meta_paths: List[str]):
    debug("Loading all trees & maps...")
    for p in tree_paths + meta_paths:
        if not os.path.exists(p):
            debug("MISSING:", p)
            raise FileNotFoundError(p)

    trees = [read_newick_file(p) for p in tree_paths]
    raw_maps = [read_leaf_type_map(p) for p in meta_paths]
    leaf_type_maps = [filter_leaf_map_to_tree(root, m) for root, m in zip(trees, raw_maps)]

    S = sorted({str(t) for m in leaf_type_maps for t in m.values()})
    debug("Type universe S:", S)
    return trees, leaf_type_maps, S

# --- CASE RUNNER ---

def process_case(map_idx: int, type_num: int, cells_n: int,
                 priors, iters=100, restarts=5, log_dir: Optional[str]=None):

    debug("\n=== process_case ===", "type", type_num, "map", map_idx, "cells", cells_n)
    fate_map_path, idx4 = build_fate_map_path(map_idx, type_num)
    tree_paths, meta_paths = build_tree_and_meta_paths(map_idx, type_num, cells_n)
    debug("Tree files:", tree_paths)
    debug("Meta files:", meta_paths)

    trees, leaf_type_maps, S = read_trees_and_maps(tree_paths, meta_paths)
    debug("Start search with |S|=", len(S))

    bestF, best_score, per_tree_logs = map_search_parallel(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=priors,
        unit_drop_edges=False,
        fixed_k=priors.fixed_k if priors.potency_mode=="fixed_k" else None,
        init_seed=123,
        iters=iters,
        restarts=restarts,
        temp_init=1.0,
        temp_decay=0.995,
        move_probs=(0.3, 0.2, 0.3, 0.2),
        prune_eps=0.0,
        n_jobs=os.cpu_count()
    )

    print(f"\n=== BEST MAP for type_{type_num}, map {idx4}, cells_{cells_n} ===")
    multi_sorted = sorted([P for P in bestF.Z_active if len(P) >= 2],
                          key=lambda x: (len(x), tuple(sorted(list(x)))))
    print("Active potencies (multi-type):")
    for P in multi_sorted: print("  ", pot_str(P))
    print("Singletons (always active):")
    for t in S: print("  ", "{" + t + "}")

    print("\nEdges:")
    edges = sorted([e for e, v in bestF.A.items() if v == 1],
                   key=lambda e: (len(e[0]), len(e[1]), tuple(sorted(list(e[0]))), tuple(sorted(list(e[1])))))
    for P, Q in edges: print(f"  {pot_str(P)} -> {pot_str(Q)}")

    print("\nScores:")
    print(f"  log posterior: {best_score:.6f}")
    for i, lg in enumerate(per_tree_logs, 1):
        print(f"  Tree {i} log P(T|F*): {lg:.6f}")

    predicted_sets = {p for p in bestF.Z_active if len(p) > 1}
    ground_truth_sets, gt_loss = score_given_map_and_trees(
        fate_map_path, trees, meta_paths, fixed_k=priors.fixed_k
    )

    pretty_print_sets("Predicted Sets", predicted_sets)
    pretty_print_sets("Ground Truth Sets", ground_truth_sets)

    jd = jaccard_distance(predicted_sets, ground_truth_sets)
    print("\n=== Jaccard Distance ===")
    print(f"Jaccard Distance (Pred vs GT): {jd:.6f}")
    print(f"Predicted map's loss: {best_score:.6f}")
    print(f"Ground truth's loss: {gt_loss:.6f}")

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"log_type{type_num}_{idx4}_cells{cells_n}.txt")
        with open(log_path, "w") as f:
            f.write(f"type_{type_num}, map {idx4}, cells_{cells_n}\n")
            f.write(f"Jaccard={jd:.6f}, GT loss={gt_loss:.6f}, Pred loss={best_score:.6f}\n")
    return jd, gt_loss, best_score


def main_multi_type(type_nums=[6, 10, 14],
                    maps_start=17, maps_end=26,
                    cells_list=[50, 100, 200],
                    out_csv="results_types_6_10_14_maps_17_26.csv",
                    log_dir="logs_types"):

    random.seed(7)
    priors = Priors(potency_mode="fixed_k", fixed_k=5, rho=0.2)
    results = []

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "MapIdx", "Cells", "Jaccard", "GT Loss", "Pred Loss"])
        for t in type_nums:
            for idx in range(maps_start, maps_end + 1):
                for cells in cells_list:
                    try:
                        jd, gt_loss, pred_loss = process_case(idx, t, cells, priors,
                                                              iters=5, restarts=7, log_dir=log_dir)
                        writer.writerow([t, idx, cells, f"{jd:.6f}", f"{gt_loss:.6f}", f"{pred_loss:.6f}"])
                        results.append((t, idx, cells, jd, gt_loss, pred_loss))
                    except Exception as e:
                        print(f"[WARN] Failed type_{t} map {idx:04d} cells_{cells}: {e}")
                        writer.writerow([t, idx, cells, "ERROR", "ERROR", "ERROR"])
                        results.append((t, idx, cells, None, None, None))

    print(f"\nSummary saved to {out_csv}")
    print("\n================= Summary Table =================")
    print(f"{'Type':<6} {'Map':<6} {'Cells':<7} {'Jaccard':<12} {'GT Loss':<14} {'Pred Loss':<14}")
    for t, idx, cells, jd, gt, pr in results:
        if jd is None:
            print(f"{t:<6} {idx:<6} {cells:<7} {'ERROR':<12} {'ERROR':<14} {'ERROR':<14}")
        else:
            print(f"{t:<6} {idx:<6} {cells:<7} {jd:<12.6f} {gt:<14.6f} {pr:<14.6f}")


if __name__ == "__main__":
    debug("Program start; cwd=", os.getcwd())
    # Example single case to match your prior run:
    main_multi_type(
        type_nums=[10],
        maps_start=2,
        maps_end=2,
        cells_list=[50],
        out_csv="cheking.csv",
        log_dir="logs_types"
    )
