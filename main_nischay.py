#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAP structure search for Carta-CDMIP model.

Score(F) = log P(F) + sum_T log P(T|F),
where P(T|F) = sum_{labelings} B(O+1, D+1), with per-node counts:
  obs = |L ∩ B(v)|, miss = |L \ B(v)|, and B(·) is Beta function.
p ~ Beta(1,1) is integrated out exactly.

- Newick parser (no external deps)
- DP over labelings with (O,D) sparse tables
- Priors: fixed-k (uniform over potency sets) OR Bernoulli(pi_P); edges Bernoulli(rho)
- Stochastic hill-climb + simulated annealing over F=(Z,A)
"""
from tqdm import trange
import math
import random
import itertools
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Set, FrozenSet
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ----------------------------
# Tree structures and Newick
# ----------------------------
def build_mid_sized_connected_dag(Z_active, keep_prob=0.3, unit_drop=False, rng=None):
    """
    Build a valid mid-density DAG:
      * Uses only admissible edges
      * Guarantees connectivity from the root node (frozenset of all singletons)
      * Keeps density moderate, controlled by `keep_prob`
    """
    if rng is None:
        rng = random.Random()

    # --- Identify root node (the potency containing all singletons) ---
    root = frozenset().union(*Z_active)  # union of all labels gives the full set
    print("ROot ",root)
    if root not in Z_active:
        raise ValueError("Root potency (all singletons) not present in Z_active.")

    nodes = list(Z_active)

    # --- Step 1: Build full admissible edge set ---
    full_edges = {
        (P, Q): 1
        for P in Z_active
        for Q in Z_active
        if P != Q and admissible_edge(P, Q, unit_drop)
    }

    # --- Step 2: Start with a spanning tree to guarantee connectivity ---
    A = {}
    visited = {root}
    to_visit = set(nodes) - {root}

    while to_visit:
        # pick a node already in the tree
        parent = rng.choice(list(visited))

        # find valid edges from parent to some unvisited node
        candidates = [(parent, q) for q in to_visit if (parent, q) in full_edges]

        if not candidates:
            # fallback: pick any edge between visited and unvisited nodes
            candidates = [
                (p, q) for p in visited for q in to_visit if (p, q) in full_edges
            ]

        edge = rng.choice(candidates)
        A[edge] = 1
        visited.add(edge[1])
        to_visit.remove(edge[1])

    # --- Step 3: Add extra edges randomly to reach desired density ---
    for edge in full_edges:
        if edge in A:
            continue
        if rng.random() < keep_prob:
            A[edge] = 1

    return A

class TreeNode:
    def __init__(self, name: Optional[str] = None):
        self.name: Optional[str] = name
        self.children: List["TreeNode"] = []
        self.parent: Optional["TreeNode"] = None

    def is_leaf(self): return len(self.children) == 0
    def add_child(self, child: "TreeNode"):
        self.children.append(child); child.parent = self

    def __repr__(self):
        return f"Leaf({self.name})" if self.is_leaf() else f"Node({self.name}, k={len(self.children)})"

# def parse_newick(newick: str) -> TreeNode:
#     s = newick.strip()
#     if not s.endswith(";"): raise ValueError("Newick must end with ';'")
#     s = s[:-1]; i = 0
#     def parse() -> TreeNode:
#         nonlocal i, s
#         if i >= len(s): raise ValueError("Unexpected end")
#         if s[i] == '(':
#             i += 1
#             node = TreeNode()
#             while True:
#                 node.add_child(parse())
#                 if i >= len(s): raise ValueError("Unbalanced")
#                 if s[i] == ',':
#                     i += 1; continue
#                 elif s[i] == ')':
#                     i += 1; break
#                 else: raise ValueError(f"Unexpected char: {s[i]} at {i}")
#             j = i
#             while j < len(s) and s[j] not in ',()': j += 1
#             name = s[i:j].strip()
#             if name: node.name = name
#             i = j
#             return node
#         else:
#             j = i
#             while j < len(s) and s[j] not in ',()': j += 1
#             name = s[i:j].strip()
#             if not name: raise ValueError("Leaf without name")
#             i = j
#             return TreeNode(name=name)
#     root = parse()
#     if i != len(s): raise ValueError(f"Trailing characters: '{s[i:]}'")
#     return root

def parse_newick(newick: str) -> TreeNode:
    # Helper: strip branch length and numeric-only labels
    def _clean_label(tok: str) -> str:
        # remove branch length: keep part before first ':'
        tok = tok.split(":", 1)[0].strip()
        # drop pure numeric internal labels like "357"
        if tok and tok.replace(".", "", 1).isdigit():
            return ""
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

            # optional internal node label (may include branch length)
            j = i
            while j < len(s) and s[j] not in ',()': j += 1
            name_raw = s[i:j].strip()
            name = _clean_label(name_raw)
            if name:  # keep non-empty, non-numeric labels only
                node.name = name
            i = j
            return node
        else:
            # leaf label (may include branch length)
            j = i
            while j < len(s) and s[j] not in ',()': j += 1
            name_raw = s[i:j].strip()
            name = _clean_label(name_raw)
            if not name:
                raise ValueError("Leaf without name")
            i = j
            return TreeNode(name=name)

    root = parse()
    if i != len(s): raise ValueError(f"Trailing characters: '{s[i:]}'")
    return root

def to_newick(root: TreeNode) -> str:
    def rec(n: TreeNode) -> str:
        if n.is_leaf(): return n.name or ""
        return f"({','.join(rec(c) for c in n.children)}){n.name or ''}"
    return rec(root) + ";"

def read_newick_file(path: str) -> TreeNode:
    with open(path, "r") as f: s = f.read().strip()
    return parse_newick(s)

def write_newick_file(path: str, root: TreeNode):
    with open(path, "w") as f: f.write(to_newick(root) + "\n")

def random_tree_newick(n_leaves: int, leaf_prefix="L") -> Tuple[TreeNode, List[str]]:
    leaves = [TreeNode(f"{leaf_prefix}{i+1}") for i in range(n_leaves)]
    nodes = leaves[:]
    while len(nodes) > 1:
        k = 2 if len(nodes) < 4 else random.choice([2,2,2,3])
        k = min(k, len(nodes))
        picks = random.sample(nodes, k)
        for p in picks: nodes.remove(p)
        parent = TreeNode()
        for p in picks: parent.add_child(p)
        nodes.append(parent)
    return nodes[0], [l.name for l in leaves]

def collect_leaf_names(root: TreeNode) -> List[str]:
    out=[]
    def dfs(v):
        if v.is_leaf(): out.append(v.name)
        else:
            for c in v.children: dfs(c)
    dfs(root); return out

# ----------------------------
# Potency universe and structure
# ----------------------------

def all_nonempty_subsets(S: List[str], max_size: Optional[int]=None) -> List[FrozenSet[str]]:
    R=len(S); max_k = R if max_size is None else min(max_size, R)
    res=[]
    for k in range(1, max_k+1):
        for comb in itertools.combinations(S, k): res.append(frozenset(comb))
    return res

def singletons(S: List[str]) -> Set[FrozenSet[str]]:
    return {frozenset([t]) for t in S}

def build_Z_active(S: List[str], fixed_k: Optional[int], max_potency_size: Optional[int], seed=0) -> Set[FrozenSet[str]]:
    rng = random.Random(seed)
    P_all = all_nonempty_subsets(S, max_potency_size)
    singles = singletons(S)
    multis = [P for P in P_all if len(P)>=2]
    Z = set(singles)
    if fixed_k is not None:
        if fixed_k > len(multis):
            raise ValueError("fixed_k too large")
        root = frozenset(S)
        Z.add(root)
        remaining_multis = [P for P in multis if P != root]
        Z.update(rng.sample(remaining_multis, fixed_k - 1))  # pick k-1 more
    else:
        Z.update(multis)
    return Z

def admissible_edge(P: FrozenSet[str], Q: FrozenSet[str], unit_drop: bool) -> bool:
    if Q == P: return False
    if not Q.issubset(P): return False
    if len(Q) >= len(P): return False
    if unit_drop and len(P - Q) != 1: return False
    return True

def build_edges(Z_active: Set[FrozenSet[str]], forbid_fn=None, unit_drop=True) -> Dict[Tuple[FrozenSet[str],FrozenSet[str]], int]:
    A={}
    for P in Z_active:
        for Q in Z_active:
            if not admissible_edge(P,Q,unit_drop): continue
            if forbid_fn and forbid_fn(P,Q): continue
            A[(P,Q)] = 1
    return A

def transitive_closure(labels: List[FrozenSet[str]], A: Dict[Tuple[FrozenSet[str],FrozenSet[str]], int]) -> Dict[FrozenSet[str], Set[FrozenSet[str]]]:
    idx = {L:i for i,L in enumerate(labels)}
    n=len(labels)
    M=[[False]*n for _ in range(n)]
    for i in range(n): M[i][i]=True
    for (P,Q),v in A.items():
        if v:
            i,j=idx[P],idx[Q]; M[i][j]=True
    for k in range(n):
        Mk=M[k]
        for i in range(n):
            if M[i][k]:
                Mi=M[i]
                for j in range(n):
                    if Mk[j]: Mi[j]=True
    Reach={L:set() for L in labels}
    for i,L in enumerate(labels):
        for j,U in enumerate(labels):
            if M[i][j]: Reach[L].add(U)
    return Reach

# ----------------------------
# DP over labelings (integrated Beta)
# ----------------------------

def compute_B_sets(root: TreeNode, leaf_to_type: Dict[str,str]) -> Dict[TreeNode, Set[str]]:
    B={}
    def post(v: TreeNode) -> Set[str]:
        if v.is_leaf():
            t = leaf_to_type.get(v.name)
            # Missing mapping? Ignore this leaf by contributing an empty set.
            B[v] = {t} if t is not None else set()
            return B[v]
        acc=set()
        for c in v.children: acc |= post(c)
        B[v]=acc; return acc
    post(root); return B

def beta_integral(O:int,D:int)->float:
    # ∫ p^O (1-p)^D dp over [0,1] = B(O+1,D+1)
    return math.exp(math.lgamma(O+1)+math.lgamma(D+1)-math.lgamma(O+D+2))

def sparse_convolve_2d(A: Dict[Tuple[int,int],float], B: Dict[Tuple[int,int],float]) -> Dict[Tuple[int,int],float]:
    if not A: return B.copy()
    if not B: return A.copy()
    out=defaultdict(float)
    for (o1,d1),w1 in A.items():
        for (o2,d2),w2 in B.items():
            out[(o1+o2,d1+d2)] += w1*w2
    return dict(out)

def dp_tree_root_table(
    root: TreeNode,
    active_labels: List[FrozenSet[str]],
    Reach: Dict[FrozenSet[str], Set[FrozenSet[str]]],
    B_sets: Dict[TreeNode, Set[str]],
    prune_eps: float = 0.0
)->Dict[Tuple[int,int],float]:
    label_index={L:i for i,L in enumerate(active_labels)}
    memo: Dict[Tuple[int,int], Dict[Tuple[int,int],float]]={}
    def nid(v:TreeNode)->int: return id(v)

    def M(v:TreeNode, P: Optional[FrozenSet[str]])->Dict[Tuple[int,int],float]:
        key=(nid(v), -1 if P is None else label_index[P])
        if key in memo: return memo[key]
        if v.is_leaf():
            memo[key] = {(0,0):1.0}; return memo[key]
        Bv=B_sets[v]
        out=defaultdict(float)
        if P is None:
            parent_reach = active_labels
        else:
            parent_reach = list(Reach[P])

        for L in parent_reach:
            if not Bv.issubset(L):  # containment constraint
                continue
            o_local=len(L & Bv); d_local=len(L - Bv)
            # children messages conditioned on parent label = L
            child_tabs=[]
            ok=True
            for u in v.children:
                tab = M(u, L)
                if not tab: ok=False; break
                child_tabs.append(tab)
            if not ok: continue
            conv = child_tabs[0] if child_tabs else {(0,0):1.0}
            for t in child_tabs[1:]:
                conv = sparse_convolve_2d(conv, t)
            for (Oc,Dc),w in conv.items():
                out[(Oc+o_local, Dc+d_local)] += w

        if prune_eps>0 and out:
            total=sum(out.values()); thresh=prune_eps*total
            out={k:v for k,v in out.items() if v>=thresh}
        memo[key]=dict(out); return memo[key]

    return M(root, None)

def tree_marginal_from_root_table(C: Dict[Tuple[int,int],float])->float:
    return sum(w * beta_integral(O,D) for (O,D),w in C.items())

# ----------------------------
# Priors and scoring
# ----------------------------

class Priors:
    def __init__(self,
                 potency_mode:str="fixed_k",  # "fixed_k" or "bernoulli"
                 fixed_k:int=2,
                 pi_P:float=0.25,    # used if potency_mode="bernoulli"
                 rho:float=0.25):    # edge Bernoulli prob
        # ------------------------------------------------------------
        # Stores hyperparameters for the prior over the structure F=(Z,A)
        #     Z: The latent assignment of "potencies" or features to nodes (the sets like {A,B,C}, {B,C,D}, etc. that you saw in the MAP output).
        #     A: The active structure (the adjacency or edge set) consistent with those potencies -- basically the graph/hypergraph that the algorithm thinks best explains the observed trees.
        #   - potency_mode: which prior to use over active multi-type potencies Z
        #       * "fixed_k": exactly k multi-type potencies are active (uniform over choices)
        #       * "bernoulli": each multi-type potency is independently active with prob pi_P
        #   - fixed_k: number of multi-type potencies when potency_mode == "fixed_k"
        #   - pi_P: inclusion probability for each multi-type potency when using "bernoulli" mode
        #   - rho: prior probability that any admissible edge (P->Q) exists
        # ------------------------------------------------------------
        self.potency_mode=potency_mode
        self.fixed_k=fixed_k
        self.pi_P=pi_P
        self.rho=rho

    def log_prior_Z(self, S: List[str], Z_active:Set[FrozenSet[str]])->float: #Z_active = the set of active potencies (both singletons and multis).
        # ------------------------------------------------------------
        # Computes log P(Z): the log prior over WHICH potencies are active.
        #
        # Inputs:
        #   - S: list of all cell types (leaf types), e.g., ["A","B","C","D"]
        #   - Z_active: set of active potencies (as frozensets). Includes singletons by construction.
        #
        # Key ideas:
        #   - Singletons are always considered active (terminal states), we don't penalize/score them.
        #   - We only place a prior over multi-type potencies (size >= 2).
        #   - Two modes:
        #       * "fixed_k": valid only if exactly `fixed_k` multis are active.
        #                    Prior is uniform over all C(M, k) choices, where M = #all possible multis.
        #       * "bernoulli": each possible multi is included independently with prob pi_P.
        #                      log prior sums log(pi_P) for included multis and log(1-pi_P) for excluded ones.
        # Returns:
        #   - log prior (float), or -inf if configuration violates "fixed_k".
        # ------------------------------------------------------------
        singles = singletons(S)
        multis = [P for P in Z_active if len(P)>=2] #P is a particular potency set
        # count available multi potencies (for fixed-k uniform)
        all_multis = [P for P in all_nonempty_subsets(S) if len(P)>=2]

        if self.potency_mode=="fixed_k":
            # ------------------------------
            # Uniform prior over all subsets of multi-type potencies with EXACTLY k elements.
            # If the current Z_active has not exactly k multis, return -inf (outside prior support).
            # Otherwise, log prior = -log( number of such subsets ) = -log( nCk ).
            # ------------------------------
            k=len(multis)
            if k!=self.fixed_k:
                return float("-inf")
            # uniform over all C(|all_multis|, k)
            total = math.comb(len(all_multis), k) #this is nCk
            return -math.log(total) if total>0 else float("-inf")
        else:
            # ------------------------------
            # Bernoulli prior on each multi-type potency:
            #   P(Z) = ∏_{P in all_multis} pi_P^{I[P in Z]} (1 - pi_P)^{I[P not in Z]}
            # We sum logs across all possible multi-type potencies (singletons ignored).
            # ------------------------------
            k_log=0.0
            for P in all_multis:
                if P in Z_active: k_log += math.log(self.pi_P)
                else: k_log += math.log(1-self.pi_P)
            return k_log

    def log_prior_A(self, Z_active:Set[FrozenSet[str]], A:Dict[Tuple[FrozenSet[str],FrozenSet[str]],int], unit_drop=True)->float:
        # ------------------------------------------------------------
        # Computes log P(A | Z): the log prior over EDGE EXISTENCE between active potencies.
        #
        # Inputs:
        #   - Z_active: set of active potencies (nodes in the potency DAG)
        #   - A: adjacency dictionary mapping (P,Q) -> {0,1}, indicating whether edge P->Q is present
        #   - unit_drop: if True, an admissible edge must drop EXACTLY one fate (|P\Q| == 1);
        #                otherwise any monotone subset drop (Q ⊂ P) is admissible.
        #
        # Prior:
        #   - For every admissible pair (P,Q):
        #         A_{P->Q} ~ Bernoulli(rho)
        #     So:
        #         log P(A|Z) = ∑_{(P,Q) admissible} [ A_{P->Q} log(rho) + (1 - A_{P->Q}) log(1 - rho) ]
        #
        # Notes:
        #   - "Admissible" enforces graph shape constraints (subset-monotone and possibly unit-drop).
        #   - If an edge (P,Q) is not admissible, it does not contribute to the product/sum at all.
        # ------------------------------------------------------------
        labels=list(Z_active)
        # admissible set is pairs with subset monotone (and optionally unit-drop)
        logp=0.0
        for P in labels:
            for Q in labels:
                if admissible_edge(P,Q,unit_drop):
                    # a == 1 if the edge is present in A, else 0
                    a = 1 if A.get((P,Q),0)==1 else 0
                    # add Bernoulli log-prob for this edge
                    logp += math.log(self.rho) if a==1 else math.log(1-self.rho)
        return logp

# ----------------------------
# Structure container and proposals
# ----------------------------

class Structure:
    def __init__(self,
                 S: List[str],
                 Z_active: Set[FrozenSet[str]],
                 A: Dict[Tuple[FrozenSet[str],FrozenSet[str]],int],
                 unit_drop: bool = True):
        self.S=S
        self.Z_active=set(Z_active)  # includes singletons
        self.A=dict(A)
        self.unit_drop=unit_drop
        self.labels_list=self._sorted_labels()
        self.Reach = transitive_closure(self.labels_list, self.A)

    def _sorted_labels(self)->List[FrozenSet[str]]:
        return sorted(list(self.Z_active), key=lambda x: (len(x), tuple(sorted(list(x)))))

    def recompute_reach(self):
        self.labels_list=self._sorted_labels()
        self.Reach = transitive_closure(self.labels_list, self.A)

    def clone(self)->"Structure":
        return Structure(self.S, set(self.Z_active), dict(self.A), self.unit_drop)

    # --- Moves ---
    def potencies_multi_all(self)->List[FrozenSet[str]]:
        return [P for P in all_nonempty_subsets(self.S) if len(P)>=2]

    def propose_add_potency(self, rng:random.Random)->Optional["Structure"]:
        candidates = [P for P in self.potencies_multi_all() if P not in self.Z_active]
        if not candidates: return None
        P = rng.choice(candidates)
        new = self.clone()
        new.Z_active.add(P)
        # add edges that respect admissibility? keep edges as-is and allow edge moves separately
        new.recompute_reach()
        return new

    def propose_remove_potency(self, rng:random.Random)->Optional["Structure"]:
        candidates = [P for P in self.Z_active if len(P)>=2]
        if not candidates: return None
        P = rng.choice(candidates)
        new = self.clone()
        # remove potency and incident edges
        new.Z_active.remove(P)
        new.A = {e:v for e,v in new.A.items() if P not in e}
        new.recompute_reach()
        return new

    def propose_swap_potency(self, rng:random.Random)->Optional["Structure"]:
        remove_candidates = [P for P in self.Z_active if len(P)>=2]
        add_candidates = [P for P in self.potencies_multi_all() if P not in self.Z_active]
        if not remove_candidates or not add_candidates: return None
        P_rm = rng.choice(remove_candidates)
        P_add = rng.choice(add_candidates)
        new = self.clone()
        new.Z_active.remove(P_rm)
        new.A = {e:v for e,v in new.A.items() if P_rm not in e}
        new.Z_active.add(P_add)
        new.recompute_reach()
        return new

    def all_edge_pairs(self)->List[Tuple[FrozenSet[str],FrozenSet[str]]]:
        L=list(self.Z_active)
        pairs=[]
        for P in L:
            for Q in L:
                if admissible_edge(P,Q,self.unit_drop):
                    pairs.append((P,Q))
        return pairs

    def propose_add_edge(self, rng:random.Random)->Optional["Structure"]:
        pairs = [e for e in self.all_edge_pairs() if self.A.get(e,0)==0]
        if not pairs: return None
        e = rng.choice(pairs)
        new = self.clone()
        new.A[e]=1
        new.recompute_reach()
        return new

    def propose_remove_edge(self, rng:random.Random)->Optional["Structure"]:
        edges = [e for e,v in self.A.items() if v==1]
        if not edges: return None
        e = rng.choice(edges)
        new = self.clone()
        del new.A[e]
        new.recompute_reach()
        return new

# ----------------------------
# Scoring: log posterior
# ----------------------------

def score_structure(struct: Structure,
                    trees: List[TreeNode],
                    leaf_type_maps: List[Dict[str,str]],
                    priors: Priors,
                    prune_eps: float = 0.0) -> Tuple[float, List[float]]:
    # log prior
    logp = priors.log_prior_Z(struct.S, struct.Z_active)
    #print(f"logp:{logp}")
    if not math.isfinite(logp):
        logp = float("-inf")
    logp += priors.log_prior_A(struct.Z_active, struct.A, unit_drop=struct.unit_drop)

    # likelihood
    logLs=[]
    for root, leaf_to_type in zip(trees, leaf_type_maps):
        B_sets = compute_B_sets(root, leaf_to_type)

        # --- NEW: if the root has no labels at all, skip this tree (neutral evidence) ---
        root_labels = B_sets.get(root, set())
        if not root_labels:
            logLs.append(0.0)
            continue
        # -------------------------------------------------------------------------------

        C = dp_tree_root_table(root, struct.labels_list, struct.Reach, B_sets, prune_eps=prune_eps)
        P_T = tree_marginal_from_root_table(C)
        #print(f"P_T:{P_T}")
        if P_T <= 0 or not math.isfinite(P_T):
            return float("-inf"), []
        logLs.append(math.log(P_T))
    return logp + sum(logLs), logLs

def map_search(
    S: List[str],
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str,str]],
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
):
    rng = random.Random(init_seed)

    best_global = None
    best_score = float("-inf")
    best_logs = None

    for rs in range(restarts):
        # --- init structure
        # print(rs)
        if priors.potency_mode=="fixed_k":
            aggregated_transitions, Z = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
        else:
            base = build_Z_active(S, fixed_k=0, max_potency_size=len(S), seed=rng.randint(0,10**9))
            Z = base
        A = build_mid_sized_connected_dag(Z,keep_prob = 0.3,rng = None)
        current = Structure(S, Z, A, unit_drop=unit_drop_edges)
        curr_score, _ = score_structure(current, trees, leaf_type_maps, priors, prune_eps)

        # fallback: if invalid, keep sampling until valid
        attempts = 0
        while not math.isfinite(curr_score) and attempts < 720:
            aggregated_transitions, Z = init_progenitors_union_fitch(S, trees, leaf_type_maps, fixed_k)
            A = build_mid_sized_connected_dag(Z,keep_prob = 0.3,rng = None)
            # A = {}
            print(f"Z:{Z}") 
            print(f"A:{A}")
            current = Structure(S, Z, A, unit_drop=unit_drop_edges)
            curr_score, _ = score_structure(current, trees, leaf_type_maps, priors, prune_eps)
            # print(curr_score)
            attempts += 1

        if not math.isfinite(curr_score):
            raise RuntimeError("Failed to initialize a valid structure; consider easing settings.")

        local_best = current.clone()
        local_best_score = curr_score

        tau = temp_init
        addP, rmP, addE, rmE = move_probs

        # iterator respects the 'progress' flag
        iterator = trange(iters, desc=f"Restart {rs+1}/{restarts}", leave=True) if progress else range(iters)

        for _ in iterator:
            # choose move
            prop = None
            r = rng.random()
            if priors.potency_mode=="fixed_k":
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
                if progress:
                    iterator.set_postfix({"Best": f"{best_score:.3f}", "Curr": f"{curr_score:.3f}", "Temp": f"{tau:.3f}"})
                continue

            prop_score, _ = score_structure(prop, trees, leaf_type_maps, priors, prune_eps)

            delta = prop_score - curr_score
            accept = (delta >= 0) or (rng.random() < math.exp(delta / max(tau,1e-12)))
            if accept:
                current = prop
                curr_score = prop_score

                if curr_score > local_best_score:
                    local_best = current.clone()
                    local_best_score = curr_score

                if curr_score > best_score:
                    best_global = current.clone()
                    best_score = curr_score
                    best_logs = None  # compute later if needed

            tau *= temp_decay
            # print(f"rs:{rs},curr{curr_score}")
            if progress:
                iterator.set_postfix({"Best": f"{best_score:.3f}", "Curr": f"{curr_score:.3f}", "Temp": f"{tau:.3f}"})

    # after restarts, recompute detailed logs for best_global
    final_score, logLs = score_structure(best_global, trees, leaf_type_maps, priors, prune_eps)
    return best_global, final_score, logLs

from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def _map_search_worker(args):
    (S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
     init_seed, iters, restarts, temp_init, temp_decay, move_probs, prune_eps) = args
    # progress=False inside workers to avoid tqdm noise
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
        progress=True,
    )

def map_search_parallel(
    S: List[str],
    trees: List[TreeNode],
    leaf_type_maps: List[Dict[str,str]],
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

    # Split restarts across jobs
    per_job = [restarts // n_jobs] * n_jobs
    for i in range(restarts % n_jobs):
        per_job[i] += 1
    per_job = [r for r in per_job if r > 0]
    n_jobs = len(per_job)

    # Unique seeds per worker to diversify trajectories
    seeds = [init_seed + 10_000 * i for i in range(n_jobs)]

    tasks = []
    for r, seed in zip(per_job, seeds):
        tasks.append((S, trees, leaf_type_maps, priors, unit_drop_edges, fixed_k,
                      seed, iters, r, temp_init, temp_decay, move_probs, prune_eps))

    best_global = None
    best_score = float("-inf")
    best_logs = None

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_map_search_worker, t) for t in tasks]
        for fut in as_completed(futures):
            bestF, score, logs = fut.result()
            if score > best_score:
                best_global, best_score, best_logs = bestF, score, logs

    return best_global, best_score, best_logs

########
#
#
# MAP TXT READING
#
########
import os
import csv
import json

def read_leaf_type_map(path: str) -> Dict[str, str]:
    """
    Read a leaf->type mapping from a file.

    Supported:
      - JSON dict: { "LeafName": "Type", ... }
      - CSV/TSV/TXT with 2 columns (header optional):
          * If header present, typical field names could be:
              - leaf, type
              - cellBC, cell_state (your .txt example)
    Returns: dict {leaf_name: type_symbol} (types are coerced to str)
    """
    import os, csv, json

    ext = os.path.splitext(path)[1].lower()
    if ext in (".json",):
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: JSON must be an object mapping leaf->type.")
        return {str(k): str(v) for k, v in data.items()}

    elif ext in (".csv", ".tsv", ".txt"):
        # treat .txt as TSV by default (your example is tab-delimited)
        delim = "\t" if ext in (".tsv", ".txt") else ","
        out = {}
        with open(path, "r", newline="") as f:
            reader = csv.reader(f, delimiter=delim)
            rows = list(reader)
            if not rows:
                raise ValueError(f"{path}: empty file")

            # Detect header
            start_idx = 0
            header = [h.strip().lower() for h in rows[0]] if rows and rows[0] else []
            has_header = False
            if len(header) >= 2:
                # Common header names we accept
                if ("leaf" in header[0] or "cellbc" in header[0]) and ("type" in header[1] or "cell_state" in header[1]):
                    has_header = True
                # Or any header line where at least one of ('leaf','cellbc') and one of ('type','cell_state') appear
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
                out[leaf] = str(typ)  # coerce types to string (handles negatives like -7, -9)
        return out

    else:
        raise ValueError(f"Unsupported mapping file type: {path} (use .csv, .tsv, .txt, or .json)")

def validate_leaf_type_map(root: TreeNode, leaf_map: Dict[str,str], S: List[str]) -> None:
    """
    Ensure mapping covers exactly the leaves in the tree, and types are in S.
    Raises ValueError if not valid.
    """
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
    return {leaf: str(typ) for leaf, typ in leaf_map.items() if leaf in leaves}

# ----------------------------
# ----------------------------
# Parsing the custom TXT "map" format and scoring
# ----------------------------

import json

def _read_json_objects_exact(path: str):
    """Read one JSON object per non-empty line (your file format)."""
    objs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(json.loads(line))
    if not objs:
        raise ValueError(f"{path}: no JSON objects found")
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
    return V, E

def _normalize_adj_remove_synthetic_root(adj: dict) -> dict:
    """Drop a synthetic 'root' node (if present) from adjacency for building F."""
    adj2 = {str(k): (list(v) if isinstance(v, list) else v) for k, v in adj.items()}
    if "root" in adj2:
        ch = adj2["root"]
        if not isinstance(ch, list) or len(ch) != 1:
            raise ValueError("Synthetic 'root' must have exactly one child")
        del adj2["root"]
    return adj2

def _resolve_id_to_set(id_str: str, comp_map: dict, memo: dict, visiting: set) -> frozenset:
    """
    Recursively resolve an id to a frozenset of base (negative-string) types.
    - negative id: returns {id}
    - list value: union of resolves
    - single value: resolve that
    Detects cycles and missing entries.
    """
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
    """
    Build F = (Z_active, A) from adjacency + hierarchical composition map.
    Returns: Z_active, A, base_types(list), potency_id_to_set(dict id->frozenset)
    """
    # Drop synthetic "root" from adjacency for structure building
    adj = _normalize_adj_remove_synthetic_root(adj)

    # Collect all ids we need to resolve
    ids_seen = set(map(str, comp_map.keys()))
    for u, chs in adj.items():
        ids_seen.add(str(u))
        if isinstance(chs, list):
            for v in chs:
                ids_seen.add(str(v))

    memo = {}
    potency_id_to_set = {}
    base_types = set()

    # Resolve every id
    for idv in ids_seen:
        if idv.startswith("-"):
            memo[idv] = frozenset([idv])
        else:
            s = _resolve_id_to_set(idv, comp_map, memo, visiting=set())
            potency_id_to_set[idv] = s

    # Gather base types
    for s in memo.values():
        for t in s:
            if t.startswith("-"):
                base_types.add(t)

    # Z: singletons for all base types + multi-type potencies (size >=2)
    Z_active = {frozenset([t]) for t in base_types}
    for pid, s in potency_id_to_set.items():
        if len(s) >= 2:
            Z_active.add(s)

    # A: only edges in adjacency, mapped via expansion; keep admissible ones
    A = {}
    def id_to_set(x: str) -> frozenset:
        x = str(x)
        if x.startswith("-"):
            return frozenset([x])
        return potency_id_to_set[x]  # safe after resolution above

    for u, chs in adj.items():
        Pu = id_to_set(u)
        for v in chs:
            Qv = id_to_set(v)
            if admissible_edge(Pu, Qv, unit_drop_edges):
                A[(Pu, Qv)] = 1

    return Z_active, A, sorted(base_types), potency_id_to_set

def score_given_map_and_trees(txt_path: str,
                              unit_drop_edges: bool = False) -> float:
    """
    Parse your EXACT file:
      1) adjacency dict
      2) node weights (ignored)
      3) composition map (CRUCIAL: defines potencies as mixtures; can reference other positives)
      4) root id
      5) leaf counts (ignored)
      6) split probs (ignored)

    Build F=(Z,A) from the composition map + adjacency, then compute
    total log-likelihood across 0002_* trees with the same DP/Beta logic.
    """
    objs = _read_json_objects_exact(txt_path)
    if len(objs) < 4:
        raise ValueError("Expected at least 4 JSON lines (adjacency, weights, composition map, root).")

    # 1) adjacency (first dict with list values)
    adj = None
    for o in objs:
        if isinstance(o, dict) and any(isinstance(v, list) for v in o.values()):
            adj = {str(k): [str(x) for x in v] for k, v in o.items() if isinstance(v, list)}
            break
    if adj is None:
        raise ValueError("Could not locate adjacency dict in the file.")

    # 2) composition map (third object)
    comp_map = objs[2]
    if not isinstance(comp_map, dict):
        raise ValueError("Third JSON must be the composition map (dict).")

    # 3) root id (fourth object) -- only for printing/sanity
    root_id = objs[3]
    if isinstance(root_id, dict) and "root_id" in root_id:
        root_id = root_id["root_id"]
    root_id = str(root_id)

    # Print vertices & edges of the given graph (raw, including 'root' if present)
    V, E = _extract_vertices_edges_from_adj(adj)
    print("=== Parsed Graph: Vertices ===")
    for v in V: print(" ", v)
    print("\n=== Parsed Graph: Edges (u -> v) ===")
    for u, v in E: print(f"  {u} -> {v}")

    # Build F = (Z, A) strictly from your map info (hierarchical potencies respected)
    Z_from_map, A_from_map, base_types_map, potency_def = _build_ZA_from_txt(
        adj=adj,
        comp_map=comp_map,
        unit_drop_edges=unit_drop_edges  # False allows multi-drop; True enforces unit-drop
    )

    # Optional: print expanded potency definitions
    print("\n=== Potency definitions (expanded) ===")
    for pid in sorted(potency_def, key=lambda x: (len(x), x)):
        s = ",".join(sorted(potency_def[pid]))
        print(f"  {pid} := {{{s}}}")

    # ----------------- Load your experimental trees + leaf maps -----------------
    trees = [read_newick_file("./0002_tree_0.txt"),
             read_newick_file("./0002_tree_1.txt"),
             read_newick_file("./0002_tree_2.txt"),
             read_newick_file("./0002_tree_3.txt"),
             read_newick_file("./0002_tree_4.txt")]

    meta_paths = ["./0002_meta_0.txt","./0002_meta_1.txt","./0002_meta_2.txt","./0002_meta_3.txt","./0002_meta_4.txt"]
    raw_maps = [read_leaf_type_map(p) for p in meta_paths]
    leaf_type_maps = [filter_leaf_map_to_tree(root, m) for root, m in zip(trees, raw_maps)]
    base_types_data = sorted({str(t) for m in leaf_type_maps for t in m.values()})

    # Final S = union of base types from data and from the map
    S_all = sorted(set(base_types_map) | set(base_types_data))

    # Ensure all singletons exist for S_all and add the map-defined potencies
    Z_active = set(Z_from_map) | {frozenset([t]) for t in S_all}
    A = dict(A_from_map)
    print(f"A:{A}")
    print(f"Z:{Z_active}")

    # Build Structure and score
    struct = Structure(S=S_all, Z_active=Z_active, A=A, unit_drop=unit_drop_edges)
    dummy_priors = Priors(potency_mode="fixed_k", fixed_k=5, rho=0.2)  # priors ignored for printed likelihoods
    k_multis = sum(1 for P in struct.Z_active if len(P) >= 2)
    print(f"k:{k_multis}")
    log_post, per_tree_logs = score_structure(
        struct=struct,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=dummy_priors,
        prune_eps=0.0
    )

    total_ll = sum(per_tree_logs)
    print("\n=== Log-likelihoods (given F from map) ===")
    for i, lg in enumerate(per_tree_logs, 1):
        print(f"Tree {i}: log P(T|F) = {lg:.6f}")
    print(f"Total log-likelihood = {total_ll:.6f}")
    return total_ll

def main():
    import random
    random.seed(7)

    # Load Newick strings from .txt (same format as .nwk)
    trees = [read_newick_file("./0002_tree_0.txt"),
             read_newick_file("./0002_tree_1.txt"),
             read_newick_file("./0002_tree_2.txt"),
             read_newick_file("./0002_tree_3.txt"),
             read_newick_file("./0002_tree_4.txt")]

    # TAB-delimited maps with header 'cellBC\tcell_state'
    map_paths = [
        "./0002_meta_0.txt",
        "./0002_meta_1.txt",
        "./0002_meta_2.txt",
        "./0002_meta_3.txt",
        "./0002_meta_4.txt",
    ]
    raw_maps = [read_leaf_type_map(p) for p in map_paths]

    # Drop dictionary entries not present in the corresponding tree
    leaf_type_maps = [filter_leaf_map_to_tree(root, m) for root, m in zip(trees, raw_maps)]

    # Build S from types that are actually used after filtering
    S = sorted({str(t) for m in leaf_type_maps for t in m.values()})

    # (Optional) soft warnings; never raise
    for idx, (root, m_raw, m_used) in enumerate(zip(trees, raw_maps, leaf_type_maps), 1):
        leaves_tree = set(collect_leaf_names(root))
        extra = sorted(set(m_raw.keys()) - leaves_tree)
        missing = sorted(leaves_tree - set(m_used.keys()))  # leaves in tree with no mapping
        if extra:
            print(f"[warn] Tree {idx}: {len(extra)} map entries are not in the tree and were ignored "
                  f"(e.g., {extra[:5]}{'...' if len(extra)>5 else ''})")
        if missing:
            print(f"[warn] Tree {idx}: {len(missing)} tree leaves have no mapping and were ignored "
                  f"(e.g., {missing[:5]}{'...' if len(missing)>5 else ''})")
        if not any(True for _ in m_used):
            print(f"[warn] Tree {idx}: no mapped leaves; treating as neutral evidence.")

    priors = Priors(potency_mode="fixed_k", fixed_k=5, rho=0.2)

    bestF, best_score, per_tree_logs = map_search_parallel(
        S=S,
        trees=trees,
        leaf_type_maps=leaf_type_maps,
        priors=priors,
        unit_drop_edges=False,
        fixed_k=priors.fixed_k if priors.potency_mode=="fixed_k" else None,
        init_seed=123,
        iters=30,
        restarts=5,
        temp_init=1.0,
        temp_decay=0.995,
        move_probs=(0.3, 0.2, 0.3, 0.2),
        prune_eps=0.0,
        n_jobs=os.cpu_count(),   # or a smaller number if memory-bound

    )

    # --- Pretty-print best map ---
    def pot_str(P): return "{" + ",".join(sorted(list(P))) + "}"
    print("\n=== BEST MAP (F*) ===")
    multi_sorted = sorted([P for P in bestF.Z_active if len(P)>=2], key=lambda x:(len(x), tuple(sorted(list(x)))))
    print("Active potencies (multi-type):")
    for P in multi_sorted: print("  ", pot_str(P))
    print("Singletons (always active):")
    for t in S: print("  ", "{"+t+"}")

    print("\nEdges:")
    edges = sorted([e for e,v in bestF.A.items() if v==1], key=lambda e:(len(e[0]), len(e[1]), tuple(sorted(list(e[0]))), tuple(sorted(list(e[1])))))
    for P,Q in edges:
        print(f"  {pot_str(P)} -> {pot_str(Q)}")

    print("\nScores:")
    print(f"  log posterior: {best_score:.6f}")
    for i,lg in enumerate(per_tree_logs,1):
        print(f"  Tree {i} log P(T|F*): {lg:.6f}")

def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    y = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{prompt} [{y}]: ").strip().lower()
        if ans == "" and default is not None:
            return default
        if ans in ("y", "yes"): return True
        if ans in ("n", "no"): return False
        print("Please answer y or n.")

def main_cli():
    print("Select mode:")
    print("  1) Run MAP search demo (uses files in code)")
    print("  2) Score a given TXT map (compute log-likelihood only)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("\n[Mode 1] Running MAP search demo...\n")
        main()  # your existing demo function
        return

    if choice == "2":
        print("\n[Mode 2] Score a given TXT map")
        txt_path = input("Path to TXT file: ").strip()
        if not txt_path:
            txt_path="main.txt"
            print("ERROR: TXT path required.")
            #return
        #unit_drop_edges = _ask_yes_no("Use unit-drop edges (|P\\Q| == 1)?", default=True)
        print("\nParsing and scoring...\n")
        try:
            score_given_map_and_trees(
                txt_path=txt_path,
                unit_drop_edges=False,
            )
        except Exception as e:
            print(f"ERROR: {e}")
        return

    print("Invalid choice. Please run again and enter 1 or 2.")

if __name__ == "__main__":
    main_cli()