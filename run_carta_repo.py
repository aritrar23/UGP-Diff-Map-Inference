#!/usr/bin/env python3
# Run CARTA on Data/<folder>s and report Jaccard vs ground truth (from main.txt).

import os, csv, glob, json, argparse, subprocess
from typing import List, Set, FrozenSet, Tuple, Optional

# ----------------- helpers: build CARTA inputs (txt only) -----------------

def write_file_locations(newicks: List[str], metas: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for n, m in zip(newicks, metas):
            f.write(f"{n}\t{m}\n")
def _canon_token(x: str) -> str:
    x = x.strip()
    # strip multiple layers of matching quotes if present:  "-7"  → -7,  '-7' → -7
    while len(x) >= 2 and x[0] == x[-1] and x[0] in "\"'":
        x = x[1:-1].strip()
    return x
def _states_from_meta(meta_path: str) -> Set[str]:
    """Collect states from 2nd column of a tab/CSV meta file (header tolerated)."""
    out=set()
    with open(meta_path, "r", encoding="utf-8") as f:
        first=f.readline()
        if not first: return out
        delim = "\t" if ("\t" in first or meta_path.endswith((".tsv",".txt"))) else ","
        hdr=[c.strip().lower() for c in first.strip().split(delim)]
        has_header = len(hdr)>=2 and ("type" in hdr[1] or "cell_state" in hdr[1])
        f.seek(0)
        for i, line in enumerate(f,1):
            if has_header and i==1: continue
            parts=[c.strip() for c in line.strip().split(delim)]
            if len(parts)>=2 and parts[1]!="": out.add(parts[1])
    return out

def write_states_file(metas: List[str], out_path: str) -> List[str]:
    """Union of states across meta files -> states.txt (one state per line)."""
    S=set()
    for p in metas: S |= _states_from_meta(p)
    S=sorted(S)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in S: f.write(s + "\n")
    return S

# ----------------- helpers: parse outputs & ground truth -----------------

def _parse_set_token(line: str) -> Optional[FrozenSet[str]]:
    l = line.strip()
    if not l or l.startswith("#"):
        return None
    # JSON array, e.g. ["-7","-9"]
    if l.startswith("[") and l.endswith("]"):
        try:
            arr = json.loads(l)
            items = [_canon_token(str(x)) for x in arr]
            return frozenset(items) if len(items) >= 2 else None
        except Exception:
            pass
    # Brace form, e.g. {"-7","-9"} or {-7,-9}
    if l.startswith("{") and l.endswith("}"):
        inside = l[1:-1].strip()
        items = [_canon_token(t) for t in inside.replace(",", " ").split() if t]
        return frozenset(items) if len(items) >= 2 else None
    # Bare tokens, e.g. -7 -9   or  -7,-9
    toks = [_canon_token(t) for t in l.replace(",", " ").split() if t]
    return frozenset(toks) if len(toks) >= 2 else None

def parse_progenitors(prog_file: str, states: List[str]) -> Set[FrozenSet[str]]:
    out = set()
    S = [_canon_token(s) for s in states]  # make sure binary rows map to canonical states
    with open(prog_file, "r", encoding="utf-8") as f:
        for line in f:
            s = _parse_set_token(line)
            if s:
                out.add(s)
                continue
            toks = [t for t in line.strip().replace(",", " ").split() if t]
            if toks and all(t in ("0", "1") for t in toks) and len(toks) == len(S):
                chosen = [S[i] for i, t in enumerate(toks) if t == "1"]
                if len(chosen) >= 2:
                    out.add(frozenset(chosen))
    return out
# --- add these helpers somewhere above ground_truth_sets ---

def _read_json_objects_exact(path: str):
    objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                objs.append(json.loads(line))
    if not objs:
        raise ValueError(f"{path}: no JSON objects found")
    return objs

def _parse_set_token(line: str) -> Optional[FrozenSet[str]]:
    l = line.strip()
    if not l or l.startswith("#"):
        return None
    # JSON array, e.g. ["-7","-9"]
    if l.startswith("[") and l.endswith("]"):
        try:
            arr = json.loads(l)
            items = [_canon_token(str(x)) for x in arr]
            return frozenset(items) if len(items) >= 2 else None
        except Exception:
            pass
    # Brace form, e.g. {"-7","-9"} or {-7,-9}
    if l.startswith("{") and l.endswith("}"):
        inside = l[1:-1].strip()
        items = [_canon_token(t) for t in inside.replace(",", " ").split() if t]
        return frozenset(items) if len(items) >= 2 else None
    # Bare tokens, e.g. -7 -9   or  -7,-9
    toks = [_canon_token(t) for t in l.replace(",", " ").split() if t]
    return frozenset(toks) if len(toks) >= 2 else None

def _normalize_adj_remove_synthetic_root(adj: dict) -> dict:
    # copy and drop "root" -> [single_child] if present
    adj2 = {str(k): (list(v) if isinstance(v, list) else v) for k, v in adj.items()}
    if "root" in adj2:
        ch = adj2["root"]
        if isinstance(ch, list) and len(ch) == 1:
            del adj2["root"]
    return adj2

def _resolve_id_to_set(id_str: str, comp_map: dict, memo: dict, visiting: set) -> frozenset:
    id_str = _canon_token(str(id_str))
    if id_str.startswith("-"):
        return frozenset([_canon_token(id_str)])
    if id_str in memo:
        return memo[id_str]
    if id_str in visiting:
        raise ValueError(f"Cycle detected while resolving '{id_str}'")
    if id_str not in comp_map:
        raise ValueError(f"Id '{id_str}' appears in adjacency but is missing in composition map")
    visiting.add(id_str)
    val = comp_map[id_str]
    acc = set()
    if isinstance(val, list):
        for child in val:
            acc |= _resolve_id_to_set(child, comp_map, memo, visiting)
    else:
        acc |= _resolve_id_to_set(val, comp_map, memo, visiting)
    visiting.remove(id_str)
    memo[id_str] = frozenset(_canon_token(x) for x in acc)
    return memo[id_str]

# --- replace your ground_truth_sets with this one ---

def ground_truth_sets(main_txt: str) -> Set[FrozenSet[str]]:
    """
    Ground truth = multi-type potencies that ACTUALLY appear as nodes in the map
    (i.e., vertices used in the adjacency), expanded to base types via the
    composition map. Singletons are excluded.
    """
    objs = _read_json_objects_exact(main_txt)

    # adjacency = first JSON that looks like a dict of lists
    adj = None
    for o in objs:
        if isinstance(o, dict) and any(isinstance(v, list) for v in o.values()):
            adj = {str(k): [str(x) for x in v] for k, v in o.items() if isinstance(v, list)}
            break
    if adj is None:
        raise ValueError(f"{main_txt}: could not locate adjacency dict (JSON #1)")

    # composition map is JSON #3 (by your format)
    if len(objs) < 3 or not isinstance(objs[2], dict):
        raise ValueError(f"{main_txt}: cannot find composition map (JSON #3)")
    comp_map = {str(k): v for k, v in objs[2].items()}

    adj = _normalize_adj_remove_synthetic_root(adj)

    # collect all vertex ids that occur in the adjacency
    ids = set(adj.keys())
    for chs in adj.values():
        for v in chs:
            ids.add(str(v))

    # resolve each id to a set of base types; keep only multi-type sets
    memo = {}
    gt_sets: Set[FrozenSet[str]] = set()
    for vid in ids:
        if str(vid).startswith("-"):    # skip base leaves
            continue
        s = _resolve_id_to_set(str(vid), comp_map, memo, visiting=set())
        if len(s) >= 2:
            gt_sets.add(s)
    return gt_sets

# def ground_truth_sets(main_txt: str) -> Set[FrozenSet[str]]:
#     """Your 4-JSON-lines file; composition map is JSON line #3."""
#     objs=[]
#     with open(main_txt,"r",encoding="utf-8") as f:
#         for ln in f:
#             ln=ln.strip()
#             if ln: objs.append(json.loads(ln))
#     if len(objs)<3 or not isinstance(objs[2], dict):
#         raise ValueError(f"{main_txt}: cannot find composition map on 3rd JSON line.")
#     comp=objs[2]; out=set()
#     for members in comp.values():
#         if isinstance(members, list):
#             s=frozenset(str(x) for x in members)
#             if len(s)>=2: out.add(s)
#     return out

def jaccard(A:Set[FrozenSet[str]], B:Set[FrozenSet[str]])->float:
    if not A and not B: return 0.0
    return 1.0 - (len(A & B) / len(A | B) if (A or B) else 0.0)

# ----------------- CARTA call -----------------

def run_carta(prefix: str, newicks: List[str], metas: List[str], states_file: str,
              klist: List[int], normalize_method: str, time_limit_sec: Optional[int],
              enforce_tree: bool, mode: str, progen_matrix: Optional[str]) -> Tuple[Set[FrozenSet[str]], str]:
    loc_file = prefix + "_file_locations.tsv"
    stdout_file = prefix + "_stdout.txt"
    write_file_locations(newicks, metas, loc_file)

    cmd = [
        "carta",
        "--prefix", prefix,
        "--file_locations", loc_file,
        "--states_file", states_file,
        "--normalize_method", normalize_method,
        "-m", mode
    ]
    if enforce_tree: cmd += ["--enforce_tree"]
    if time_limit_sec is not None: cmd += ["--time_limit_sec", str(time_limit_sec)]
    if klist: cmd += ["--klist", *map(str, klist)]
    if progen_matrix: cmd += ["--progen_matrix", progen_matrix]

    print(">>", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    with open(stdout_file, "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n" + (proc.stdout or "") + "\n\n=== STDERR ===\n" + (proc.stderr or ""))
    if proc.returncode != 0:
        raise RuntimeError(f"CARTA failed (code {proc.returncode}). See {stdout_file}.")

    # find newest progenitors file for this prefix
    base = os.path.dirname(prefix) or "."
    stem = os.path.basename(prefix)
    cands = sorted(set(glob.glob(os.path.join(base, f"{stem}*progenitors*.txt"))),
                   key=lambda p: (-os.path.getmtime(p), len(p)))
    if not cands:
        raise FileNotFoundError(f"No progenitors file produced for '{prefix}'. Check {stdout_file}.")
    return cands[0], stdout_file

# ----------------- per-folder pipeline -----------------

def discover_folder(data_root: str, folder: str):
    d = os.path.join(data_root, folder)
    t = sorted(glob.glob(os.path.join(d, f"{folder}_tree_*.txt")))
    m = sorted(glob.glob(os.path.join(d, f"{folder}_meta_*.txt")))
    main = os.path.join(d, "main.txt")
    if not t or not m or len(t)!=len(m): raise FileNotFoundError(f"{folder}: need equal #tree and #meta files.")
    if not os.path.isfile(main): raise FileNotFoundError(f"{folder}: missing main.txt")
    return d, t, m, main

def process_folder(data_root: str, out_root: str, folder: str, klist: List[int],
                   normalize_method: str, time_limit_sec: Optional[int],
                   enforce_tree: bool, mode: str, progen_matrix: Optional[str],
                   states_override: Optional[str]) -> Tuple[str, float]:
    d, newicks, metas, main = discover_folder(data_root, folder)

    # states file: use override if provided, else derive from meta txts
    os.makedirs(os.path.join(out_root, folder), exist_ok=True)
    prefix = os.path.join(out_root, folder, folder)
    if states_override:
        states_file = states_override
        with open(states_file, "r", encoding="utf-8") as f:
            states = [ln.strip() for ln in f if ln.strip()]
    else:
        states_file = prefix + "_states.txt"
        states = write_states_file(metas, states_file)

    prog_file, log_path = run_carta(prefix, newicks, metas, states_file, klist,
                                    normalize_method, time_limit_sec, enforce_tree, mode, progen_matrix)
    pred_sets = parse_progenitors(prog_file, states)
    gt_sets = ground_truth_sets(main)
    print("\nPredicted progenitors:")
    for s in sorted(pred_sets, key=lambda x: (len(x), tuple(sorted(x)))): print("  ", s)
    print("\nGround-truth progenitors:")
    for s in sorted(gt_sets, key=lambda x: (len(x), tuple(sorted(x)))): print("  ", s)

    jd = jaccard(pred_sets, gt_sets)

    print(f"\n=== {folder} ===")
    print(f"Pred file : {prog_file}")
    print(f"Jaccard   : {jd:.6f}")
    return folder, jd

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="CARTA batch runner on repo-style Data/<ID> folders.")
    ap.add_argument("--data-root", default="Data")
    ap.add_argument("--folders", nargs="+", help="Folder names under data-root (e.g., 0002 0003).")
    ap.add_argument("--folders-file", help="Text file: one folder per line.")
    ap.add_argument("-k", "--klist", nargs="+", type=int, default=[5])
    ap.add_argument("--normalize_method",
                    choices=["no_normalization","cell_proportion_before_pruning","cell_proportion_after_pruning"],
                    default="no_normalization")
    ap.add_argument("--time_limit_sec", type=int)
    ap.add_argument("--no_enforce_tree", action="store_true")
    ap.add_argument("-m", "--mode", choices=["exact","round"], default="exact")
    ap.add_argument("--progen_matrix", help="Optional CSV putative progenitors.")
    ap.add_argument("--states_file", help="OPTIONAL: use an existing states.txt (skip deriving).")
    ap.add_argument("--out-root", default="carta_runs")
    ap.add_argument("--csv", default="carta_runs/summary_results.csv")
    args = ap.parse_args()

    if args.folders:
        folders = args.folders
    elif args.folders_file:
        with open(args.folders_file, "r", encoding="utf-8") as f:
            folders = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    else:
        folders = sorted([d for d in os.listdir(args.data_root)
                          if os.path.isdir(os.path.join(args.data_root,d)) and d.isdigit()])

    results=[]
    for folder in folders:
        try:
            results.append(process_folder(
                data_root=args.data_root, out_root=args.out_root, folder=folder,
                klist=args.klist, normalize_method=args.normalize_method,
                time_limit_sec=args.time_limit_sec, enforce_tree=(not args.no_enforce_tree),
                mode=args.mode, progen_matrix=args.progen_matrix,
                states_override=args.states_file))
        except Exception as e:
            print(f"\n=== {folder} ERROR: {e}")
            results.append((folder, None))

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    print("\n===== Summary =====")
    print(f"{'Folder':<10} {'Jaccard':<12}")
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["Folder","Jaccard"])
        for folder, jd in results:
            s = "ERROR" if jd is None else f"{jd:.6f}"
            print(f"{folder:<10} {s:<12}")
            w.writerow([folder, "" if jd is None else s])
    print(f"Saved {args.csv}")

if __name__ == "__main__":
    main()
