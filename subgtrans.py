import numpy as np
from itertools import product, combinations
from scipy.stats import qmc

# =========================
# Dyadic utilities
# =========================

def generate_dyadic_intervals(h):
    """Generate 1D dyadic intervals up to depth h"""
    intervals = []
    for level in range(h + 1):
        m = 1 << level
        inv_m = 1.0 / m
        intervals.extend([(i * inv_m, (i + 1) * inv_m) for i in range(m)])
    return intervals


def generate_full_dyadic_boxes(h, d):
    """All d-dim dyadic boxes with level ≤ h in each coordinate (no restrictions)."""
    one_d = generate_dyadic_intervals(h)
    return list(product(*([one_d] * d)))


def generate_filtered_dyadic_boxes(h, d, truncation_mask):
    """
    TRUNCATION: per-axis mask; 0 => only [0,1] (no refinement) on that axis.
    """
    one_d = generate_dyadic_intervals(h)
    trivial = [(0.0, 1.0)]
    allowed = [trivial if (truncation_mask is not None and truncation_mask[j] == 0) else one_d
               for j in range(d)]
    return list(product(*allowed))


def generate_superposition_boxes(h, d, s):
    """
    SUPERPOSITION: use ALL axes but restrict to interactions |u| ≤ s (others fixed to [0,1]).
    """
    if not (1 <= s <= d):
        raise ValueError("s must be in [1, d] for superposition.")
    one_d = generate_dyadic_intervals(h)
    trivial = [(0.0, 1.0)]
    boxes = []
    for k_order in range(1, s + 1):
        for u in combinations(range(d), k_order):
            grids = [one_d if j in u else trivial for j in range(d)]
            boxes.extend(product(*grids))
    return boxes


def compute_box_weights(boxes, gamma=None):
    """
    Product weights: γ(B) = ∏_{j nontrivial in B} γ_j. If gamma is None → all ones.
    """
    if gamma is None:
        return np.ones(len(boxes), dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    d = gamma.size
    w = np.ones(len(boxes), dtype=float)
    for i, B in enumerate(boxes):
        prod_w = 1.0
        for j in range(d):
            a, b = B[j]
            if not (a == 0.0 and b == 1.0):  # nontrivial in dim j
                prod_w *= gamma[j]
        w[i] = prod_w
    return w


def get_point_box_incidence(points, boxes):
    """
    Boolean incidence matrix: mask[i, j] = True iff point i lies in box j.
    points: (n, d), boxes: list of length m each with d tuples (a,b)
    """
    n, d = points.shape
    m = len(boxes)
    box_bounds = np.array(boxes)         # (m, d, 2)
    a = box_bounds[:, :, 0]              # (m, d)
    b = box_bounds[:, :, 1]              # (m, d)
    p = points[:, None, :]               # (n, 1, d)
    mask = np.all((a <= p) & (p < b), axis=2)  # (n, m)
    return mask


def _point_membership_indices(point, boxes, a=None, b=None, *, batch_cols=4096, float_dtype=np.float64):
    """
    Return indices J ⊆ {0,...,M-1} of boxes that contain `point`.
    If a,b (shape (M,d)) are provided, uses them directly (fast, but uses memory).
    Otherwise, streams over `boxes` in column batches of size `batch_cols`.
    """
    point = np.asarray(point, dtype=float_dtype)

    if a is not None and b is not None:
        # a,b are (M,d). Vectorized membership test, then nonzeros.
        mask = np.all((a <= point) & (point < b), axis=1)
        return np.nonzero(mask)[0]

    M = len(boxes)
    hits_blocks = []
    for start in range(0, M, batch_cols):
        bb = np.array(boxes[start:start+batch_cols], dtype=float_dtype)  # (B,d,2)
        a_blk = bb[:, :, 0]  # (B,d)
        b_blk = bb[:, :, 1]
        # (B,d) vs (d,) broadcast → (B,d)
        mask_blk = np.all((a_blk <= point) & (point < b_blk), axis=1)
        J_local = np.nonzero(mask_blk)[0]
        if J_local.size:
            hits_blocks.append(J_local + start)

    if hits_blocks:
        return np.concatenate(hits_blocks)
    return np.empty(0, dtype=np.int64)


def balanced_self_balancing_walk_streaming(points, boxes, box_weights, *,
                                           c_local=0.01, rng=None,
                                           batch_cols=4096, precompute=False,
                                           float_dtype=np.float64):
    """
    Streaming version of the self-balancing walk.
    Computes signs without materializing a full incidence matrix.

    points: (N,d)
    boxes:  list length M, each element is length-d list of (a,b)
    box_weights: array length M
    """
    points = np.asarray(points, dtype=float_dtype)
    N, d = points.shape
    assert N % 2 == 0, "n must be even"
    M = len(boxes)
    rng = np.random.default_rng() if rng is None else rng

    # Running weights for [I | features]
    wI = np.zeros(N, dtype=float_dtype)        # identity columns part
    wF = np.zeros(M, dtype=float_dtype)        # feature columns part
    weights = np.asarray(box_weights, dtype=float_dtype)

    coloring = np.zeros(N, dtype=np.int8)

    # Optional precompute of all box bounds (cost: ~ M*d*2*8 bytes)
    a = b = None
    if precompute and M > 0:
        bb = np.array(boxes, dtype=float_dtype)  # (M,d,2)
        a = bb[:, :, 0]
        b = bb[:, :, 1]

    for i in range(0, N, 2):
        # Membership sets for the pair (i, i+1)
        J_i   = _point_membership_indices(points[i],   boxes, a, b, batch_cols=batch_cols, float_dtype=float_dtype)
        J_ip1 = _point_membership_indices(points[i+1], boxes, a, b, batch_cols=batch_cols, float_dtype=float_dtype)

        # Current dot = <w, v_i - v_{i+1}>
        dot = wI[i] - wI[i+1]
        if J_i.size:
            dot += (wF[J_i] * weights[J_i]).sum()
        if J_ip1.size:
            dot -= (wF[J_ip1] * weights[J_ip1]).sum()

        # Sampling sign with the same rule as before
        p = 0.5 - dot / (2 * c_local)
        if p <= 0.0:
            sign = -1
        elif p >= 1.0:
            sign = 1
        else:
            sign = 1 if rng.random() < p else -1

        coloring[i]     = sign
        coloring[i + 1] = -sign

        # Update w for [I | features]
        wI[i]     += sign
        wI[i + 1] += -sign
        if J_i.size:
            wF[J_i]     += sign * weights[J_i]
        if J_ip1.size:
            wF[J_ip1]   += -sign * weights[J_ip1]

    return coloring


def balanced_self_balancing_walk(V, delta: float = 0.01):
    """
    Perform a self-balancing walk on a sparse incidence matrix V.
    This is kept for completeness; the streaming version is used in practice.
    """
    n_rows, n_cols = V.shape
    assert n_rows % 2 == 0, "n must be even"

    c_local = 0.01
    coloring = np.zeros(n_rows, dtype=int)
    w = np.zeros(n_cols)

    for i in range(0, n_rows, 2):
        v_i = V.getrow(i)
        v_ip1 = V.getrow(i + 1)

        dot = v_i.dot(w).item() - v_ip1.dot(w).item()
        p = np.clip(0.5 - dot / (2 * c_local), 0.0, 1.0)
        sign = 1 if np.random.rand() < p else -1
        coloring[i] = sign
        coloring[i + 1] = -sign

        for idx, data in zip(v_i.indices, v_i.data):
            w[idx] += sign * data
        for idx, data in zip(v_ip1.indices, v_ip1.data):
            w[idx] += -sign * data

    return coloring

# =========================
# Core SubgTrans engine
# =========================

def _prepare_feature_family(h, d, family, gamma=None, s=None,
                            rng=None, max_boxes=None):
    """
    Build (boxes, box_weights) according to desired family:
      - 'unweighted'      : full boxes, weights=1
      - 'weighted'        : full boxes, weights=product(gamma_j)
      - 'superposition'   : boxes with |u| ≤ s,   weights=1
      - 'truncation'      : refine only first s axes (mask), weights=1
    """
    if family == "unweighted":
        boxes = generate_full_dyadic_boxes(h, d)
        weights = np.ones(len(boxes), dtype=float)

    elif family == "weighted":
        if gamma is None or len(gamma) != d:
            raise ValueError("For 'weighted', provide gamma of length d.")
        boxes = generate_full_dyadic_boxes(h, d)
        weights = compute_box_weights(boxes, gamma)

    elif family == "superposition":
        if s is None:
            raise ValueError("Provide s (order) for 'superposition'.")
        boxes = generate_superposition_boxes(h, d, s)
        weights = np.ones(len(boxes), dtype=float)

    elif family == "truncation":
        if s is None:
            raise ValueError("Provide s (active leading coordinates) for 'truncation'.")
        mask = [1] * s + [0] * (d - s)
        boxes = generate_filtered_dyadic_boxes(h, d, mask)
        weights = np.ones(len(boxes), dtype=float)

    else:
        raise ValueError("family must be one of {'unweighted','weighted','superposition','truncation'}")

    # Optional subsampling to control feature count
    if max_boxes is not None and len(boxes) > max_boxes:
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.choice(len(boxes), size=max_boxes, replace=False)
        boxes = [boxes[i] for i in idx]
        weights = weights[idx]

    return boxes, weights


def subg_transference_sparse(
    n, d, h, *,            # n must be power of two
    family="unweighted",   # 'unweighted' | 'weighted' | 'superposition' | 'truncation'
    gamma=None,            # for 'weighted'
    s=None,                # for 'superposition' (order) or 'truncation' (leading dims)
    shift=True, seed=None, initial_sampler='uniform',
    init='k', k=8,         # init='n2' uses k=n; init='k' uses provided k (both must be powers of two)
    max_boxes=None
):
    """
    Generic SubgTrans / WSubgTrans engine returning a list of k sets each of size (n, d).

    - When init='n2': uses n0 = n^2 and T = log2(n)  → returns n sets.
    - When init='k' : uses n0 = k*n and T = log2(k)  → returns k sets.
    """
    # ---- Validate powers of two
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two.")

    if init == 'n2':
        k_eff = n
    elif init == 'k':
        k_eff = k
        if k_eff <= 0 or (k_eff & (k_eff - 1)) != 0:
            raise ValueError("k must be a positive power of two when init='k'.")
    else:
        raise ValueError("init must be 'n2' or 'k'.")

    T = int(np.log2(k_eff))
    n0 = k_eff * n
    rng = np.random.default_rng(seed)

    # ---- Initial sampling
    if initial_sampler == 'uniform':
        A = rng.uniform(0, 1, size=(n0, d))
    elif initial_sampler == 'sobol':
        sob = qmc.Sobol(d, scramble=True, seed=seed)
        A = sob.random(n=n0)
    else:
        raise ValueError("initial_sampler must be 'uniform' or 'sobol'")

    if shift:
        sft = rng.uniform(0, 1, size=(1, d))
        A = (A + sft) % 1.0

    # ---- Feature family
    boxes, box_weights = _prepare_feature_family(
        h=h, d=d, family=family, gamma=gamma, s=s, rng=rng, max_boxes=max_boxes
    )

    # ---- T rounds of halving with streaming balancing
    sets = [A.copy()]
    for _ in range(T):
        new_sets = []
        for At in sets:
            # Heuristic: precompute a,b if it fits in ~250MB
            approx_bytes = len(boxes) * d * 2 * 8  # float64
            precompute = approx_bytes <= 250 * 1024**2

            x = balanced_self_balancing_walk_streaming(
                At, boxes, box_weights,
                c_local=0.01,
                rng=rng,                     # reuse the rng you created earlier
                batch_cols=2048,             # tune this if you like
                precompute=precompute,
                float_dtype=np.float64
            )

            At_minus = At[x == -1]
            At_plus  = At[x == +1]
            new_sets.extend([At_minus, At_plus])

        sets = new_sets

    return sets  # list of length k_eff, each (n, d)


def run_subgtrans(n, d, *,
                  family="unweighted",   # 'unweighted' | 'weighted' | 'superposition' | 'truncation'
                  gamma=None,            # for 'weighted'
                  s=None,                # order (superposition) or leading dims (truncation)
                  h=None,                # default: ceil(log2(d*n))
                  init="n2", k=16,       # 'n2' → n^2;  'k' → k*n (k power of two)
                  sampler="uniform",     # 'uniform' | 'sobol'
                  shift=True, seed=42,
                  max_boxes=None):
    """
    Convenience wrapper. Returns list of point sets (each (n,d)).
    """
    if h is None:
        # Reasonable default: h ≈ ceil(log2(d*n))
        h = int(np.ceil(np.log2(d * n)))

    print(f"Running SubgTrans ({family}), n={n}, d={d}, h={h}, init={init}, sampler={sampler}")
    if family == "weighted" and gamma is not None:
        print(f"  Using product weights: len(gamma)={len(gamma)}")
    if family in ("superposition", "truncation") and s is not None:
        print(f"  Using parameter s={s}")

    sets = subg_transference_sparse(
        n=n, d=d, h=h,
        family=family, gamma=gamma, s=s,
        shift=shift, seed=seed, initial_sampler=sampler,
        init=init, k=k, max_boxes=max_boxes
    )
    print(f"Produced {len(sets)} sets of shape ({n}, {d}).")
    return sets



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SubgTrans or WSubgTrans construction.")
    parser.add_argument("-n", type=int, default=8, help="Number of points per set (must be power of two)")
    parser.add_argument("-d", type=int, default=2, help="Dimension")
    parser.add_argument("-family", type=str, default="unweighted",
                        choices=["unweighted", "weighted", "superposition", "truncation"],
                        help="Feature family")
    parser.add_argument("-gamma", type=float, nargs='*', default=None, help="Product weights for 'weighted' family")
    parser.add_argument("-s", type=int, default=None, help="Order for 'superposition' or leading dims for 'truncation'")
    parser.add_argument("-h", type=int, default=None, help="Dyadic depth (default: ceil(log2(d*n)))")
    parser.add_argument("-init", type=str, default="n2", choices=["n2", "k"], help="Initialization mode")
    parser.add_argument("-k", type=int, default=8, help="k for init='k' (must be power of two)")
    parser.add_argument("-sampler", type=str, default="uniform", choices=["uniform", "sobol"], help="Initial sampler")
    parser.add_argument("-shift", action="store_true", help="Apply random shift")
    parser.add_argument("-seed", type=int, default=42, help="Random seed")
    parser.add_argument("-max_boxes", type=int, default=None, help="Maximum number of boxes (features)")

    args = parser.parse_args()

    sets = run_subgtrans(
        n=args.n,
        d=args.d,
        family=args.family,
        gamma=args.gamma,
        s=args.s,
        h=args.h,
        init=args.init,
        k=args.k,
        sampler=args.sampler,
        shift=args.shift,
        seed=args.seed,
        max_boxes=args.max_boxes
    )

    for i, S in enumerate(sets[:4], 1):
        print(f"Set {i}: shape={S.shape}, first row={np.round(S[0], 4)}")
