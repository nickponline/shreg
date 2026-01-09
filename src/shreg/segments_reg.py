"""Segment regularization using quadratic programming.

This module implements the CGAL Shape Regularization algorithm for line segments,
which regularizes angles and offsets between nearby segments using OSQP optimization.

Reference: https://doc.cgal.org/latest/Shape_regularization/
"""
from __future__ import annotations

import importlib.resources
import random
from typing import Callable, List

import numpy as np
import osqp
from scipy import sparse
from scipy.spatial import Delaunay

from . import geometry
from . import logger
from . import plotting
from .geometry import Segment

log = logger.get_custom_logger(__name__)


def generate_random_point_on_line_segment(
    segment: Segment, i: int, n: int
) -> tuple[float, float]:
    x1, y1, x2, y2 = segment
    t = (i + 1) / n
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return x, y


def create_example_angles() -> List[Segment]:
    segments: List[Segment] = []

    s1 = np.array([-250, -50, -250, 50], dtype=np.float64)
    s2 = np.array([250, -250, 250, 250], dtype=np.float64)
    points1 = [generate_random_point_on_line_segment(s1, i, 50) for i in range(50)]
    points2 = [generate_random_point_on_line_segment(s2, i, 50) for i in range(50)]
    points1 = sorted(points1, key=lambda x: x[1])
    points2 = sorted(points2, key=lambda x: x[1])
    for i in range(50):
        segments.append(
            np.array([
                points1[i][0] + random.random() / 1000,
                points1[i][1],
                points2[i][0],
                points2[i][1],
            ], dtype=np.float64)
        )

    s1 = np.array([-50, -250, 50, -250], dtype=np.float64)
    s2 = np.array([-250, 250, 250, 250], dtype=np.float64)
    points1 = [generate_random_point_on_line_segment(s1, i, 50) for i in range(50)]
    points2 = [generate_random_point_on_line_segment(s2, i, 50) for i in range(50)]
    points1 = sorted(points1, key=lambda x: x[0])
    points2 = sorted(points2, key=lambda x: x[0])
    for i in range(50):
        segments.append(
            np.array([
                points1[i][0],
                points1[i][1] + random.random() / 1000,
                points2[i][0],
                points2[i][1],
            ], dtype=np.float64)
        )
    return segments


def get_coefficient_value(theta: float, iterator: float) -> float:
    if (
        theta == 0.0
        or theta == np.pi / 2.0
        or theta == np.pi
        or theta == 3.0 * np.pi / 2.0
    ):
        iterator = 0.0
    elif (
        theta == np.pi / 4.0
        or theta == 3.0 * np.pi / 4.0
        or theta == 5.0 * np.pi / 4.0
        or theta == 7.0 * np.pi / 4.0
    ):
        iterator = 0.22
    elif (
        (theta > 0.0 and theta < np.pi / 4.0)
        or (theta > np.pi / 2.0 and theta < 3.0 * np.pi / 4.0)
        or (theta > np.pi and theta < 5.0 * np.pi / 4.0)
        or (theta > 3.0 * np.pi / 2.0 and theta < 7.0 * np.pi / 4.0)
    ):
        iterator += 0.02
    else:
        iterator -= 0.02

    if theta < np.pi:
        return -1.0 * iterator
    else:
        return iterator


def seg(x1: float, y1: float, x2: float, y2: float) -> Segment:
    """Helper function to create a segment array."""
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def create_cgal_example() -> tuple[List[Segment], List[List[int]]]:
    """Create the CGAL 2.4 Angle + Offset Regularization example.

    This example from the CGAL documentation shows 15 segments organized into
    three contextual groups:
    - Outer boundary (7 segments): indices 0-6
    - Top rhombus (4 segments): indices 7-10
    - Bottom rhombus (4 segments): indices 11-14

    Reference: https://doc.cgal.org/latest/Shape_regularization/index.html#title10

    Returns:
        Tuple of (segments, groups) where groups define which segments
        should be regularized together.
    """
    segments = [
        # Outer boundary
        seg(1.000000, 1.000000, 0.925377, 2.995179),
        seg(1.000000, 3.000000, 1.066662, 4.951894),
        seg(1.000000, 5.000000, 2.950000, 4.930389),
        seg(3.000000, 4.950000, 2.934996, 3.008203),
        seg(3.085452, 3.003266, 2.969782, 1.002004),
        seg(0.948866, 3.033161, 2.900000, 3.000000),
        seg(0.930000, 1.000000, 2.860000, 1.002004),
        # Top rhombus
        seg(1.600000, 4.000000, 1.932136, 4.364718),
        seg(1.598613, 3.982686, 2.018220, 3.686595),
        seg(1.951872, 4.363094, 2.290848, 4.054154),
        seg(2.018220, 3.686595, 2.304517, 4.045054),
        # Bottom rhombus
        seg(1.642059, 1.928505, 1.993860, 2.247986),
        seg(1.993860, 2.247986, 2.259099, 1.919966),
        seg(1.629845, 1.923077, 1.968759, 1.599174),
        seg(2.259099, 1.919966, 1.968759, 1.599170),
    ]
    groups = [
        [0, 1, 2, 3, 4, 5, 6],  # Outer boundary
        [7, 8, 9, 10],          # Top rhombus
        [11, 12, 13, 14],       # Bottom rhombus
    ]
    return segments, groups


def create_example_offsets() -> List[Segment]:
    segments: List[Segment] = []

    theta = 0.0
    iterator = 0.0
    theta_step = np.pi / 25.0

    while theta < 2.0 * np.pi:
        st = np.sin(theta)
        ct = np.cos(theta)

        a = [0.0, 0.0]
        b = [ct, st]

        coef = get_coefficient_value(theta, iterator)
        c = [ct, st + coef]
        d = [2.0 * ct, 2.0 * st + coef]
        theta += theta_step

        segments.append(np.array([a[0], a[1], b[0], b[1]], dtype=np.float64))
        segments.append(np.array([c[0], c[1], d[0], d[1]], dtype=np.float64))
    return segments


def get_nearest_neighbour(points: np.ndarray) -> List[tuple[int, int]]:
    """Calculate nearest neighbour for each point."""
    n = len(points)
    nearest_neighbour = np.zeros(n, dtype=int)
    nearest_distance = np.zeros(n, dtype=float)
    for i in range(n):
        nearest_distance[i] = np.inf
        for j in range(n):
            if i == j:
                continue
            d = np.linalg.norm(points[i] - points[j])
            if d < nearest_distance[i]:
                nearest_distance[i] = d
                nearest_neighbour[i] = j

    ret = [(i, nearest_neighbour[i]) for i in range(n)]
    ret = set([tuple(sorted(x)) for x in ret])
    return list(ret)


def angle_error(s1: Segment, s2: Segment) -> float:
    return geometry.angle_difference(geometry.orientation(s1), geometry.orientation(s2))


def get_reference(frame: List[float], line: Segment) -> List[float]:
    mid = geometry.midpoint(line)
    dx = mid[0] - frame[0]
    dy = mid[1] - frame[1]
    angle_rad = np.radians(geometry.orientation(line))
    sin_val = np.sin(angle_rad)
    cos_val = np.cos(angle_rad)

    rx = dx * cos_val + dy * sin_val
    ry = dy * cos_val - dx * sin_val
    return [rx, ry]


def offset_error(s1: Segment, s2: Segment, debug: bool = False) -> float:
    frame = [0, 0]
    ref1 = get_reference(frame, s1)
    ref2 = get_reference(frame, s2)

    diff_orientation = abs(geometry.orientation(s1) - geometry.orientation(s2))

    if diff_orientation < 0.5:
        return ref2[1] - ref1[1]
    else:
        return np.inf


def solve(
    segments: List[Segment],
    edges: List[tuple[int, int]],
    max_bound: float = 25,
    error_func: Callable[[Segment, Segment], float] | None = None,
    update_func: Callable[[List[Segment], int, float], None] | None = None,
    debug: bool = False,
) -> List[Segment]:
    weight = 100000
    lam = 4 / 5.0
    neg_inf = -1e12
    pos_inf = 1e12
    val_neg = -2 * lam
    val_pos = +2 * lam

    k = len(segments)
    e = len(edges)
    n = k + e
    m = 2 * e + n

    if debug:
        log.info(f"{k} segments, {e} edges")
    if debug:
        log.info(f"{n} variables, {m} constraints")

    D = np.zeros((n, n))
    q = np.zeros(n)

    for i in range(n):
        if i < k:
            D[i, i] = 2.0 * weight * (1 - lam) / (max_bound * max_bound * k)
        else:
            D[i, i] = 0.0

    for i in range(n):
        if i >= k:
            q[i] = lam * weight / (4 * max_bound * e)
        else:
            q[i] = 0.0

    P = sparse.csc_matrix(D)

    X = np.zeros((m, n))
    ij = k
    it = 0

    for i, j in edges:
        X[it, i] = val_neg
        if debug:
            log.info(f"{it} {i} {val_neg}")
        X[it, j] = val_pos
        if debug:
            log.info(f"{it} {j} {val_pos}")
        X[it, ij] = -1
        if debug:
            log.info(f"{it} {ij} -1")
        it += 1

        X[it, i] = val_pos
        if debug:
            log.info(f"{it} {i} {val_pos}")
        X[it, j] = val_neg
        if debug:
            log.info(f"{it} {j} {val_neg}")
        X[it, ij] = -1
        if debug:
            log.info(f"{it} {ij} -1")
        ij += 1
        it += 1

    s = len(edges) * 2
    for i in range(n):
        X[s + i, i] = 1
        if debug:
            log.info(f"{s + i} {i} 1")

    A = sparse.csc_matrix(X)
    L = np.zeros(m, dtype="float64")
    U = np.zeros(m, dtype="float64")

    if debug:
        log.info(f"{L.shape} {U.shape}")

    tit = 0
    for i in range(m):
        if i < 2 * e:
            edge = edges[tit]
            value = error_func(segments[edge[0]], segments[edge[1]])
            if i % 2 == 0:
                L[i] = neg_inf
                U[i] = val_neg * value
            else:
                L[i] = neg_inf
                U[i] = val_pos * value
                tit += 1
        elif i < 2 * e + k:
            L[i] = -(1) * max_bound
            U[i] = +(1) * max_bound
        else:
            L[i] = neg_inf
            U[i] = pos_inf

    prob = osqp.OSQP()

    try:
        prob.setup(P, q, A, L, U, eps_abs=1e-05, eps_rel=1e-05, verbose=False)
    except Exception as ex:
        log.info(f"Problem is not feasible, check constraints: {ex}")
        return segments
    res = prob.solve()
    solution = res.x

    if debug:
        log.info(f"solution: {solution}")
    if debug:
        log.info(f"objective: {solution.T.dot(D).dot(solution) + q.dot(solution)}")

    for i in range(k):
        correction = solution[i]
        update_func(segments, i, correction)

    return segments


def angle_update_func(segments: List[Segment], idx: int, correction: float) -> None:
    segments[idx] = geometry.rotate(segments[idx], correction)


def offset_update_func(segments: List[Segment], idx: int, correction: float) -> None:
    segments[idx] = geometry.translate_by_normal(segments[idx], correction)


def get_edges(
    segments: List[Segment],
    error_func: Callable[[Segment, Segment], float],
    bound: float = 25,
    debug: bool = False,
) -> List[tuple[int, int]]:
    points = np.array([geometry.midpoint(s) for s in segments])
    try:
        tri = Delaunay(points)
    except Exception:
        edges = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                error = error_func(segments[i], segments[j])
                if abs(error) < 2 * bound:
                    edges.append((i, j))

        return edges

    tripoints = tri.simplices

    if debug:
        plotting.delaunay_debug(segments, tripoints, error_func)

    edges = set()
    for v1, v2, v3 in tripoints:
        a, b = min(v1, v2), max(v1, v2)
        target = error_func(segments[a], segments[b])
        if abs(target) < bound * 2:
            edges.add((a, b))
        a, b = min(v2, v3), max(v2, v3)
        target = error_func(segments[a], segments[b])
        if abs(target) < bound * 2:
            edges.add((a, b))
        a, b = min(v3, v1), max(v3, v1)
        target = error_func(segments[a], segments[b])
        if abs(target) < bound * 2:
            edges.add((a, b))

    return list(edges)


def solve_line_segments(
    segments: List[Segment],
    offset: bool = True,
    angle: bool = True,
    maximum_offset: float = 0.5,
    maximum_angle: float = 25,
    debug: bool = False,
) -> List[Segment]:
    if len(segments) < 2:
        return segments

    # Convert to numpy arrays if needed
    segments = [
        np.array(s, dtype=np.float64) if not isinstance(s, np.ndarray) else s.astype(np.float64)
        for s in segments
    ]

    # Standardize segments (ensure x1 <= x2)
    segments = [geometry.standardize_segment(s) for s in segments]

    if angle:
        edges = get_edges(segments, angle_error, bound=maximum_angle, debug=debug)
        if debug:
            plotting.constraints(segments, edges, angle_error, title="Angle Constraints")
        solve(
            segments,
            edges,
            max_bound=maximum_angle,
            error_func=angle_error,
            update_func=angle_update_func,
        )

    if offset:
        edges = get_edges(segments, offset_error, bound=maximum_offset, debug=debug)
        if debug:
            plotting.constraints(segments, edges, offset_error, title="Offset Constraints")
        segments = solve(
            segments,
            edges,
            max_bound=maximum_offset,
            error_func=offset_error,
            update_func=offset_update_func,
        )

    return segments


def find_endpoint_clusters(
    segments: List[Segment],
    epsilon: float = 1.0,
) -> List[List[tuple[int, int]]]:
    """Find clusters of nearby endpoints using spatial indexing.

    Uses a KD-Tree to efficiently find endpoints within epsilon distance of each other,
    then groups them into clusters using union-find.

    Args:
        segments: List of segments, each with shape (4,): [x1, y1, x2, y2]
        epsilon: Maximum distance for endpoints to be considered "close"

    Returns:
        List of clusters, where each cluster is a list of (segment_idx, endpoint_idx) tuples.
        endpoint_idx is 0 for the first endpoint (x1, y1) and 1 for the second (x2, y2).
    """
    from scipy.spatial import KDTree

    # Extract all endpoints: [(segment_idx, endpoint_idx, x, y), ...]
    endpoints = []
    coords = []
    for i, seg in enumerate(segments):
        endpoints.append((i, 0))  # First endpoint
        coords.append([seg[0], seg[1]])
        endpoints.append((i, 1))  # Second endpoint
        coords.append([seg[2], seg[3]])

    coords = np.array(coords)
    n_endpoints = len(endpoints)

    # Build KD-Tree and find pairs within epsilon
    tree = KDTree(coords)
    pairs = tree.query_pairs(epsilon)

    # Union-Find to group endpoints into clusters
    parent = list(range(n_endpoints))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i, j in pairs:
        union(i, j)

    # Group endpoints by their root
    from collections import defaultdict
    clusters_dict = defaultdict(list)
    for idx in range(n_endpoints):
        root = find(idx)
        clusters_dict[root].append(endpoints[idx])

    # Filter to clusters with more than one endpoint
    clusters = [c for c in clusters_dict.values() if len(c) > 1]

    return clusters


def find_t_junctions(
    segments: List[Segment],
    epsilon: float = 1.0,
    exclude_clusters: List[List[tuple[int, int]]] | None = None,
) -> List[tuple[tuple[int, int], int]]:
    """Find T-junctions where an endpoint is close to another segment's interior.

    Args:
        segments: List of segments
        epsilon: Maximum distance for endpoint to segment
        exclude_clusters: Endpoint clusters to exclude (already snapped endpoint-to-endpoint)

    Returns:
        List of (endpoint, segment_idx) tuples where endpoint = (seg_idx, endpoint_idx)
        and segment_idx is the segment the endpoint should snap onto.
    """
    # Build set of clustered endpoints to exclude
    clustered = set()
    if exclude_clusters:
        for cluster in exclude_clusters:
            for ep in cluster:
                clustered.add(ep)

    t_junctions = []
    n = len(segments)

    for i, seg in enumerate(segments):
        for ep_idx in [0, 1]:
            if (i, ep_idx) in clustered:
                continue

            # Get endpoint coordinates
            if ep_idx == 0:
                px, py = seg[0], seg[1]
            else:
                px, py = seg[2], seg[3]
            point = np.array([[px, py]])

            # Check distance to all other segments
            for j in range(n):
                if i == j:
                    continue

                dist = geometry.distance_segment_to_points(segments[j], point)
                if dist < epsilon:
                    # Check it's not near the endpoints of segment j
                    x1, y1, x2, y2 = segments[j]
                    dist_to_ep1 = np.sqrt((px - x1)**2 + (py - y1)**2)
                    dist_to_ep2 = np.sqrt((px - x2)**2 + (py - y2)**2)

                    # Only consider it a T-junction if not close to endpoints
                    if dist_to_ep1 > epsilon and dist_to_ep2 > epsilon:
                        t_junctions.append(((i, ep_idx), j))
                        break  # One T-junction per endpoint

    return t_junctions


def snap_regularize_segments(
    segments: List[Segment],
    epsilon: float = 1.0,
    method: str = "hard",
    soft_weight: float = 100.0,
    t_junctions: bool = False,
    debug: bool = False,
) -> List[Segment]:
    """Regularize segments by snapping nearby endpoints together.

    This implements Snap (Connectivity) regularization as a Quadratic Programming problem.
    Endpoints within epsilon distance are clustered and forced to coincide, minimizing
    total endpoint movement while satisfying connectivity constraints.

    Args:
        segments: List of segments to regularize
        epsilon: Maximum distance for endpoints to be considered "close" and snapped
        method: "hard" for exact equality constraints (Method A - perfectly watertight),
               "soft" for penalty-based constraints (Method B - elastic connections),
               "cluster" for cluster-then-solve approach (fastest, guaranteed watertight)
        soft_weight: Weight (lambda) for soft constraints. Higher = stiffer snap.
        t_junctions: If True, also detect and snap T-junctions (endpoints onto segments)
        debug: If True, print debug information

    Returns:
        List of regularized segments with snapped endpoints

    Reference:
        The formulation minimizes (1/2)||x - x_hat||^2 subject to snap constraints,
        where x contains all endpoint coordinates and x_hat are the original positions.
    """
    if len(segments) < 2:
        return segments

    # Convert to numpy arrays if needed
    segments = [
        np.array(s, dtype=np.float64) if not isinstance(s, np.ndarray) else s.astype(np.float64)
        for s in segments
    ]

    # Find endpoint clusters
    clusters = find_endpoint_clusters(segments, epsilon)

    if debug:
        log.info(f"Found {len(clusters)} endpoint clusters")
        for i, cluster in enumerate(clusters):
            log.info(f"  Cluster {i}: {cluster}")

    # Find T-junctions if requested
    tjunc_list = []
    if t_junctions:
        tjunc_list = find_t_junctions(segments, epsilon, exclude_clusters=clusters)
        if debug:
            log.info(f"Found {len(tjunc_list)} T-junctions")

    if not clusters and not tjunc_list:
        if debug:
            log.info("No endpoints to snap")
        return segments

    # Choose solving method
    if method == "cluster":
        return _solve_snap_cluster(segments, clusters, tjunc_list, debug)
    elif method == "hard":
        return _solve_snap_hard(segments, clusters, tjunc_list, debug)
    elif method == "soft":
        return _solve_snap_soft(segments, clusters, tjunc_list, soft_weight, debug)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hard', 'soft', or 'cluster'.")


def _solve_snap_cluster(
    segments: List[Segment],
    clusters: List[List[tuple[int, int]]],
    t_junctions: List[tuple[tuple[int, int], int]],
    debug: bool = False,
) -> List[Segment]:
    """Solve snap regularization using cluster-then-solve approach.

    This is the most efficient method. Instead of using constraints, we:
    1. Replace clustered endpoints with a single "cluster variable"
    2. Solve unconstrained optimization on the reduced variable set
    3. The optimal position for each cluster is simply the centroid

    This guarantees mathematically watertight results without Lagrange multipliers.
    """
    n = len(segments)
    result = [seg.copy() for seg in segments]

    # For each cluster, compute centroid and assign to all endpoints in cluster
    for cluster in clusters:
        # Compute centroid of all endpoints in cluster
        coords = []
        for seg_idx, ep_idx in cluster:
            if ep_idx == 0:
                coords.append([segments[seg_idx][0], segments[seg_idx][1]])
            else:
                coords.append([segments[seg_idx][2], segments[seg_idx][3]])

        centroid = np.mean(coords, axis=0)

        if debug:
            log.info(f"Cluster centroid: {centroid}")

        # Assign centroid to all endpoints in cluster
        for seg_idx, ep_idx in cluster:
            if ep_idx == 0:
                result[seg_idx][0] = centroid[0]
                result[seg_idx][1] = centroid[1]
            else:
                result[seg_idx][2] = centroid[0]
                result[seg_idx][3] = centroid[1]

    # Handle T-junctions: project endpoint onto target segment
    for (seg_idx, ep_idx), target_seg_idx in t_junctions:
        if ep_idx == 0:
            point = np.array([[result[seg_idx][0], result[seg_idx][1]]])
        else:
            point = np.array([[result[seg_idx][2], result[seg_idx][3]]])

        # Project onto target segment
        proj_x, proj_y = geometry.distance_segment_to_points(
            result[target_seg_idx], point, return_xy=True
        )

        if ep_idx == 0:
            result[seg_idx][0] = proj_x
            result[seg_idx][1] = proj_y
        else:
            result[seg_idx][2] = proj_x
            result[seg_idx][3] = proj_y

    return result


def _solve_snap_hard(
    segments: List[Segment],
    clusters: List[List[tuple[int, int]]],
    t_junctions: List[tuple[tuple[int, int], int]],
    debug: bool = False,
) -> List[Segment]:
    """Solve snap regularization with hard equality constraints (Method A).

    Formulation:
        minimize    (1/2) x'Hx + f'x
        subject to  A_eq @ x = b_eq

    Where:
        - x = [x11, y11, x12, y12, ..., xN2, yN2]' (all 4N coordinates)
        - H = I (identity matrix for equal weight on all points)
        - f = -x_hat (negative of original coordinates)
        - A_eq encodes equality constraints: v_i - u_j = 0 for snapped endpoints
    """
    n = len(segments)
    num_vars = 4 * n  # 4 coordinates per segment

    # Build original coordinate vector x_hat
    x_hat = np.zeros(num_vars)
    for i, seg in enumerate(segments):
        x_hat[4*i:4*i+4] = seg

    # Build equality constraints for clusters
    # For cluster with endpoints [(i1, e1), (i2, e2), ...]:
    # All must equal each other, so we use constraints:
    # endpoint[0] - endpoint[1] = 0
    # endpoint[0] - endpoint[2] = 0
    # etc.

    eq_rows = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue

        # Use first endpoint as reference
        ref_seg, ref_ep = cluster[0]
        ref_x_idx = 4 * ref_seg + (0 if ref_ep == 0 else 2)
        ref_y_idx = ref_x_idx + 1

        # Constrain all other endpoints to equal the reference
        for seg_idx, ep_idx in cluster[1:]:
            x_idx = 4 * seg_idx + (0 if ep_idx == 0 else 2)
            y_idx = x_idx + 1

            # x constraint: ref_x - this_x = 0
            row_x = np.zeros(num_vars)
            row_x[ref_x_idx] = 1
            row_x[x_idx] = -1
            eq_rows.append(row_x)

            # y constraint: ref_y - this_y = 0
            row_y = np.zeros(num_vars)
            row_y[ref_y_idx] = 1
            row_y[y_idx] = -1
            eq_rows.append(row_y)

    # Note: T-junctions with hard constraints are more complex (collinearity is quadratic)
    # For simplicity, we handle them approximately after the main solve
    # A full implementation would linearize the collinearity constraint

    if not eq_rows:
        # No constraints, just return original segments
        return [seg.copy() for seg in segments]

    A_eq = np.array(eq_rows)
    b_eq = np.zeros(len(eq_rows))

    # Solve using OSQP with equality constraints
    # OSQP handles equality as: l <= Ax <= u with l = u
    H = sparse.eye(num_vars, format='csc')
    f = -x_hat

    # Add box constraints (no bounds, but OSQP needs them)
    # Full constraint matrix: [A_eq; I]
    # Bounds: equality rows have l=u=0, identity rows have large bounds
    A_full = sparse.vstack([
        sparse.csc_matrix(A_eq),
        sparse.eye(num_vars)
    ], format='csc')

    l_full = np.concatenate([b_eq, np.full(num_vars, -1e12)])
    u_full = np.concatenate([b_eq, np.full(num_vars, 1e12)])

    prob = osqp.OSQP()
    try:
        prob.setup(H, f, A_full, l_full, u_full, eps_abs=1e-08, eps_rel=1e-08, verbose=False)
        res = prob.solve()
        x_opt = res.x
    except Exception as ex:
        log.info(f"Snap optimization failed: {ex}")
        return [seg.copy() for seg in segments]

    if debug:
        log.info(f"Snap solution status: {res.info.status}")

    # Reconstruct segments from solution
    result = []
    for i in range(n):
        result.append(np.array(x_opt[4*i:4*i+4], dtype=np.float64))

    # Handle T-junctions by projection (approximate)
    for (seg_idx, ep_idx), target_seg_idx in t_junctions:
        if ep_idx == 0:
            point = np.array([[result[seg_idx][0], result[seg_idx][1]]])
        else:
            point = np.array([[result[seg_idx][2], result[seg_idx][3]]])

        proj_x, proj_y = geometry.distance_segment_to_points(
            result[target_seg_idx], point, return_xy=True
        )

        if ep_idx == 0:
            result[seg_idx][0] = proj_x
            result[seg_idx][1] = proj_y
        else:
            result[seg_idx][2] = proj_x
            result[seg_idx][3] = proj_y

    return result


def _solve_snap_soft(
    segments: List[Segment],
    clusters: List[List[tuple[int, int]]],
    t_junctions: List[tuple[tuple[int, int], int]],
    soft_weight: float = 100.0,
    debug: bool = False,
) -> List[Segment]:
    """Solve snap regularization with soft penalty constraints (Method B).

    Formulation:
        minimize    (1/2) x'Hx + f'x

    Where the objective combines:
        - Fidelity: (1/2) sum_i ||p_i - p_hat_i||^2
        - Snap penalty: (lambda/2) sum_{(i,j) in pairs} ||v_i - u_j||^2

    This creates "spring" forces between endpoints that should connect.
    Higher lambda = stiffer springs = closer to hard constraints.
    """
    n = len(segments)
    num_vars = 4 * n

    # Build original coordinate vector
    x_hat = np.zeros(num_vars)
    for i, seg in enumerate(segments):
        x_hat[4*i:4*i+4] = seg

    # Build H matrix
    # H = I + lambda * (sum over pairs of: difference matrices)
    # For a pair (v_i, u_j), the penalty ||v_i - u_j||^2 adds to H:
    #   +lambda at (v_i_x, v_i_x), (v_i_y, v_i_y), (u_j_x, u_j_x), (u_j_y, u_j_y)
    #   -lambda at (v_i_x, u_j_x), (u_j_x, v_i_x), (v_i_y, u_j_y), (u_j_y, v_i_y)

    H = np.eye(num_vars)

    for cluster in clusters:
        # Add springs between all pairs in cluster
        for idx1, (seg1, ep1) in enumerate(cluster):
            for seg2, ep2 in cluster[idx1+1:]:
                x1_idx = 4 * seg1 + (0 if ep1 == 0 else 2)
                y1_idx = x1_idx + 1
                x2_idx = 4 * seg2 + (0 if ep2 == 0 else 2)
                y2_idx = x2_idx + 1

                # Add spring terms
                H[x1_idx, x1_idx] += soft_weight
                H[y1_idx, y1_idx] += soft_weight
                H[x2_idx, x2_idx] += soft_weight
                H[y2_idx, y2_idx] += soft_weight
                H[x1_idx, x2_idx] -= soft_weight
                H[x2_idx, x1_idx] -= soft_weight
                H[y1_idx, y2_idx] -= soft_weight
                H[y2_idx, y1_idx] -= soft_weight

    # Add T-junction springs (endpoint to projected point on segment)
    # This is an approximation - we use the current projection point
    for (seg_idx, ep_idx), target_seg_idx in t_junctions:
        if ep_idx == 0:
            point = np.array([[segments[seg_idx][0], segments[seg_idx][1]]])
        else:
            point = np.array([[segments[seg_idx][2], segments[seg_idx][3]]])

        # Get projection point (fixed for linearization)
        proj_x, proj_y = geometry.distance_segment_to_points(
            segments[target_seg_idx], point, return_xy=True
        )

        # This adds a spring from the endpoint to a fixed projection point
        # which is an approximation of the true sliding constraint
        x_idx = 4 * seg_idx + (0 if ep_idx == 0 else 2)
        y_idx = x_idx + 1

        # Add to diagonal (endpoint is pulled toward projection)
        H[x_idx, x_idx] += soft_weight
        H[y_idx, y_idx] += soft_weight

        # Adjust linear term to pull toward projection point
        # The penalty lambda * ||p - proj||^2 = lambda * (p'p - 2*p'*proj + proj'*proj)
        # Linear term contribution: -lambda * proj
        x_hat[x_idx] += soft_weight * proj_x
        x_hat[y_idx] += soft_weight * proj_y

    H = sparse.csc_matrix(H)
    f = -x_hat

    # No inequality constraints, just solve unconstrained QP
    A = sparse.eye(num_vars, format='csc')
    l = np.full(num_vars, -1e12)
    u = np.full(num_vars, 1e12)

    prob = osqp.OSQP()
    try:
        prob.setup(H, f, A, l, u, eps_abs=1e-08, eps_rel=1e-08, verbose=False)
        res = prob.solve()
        x_opt = res.x
    except Exception as ex:
        log.info(f"Snap optimization failed: {ex}")
        return [seg.copy() for seg in segments]

    if debug:
        log.info(f"Snap solution status: {res.info.status}")

    # Reconstruct segments
    result = []
    for i in range(n):
        result.append(np.array(x_opt[4*i:4*i+4], dtype=np.float64))

    return result


def fit_line_segment(points: np.ndarray) -> tuple[float, float, float, float]:
    """Fit a line segment to set of 2d points and return x1, y1, x2, y2."""
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    x1 = x.min()
    x2 = x.max()
    y1 = m * x1 + c
    y2 = m * x2 + c
    return x1, y1, x2, y2


def process_real(plot: bool = False) -> List[Segment]:
    """Load and process real segment data from bundled data file."""
    last_group = None
    segments: List[Segment] = []
    all_points: List[tuple[float, float]] = []
    points = []

    # Load data file from package resources
    data_file = importlib.resources.files(__package__).joinpath("real_data_2.xyzi")
    content = data_file.read_text()

    for line in content.strip().split("\n"):
        line = line.strip()
        parts = line.split()
        x = float(parts[0])
        y = float(parts[1])
        z = float(parts[2])
        i = int(parts[3])

        if plot:
            all_points.append((x, y))

        if i != last_group:
            if len(points) > 1:
                pts = np.array(points)
                x1, y1, x2, y2 = fit_line_segment(pts)
                segments.append(np.array([x1, y1, x2, y2], dtype=np.float64))

            last_group = i
            points = [(x, y)]
        else:
            points.append((x, y))

    if plot:
        plotting.points_and_segments(segments, all_points, title="Real Data")

    return segments


