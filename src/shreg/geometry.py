from __future__ import annotations

import collections
from typing import Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Type aliases
Segment = NDArray[np.floating]  # shape (4,): [x1, y1, x2, y2]
Point = NDArray[np.floating]    # shape (2,): [x, y]
SegmentLike = Union[Segment, Sequence[float], Tuple[float, float, float, float]]

def angle_difference(anglei: float, anglej: float) -> float:
    """Calculate the angle difference modulo 90 degrees."""
    diff_ij = anglei - anglej
    diff_90 = np.floor(diff_ij / 90)

    to_lower = 90 * (diff_90 + 0) - diff_ij
    to_upper = 90 * (diff_90 + 1) - diff_ij

    abs_lower = np.abs(to_lower)
    abs_upper = np.abs(to_upper)

    angle_deg = to_lower if abs_lower < abs_upper else to_upper
    return angle_deg


def point_side_of_line(x1, y1, x2, y2, x, y):
    """
    Determine which side of the line segment (x1, y1) to (x2, y2) the point (x, y) is on.

    Returns:
        -1 if the point is on the left side
        1 if the point is on the right side
        0 if the point is on the line
    """
    # Calculate the cross product
    cross_product = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    if cross_product > 0:
        return 1  # Right side
    elif cross_product < 0:
        return -1  # Left side
    else:
        return 0  # On the line


def is_clockwise(a, b, c):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c

    determinant = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    if determinant > 0:
        return False  # Counter-clockwise
    elif determinant < 0:
        return True  # Clockwise
    else:
        return None  # Collinear


def standardize(segments):
    ret = []

    for x1, y1, x2, y2 in segments:
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        ret.append([x1, y1, x2, y2])
    return ret


#   // Converts an angle in degrees from the range [-180, 180]
#   // into the mod 90 angle.
def convert_angle_2(angle_2):
    angle = angle_2
    if angle > 90:
        angle = 180 - angle
    elif angle < -90:
        angle = 180 + angle
    return angle


def is_orthogonal(line1, line2):
    o1 = orientation(line1)
    o2 = orientation(line2)
    return abs(o1 - o2) > 89 and abs(o1 - o2) < 91


def angle_between_vectors(a, b):
    """
    Calculate angle between two vectors in degrees

    Args:
        a (np.array): vector
        b (np.array): vector


    Returns:
        float, angle between the vector in range [-180, 180]

    Example usage:

    ```
    a = np.array([0, 1])
    b = np.array([1, 0])
    angle_between_vector(a, b)
    ```
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_theta = dot_product / (norm_a * norm_b)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    cross_product = np.cross(a, b)
    if cross_product < 0:
        angle_deg = -angle_deg
    return angle_deg


def distance_along(line, points, clip=True):
    """
    Calculate the normalized distance along a line segment for each point.

    Args:
        line (tuple): A tuple of four elements (x1, y1, x2, y2) representing the line segment.
        points (np.array): An array of points to be projected onto the line.

    Returns:
        np.array: An array of normalized distances (between 0 and 1) along the line segment for each point.

    Example usage:

    ```
    line = (0, 0, 1, 1)
    points = np.array([[0.5, 0.5], [1, 0], [0, 1]])
    distance_along(line, points)
    ```
    """
    points = project_points_to_line(points, line)
    x1, y1, x2, y2 = line

    if x2 == x1:
        t = (points[:, 1] - y1) / (y2 - y1)
    else:
        t = (points[:, 0] - x1) / (x2 - x1)

    if clip:
        return np.clip(t, 0, 1)
    else:
        return t


def project_points_to_line(points, line):
    """
    Project a set of points onto a given line segment.

    Args:
        points (np.array): An array of points to be projected onto the line.
        line (tuple): A tuple of four elements (x1, y1, x2, y2) representing the line segment.

    Returns:
        np.array: An array of points projected onto the line segment.

    Example usage:

    ```
    points = np.array([[1, 2], [2, 3], [3, 4]])
    line = (0, 0, 4, 4)
    project_points_to_line(points, line)
    ```
    """
    x1, y1, x2, y2 = line
    point1 = np.array([x1, y1])
    point2 = np.array([x2, y2])
    vec = point2 - point1
    unit_vec = vec / np.linalg.norm(vec)
    direction = points - point1
    projected_vector = np.dot(direction, unit_vec)[:, np.newaxis] * unit_vec
    projected = point1 + projected_vector
    return projected


def distance_segment_to_points(segment, xy, return_xy=False):
    """
    Calculate the distance from a line segment to a point and optionally return the projection points.

    Args:
        segment (tuple): A tuple of four elements (x1, y1, x2, y2) representing the line segment.
        xy (np.array): A point to measure the distance to the line segment.
        return_xy (bool): If True, return the projection points instead of the distances. Default is False.

    Returns:
        float or tuple: The distance from the segment to the points, or the coordinates of the projection points
        if return_xy is True. If return_xy is False, returns a float representing the distance.

    Example usage:

    ```
    segment = (0, 0, 1, 1)
    xy = np.array([[0.5, 0.5]])
    distance_segment_to_points(segment, xy)
    distance_segment_to_points(segment, xy, return_xy=True)
    ```
    """
    x1, y1, x2, y2 = segment
    x = xy[0, 0]
    y = xy[0, 1]

    projected = project_points_to_line(xy, segment)
    px, py = projected[0]

    if x2 == x1:
        t = (py - y1) / (y2 - y1)
    else:
        t = (px - x1) / (x2 - x1)

    if t < 0:
        dx = x - x1
        dy = y - y1
        if return_xy:
            return x1, y1
        else:
            return np.sqrt(dx * dx + dy * dy)
    elif t > 1:
        dx = x - x2
        dy = y - y2
        if return_xy:
            return x2, y2
        else:
            return np.sqrt(dx * dx + dy * dy)
    else:
        dx = x - px
        dy = y - py
        if return_xy:
            return px, py
        else:
            return np.sqrt(dx * dx + dy * dy)


def affine(points, translation, scale):
    """
    Apply an affine transformation to a set of points.

    Args:
        points (np.array): An array of points to be transformed.
        translation (np.array): A translation vector to be subtracted from the points.
        scale (float or np.array): A scaling factor to divide the points.

    Returns:
        np.array: An array of transformed points with the applied affine transformation.

    Example usage:

    ```
    points = np.array([[2, 4], [6, 8]])
    translation = np.array([1, 1])
    scale = 2
    affine(points, translation, scale)
    ```
    """
    points = (points - translation) / scale
    points = np.rint(points)
    points = points.astype(int)
    return points


def which_side(segment, normal):
    """
    Determine the side of a line segment relative to a normal vector.

    Args:
        segment (tuple): A tuple of four elements (x1, y1, x2, y2) representing the line segment.
        normal (np.array): A normal vector to compare against.

    Returns:
        float: A value indicating the side of the line segment relative to the normal vector.
               Returns 1 if the normal vector is to the left, -1 if to the right, and 0 if it is collinear.

    Example usage:

    ```
    segment = (0, 0, 1, 1)
    normal = np.array([1, 0])
    which_side(segment, normal)
    ```
    """
    x1, y1, x2, y2 = segment
    vec = np.array([x2 - x1, y2 - y1])
    vec = vec / np.linalg.norm(vec)
    return np.sign(np.cross(vec, normal))


def line_segment_intersection(line1, line2, return_ts=False, ignore_extent=False):
    """
    line1 and line2 should be defined in the format: ((x1, y1), (x2, y2))
    """
    l1x1, l1y1, l1x2, l1y2 = line1
    l2x1, l2y1, l2x2, l2y2 = line2

    xdiff = (l1x1 - l1x2, l2x1 - l2x2)
    ydiff = (l1y1 - l1y2, l2y1 - l2y2)

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (
        det((l1x1, l1y1), (l1x2, l1y2)),
        det((l2x1, l2y1), (l2x2, l2y2)),
    )
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    t = distance_along(line1, np.array([[x, y]]), clip=False)[0]
    s = distance_along(line2, np.array([[x, y]]), clip=False)[0]

    if return_ts:
        return t, s

    if ignore_extent:
        return x, y

    if t >= 0 and t <= 1 and s >= 0 and s <= 1:
        return x, y
    else:
        return None


def intersection_ears(line1, line2, xy):
    x, y = xy
    l1x1, l1y1, l1x2, l1y2 = line1
    l2x1, l2y1, l2x2, l2y2 = line2

    ears = [
        [l1x1, l1y1, x, y],
        [x, y, l1x2, l1y2],
        [l2x1, l2y1, x, y],
        [x, y, l2x2, l2y2],
    ]

    return ears


def direction(segment: SegmentLike) -> Point:
    """Return the direction vector of a segment, normalized to have positive y (or positive x if y=0)."""
    x1, y1, x2, y2 = segment
    vec = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    if (vec[1] < 0) or ((vec[1] == 0) and (vec[0] < 0)):
        vec = -vec
    return vec


def orientation(segment: SegmentLike) -> float:
    """Return the orientation of a segment in degrees, in range [0, 180)."""
    direc = direction(segment)
    ret = np.arctan2(direc[1], direc[0])
    ret = np.degrees(ret)
    if ret < 0:
        ret += 180
    ret = ret % 180
    return float(ret)


def midpoint(line: SegmentLike) -> Point:
    x1, y1, x2, y2 = line
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def length(segment: SegmentLike) -> float:
    """Calculate the length of a segment."""
    x1, y1, x2, y2 = segment
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def normal(segment: SegmentLike) -> Point:
    """Return the unit normal vector of a segment (perpendicular to direction)."""
    d = direction(segment)
    n = np.array([-d[1], d[0]], dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm == 0:
        raise ValueError("Cannot compute normal of zero-length segment")
    return n / norm


def flip(segment: SegmentLike) -> Segment:
    """Return a new segment with endpoints swapped."""
    x1, y1, x2, y2 = segment
    return np.array([x2, y2, x1, y1], dtype=np.float64)


def standardize_segment(segment: SegmentLike) -> Segment:
    """Return segment with x1 <= x2 (swap endpoints if needed)."""
    x1, y1, x2, y2 = segment
    if x1 > x2:
        return np.array([x2, y2, x1, y1], dtype=np.float64)
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def rotate(segment: SegmentLike, angle_deg: float) -> Segment:
    """Rotate segment around its midpoint by angle_deg degrees."""
    x1, y1, x2, y2 = segment
    mid = midpoint(segment)
    seg_length = length(segment)
    current_orientation = orientation(segment)
    new_angle = np.radians(angle_deg + current_orientation)
    half = seg_length / 2

    new_x1 = mid[0] + np.cos(new_angle) * half
    new_y1 = mid[1] + np.sin(new_angle) * half
    new_x2 = mid[0] - np.cos(new_angle) * half
    new_y2 = mid[1] - np.sin(new_angle) * half

    return np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float64)


def translate_by_normal(segment: SegmentLike, distance: float) -> Segment:
    """Translate segment along its normal by the given distance."""
    x1, y1, x2, y2 = segment
    n = normal(segment)
    return np.array([
        x1 + distance * n[0],
        y1 + distance * n[1],
        x2 + distance * n[0],
        y2 + distance * n[1],
    ], dtype=np.float64)


def signed_distance_to_point(segment: SegmentLike, point: Point) -> float:
    """
    Calculate the signed perpendicular distance from a point to the line defined by segment.

    Positive distance means the point is on the right side of the segment direction,
    negative means left side.
    """
    x1, y1, x2, y2 = segment
    px, py = point

    line_dir = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    line_len = np.linalg.norm(line_dir)
    if line_len == 0:
        raise ValueError("Cannot compute distance for zero-length segment")
    line_dir = line_dir / line_len

    point_vec = np.array([px - x1, py - y1], dtype=np.float64)
    proj = np.dot(point_vec, line_dir)
    ortho_vec = point_vec - proj * line_dir
    ortho_dist = np.linalg.norm(ortho_vec)

    cross = np.cross(line_dir, point_vec)
    sign = np.sign(cross) if cross != 0 else 0

    return float(sign * ortho_dist)


def validate_segment(segment: SegmentLike) -> None:
    """Validate that segment is a valid segment array."""
    if isinstance(segment, np.ndarray):
        if segment.shape != (4,):
            raise ValueError(f"Segment must have shape (4,), got {segment.shape}")
    elif hasattr(segment, '__len__'):
        if len(segment) != 4:
            raise ValueError(f"Segment must have 4 elements, got {len(segment)}")
    else:
        raise TypeError(f"Segment must be array-like, got {type(segment)}")


def clip_and_close(segments, cut_length=2):
    ret = []
    num_segments = len(segments)

    for i in range(num_segments):
        intersections = []
        for j in range(num_segments):
            if abs(orientation(segments[i]) - orientation(segments[j])) < 1:
                continue
            xy = line_segment_intersection(segments[i], segments[j])

            if xy is not None:
                ts = line_segment_intersection(segments[i], segments[j], return_ts=True)
                intersections.append(ts[0])

        if len(intersections) == 0:
            ret.append(segments[i])
        else:
            intersections.append(1)
            sections = []
            x1, y1, x2, y2 = segments[i]
            dx = x2 - x1
            dy = y2 - y1

            x = x1
            y = y1
            for t in intersections:
                x2 = x1 + t * dx
                y2 = y1 + t * dy

                sections.append([x, y, x2, y2])
                x = x2
                y = y2

            sections = np.array(sections)
            section_lengths = np.linalg.norm(sections[:, 2:] - sections[:, :2], axis=1)
            start = None
            end = None
            if section_lengths[0] < cut_length:
                start = 1
            if section_lengths[-1] < cut_length:
                end = -1

            ret.extend(sections[start:end])

    return ret


def group_segments(segments, max_orientation=5.0, max_distance=5.0, collate=False):
    num_segments = len(segments)
    group = [None] * num_segments
    group_counter = 0

    for i in range(num_segments):
        if group[i] is not None:
            continue

        group[i] = group_counter

        # traverse group
        for j in range(num_segments):
            if group[j] is not None:
                continue

            x1, y1, x2, y2 = segments[i]
            d1 = (
                distance_segment_to_points(segments[j], np.array([[x1, y1]]))
                < max_distance
            )
            d2 = (
                distance_segment_to_points(segments[j], np.array([[x2, y2]]))
                < max_distance
            )
            x1, y1, x2, y2 = segments[j]
            d3 = (
                distance_segment_to_points(segments[i], np.array([[x1, y1]]))
                < max_distance
            )
            d4 = (
                distance_segment_to_points(segments[i], np.array([[x2, y2]]))
                < max_distance
            )
            orientation_v = (
                np.abs(orientation(segments[i]) - orientation(segments[j]))
                < max_orientation
            )

            if orientation_v and (d1 or d2 or d3 or d4):
                group[j] = group_counter
        group_counter += 1

    if collate:
        group_lookup = collections.defaultdict(list)
        for i, g in enumerate(group):
            group_lookup[g].append(i)
        return group_lookup

    return group


def weld_segments_group(segments):
    OPEN_TOKEN = 0
    CLOSE_TOKEN = 1
    endpoints = []

    # segment point left to right ->
    segments = standardize(segments)
    for i, (x1, y1, x2, y2) in enumerate(segments):
        endpoints.append((x1, y1, OPEN_TOKEN))
        endpoints.append((x2, y2, CLOSE_TOKEN))

    endpoints = sorted(endpoints)

    num_open = 0
    welded_segments = []
    for x, y, event in endpoints:
        if event == OPEN_TOKEN:
            if num_open == 0:
                x1, y1 = x, y
            num_open += 1
        elif event == CLOSE_TOKEN:
            num_open -= 1
            if num_open == 0:
                x2, y2 = x, y
                welded_segments.append([x1, y1, x2, y2])

    return welded_segments


def weld_segments(segments, max_orientation=5.0, max_distance=5.0):
    segments = standardize(segments)
    group = group_segments(
        segments,
        max_orientation=max_orientation,
        max_distance=max_distance,
        collate=True,
    )

    ret = flatten(
        [
            weld_segments_group([segments[i] for i in group[group_key]])
            for group_key in group.keys()
        ]
    )
    return ret


def flatten(xss):
    return [x for xs in xss for x in xs]


def find_crossing(segments, i):
    crossings = []
    x1, y1, x2, y2 = segments[i]
    crossings.append((0, (x1, y1)))
    crossings.append((1, (x2, y2)))
    for j in range(len(segments)):
        if i == j:
            continue
        xy = line_segment_intersection(segments[i], segments[j])
        if xy is not None:
            t, s = line_segment_intersection(segments[i], segments[j], return_ts=True)
            crossings.append((t, xy))

    crossings = list(set(crossings))
    return sorted(crossings, key=lambda x: x[0])


def clip_segments(segments, clip_under=0.5):
    segments = standardize(segments)
    ret = []
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        crossings = find_crossing(lines, i)

        atstart = True
        start = 0
        for j in range(len(crossings) - 1):
            t, xy = crossings[j]
            x, y = xy
            t_next, xy_next = crossings[j + 1]
            x_next, y_next = xy_next

            dist = np.linalg.norm(np.array([x, y]) - np.array([x_next, y_next]))

            if dist < clip_under and atstart:
                # pylab.plot([x, x_next], [y, y_next], "r-")
                start += 1
            atstart = False

        atstart = True
        end = len(crossings) - 1
        for j in range(len(crossings) - 1, 0, -1):
            t, xy = crossings[j]
            x, y = xy
            t_next, xy_next = crossings[j - 1]
            x_next, y_next = xy_next

            dist = np.linalg.norm(np.array([x, y]) - np.array([x_next, y_next]))

            if dist < clip_under and atstart:
                # pylab.plot([x, x_next], [y, y_next], "r-")
                end -= 1
            atstart = False

        x1, y1 = crossings[start][1]
        x2, y2 = crossings[end][1]
        ret.append([x1, y1, x2, y2])
    return ret


def close_points(segments, i):
    num_segments = len(segments)

    best_extensions = [None, None]
    best_extension_distances = [None, None]

    for j in range(num_segments):
        if i == j:
            continue

        if not is_orthogonal(segments[i], segments[j]):
            continue

        xy = line_segment_intersection(segments[i], segments[j])

        if xy is not None:
            continue

        xy = line_segment_intersection(segments[i], segments[j], ignore_extent=True)

        if xy is not None:
            x1, y1, x2, y2 = segments[i]
            x, y = xy

            # if the extension point is on our segment
            if distance_segment_to_points(segments[i], np.array([[x, y]])) < 0.01:
                continue

            d1 = np.linalg.norm(np.array([x1, y1]) - np.array([xy[0], xy[1]]))
            d2 = np.linalg.norm(np.array([x2, y2]) - np.array([xy[0], xy[1]]))

            nearest_end = 0 if d1 < d2 else 1
            d = d1 if d1 < d2 else d2

            if (
                best_extension_distances[nearest_end] is None
                or d < best_extension_distances[nearest_end]
            ) and d < 1.0:
                best_extensions[nearest_end] = xy
                best_extension_distances[nearest_end] = d

    return best_extensions


