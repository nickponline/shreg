"""Tests for Segment Regularization (angle/offset, snap, and helper functions).

Tests for:
- solve_line_segments (angle and offset regularization)
- snap_regularize_segments (endpoint snapping)
- Helper functions (edges, errors, example data)
"""
import numpy as np
import pytest

from shreg import (
    solve_line_segments,
    snap_regularize_segments,
    geometry,
)
from shreg.segments_reg import (
    generate_random_point_on_line_segment,
    create_example_angles,
    create_example_offsets,
    create_cgal_example,
    get_coefficient_value,
    seg,
    get_nearest_neighbour,
    angle_error,
    offset_error,
    get_reference,
    get_edges,
    solve,
    angle_update_func,
    offset_update_func,
    find_endpoint_clusters,
    find_t_junctions,
    fit_line_segment,
    process_real,
)


class TestSeg:
    def test_creates_numpy_array(self):
        s = seg(0, 0, 1, 1)
        assert isinstance(s, np.ndarray)
        assert s.dtype == np.float64
        assert len(s) == 4

    def test_values_correct(self):
        s = seg(1.5, 2.5, 3.5, 4.5)
        assert np.allclose(s, [1.5, 2.5, 3.5, 4.5])


class TestGenerateRandomPointOnLineSegment:
    def test_first_point(self):
        segment = np.array([0, 0, 10, 0])
        x, y = generate_random_point_on_line_segment(segment, 0, 10)
        assert np.isclose(x, 1.0)
        assert np.isclose(y, 0.0)

    def test_last_point(self):
        segment = np.array([0, 0, 10, 0])
        x, y = generate_random_point_on_line_segment(segment, 9, 10)
        assert np.isclose(x, 10.0)
        assert np.isclose(y, 0.0)

    def test_middle_point(self):
        segment = np.array([0, 0, 10, 10])
        x, y = generate_random_point_on_line_segment(segment, 4, 10)
        assert np.isclose(x, 5.0)
        assert np.isclose(y, 5.0)


class TestGetCoefficientValue:
    def test_zero_angle(self):
        result = get_coefficient_value(0.0, 1.0)
        assert result == 0.0

    def test_pi_half_angle(self):
        result = get_coefficient_value(np.pi / 2.0, 1.0)
        assert result == 0.0

    def test_pi_angle(self):
        result = get_coefficient_value(np.pi, 1.0)
        assert result == 0.0

    def test_three_pi_half_angle(self):
        result = get_coefficient_value(3.0 * np.pi / 2.0, 1.0)
        assert result == 0.0

    def test_pi_quarter_angle(self):
        result = get_coefficient_value(np.pi / 4.0, 1.0)
        assert result == -0.22

    def test_three_pi_quarter_angle(self):
        result = get_coefficient_value(3.0 * np.pi / 4.0, 1.0)
        assert result == -0.22

    def test_five_pi_quarter_angle(self):
        result = get_coefficient_value(5.0 * np.pi / 4.0, 1.0)
        assert result == 0.22

    def test_seven_pi_quarter_angle(self):
        result = get_coefficient_value(7.0 * np.pi / 4.0, 1.0)
        assert result == 0.22

    def test_increment_case(self):
        # theta in first quadrant (between 0 and pi/4)
        theta = 0.1
        result = get_coefficient_value(theta, 0.5)
        assert result == -0.52  # -(0.5 + 0.02)

    def test_decrement_case(self):
        # theta between pi/4 and pi/2 triggers decrement
        # But theta=0.5 is actually between 0 and pi/4 (0.785), so it increments
        # Let's use theta = 0.9 which is between pi/4 and pi/2
        theta = 0.9
        result = get_coefficient_value(theta, 0.5)
        assert result == -0.48  # -(0.5 - 0.02)

    def test_theta_greater_than_pi(self):
        theta = 4.0  # Greater than pi
        result = get_coefficient_value(theta, 0.5)
        assert result > 0  # Should be positive


class TestCreateExamples:
    def test_create_example_angles(self):
        segments = create_example_angles()
        assert len(segments) == 100
        for s in segments:
            assert isinstance(s, np.ndarray)
            assert len(s) == 4

    def test_create_example_offsets(self):
        segments = create_example_offsets()
        assert len(segments) > 0
        for s in segments:
            assert isinstance(s, np.ndarray)
            assert len(s) == 4

    def test_create_cgal_example(self):
        segments, groups = create_cgal_example()
        assert len(segments) == 15
        assert len(groups) == 3
        assert groups[0] == [0, 1, 2, 3, 4, 5, 6]
        assert groups[1] == [7, 8, 9, 10]
        assert groups[2] == [11, 12, 13, 14]


class TestGetNearestNeighbour:
    def test_two_points(self):
        points = np.array([[0, 0], [1, 0]])
        pairs = get_nearest_neighbour(points)
        assert len(pairs) == 1
        assert (0, 1) in pairs

    def test_three_points_in_line(self):
        points = np.array([[0, 0], [1, 0], [2, 0]])
        pairs = get_nearest_neighbour(points)
        # Should have pairs (0,1) and (1,2)
        assert len(pairs) == 2

    def test_four_points_square(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        pairs = get_nearest_neighbour(points)
        assert len(pairs) >= 2


class TestAngleError:
    def test_same_orientation(self):
        s1 = np.array([0, 0, 1, 0])
        s2 = np.array([0, 1, 1, 1])
        error = angle_error(s1, s2)
        assert np.isclose(error, 0.0)

    def test_perpendicular(self):
        s1 = np.array([0, 0, 1, 0])  # horizontal
        s2 = np.array([0, 0, 0, 1])  # vertical
        error = angle_error(s1, s2)
        # angle_difference returns min distance to 0, 90, 180
        assert np.isclose(error, 0.0)

    def test_45_degree_difference(self):
        s1 = np.array([0, 0, 1, 0])  # 0 degrees
        s2 = np.array([0, 0, 1, 1])  # 45 degrees
        error = angle_error(s1, s2)
        assert np.isclose(error, 45.0)


class TestGetReference:
    def test_horizontal_segment(self):
        frame = [0, 0]
        line = np.array([0, 0, 2, 0])
        ref = get_reference(frame, line)
        assert len(ref) == 2

    def test_vertical_segment(self):
        frame = [0, 0]
        line = np.array([0, 0, 0, 2])
        ref = get_reference(frame, line)
        assert len(ref) == 2


class TestOffsetError:
    def test_parallel_segments(self):
        s1 = np.array([0, 0, 2, 0])
        s2 = np.array([0, 1, 2, 1])
        error = offset_error(s1, s2)
        assert error != np.inf

    def test_non_parallel_segments(self):
        s1 = np.array([0, 0, 1, 0])
        s2 = np.array([0, 0, 0, 1])
        error = offset_error(s1, s2)
        assert error == np.inf


class TestGetEdges:
    def test_basic_edges(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([0, 0.5, 1, 0.5]),
            np.array([0, 1, 1, 1]),
        ]
        edges = get_edges(segments, angle_error, bound=45)
        assert len(edges) >= 0

    def test_no_edges_with_strict_bound(self):
        # angle_error uses angle_difference which is modulo 90, so horizontal and
        # vertical have 0 difference. Use a 45-degree segment instead.
        segments = [
            np.array([0, 0, 1, 0]),  # horizontal (0 degrees)
            np.array([0, 0, 1, 1]),  # diagonal (45 degrees)
        ]
        # These have 45 degree difference which exceeds bound=1
        edges = get_edges(segments, angle_error, bound=1)
        assert len(edges) == 0

    def test_collinear_points_fallback(self):
        """Test fallback when Delaunay triangulation fails."""
        # Create collinear points (midpoints on same line)
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([2, 0, 3, 0]),
            np.array([4, 0, 5, 0]),
        ]
        edges = get_edges(segments, angle_error, bound=45)
        assert isinstance(edges, list)


class TestUpdateFuncs:
    def test_angle_update_func(self):
        segments = [np.array([0, 0, 2, 0])]
        original_orientation = geometry.orientation(segments[0])
        angle_update_func(segments, 0, 10.0)
        new_orientation = geometry.orientation(segments[0])
        assert np.isclose(new_orientation - original_orientation, 10.0) or \
               np.isclose(abs(new_orientation - original_orientation - 180), 10.0)

    def test_offset_update_func(self):
        segments = [np.array([0, 0, 2, 0])]
        offset_update_func(segments, 0, 1.0)
        # Y coordinates should change
        assert segments[0][1] != 0 or segments[0][3] != 0


class TestSolve:
    def test_solve_basic(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([0, 0.1, 1, 0.1]),
        ]
        edges = [(0, 1)]
        result = solve(
            segments,
            edges,
            max_bound=5,
            error_func=angle_error,
            update_func=angle_update_func,
        )
        assert len(result) == 2


class TestSolveLineSegments:
    def test_empty_segments(self):
        result = solve_line_segments([])
        assert result == []

    def test_single_segment(self):
        segments = [np.array([0, 0, 1, 1])]
        result = solve_line_segments(segments)
        assert len(result) == 1

    def test_parallel_segments_stay_parallel(self):
        segments = [
            np.array([0, 0, 2, 0]),
            np.array([0, 1, 2, 1]),
        ]
        result = solve_line_segments(segments, offset=True, angle=True)
        # Check segments are still roughly parallel
        o1 = geometry.orientation(result[0])
        o2 = geometry.orientation(result[1])
        assert abs(o1 - o2) < 5.0 or abs(abs(o1 - o2) - 180) < 5.0

    def test_angle_only(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([0, 1, 1.01, 1.02]),  # slightly tilted
        ]
        result = solve_line_segments(segments, offset=False, angle=True, maximum_angle=25)
        assert len(result) == 2

    def test_offset_only(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([0, 0.1, 1, 0.1]),
        ]
        result = solve_line_segments(segments, offset=True, angle=False, maximum_offset=0.5)
        assert len(result) == 2

    def test_accepts_lists(self):
        """Test that regular lists work, not just numpy arrays."""
        segments = [
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ]
        result = solve_line_segments(segments)
        assert len(result) == 2


class TestFindEndpointClusters:
    def test_no_clusters(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([5, 5, 6, 5]),  # Far away
        ]
        clusters = find_endpoint_clusters(segments, epsilon=0.5)
        assert len(clusters) == 0

    def test_simple_cluster(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([1.05, 0, 2, 0]),  # Start near end of first
        ]
        clusters = find_endpoint_clusters(segments, epsilon=0.15)
        assert len(clusters) == 1
        # Should cluster endpoint 1 of seg 0 with endpoint 0 of seg 1
        assert len(clusters[0]) == 2

    def test_multiple_clusters(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([1.05, 0, 2, 0]),
            np.array([2.05, 0, 3, 0]),
        ]
        clusters = find_endpoint_clusters(segments, epsilon=0.15)
        assert len(clusters) == 2


class TestFindTJunctions:
    def test_simple_t_junction(self):
        segments = [
            np.array([0, 0, 2, 0]),  # horizontal
            np.array([1, -0.05, 1, 1]),  # vertical touching middle
        ]
        tjunctions = find_t_junctions(segments, epsilon=0.15)
        assert len(tjunctions) >= 1

    def test_no_t_junction(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([5, 5, 6, 6]),  # Far away
        ]
        tjunctions = find_t_junctions(segments, epsilon=0.15)
        assert len(tjunctions) == 0

    def test_excludes_clustered_endpoints(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([1.05, 0, 2, 0]),
        ]
        clusters = find_endpoint_clusters(segments, epsilon=0.15)
        tjunctions = find_t_junctions(segments, epsilon=0.15, exclude_clusters=clusters)
        # The close endpoints should be excluded
        for (seg_idx, ep_idx), _ in tjunctions:
            assert (seg_idx, ep_idx) not in [(0, 1), (1, 0)]


class TestSnapRegularizeSegments:
    def test_single_segment(self):
        segments = [np.array([0, 0, 1, 1])]
        result = snap_regularize_segments(segments)
        assert len(result) == 1

    def test_cluster_method(self):
        segments = [
            np.array([0, 0, 1, 0.05]),
            np.array([1.08, 0, 1.05, 1]),
            np.array([1, 1.08, 0, 0.95]),
            np.array([-0.05, 1, 0, 0]),
        ]
        result = snap_regularize_segments(segments, epsilon=0.15, method="cluster")
        # Check corners are now coincident
        assert np.allclose(result[0][2:4], result[1][0:2], atol=0.01)

    def test_hard_method(self):
        segments = [
            np.array([0, 0, 1, 0.05]),
            np.array([1.08, 0, 1.05, 1]),
        ]
        result = snap_regularize_segments(segments, epsilon=0.15, method="hard")
        assert len(result) == 2

    def test_soft_method(self):
        segments = [
            np.array([0, 0, 1, 0.05]),
            np.array([1.08, 0, 1.05, 1]),
        ]
        result = snap_regularize_segments(segments, epsilon=0.25, method="soft", soft_weight=50.0)
        assert len(result) == 2

    def test_invalid_method_raises(self):
        # Need at least 2 segments with close endpoints to trigger method check
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([1.05, 0, 2, 0]),
        ]
        with pytest.raises(ValueError):
            snap_regularize_segments(segments, epsilon=0.15, method="invalid")

    def test_t_junction_snapping(self):
        segments = [
            np.array([0, 0, 2, 0]),
            np.array([0, 1, 2, 1]),
            np.array([0.95, -0.08, 1.05, 1.1]),
        ]
        result = snap_regularize_segments(segments, epsilon=0.15, method="cluster", t_junctions=True)
        assert len(result) == 3

    def test_no_clusters_returns_original(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([5, 5, 6, 6]),  # Far away
        ]
        result = snap_regularize_segments(segments, epsilon=0.01)
        for orig, res in zip(segments, result):
            assert np.allclose(orig, res)


class TestFitLineSegment:
    def test_horizontal_line(self):
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        x1, y1, x2, y2 = fit_line_segment(points)
        assert np.isclose(y1, 0, atol=1e-6)
        assert np.isclose(y2, 0, atol=1e-6)
        assert x1 == 0
        assert x2 == 3

    def test_diagonal_line(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        x1, y1, x2, y2 = fit_line_segment(points)
        assert np.isclose(y1, x1, atol=1e-6)
        assert np.isclose(y2, x2, atol=1e-6)


class TestProcessReal:
    def test_loads_data(self):
        segments = process_real(plot=False)
        assert len(segments) > 0
        for s in segments:
            assert isinstance(s, np.ndarray)
            assert len(s) == 4
