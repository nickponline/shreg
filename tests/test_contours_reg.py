"""Tests for Contour Regularization."""
import numpy as np
import pytest

from shreg import regularize_contour
from shreg.contours_reg import (
    segments_from_points,
    segment_offset,
    is_parallel,
    make_segment_from_midpoint,
    line_intersection,
    project_point_onto_line,
    longest_segment,
    load_polylines,
)


class TestSegmentsFromPoints:
    def test_triangle(self):
        points = [(0, 0), (1, 0), (0.5, 1)]
        segments = segments_from_points(points)
        assert len(segments) == 3
        # First segment connects point 0 to point 1
        assert np.allclose(segments[0], [0, 0, 1, 0])
        # Second segment connects point 1 to point 2
        assert np.allclose(segments[1], [1, 0, 0.5, 1])
        # Third segment connects point 2 to point 0 (closing)
        assert np.allclose(segments[2], [0.5, 1, 0, 0])

    def test_square(self):
        points = [(0, 0), (1, 0), (1, 1), (0, 1)]
        segments = segments_from_points(points)
        assert len(segments) == 4


class TestSegmentOffset:
    def test_parallel_horizontal_lines(self):
        s = np.array([0, 0, 2, 0])
        t = np.array([0, 1, 2, 1])
        offset = segment_offset(s, t)
        assert np.isclose(offset, 1.0)

    def test_same_line(self):
        s = np.array([0, 0, 2, 0])
        t = np.array([1, 0, 3, 0])
        offset = segment_offset(s, t)
        assert np.isclose(offset, 0.0)


class TestIsParallel:
    def test_parallel_horizontal_lines(self):
        s = np.array([0, 0, 2, 0])
        t = np.array([0, 1, 2, 1])
        assert is_parallel(s, t) is True

    def test_non_parallel_lines(self):
        s = np.array([0, 0, 2, 0])
        t = np.array([0, 0, 0, 2])
        assert is_parallel(s, t) is False

    def test_nearly_parallel_lines(self):
        s = np.array([0, 0, 2, 0])
        t = np.array([0, 0, 2, 0.01])  # Slightly tilted
        assert is_parallel(s, t, tolerance_degrees=1.0) is True


class TestMakeSegmentFromMidpoint:
    def test_horizontal_segment(self):
        midpoint = np.array([1, 0])
        seg = make_segment_from_midpoint(midpoint, 0.0, 2.0)
        # Midpoint should be at (1, 0) with length 2, horizontal
        assert np.isclose((seg[0] + seg[2]) / 2, 1.0)
        assert np.isclose((seg[1] + seg[3]) / 2, 0.0)

    def test_vertical_segment(self):
        midpoint = np.array([0, 1])
        seg = make_segment_from_midpoint(midpoint, 90.0, 2.0)
        # Midpoint should be at (0, 1), vertical
        assert np.isclose((seg[0] + seg[2]) / 2, 0.0)
        assert np.isclose((seg[1] + seg[3]) / 2, 1.0)


class TestLineIntersection:
    def test_perpendicular_lines(self):
        s = np.array([0, 0, 2, 0])  # horizontal through origin
        t = np.array([1, -1, 1, 1])  # vertical through x=1
        result = line_intersection(s, t)
        assert result is not None
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 0.0)

    def test_parallel_lines(self):
        s = np.array([0, 0, 2, 0])
        t = np.array([0, 1, 2, 1])
        result = line_intersection(s, t)
        assert result is None

    def test_diagonal_lines(self):
        s = np.array([0, 0, 2, 2])
        t = np.array([0, 2, 2, 0])
        result = line_intersection(s, t)
        assert result is not None
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 1.0)


class TestProjectPointOntoLine:
    def test_project_onto_horizontal(self):
        point = (1, 1)
        line = np.array([0, 0, 2, 0])
        result = project_point_onto_line(point, line)
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 0.0)

    def test_project_onto_vertical(self):
        point = (1, 1)
        line = np.array([0, 0, 0, 2])
        result = project_point_onto_line(point, line)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 1.0)

    def test_degenerate_line(self):
        point = (1, 1)
        line = np.array([0, 0, 0, 0])  # Zero length
        result = project_point_onto_line(point, line)
        assert result == (0, 0)


class TestLongestSegment:
    def test_basic(self):
        segments = [
            np.array([0, 0, 1, 0]),  # length 1
            np.array([0, 0, 3, 4]),  # length 5
            np.array([0, 0, 2, 0]),  # length 2
        ]
        result = longest_segment(segments)
        assert np.allclose(result, [0, 0, 3, 4])


class TestRegularizeContour:
    def test_simple_rectangle(self):
        # A slightly noisy rectangle
        points = [
            (0, 0), (10, 0.5), (10.5, 10), (0.2, 9.8)
        ]
        result = regularize_contour(points, "axis", max_offset=20, visualize=False)
        assert len(result) >= 3  # Should have at least 3 corners

    def test_with_longest_principle(self):
        points = [
            (0, 0), (10, 0), (10, 10), (0, 10)
        ]
        result = regularize_contour(points, "longest", max_offset=20, visualize=False)
        assert len(result) >= 3

    def test_with_cardinal_principle(self):
        points = [
            (0, 0), (10, 0), (10, 10), (0, 10)
        ]
        result = regularize_contour(points, "cardinal", max_offset=20, visualize=False)
        assert len(result) >= 3

    def test_unknown_principle_raises(self):
        points = [(0, 0), (1, 0), (1, 1)]
        with pytest.raises(ValueError):
            regularize_contour(points, "unknown", visualize=False)


class TestLoadPolylines:
    def test_load_default(self):
        # Should load bundled data file
        points = load_polylines()
        assert len(points) > 0
        for p in points:
            assert len(p) == 2
