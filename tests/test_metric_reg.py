"""Tests for Metric & Pattern Regularization.

Tests the following features:
- Equal Length: Forces distinct segments to have the same length
- Length Quantization: Forces lengths to be multiples of a base unit
- Equal Spacing: Forces equal gaps between parallel lines
"""
import numpy as np
import pytest

from shreg import (
    metric_regularize_segments,
    find_equal_length_pairs,
    find_length_quantization_targets,
    find_parallel_line_groups,
    geometry,
)
from shreg.segments_reg import (
    _get_unit_vector,
    _get_segment_length,
    _sort_parallel_lines_by_position,
)


class TestGetUnitVector:
    def test_horizontal_segment(self):
        segment = np.array([0, 0, 3, 0])
        dx, dy = _get_unit_vector(segment)
        assert np.isclose(dx, 1.0)
        assert np.isclose(dy, 0.0)

    def test_vertical_segment(self):
        segment = np.array([0, 0, 0, 4])
        dx, dy = _get_unit_vector(segment)
        assert np.isclose(dx, 0.0)
        assert np.isclose(dy, 1.0)

    def test_diagonal_segment(self):
        segment = np.array([0, 0, 3, 4])
        dx, dy = _get_unit_vector(segment)
        assert np.isclose(dx, 0.6)
        assert np.isclose(dy, 0.8)

    def test_degenerate_segment(self):
        segment = np.array([1, 1, 1, 1])
        dx, dy = _get_unit_vector(segment)
        assert dx == 0.0
        assert dy == 0.0


class TestGetSegmentLength:
    def test_horizontal_segment(self):
        segment = np.array([0, 0, 5, 0])
        assert _get_segment_length(segment) == 5.0

    def test_pythagorean_triple(self):
        segment = np.array([0, 0, 3, 4])
        assert _get_segment_length(segment) == 5.0


class TestFindEqualLengthPairs:
    def test_no_segments(self):
        pairs = find_equal_length_pairs([])
        assert pairs == []

    def test_one_segment(self):
        segments = [np.array([0, 0, 2, 0])]
        pairs = find_equal_length_pairs(segments)
        assert pairs == []

    def test_two_equal_segments(self):
        segments = [
            np.array([0, 0, 2, 0]),
            np.array([0, 1, 2, 1]),
        ]
        pairs = find_equal_length_pairs(segments, tolerance=0.1)
        assert len(pairs) == 1
        assert (0, 1) in pairs

    def test_two_different_segments(self):
        segments = [
            np.array([0, 0, 2, 0]),  # length 2
            np.array([0, 1, 5, 1]),  # length 5
        ]
        pairs = find_equal_length_pairs(segments, tolerance=0.1)
        assert len(pairs) == 0

    def test_similar_lengths_within_tolerance(self):
        segments = [
            np.array([0, 0, 2.0, 0]),  # length 2.0
            np.array([0, 1, 2.1, 1]),  # length 2.1, diff = 5%
        ]
        pairs = find_equal_length_pairs(segments, tolerance=0.1)
        assert len(pairs) == 1

    def test_min_length_filter(self):
        segments = [
            np.array([0, 0, 0.3, 0]),  # length 0.3, below min
            np.array([0, 1, 0.3, 1]),  # length 0.3, below min
        ]
        pairs = find_equal_length_pairs(segments, min_length=0.5)
        assert len(pairs) == 0


class TestFindLengthQuantizationTargets:
    def test_exact_multiple(self):
        segments = [np.array([0, 0, 2, 0])]  # length 2.0
        targets = find_length_quantization_targets(segments, base_unit=1.0)
        assert len(targets) == 1
        assert targets[0] == (0, 2.0)

    def test_near_multiple(self):
        segments = [np.array([0, 0, 2.1, 0])]  # length 2.1, near 2.0
        targets = find_length_quantization_targets(segments, base_unit=1.0, tolerance=0.3)
        assert len(targets) == 1
        assert targets[0][0] == 0
        assert targets[0][1] == 2.0

    def test_far_from_multiple(self):
        segments = [np.array([0, 0, 2.5, 0])]  # length 2.5, far from both 2.0 and 3.0
        targets = find_length_quantization_targets(segments, base_unit=1.0, tolerance=0.2)
        assert len(targets) == 0

    def test_custom_base_unit(self):
        segments = [np.array([0, 0, 1.5, 0])]  # length 1.5
        targets = find_length_quantization_targets(segments, base_unit=0.5, tolerance=0.3)
        assert len(targets) == 1
        assert targets[0][1] == 1.5  # 3 * 0.5


class TestFindParallelLineGroups:
    def test_no_parallel_lines(self):
        segments = [
            np.array([0, 0, 1, 0]),  # horizontal
            np.array([0, 0, 0, 1]),  # vertical
        ]
        groups = find_parallel_line_groups(segments, min_group_size=2)
        assert len(groups) == 0

    def test_three_horizontal_lines(self):
        segments = [
            np.array([0, 0, 2, 0]),
            np.array([0, 1, 2, 1]),
            np.array([0, 2, 2, 2]),
        ]
        groups = find_parallel_line_groups(segments, angle_tolerance=5.0, min_group_size=3)
        assert len(groups) == 1
        assert set(groups[0]) == {0, 1, 2}

    def test_two_groups_of_parallel_lines(self):
        segments = [
            # Horizontal group
            np.array([0, 0, 2, 0]),
            np.array([0, 1, 2, 1]),
            np.array([0, 2, 2, 2]),
            # Vertical group
            np.array([0, 0, 0, 2]),
            np.array([1, 0, 1, 2]),
            np.array([2, 0, 2, 2]),
        ]
        groups = find_parallel_line_groups(segments, angle_tolerance=5.0, min_group_size=3)
        assert len(groups) == 2


class TestSortParallelLinesByPosition:
    def test_horizontal_lines_sorted_by_y(self):
        segments = [
            np.array([0, 2, 1, 2]),  # y=2
            np.array([0, 0, 1, 0]),  # y=0
            np.array([0, 1, 1, 1]),  # y=1
        ]
        indices = [0, 1, 2]
        sorted_indices = _sort_parallel_lines_by_position(segments, indices)
        # Should be sorted by position along normal
        # Horizontal lines: orientation=0, normal at 90°, so sorted by y
        assert sorted_indices == [1, 2, 0]  # y=0, y=1, y=2

    def test_vertical_lines_sorted_by_position(self):
        """Vertical lines should be sorted by position along their normal.

        For vertical lines (orientation=90°), the normal is at 180° = (-1, 0),
        so position = -x. Lines are sorted by increasing position.
        """
        segments = [
            np.array([2, 0, 2, 1]),  # x=2, position=-2
            np.array([0, 0, 0, 1]),  # x=0, position=0
            np.array([1, 0, 1, 1]),  # x=1, position=-1
        ]
        indices = [0, 1, 2]
        sorted_indices = _sort_parallel_lines_by_position(segments, indices)
        # Sorted by position: -2, -1, 0 → indices 0, 2, 1
        assert sorted_indices == [0, 2, 1]


class TestMetricRegularizeSegmentsEqualLength:
    def test_two_similar_length_segments(self):
        """Two segments with slightly different lengths should become equal."""
        segments = [
            np.array([0, 0, 2.0, 0]),  # length 2.0
            np.array([0, 1, 2.1, 1]),  # length 2.1
        ]
        result = metric_regularize_segments(
            segments,
            equal_length=True,
            length_quantization=False,
            equal_spacing=False,
            length_tolerance=0.1,
        )
        # After regularization, lengths should be equal
        len1 = _get_segment_length(result[0])
        len2 = _get_segment_length(result[1])
        assert np.isclose(len1, len2, atol=1e-6)

    def test_preserves_approximately_original_lengths(self):
        """The average length should be preserved approximately."""
        segments = [
            np.array([0, 0, 2.0, 0]),
            np.array([0, 1, 2.1, 1]),
        ]
        avg_original = (2.0 + 2.1) / 2

        result = metric_regularize_segments(
            segments,
            equal_length=True,
            length_quantization=False,
            equal_spacing=False,
        )

        len1 = _get_segment_length(result[0])
        len2 = _get_segment_length(result[1])
        avg_result = (len1 + len2) / 2
        assert np.isclose(avg_result, avg_original, atol=0.1)


class TestMetricRegularizeSegmentsQuantization:
    def test_snap_to_integer(self):
        """Segment length should snap to nearest integer."""
        segments = [
            np.array([0, 0, 2.1, 0]),  # length 2.1, should snap to 2.0
        ]
        result = metric_regularize_segments(
            segments,
            equal_length=False,
            length_quantization=True,
            equal_spacing=False,
            base_unit=1.0,
            quantization_tolerance=0.3,
        )
        length = _get_segment_length(result[0])
        assert np.isclose(length, 2.0, atol=1e-5)

    def test_custom_base_unit(self):
        """Segment should snap to multiples of custom base unit."""
        segments = [
            np.array([0, 0, 1.45, 0]),  # length 1.45, should snap to 1.5
        ]
        result = metric_regularize_segments(
            segments,
            equal_length=False,
            length_quantization=True,
            equal_spacing=False,
            base_unit=0.5,
            quantization_tolerance=0.3,
        )
        length = _get_segment_length(result[0])
        assert np.isclose(length, 1.5, atol=1e-5)


class TestMetricRegularizeSegmentsEqualSpacing:
    def test_three_horizontal_lines(self):
        """Three horizontal lines should become equally spaced."""
        segments = [
            np.array([0, 0, 2, 0]),    # y=0
            np.array([0, 0.9, 2, 0.9]),  # y=0.9 (should be 1.0)
            np.array([0, 2, 2, 2]),    # y=2
        ]
        result = metric_regularize_segments(
            segments,
            equal_length=False,
            length_quantization=False,
            equal_spacing=True,
            angle_tolerance=5.0,
        )

        # After regularization, spacing should be equal
        mid1 = geometry.midpoint(result[0])
        mid2 = geometry.midpoint(result[1])
        mid3 = geometry.midpoint(result[2])

        spacing1 = mid2[1] - mid1[1]
        spacing2 = mid3[1] - mid2[1]
        assert np.isclose(spacing1, spacing2, atol=1e-5)

    def test_four_vertical_lines(self):
        """Four vertical lines with uneven spacing should become equally spaced."""
        segments = [
            np.array([0, 0, 0, 2]),    # x=0
            np.array([0.9, 0, 0.9, 2]),  # x=0.9 (should be 1.0)
            np.array([2.1, 0, 2.1, 2]),  # x=2.1 (should be 2.0)
            np.array([3, 0, 3, 2]),    # x=3
        ]
        result = metric_regularize_segments(
            segments,
            equal_length=False,
            length_quantization=False,
            equal_spacing=True,
            angle_tolerance=5.0,
        )

        # Get x positions of midpoints
        xs = sorted([geometry.midpoint(s)[0] for s in result])

        # Check equal spacing
        spacing1 = xs[1] - xs[0]
        spacing2 = xs[2] - xs[1]
        spacing3 = xs[3] - xs[2]
        assert np.isclose(spacing1, spacing2, atol=1e-5)
        assert np.isclose(spacing2, spacing3, atol=1e-5)


class TestMetricRegularizeSegmentsCombined:
    def test_equal_length_and_quantization(self):
        """Segments should be both equal and quantized."""
        segments = [
            np.array([0, 0, 1.9, 0]),  # length 1.9
            np.array([0, 1, 2.1, 1]),  # length 2.1
        ]
        result = metric_regularize_segments(
            segments,
            equal_length=True,
            length_quantization=True,
            equal_spacing=False,
            base_unit=1.0,
            length_tolerance=0.2,
            quantization_tolerance=0.3,
        )

        len1 = _get_segment_length(result[0])
        len2 = _get_segment_length(result[1])

        # Lengths should be equal
        assert np.isclose(len1, len2, atol=1e-5)

        # And close to 2.0 (quantized)
        assert np.isclose(len1, 2.0, atol=0.1)

    def test_no_change_when_no_constraints(self):
        """Segments should remain unchanged if no constraints apply."""
        segments = [
            np.array([0, 0, 1, 0]),  # length 1
            np.array([0, 1, 3, 1]),  # length 3 (very different)
        ]
        original = [s.copy() for s in segments]
        result = metric_regularize_segments(
            segments,
            equal_length=True,  # Won't apply, lengths too different
            length_quantization=False,
            equal_spacing=False,
            length_tolerance=0.05,  # Very strict tolerance
        )

        # Segments should be unchanged or nearly unchanged
        for orig, res in zip(original, result):
            assert np.allclose(orig, res, atol=1e-5)


class TestMetricRegularizeSegmentsEdgeCases:
    def test_empty_list(self):
        result = metric_regularize_segments([])
        assert result == []

    def test_single_segment_no_constraints(self):
        """Single segment with no applicable constraints returns unchanged."""
        segments = [np.array([0, 0, 1, 0])]
        result = metric_regularize_segments(
            segments,
            equal_length=True,  # Can't apply with one segment
            length_quantization=False,
            equal_spacing=False,
        )
        assert len(result) == 1
        assert np.allclose(result[0], segments[0])

    def test_single_segment_with_quantization(self):
        """Single segment can be quantized."""
        segments = [np.array([0, 0, 2.1, 0])]  # length 2.1
        result = metric_regularize_segments(
            segments,
            equal_length=False,
            length_quantization=True,
            equal_spacing=False,
            base_unit=1.0,
        )
        assert len(result) == 1
        length = _get_segment_length(result[0])
        assert np.isclose(length, 2.0, atol=1e-5)

    def test_iteration_convergence(self):
        """Verify that iterative refinement converges."""
        segments = [
            np.array([0, 0, 2.0, 0]),
            np.array([0, 1, 2.2, 1]),
        ]
        result = metric_regularize_segments(
            segments,
            equal_length=True,
            length_quantization=False,
            equal_spacing=False,
            max_iterations=5,
            debug=False,
        )

        len1 = _get_segment_length(result[0])
        len2 = _get_segment_length(result[1])
        assert np.isclose(len1, len2, atol=1e-6)
