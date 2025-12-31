import numpy as np
import pytest

from shreg import geometry


class TestLength:
    def test_horizontal_segment(self):
        segment = np.array([0, 0, 3, 0])
        assert geometry.length(segment) == 3.0

    def test_vertical_segment(self):
        segment = np.array([0, 0, 0, 4])
        assert geometry.length(segment) == 4.0

    def test_diagonal_segment(self):
        segment = np.array([0, 0, 3, 4])
        assert geometry.length(segment) == 5.0

    def test_zero_length(self):
        segment = np.array([1, 1, 1, 1])
        assert geometry.length(segment) == 0.0


class TestNormal:
    def test_horizontal_segment_normal(self):
        segment = np.array([0, 0, 1, 0])
        n = geometry.normal(segment)
        assert np.allclose(np.abs(n), [0, 1])
        assert np.allclose(np.linalg.norm(n), 1.0)

    def test_vertical_segment_normal(self):
        segment = np.array([0, 0, 0, 1])
        n = geometry.normal(segment)
        assert np.allclose(np.abs(n), [1, 0])
        assert np.allclose(np.linalg.norm(n), 1.0)

    def test_diagonal_segment_normal(self):
        segment = np.array([0, 0, 1, 1])
        n = geometry.normal(segment)
        direction = geometry.direction(segment)
        assert np.allclose(np.dot(n, direction), 0)
        assert np.allclose(np.linalg.norm(n), 1.0)

    def test_zero_length_raises(self):
        segment = np.array([1, 1, 1, 1])
        with pytest.raises(ValueError):
            geometry.normal(segment)


class TestOrientation:
    def test_horizontal_orientation(self):
        segment = np.array([0, 0, 1, 0])
        assert geometry.orientation(segment) == 0.0

    def test_vertical_orientation(self):
        segment = np.array([0, 0, 0, 1])
        assert geometry.orientation(segment) == 90.0

    def test_45_degree_orientation(self):
        segment = np.array([0, 0, 1, 1])
        assert np.isclose(geometry.orientation(segment), 45.0)

    def test_orientation_range(self):
        for angle in range(0, 360, 15):
            rad = np.radians(angle)
            segment = np.array([0, 0, np.cos(rad), np.sin(rad)])
            o = geometry.orientation(segment)
            assert 0 <= o < 180


class TestRotate:
    def test_rotate_preserves_midpoint(self):
        segment = np.array([0, 0, 2, 0])
        original_mid = geometry.midpoint(segment)
        rotated = geometry.rotate(segment, 45)
        new_mid = geometry.midpoint(rotated)
        assert np.allclose(original_mid, new_mid)

    def test_rotate_preserves_length(self):
        segment = np.array([0, 0, 3, 4])
        original_length = geometry.length(segment)
        rotated = geometry.rotate(segment, 30)
        new_length = geometry.length(rotated)
        assert np.isclose(original_length, new_length)

    def test_rotate_90_degrees(self):
        segment = np.array([0, 0, 2, 0])
        rotated = geometry.rotate(segment, 90)
        assert np.isclose(geometry.orientation(rotated), 90.0)


class TestTranslateByNormal:
    def test_translate_preserves_orientation(self):
        segment = np.array([0, 0, 1, 1])
        original_orientation = geometry.orientation(segment)
        translated = geometry.translate_by_normal(segment, 1.0)
        new_orientation = geometry.orientation(translated)
        assert np.isclose(original_orientation, new_orientation)

    def test_translate_preserves_length(self):
        segment = np.array([0, 0, 3, 4])
        original_length = geometry.length(segment)
        translated = geometry.translate_by_normal(segment, 2.0)
        new_length = geometry.length(translated)
        assert np.isclose(original_length, new_length)

    def test_translate_horizontal_segment(self):
        segment = np.array([0, 0, 2, 0])
        translated = geometry.translate_by_normal(segment, 1.0)
        assert np.isclose(translated[1], 1.0) or np.isclose(translated[1], -1.0)
        assert np.isclose(translated[3], 1.0) or np.isclose(translated[3], -1.0)


class TestSignedDistanceToPoint:
    def test_point_on_line(self):
        segment = np.array([0, 0, 2, 0])
        point = np.array([1, 0])
        assert geometry.signed_distance_to_point(segment, point) == 0.0

    def test_point_above_horizontal_line(self):
        segment = np.array([0, 0, 2, 0])
        point = np.array([1, 1])
        dist = geometry.signed_distance_to_point(segment, point)
        assert np.isclose(np.abs(dist), 1.0)

    def test_point_below_horizontal_line(self):
        segment = np.array([0, 0, 2, 0])
        point = np.array([1, -1])
        dist = geometry.signed_distance_to_point(segment, point)
        assert np.isclose(np.abs(dist), 1.0)

    def test_opposite_sides_have_opposite_signs(self):
        segment = np.array([0, 0, 2, 0])
        point_above = np.array([1, 1])
        point_below = np.array([1, -1])
        dist_above = geometry.signed_distance_to_point(segment, point_above)
        dist_below = geometry.signed_distance_to_point(segment, point_below)
        assert np.sign(dist_above) != np.sign(dist_below)


class TestStandardizeSegment:
    def test_already_standardized(self):
        segment = np.array([0, 0, 1, 1])
        result = geometry.standardize_segment(segment)
        assert np.allclose(result, [0, 0, 1, 1])

    def test_needs_flip(self):
        segment = np.array([2, 3, 1, 4])
        result = geometry.standardize_segment(segment)
        assert result[0] <= result[2]
        assert np.allclose(result, [1, 4, 2, 3])

    def test_vertical_segment_unchanged(self):
        segment = np.array([1, 0, 1, 2])
        result = geometry.standardize_segment(segment)
        assert np.allclose(result, segment)


class TestFlip:
    def test_flip_swaps_endpoints(self):
        segment = np.array([0, 1, 2, 3])
        flipped = geometry.flip(segment)
        assert np.allclose(flipped, [2, 3, 0, 1])

    def test_double_flip_returns_original(self):
        segment = np.array([0, 1, 2, 3])
        double_flipped = geometry.flip(geometry.flip(segment))
        assert np.allclose(double_flipped, segment)


class TestValidateSegment:
    def test_valid_numpy_array(self):
        segment = np.array([0, 0, 1, 1])
        geometry.validate_segment(segment)

    def test_valid_list(self):
        segment = [0, 0, 1, 1]
        geometry.validate_segment(segment)

    def test_valid_tuple(self):
        segment = (0, 0, 1, 1)
        geometry.validate_segment(segment)

    def test_invalid_shape(self):
        segment = np.array([0, 0, 1])
        with pytest.raises(ValueError):
            geometry.validate_segment(segment)

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            geometry.validate_segment(42)


class TestMidpoint:
    def test_horizontal_segment(self):
        segment = np.array([0, 0, 2, 0])
        mid = geometry.midpoint(segment)
        assert np.allclose(mid, [1, 0])

    def test_diagonal_segment(self):
        segment = np.array([0, 0, 2, 2])
        mid = geometry.midpoint(segment)
        assert np.allclose(mid, [1, 1])


class TestDirection:
    def test_horizontal_direction(self):
        segment = np.array([0, 0, 2, 0])
        d = geometry.direction(segment)
        assert np.allclose(d, [2, 0])

    def test_vertical_direction(self):
        segment = np.array([0, 0, 0, 2])
        d = geometry.direction(segment)
        assert np.allclose(d, [0, 2])

    def test_direction_normalized_y_positive(self):
        segment = np.array([0, 0, -1, -1])
        d = geometry.direction(segment)
        assert d[1] >= 0 or (d[1] == 0 and d[0] >= 0)
