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


class TestPointSideOfLine:
    def test_point_below_line(self):
        # For a line going left to right, point below has negative cross product
        result = geometry.point_side_of_line(0, 0, 1, 0, 0.5, -1)
        assert result == -1

    def test_point_above_line(self):
        # For a line going left to right, point above has positive cross product
        result = geometry.point_side_of_line(0, 0, 1, 0, 0.5, 1)
        assert result == 1

    def test_point_on_line(self):
        result = geometry.point_side_of_line(0, 0, 1, 0, 0.5, 0)
        assert result == 0


class TestIsClockwise:
    def test_clockwise(self):
        a = (0, 0)
        b = (1, 0)
        c = (1, -1)
        assert geometry.is_clockwise(a, b, c) is True

    def test_counter_clockwise(self):
        a = (0, 0)
        b = (1, 0)
        c = (1, 1)
        assert geometry.is_clockwise(a, b, c) is False

    def test_collinear(self):
        a = (0, 0)
        b = (1, 0)
        c = (2, 0)
        assert geometry.is_clockwise(a, b, c) is None


class TestStandardize:
    def test_standardize_list_of_segments(self):
        segments = [[1, 2, 0, 1], [0, 0, 1, 1]]
        result = geometry.standardize(segments)
        assert result[0] == [0, 1, 1, 2]
        assert result[1] == [0, 0, 1, 1]


class TestConvertAngle2:
    def test_angle_over_90(self):
        result = geometry.convert_angle_2(120)
        assert result == 60

    def test_angle_under_minus_90(self):
        result = geometry.convert_angle_2(-120)
        assert result == 60

    def test_angle_in_range(self):
        result = geometry.convert_angle_2(45)
        assert result == 45


class TestIsOrthogonal:
    def test_orthogonal_lines(self):
        line1 = np.array([0, 0, 1, 0])
        line2 = np.array([0, 0, 0, 1])
        assert geometry.is_orthogonal(line1, line2) is True

    def test_parallel_lines(self):
        line1 = np.array([0, 0, 1, 0])
        line2 = np.array([0, 1, 1, 1])
        assert geometry.is_orthogonal(line1, line2) is False


class TestAngleBetweenVectors:
    def test_perpendicular_vectors(self):
        a = np.array([1, 0])
        b = np.array([0, 1])
        result = geometry.angle_between_vectors(a, b)
        assert np.isclose(abs(result), 90, atol=0.01)

    def test_parallel_vectors(self):
        a = np.array([1, 0])
        b = np.array([2, 0])
        result = geometry.angle_between_vectors(a, b)
        assert np.isclose(result, 0, atol=0.01)


class TestDistanceAlong:
    def test_midpoint(self):
        line = (0, 0, 2, 0)
        points = np.array([[1, 0]])
        result = geometry.distance_along(line, points)
        assert np.isclose(result[0], 0.5)

    def test_start_point(self):
        line = (0, 0, 2, 0)
        points = np.array([[0, 0]])
        result = geometry.distance_along(line, points)
        assert np.isclose(result[0], 0.0)

    def test_vertical_line(self):
        line = (0, 0, 0, 2)
        points = np.array([[0, 1]])
        result = geometry.distance_along(line, points)
        assert np.isclose(result[0], 0.5)

    def test_no_clip(self):
        line = (0, 0, 2, 0)
        points = np.array([[3, 0]])  # Beyond endpoint
        result = geometry.distance_along(line, points, clip=False)
        assert result[0] > 1.0


class TestProjectPointsToLine:
    def test_project_on_line(self):
        points = np.array([[0.5, 1]])
        line = (0, 0, 1, 0)
        result = geometry.project_points_to_line(points, line)
        assert np.allclose(result, [[0.5, 0]])


class TestDistanceSegmentToPointsReturnXY:
    def test_return_projection_in_middle(self):
        segment = (0, 0, 2, 0)
        xy = np.array([[1, 1]])
        px, py = geometry.distance_segment_to_points(segment, xy, return_xy=True)
        assert np.isclose(px, 1.0)
        assert np.isclose(py, 0.0)

    def test_return_projection_before_start(self):
        segment = (0, 0, 2, 0)
        xy = np.array([[-1, 0]])
        px, py = geometry.distance_segment_to_points(segment, xy, return_xy=True)
        assert np.isclose(px, 0.0)
        assert np.isclose(py, 0.0)

    def test_return_projection_after_end(self):
        segment = (0, 0, 2, 0)
        xy = np.array([[3, 0]])
        px, py = geometry.distance_segment_to_points(segment, xy, return_xy=True)
        assert np.isclose(px, 2.0)
        assert np.isclose(py, 0.0)


class TestAffine:
    def test_basic_transform(self):
        points = np.array([[2, 4], [6, 8]])
        translation = np.array([1, 1])
        scale = 2
        result = geometry.affine(points, translation, scale)
        expected = np.array([[0, 2], [2, 4]])
        assert np.allclose(result, expected)


class TestWhichSide:
    def test_normal_to_left(self):
        segment = (0, 0, 1, 0)
        normal = np.array([0, 1])
        result = geometry.which_side(segment, normal)
        assert result != 0


class TestLineSegmentIntersection:
    def test_intersecting_segments(self):
        line1 = (0, 0, 2, 2)
        line2 = (0, 2, 2, 0)
        result = geometry.line_segment_intersection(line1, line2)
        assert result is not None
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 1.0)

    def test_parallel_segments(self):
        line1 = (0, 0, 1, 0)
        line2 = (0, 1, 1, 1)
        result = geometry.line_segment_intersection(line1, line2)
        assert result is None

    def test_non_intersecting_segments(self):
        line1 = (0, 0, 1, 0)
        line2 = (2, 2, 3, 3)
        result = geometry.line_segment_intersection(line1, line2)
        assert result is None

    def test_return_ts(self):
        line1 = (0, 0, 2, 2)
        line2 = (0, 2, 2, 0)
        result = geometry.line_segment_intersection(line1, line2, return_ts=True)
        assert result is not None
        t, s = result
        assert np.isclose(t, 0.5)
        assert np.isclose(s, 0.5)

    def test_ignore_extent(self):
        line1 = (0, 0, 1, 0)
        line2 = (2, -1, 2, 1)
        result = geometry.line_segment_intersection(line1, line2, ignore_extent=True)
        assert result is not None


class TestIntersectionEars:
    def test_basic(self):
        line1 = (0, 0, 2, 0)
        line2 = (1, -1, 1, 1)
        xy = (1, 0)
        result = geometry.intersection_ears(line1, line2, xy)
        assert len(result) == 4


class TestFlatten:
    def test_flatten(self):
        nested = [[1, 2], [3, 4], [5]]
        result = geometry.flatten(nested)
        assert result == [1, 2, 3, 4, 5]


class TestGroupSegments:
    def test_group_parallel_segments(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([0, 0.1, 1, 0.1]),
            np.array([5, 5, 6, 5]),  # Same orientation, far away
        ]
        result = geometry.group_segments(segments, max_orientation=5.0, max_distance=1.0)
        assert len(result) == 3
        # First two should be same group
        assert result[0] == result[1]

    def test_group_collate(self):
        segments = [
            np.array([0, 0, 1, 0]),
            np.array([0, 0.1, 1, 0.1]),
        ]
        result = geometry.group_segments(segments, max_orientation=5.0, max_distance=1.0, collate=True)
        assert isinstance(result, dict)


class TestWeldSegmentsGroup:
    def test_overlapping_segments(self):
        segments = [[0, 0, 2, 0], [1, 0, 3, 0]]
        result = geometry.weld_segments_group(segments)
        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][2] == 3


class TestWeldSegments:
    def test_weld_close_segments(self):
        segments = [
            [0, 0, 1, 0],
            [0.5, 0, 1.5, 0],
        ]
        result = geometry.weld_segments(segments, max_orientation=5.0, max_distance=1.0)
        assert len(result) >= 1


class TestFindCrossing:
    def test_find_crossings(self):
        segments = [
            [0, 0, 2, 0],
            [1, -1, 1, 1],
        ]
        result = geometry.find_crossing(segments, 0)
        assert len(result) >= 2  # At least start and end


class TestClipAndClose:
    def test_basic(self):
        segments = [
            [0, 0, 2, 0],
            [0, 1, 2, 1],
        ]
        result = geometry.clip_and_close(segments, cut_length=0.1)
        assert len(result) >= 0
