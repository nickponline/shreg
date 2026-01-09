"""Tests for Plotting utilities."""
import tempfile
import os
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from shreg import plotting, geometry


# Disable interactive display for all tests
@pytest.fixture(autouse=True)
def disable_plotting():
    """Disable interactive plots and ensure clean state."""
    plotting.disable()
    yield
    plotting.enable()


class TestStyle:
    def test_style_attributes_exist(self):
        assert hasattr(plotting.Style, 'SEGMENT_COLOR')
        assert hasattr(plotting.Style, 'SEGMENT_WIDTH')
        assert hasattr(plotting.Style, 'FIGURE_SIZE')
        assert hasattr(plotting.Style, 'COMPARISON_SIZE')


class TestGetSegmentsBounds:
    def test_empty_segments(self):
        xmin, xmax, ymin, ymax = plotting._get_segments_bounds([])
        assert xmin == 0
        assert xmax == 1
        assert ymin == 0
        assert ymax == 1

    def test_single_segment(self):
        segs = [np.array([0, 0, 2, 2])]
        # Even with margin=0, if the calculation results in dx=0, it defaults to 1
        # So the actual bounds include a minimum margin of 1
        xmin, xmax, ymin, ymax = plotting._get_segments_bounds(segs, margin=0.1)
        # Should have margin applied
        assert xmin < 0
        assert xmax > 2
        assert ymin < 0
        assert ymax > 2

    def test_with_margin(self):
        segs = [np.array([0, 0, 10, 10])]
        xmin, xmax, ymin, ymax = plotting._get_segments_bounds(segs, margin=0.1)
        assert xmin < 0
        assert xmax > 10


class TestMakeBoundsSquare:
    def test_wider_than_tall(self):
        xmin, xmax, ymin, ymax = plotting._make_bounds_square(0, 10, 0, 5)
        # Y range should expand to match X
        assert ymax - ymin == xmax - xmin
        # Should be centered
        y_center = (ymin + ymax) / 2
        assert np.isclose(y_center, 2.5)

    def test_taller_than_wide(self):
        xmin, xmax, ymin, ymax = plotting._make_bounds_square(0, 5, 0, 10)
        # X range should expand to match Y
        assert xmax - xmin == ymax - ymin
        # Should be centered
        x_center = (xmin + xmax) / 2
        assert np.isclose(x_center, 2.5)


class TestSegments:
    def test_basic_plot(self):
        segs = [np.array([0, 0, 1, 1]), np.array([1, 0, 0, 1])]
        fig = plotting.segments(segs, title="Test", show=False)
        # When disabled and no save_path, returns None
        assert fig is None

    def test_with_save_path(self):
        segs = [np.array([0, 0, 1, 1])]
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            fig = plotting.segments(segs, title="Test", show=False, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


class TestComparison:
    def test_basic_comparison(self):
        before = [np.array([0, 0, 1, 0])]
        after = [np.array([0, 0, 1.1, 0])]
        fig = plotting.comparison(before, after, show=False)
        assert fig is None  # Disabled

    def test_with_save_path(self):
        before = [np.array([0, 0, 1, 0])]
        after = [np.array([0, 0, 1.1, 0])]
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            fig = plotting.comparison(
                before, after,
                show=False,
                save_path=path,
                show_titles=True,
                show_ticks=True,
                show_grid=True
            )
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_without_titles(self):
        before = [np.array([0, 0, 1, 0])]
        after = [np.array([0, 0, 1.1, 0])]
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            fig = plotting.comparison(
                before, after,
                show=False,
                save_path=path,
                show_titles=False,
                show_ticks=False,
                show_grid=False
            )
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestContourComparison:
    def test_basic_contour_comparison(self):
        before = [(0, 0), (1, 0), (1, 1), (0, 1)]
        after = [(0, 0), (1, 0), (1, 1), (0, 1)]
        fig = plotting.contour_comparison(before, after, show=False)
        assert fig is None  # Disabled

    def test_with_save_path(self):
        before = [(0, 0), (1, 0), (1, 1), (0, 1)]
        after = [(0, 0), (1, 0), (1, 1), (0, 1)]
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            fig = plotting.contour_comparison(
                before, after,
                show=False,
                save_path=path,
                show_point_counts=True
            )
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestMetricComparison:
    def test_basic_metric_comparison(self):
        before = [np.array([0, 0, 2, 0])]
        after = [np.array([0, 0, 2.1, 0])]
        fig = plotting.metric_comparison(before, after, show=False)
        assert fig is None  # Disabled

    def test_with_save_path(self):
        before = [np.array([0, 0, 2, 0]), np.array([0, 1, 2.15, 1])]
        after = [np.array([0, 0, 2.05, 0]), np.array([0, 1, 2.05, 1])]
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            fig = plotting.metric_comparison(
                before, after,
                show=False,
                save_path=path,
                show_lengths=True
            )
            assert os.path.exists(path)
        finally:
            os.unlink(path)


class TestConstraints:
    def test_basic_constraints_plot(self):
        def error_func(s1, s2):
            return geometry.angle_difference(geometry.orientation(s1), geometry.orientation(s2))

        segs = [np.array([0, 0, 1, 0]), np.array([0, 1, 1, 1])]
        edges = [(0, 1)]
        fig = plotting.constraints(segs, edges, error_func, show=False)
        assert fig is None  # Disabled


class TestDelaunayDebug:
    def test_basic_delaunay_debug(self):
        def error_func(s1, s2):
            return 0.0

        segs = [np.array([0, 0, 1, 0]), np.array([1, 0, 2, 0]), np.array([0.5, 1, 1.5, 1])]
        tripoints = np.array([[0, 1, 2]])
        fig = plotting.delaunay_debug(segs, tripoints, error_func, show=False)
        assert fig is None  # Disabled


class TestPointsAndSegments:
    def test_basic(self):
        segs = [np.array([0, 0, 1, 1])]
        points = [(0, 0), (1, 1)]
        fig = plotting.points_and_segments(segs, points, show=False)
        assert fig is None  # Disabled


class TestEnableDisable:
    def test_enable_disable(self):
        plotting.disable()
        assert plotting.ENABLED is False
        plotting.enable()
        assert plotting.ENABLED is True
        plotting.disable()


class TestBeforeAfter:
    def test_context_manager(self):
        segs = [np.array([0, 0, 1, 0])]
        with plotting.BeforeAfter("Test") as plot:
            plot.before(segs)
            segs_after = [s + 0.1 for s in segs]
            plot.after(segs_after)
        # Should complete without error

    def test_before_after_function(self):
        before = [np.array([0, 0, 1, 0])]
        after = [np.array([0, 0, 1.1, 0])]
        plotting.before_after(before, after, title="Test")
        # Should complete without error when disabled
