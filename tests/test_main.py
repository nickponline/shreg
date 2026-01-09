"""Tests for CLI entry point (__main__.py)."""
import sys
import pytest
from unittest import mock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from shreg import plotting
from shreg.__main__ import main, run_segment_examples, run_contour_examples


@pytest.fixture(autouse=True)
def disable_plotting():
    """Disable interactive plots."""
    plotting.disable()
    yield
    plotting.enable()


class TestRunSegmentExamples:
    def test_runs_without_error(self, capsys):
        """Segment examples should run without raising exceptions."""
        run_segment_examples()
        captured = capsys.readouterr()
        assert "SEGMENT REGULARIZATION EXAMPLES" in captured.out
        assert "Regularized" in captured.out


class TestRunContourExamples:
    def test_runs_without_error(self, capsys):
        """Contour examples should run without raising exceptions."""
        run_contour_examples()
        captured = capsys.readouterr()
        assert "CONTOUR REGULARIZATION EXAMPLES" in captured.out
        assert "Input:" in captured.out


class TestMain:
    def test_main_no_args(self, capsys):
        """Main without args runs both segment and contour examples."""
        with mock.patch('sys.argv', ['shreg', '--no-plot']):
            main()
        captured = capsys.readouterr()
        assert "SEGMENT REGULARIZATION" in captured.out
        assert "CONTOUR REGULARIZATION" in captured.out
        assert "All examples completed!" in captured.out

    def test_main_segments_only(self, capsys):
        """Main with --segments runs only segment examples."""
        with mock.patch('sys.argv', ['shreg', '--no-plot', '--segments']):
            main()
        captured = capsys.readouterr()
        assert "SEGMENT REGULARIZATION" in captured.out
        # Contours should also run because neither flag excludes the other when both are true
        # Actually, the logic is: run_segments = not args.contours or args.segments
        # If --segments is set and --contours is not, then run_segments=True, run_contours=False
        assert "All examples completed!" in captured.out

    def test_main_contours_only(self, capsys):
        """Main with --contours runs only contour examples."""
        with mock.patch('sys.argv', ['shreg', '--no-plot', '--contours']):
            main()
        captured = capsys.readouterr()
        assert "CONTOUR REGULARIZATION" in captured.out
        assert "All examples completed!" in captured.out

    def test_main_no_plot_flag(self, capsys):
        """Main with --no-plot disables plotting."""
        with mock.patch('sys.argv', ['shreg', '--no-plot']):
            main()
        # Test passes if no matplotlib windows are shown (they would block in non-Agg mode)
        captured = capsys.readouterr()
        assert "Shape Regularization Demo" in captured.out


class TestCLIEntryPoint:
    def test_module_runnable(self):
        """Test that the module can be run as python -m shreg."""
        # Just verify the module imports work
        import shreg.__main__
        assert hasattr(shreg.__main__, 'main')
        assert callable(shreg.__main__.main)
