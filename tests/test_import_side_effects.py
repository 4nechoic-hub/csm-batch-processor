import importlib
import sys


def _drop_modules(*names):
    for name in names:
        sys.modules.pop(name, None)


def test_top_level_import_does_not_eagerly_import_plotting_modules():
    _drop_modules(
        "csm_processor",
        "csm_processor.plotting",
        "csm_processor.anomaly_plotting",
    )

    pkg = importlib.import_module("csm_processor")

    assert "csm_processor.plotting" not in sys.modules
    assert "csm_processor.anomaly_plotting" not in sys.modules
    assert callable(pkg.csm_calculator)


def test_plotting_export_is_loaded_lazily():
    _drop_modules("csm_processor", "csm_processor.plotting")

    pkg = importlib.import_module("csm_processor")
    assert "csm_processor.plotting" not in sys.modules

    _ = pkg.plot_autospectra
    assert "csm_processor.plotting" in sys.modules