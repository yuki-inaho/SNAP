"""Pytest fixtures shared by the SNAP loader / GUI tests."""

import os
import sys
from pathlib import Path

# Qt must be told to stay off screen before it is imported.  Under
# ``xvfb-run`` a DISPLAY exists and Qt can use its normal xcb backend with a
# software OpenGL context, which is required for the real Qt widget tests.
if "QT_QPA_PLATFORM" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXAMPLES = ROOT / "data_examples"


def pytest_configure(config):
    config.addinivalue_line("markers", "gui: needs a working off-screen Qt/VTK stack")


import pytest  # noqa: E402  (must come after the environment tweaks)


@pytest.fixture
def examples() -> Path:
    return EXAMPLES
