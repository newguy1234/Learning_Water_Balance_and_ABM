import matplotlib

# Tests never open windows; this Python has no Tcl/Tk, so pyplot must not pick TkAgg.
matplotlib.use("Agg")

import pytest


@pytest.fixture
def abm_builtins_only():
    """Hide farmers and stations registered outside the abm package (e.g. by example.py,
    which test_register imports at collection) so tests of the built-in defaults don't
    depend on what else pytest imported."""
    from unittest import mock
    from abm.agents import AGENTS
    from abm.station import STATIONS, _is_builtin

    farmers = {k: v for k, v in AGENTS.items() if (getattr(v, "__module__", "") or "").startswith("abm")}
    stations = {k: v for k, v in STATIONS.items() if _is_builtin(v)}
    with mock.patch.dict(AGENTS, farmers, clear=True), mock.patch.dict(STATIONS, stations, clear=True):
        yield
