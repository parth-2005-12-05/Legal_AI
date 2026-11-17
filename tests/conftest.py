"""Shared pytest fixtures.

Every test that needs the full application gets a CaseWay instance whose var/
folder (SQLite, keys, encrypted objects, audit log) lives in a temporary
directory, so tests never touch real data and can run in parallel.
"""

from __future__ import annotations

import pytest

from caseway.config import ROOT, load_settings
from caseway.extraction.statutes import StatuteTable
from caseway.gazetteer import load_gazetteer
from caseway.gazetteer.fuzzy import FuzzyMatcher
from caseway.pipeline import CaseWay

SAMPLES = ROOT / "data" / "samples"


def make_app(tmp_path, **overrides) -> CaseWay:
    settings = load_settings(overrides=overrides)
    settings.var_dir = tmp_path / "var"
    return CaseWay(settings)


@pytest.fixture
def app(tmp_path):
    return make_app(tmp_path)


@pytest.fixture
def partner_app(tmp_path):
    return make_app(tmp_path, matching={"partner_review": True})


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def gaz(settings):
    return load_gazetteer(settings.data_dir / "gazetteer" / "sample_gazetteer.json")


@pytest.fixture(scope="session")
def fuzzy(gaz):
    return FuzzyMatcher(gaz)


@pytest.fixture(scope="session")
def statutes(settings):
    return StatuteTable(settings.data_dir / "classifier" / "statutes.json")
