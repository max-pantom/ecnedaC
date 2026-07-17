import pytest


@pytest.fixture(autouse=True)
def allow_temporary_pilot_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy pilot unit tests independent of the host's available disk."""
    monkeypatch.setattr("cadence.ingestion.dataset_pilot.MIN_FREE_BYTES", 0)
