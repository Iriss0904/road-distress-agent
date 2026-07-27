from pathlib import Path

from road_distress_agent.api.paths import data_dir, web_db_path


def test_default_runtime_data_is_under_data_directory(monkeypatch) -> None:
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("ROAD_DISTRESS_WEB_DB", raising=False)

    assert data_dir().name == "data"
    assert web_db_path() == data_dir() / "runtime/web-workflow.sqlite3"


def test_configured_web_database_path_is_relative_to_data_dir(monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", "workspace-data")
    monkeypatch.setenv("ROAD_DISTRESS_WEB_DB", "state/workbench.sqlite3")

    assert web_db_path() == data_dir() / Path("state/workbench.sqlite3")
