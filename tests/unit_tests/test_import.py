"""Scaffold checks for the support triage package."""

import importlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_import_support_triage_not_react_agent() -> None:
    module = importlib.import_module("support_triage")
    assert module.graph is not None
    assert importlib.util.find_spec("react_agent") is None


def test_langgraph_json_points_at_support_triage_graph() -> None:
    config = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))
    assert config["graphs"]["agent"] == "./src/support_triage/graph.py:graph"


def test_langgraph_file_spec_loads_graph() -> None:
    """LangGraph dev loads the graph path as a file, with no package parent."""
    spec = importlib.util.spec_from_file_location(
        "src__support_triage__graph",
        ROOT / "src" / "support_triage" / "graph.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.graph is not None


def test_scaffold_does_not_include_tavily() -> None:
    tools = (ROOT / "src/support_triage/tools.py").read_text(encoding="utf-8").lower()
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "tavily" not in tools
    assert "tavily" not in project
    assert "TAVILY" not in env_example
