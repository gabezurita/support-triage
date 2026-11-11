"""Submission docs and the documentation index."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_readme_has_quickstart_diagram_and_traces() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Quickstart" in readme
    assert "uv run langgraph dev" in readme
    assert "SUPPORT_TRIAGE_PROVIDER" in readme
    assert "GEMINI_API_KEY" in readme
    assert "ANTHROPIC_API_KEY" in readme
    assert "OPENAI_API_KEY" in readme
    assert "data/evaluation/live_judge.json" in readme
    assert "billing-double-charge.json" in readme
    assert "LANGSMITH_API_KEY" in readme
    assert "LANGSMITH_PROJECT" in readme
    assert "```mermaid" in readme
    assert "citation_gate" in readme
    assert "docs/demo.md" in readme
    assert "land with the submission docs" not in readme
    assert (ROOT / "docs" / "demo.md").is_file()


def test_docs_index_lists_adr_plans_and_evaluations() -> None:
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "adr/" in index
    assert "plans/" in index
    assert "evaluations/" in index
    assert "EVAL-001-citation-gate.md" in index
    assert "demo.md" in index
    assert (ROOT / "docs" / "adr" / "ADR-001-langgraph-react-agent.md").is_file()
    assert (ROOT / "docs" / "evaluations" / "EVAL-001-citation-gate.md").is_file()
