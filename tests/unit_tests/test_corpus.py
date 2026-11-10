"""Corpus retrieval without a model."""

from support_triage.corpus import _KEYWORDS, _read_article, retrieve


def test_billing_ticket_returns_billing_article_not_shipping() -> None:
    ids = [passage.id for passage in retrieve("Why was I charged twice for order 1001?")]
    assert "billing-001" in ids
    assert "shipping-001" not in ids


def test_empty_ticket_returns_no_passages() -> None:
    assert retrieve("") == []
    assert retrieve("   ") == []


def test_refund_article_states_thirty_day_window() -> None:
    refund_request = "Refund order 1002 in full."
    matches = [passage for passage in retrieve(refund_request) if passage.id == "refunds-001"]
    assert matches
    text = matches[0].text.casefold()
    assert "30" in text
    assert "day" in text


def test_all_configured_articles_exist_in_package() -> None:
    for article_id in _KEYWORDS:
        text = _read_article(article_id)
        assert text.strip()
        assert article_id in text
