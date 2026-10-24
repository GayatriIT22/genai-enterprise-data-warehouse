"""Tests for the GenAI layer (no API key or internet needed - the models are mocked).

Run from the project root:   python -m tests.test_genai
"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import genai.text_to_sql as t2s
from genai.fallback_questions import FALLBACK
from genai.text_to_sql import (ClaudeProvider, GroqProvider, ask, clean_sql, get_providers,
                               guard_question, match_fallback, run_query, validate_sql)

GOOD_SQL = ("SELECT l.region, ROUND(SUM(f.sales),2) AS s FROM fact_sales f "
            "JOIN dim_location l ON f.location_key=l.location_key GROUP BY 1")


class FakeClaude:
    """Mimics anthropic.Anthropic().messages.create."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_):
        self.calls += 1
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.replies.pop(0))])


class DownProvider:
    name, model = "claude", "x"

    def complete(self, *a, **k):
        raise ConnectionError("no credits / network down")


def test_safety():
    for q in ["SELECT * FROM fact_sales LIMIT 3",
              "WITH t AS (SELECT region FROM dim_location) SELECT * FROM t"]:
        assert validate_sql(q)[0], q
    for q in ["DELETE FROM fact_sales", "DROP TABLE dim_date", "UPDATE fact_sales SET sales = 0",
              "SELECT 1; DROP TABLE dim_date", "INSERT INTO dim_date VALUES (1)",
              "SELECT * FROM fact_sales -- hidden", "PRAGMA table_info(fact_sales)", ""]:
        assert not validate_sql(q)[0], q
    print("safety validator ............ OK")


def test_read_only_connection():
    from db import connect
    conn = connect(read_only=True)
    try:
        conn.execute("DELETE FROM fact_sales")
        raise AssertionError("write succeeded on a read-only connection!")
    except Exception as exc:
        assert "readonly" in str(exc).lower(), exc
    finally:
        conn.close()
    print("read-only connection ........ OK")


def test_rule_check():
    assert guard_question("What is the total profit?")["summary"]
    assert guard_question("average discount per customer")["summary"]
    assert "Blocked" in guard_question("Delete all rows from the sales table")["error"]
    assert "Blocked" in guard_question("update the data so sales are zero")["error"]
    assert guard_question("Which region has the highest sales?") is None
    assert guard_question("How did sales change from 2016 to 2017?") is None
    r = ask("What is the total profit?", providers=[], use_cache=False)       # no keys at all
    assert r["source"] == "rule check" and "profit" in r["summary"].lower()
    print("rule check (profit/delete) .. OK  (works with NO api key)")


def test_clean_sql():
    messy = "Sure! Here is the query:\n```sql\nSELECT 1 AS x;\n```\nThis returns one."
    assert clean_sql(messy) == "SELECT 1 AS x;"
    assert clean_sql("SELECT region FROM dim_location\n\nThis query lists regions.") == \
        "SELECT region FROM dim_location"
    assert clean_sql("NOT_AVAILABLE") == "NOT_AVAILABLE"
    print("clean_sql (chatty output) ... OK")


def test_fallback_sql_runs():
    for item in FALLBACK:
        assert len(run_query(item["sql"])) > 0, item["question"]
    assert match_fallback("which region has highest sales")["question"].startswith("Which region")
    print(f"{len(FALLBACK)} backup queries ......... OK")


def test_model_flow():
    r = ask("sales by region", client=FakeClaude(["```sql\n" + GOOD_SQL + "\n```", "West leads."]),
            use_cache=False)
    assert r["source"] == "claude" and len(r["df"]) == 4 and r["summary"] == "West leads."

    r = ask("remove every sales row", client=FakeClaude(["DELETE FROM fact_sales"]), use_cache=False)
    assert r["df"] is None and "Blocked" in r["error"]            # validator catches model output

    r = ask("predict sales for 2030", client=FakeClaude(["NOT_AVAILABLE"]), use_cache=False)
    assert r["df"] is None and "can't be answered" in r["summary"]

    r = ask("sales by region", client=FakeClaude(["SELECT nope FROM fact_sales", GOOD_SQL, "Fixed."]),
            use_cache=False)
    assert r["df"] is not None and r["summary"] == "Fixed."       # error -> retry worked
    print("model flow (mocked) ......... OK  (generate, block, NOT_AVAILABLE, retry)")


def test_provider_switch_and_offline():
    good = ClaudeProvider(FakeClaude([GOOD_SQL, "Backup model answered."]))
    good.name = "groq"
    r = ask("sales by region", providers=[DownProvider(), good], use_cache=False)
    assert r["source"] == "groq" and "Switched provider" in r["note"]

    r = ask("Which region has the highest total sales?", providers=[DownProvider()], use_cache=False)
    assert r["source"] == "offline backup" and r["df"] is not None
    print("provider switch + offline ... OK")


def test_cache():
    old = t2s.CACHE_PATH
    t2s.CACHE_PATH = Path(tempfile.mkdtemp()) / "cache.json"
    try:
        fake = FakeClaude([GOOD_SQL, "Cached answer."])
        first = ask("Sales by region please", client=fake)
        second = ask("sales by region, please!", client=fake)       # same question, new wording
        assert fake.calls == 2 and second["cached"] and second["summary"] == "Cached answer."
        assert len(second["df"]) == 4
    finally:
        t2s.CACHE_PATH = old
    print("answer cache ................ OK  (repeat question = 0 API calls)")


def test_groq_provider_and_key_routing():
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        if json["model"] == "bad-model":
            raise ConnectionError("model not found")
        body = {"choices": [{"message": {"content": GOOD_SQL}}]}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    real = t2s.requests
    t2s.requests = SimpleNamespace(post=fake_post)
    try:
        g = GroqProvider("gsk_test", models=["bad-model", "llama-3.3-70b-versatile"])
        assert g.complete("sys", [{"role": "user", "content": "hi"}]) == GOOD_SQL
        assert calls == ["bad-model", "llama-3.3-70b-versatile"] and g.model == "llama-3.3-70b-versatile"
    finally:
        t2s.requests = real

    saved = {k: os.environ.pop(k, None) for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY")}
    try:
        os.environ["ANTHROPIC_API_KEY"] = "gsk_pasted_in_wrong_box"      # Groq key, wrong variable
        assert [p.name for p in get_providers()] == ["groq"]
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        os.environ["GROQ_API_KEY"] = "gsk_test"
        assert [p.name for p in get_providers()] == ["claude", "groq"]
    finally:
        for k in ("ANTHROPIC_API_KEY", "GROQ_API_KEY"):
            os.environ.pop(k, None)
            if saved[k]:
                os.environ[k] = saved[k]
    print("groq provider + key routing . OK  (model fallback, gsk_ detected)")


if __name__ == "__main__":
    test_safety()
    test_read_only_connection()
    test_rule_check()
    test_clean_sql()
    test_fallback_sql_runs()
    test_model_flow()
    test_provider_switch_and_offline()
    test_cache()
    test_groq_provider_and_key_routing()
    print("\nAll GenAI tests passed.")
