"""GenAI layer: plain-English question -> SQL -> result -> plain-English answer.

Built so that the demo does NOT fail while you test again and again:

  0. Rule check   - questions about profit/quantity/discount/cost, or asking to delete/modify
                    data, are answered instantly (no API call, no credits used).
  1. Cache        - a question that was answered before is served from cache (no API call).
  2. Providers    - Claude (Anthropic key) and/or Llama via Groq (gsk_ key). If the first one
                    fails (no credits, rate limit, network), the next one is tried.
  3. Safety       - generated SQL must be a single SELECT, then runs on a READ-ONLY connection.
  4. Retry        - if the SQL errors, the error is sent back to the model once.
  5. Offline      - if every provider fails, the closest pre-written question/SQL pair is used.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from contextlib import closing

import pandas as pd
from dotenv import load_dotenv

try:
    import requests
except ImportError:                      # only needed for the Groq provider
    requests = None

from config import ROOT
from db import connect
from genai.fallback_questions import FALLBACK

load_dotenv(ROOT / ".env")

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")        # claude-haiku-5-5 = cheaper/faster
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = list(dict.fromkeys(m for m in [os.getenv("GROQ_MODEL"), "llama-3.3-70b-versatile",
                                             "llama-3.1-8b-instant"] if m))
MAX_ROWS = 500
CACHE_PATH = ROOT / ".answer_cache.json"

NOT_AVAILABLE_MSG = ("This can't be answered from the warehouse. The dataset only has the Sales "
                     "amount - there is no profit, quantity, discount or cost data.")
READ_ONLY_MSG = "Blocked for safety: this assistant is read-only and cannot modify the warehouse."


# ------------------------------------------------------- 0. Rule check
_UNAVAILABLE = re.compile(
    r"\b(profit\w*|margins?|quantit(?:y|ies)|units?\s+sold|discounts?|costs?|inventory)\b", re.I)
_WRITE_INTENT = re.compile(
    r"\b(delete|drop|truncate|erase|wipe)\b|"
    r"\b(insert|update|alter)\b.*\b(tables?|rows?|records?|data|database)\b", re.I)


def guard_question(question: str) -> dict | None:
    """Instant answer for questions that must never reach the database or an API."""
    if _WRITE_INTENT.search(question):
        return {"error": READ_ONLY_MSG}
    if _UNAVAILABLE.search(question):
        return {"summary": NOT_AVAILABLE_MSG}
    return None


# ------------------------------------------------------ 3. Safety layer
BLOCKED = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|vacuum|reindex|"
    r"grant|revoke)\b", re.IGNORECASE)


def validate_sql(sql: str) -> tuple[bool, str]:
    """Return (ok, cleaned_sql_or_reason)."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return False, "Empty query."
    if ";" in s:
        return False, "Only a single SQL statement is allowed."
    if "--" in s or "/*" in s:
        return False, "SQL comments are not allowed."
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        return False, "Only SELECT queries are allowed."
    hit = BLOCKED.search(s)
    if hit:
        return False, f"Blocked keyword: {hit.group(0).upper()}"
    return True, s


def clean_sql(text: str) -> str:
    """Pull the SQL out of a model reply (handles ``` fences and chatty explanations)."""
    text = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    if not re.match(r"(?is)^\s*(select|with|not_available)\b", text):
        start = re.search(r"(?is)\b(select|with\s+\w+\s+as)\b", text)
        if start:
            text = text[start.start():]
    first_block = text.split("\n\n")[0].strip()
    if re.match(r"(?is)^(select|with)\b", first_block):
        text = first_block
    return text.strip()


# ------------------------------------------------------------ Prompt
def build_system_prompt() -> str:
    with closing(connect()) as conn:
        ddl = "\n\n".join(r[0] for r in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"))

        def distinct(sql: str) -> str:
            return ", ".join(str(r[0]) for r in conn.execute(sql))

        regions = distinct("SELECT DISTINCT region FROM dim_location ORDER BY 1")
        categories = distinct("SELECT DISTINCT category FROM dim_product ORDER BY 1")
        sub_cats = distinct("SELECT DISTINCT sub_category FROM dim_product ORDER BY 1")
        segments = distinct("SELECT DISTINCT segment FROM dim_customer ORDER BY 1")
        modes = distinct("SELECT DISTINCT ship_mode FROM fact_sales ORDER BY 1")
        first, last = conn.execute(
            "SELECT MIN(d.full_date), MAX(d.full_date) FROM fact_sales f "
            "JOIN dim_date d ON d.date_key = f.order_date_key").fetchone()
        try:
            rfm_names = distinct("SELECT DISTINCT rfm_segment FROM customer_segments ORDER BY 1")
            rfm_note = (
                "\n- customer_segments(customer_key, customer_id, customer_name, recency, frequency, "
                "monetary, cluster, rfm_segment): ML result of K-Means RFM clustering, one row per "
                f"customer. recency = days since last order, frequency = number of orders, monetary = "
                f"total sales. rfm_segment values: {rfm_names}. Join to dim_customer on customer_key. "
                "Do NOT confuse rfm_segment with dim_customer.segment.")
        except Exception:
            rfm_note = ""

    return f"""You are an expert SQL analyst for a retail data warehouse stored in SQLite.
Convert the user's question into exactly ONE read-only SQLite SELECT query.

DATABASE SCHEMA (star schema):
{ddl}

VALUES AND NOTES:
- dim_location.region: {regions}
- dim_product.category: {categories}
- dim_product.sub_category: {sub_cats}
- dim_customer.segment (business segment): {segments}
- fact_sales.ship_mode: {modes}
- Orders cover {first} to {last}. "Latest year" = {last[:4]}; "last quarter" = Q4 of {last[:4]}.{rfm_note}

RULES:
1. Output ONLY the SQL query. No explanation, no markdown fences.
2. Only SELECT (or WITH ... SELECT). Never modify data or schema.
3. Join fact_sales to dimensions with the *_key columns; use order_date_key for order dates.
4. "Sales"/"revenue" means SUM(fact_sales.sales). Use ROUND(..., 2) for money.
5. "Number of orders" means COUNT(DISTINCT fact_sales.order_id).
6. For top-N / lowest-N questions use ORDER BY and LIMIT. Always give columns readable aliases.
7. The data has NO profit, quantity, discount, cost, inventory or forecast columns. If the
   question needs them, cannot be answered from this schema, or asks to change/delete data,
   output exactly: NOT_AVAILABLE
"""


# --------------------------------------------------------- 2. Providers
def _text(response) -> str:
    return "".join(getattr(block, "text", "") for block in response.content).strip()


class ClaudeProvider:
    name = "claude"

    def __init__(self, client, model: str | None = None):
        self.client, self.model = client, model or MODEL

    def complete(self, system: str | None, messages: list, max_tokens: int = 600) -> str:
        kwargs = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        return _text(self.client.messages.create(**kwargs))


class GroqProvider:
    """Llama models on Groq (OpenAI-compatible API). Tries each model name in turn."""
    name = "groq"

    def __init__(self, api_key: str, models: list[str] | None = None):
        self.api_key, self.models = api_key, models or GROQ_MODELS
        self.model = self.models[0]

    def complete(self, system: str | None, messages: list, max_tokens: int = 600) -> str:
        if requests is None:
            raise RuntimeError("The 'requests' package is not installed")
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        last_error: Exception | None = None
        for model in self.models:
            try:
                resp = requests.post(
                    GROQ_URL, timeout=45,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": model, "messages": msgs, "max_tokens": max_tokens,
                          "temperature": 0})
                resp.raise_for_status()
                self.model = model
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:                 # wrong model name, rate limit, network...
                last_error = exc
        raise last_error


def get_providers() -> list:
    """Providers from environment variables. A key pasted into the wrong variable is routed
    by its prefix: gsk_... -> Groq, anything else -> Anthropic."""
    keys = [(os.getenv("ANTHROPIC_API_KEY") or "").strip(), (os.getenv("GROQ_API_KEY") or "").strip()]
    claude_key = next((k for k in keys if k and not k.startswith("gsk_")), None)
    groq_key = next((k for k in keys if k.startswith("gsk_")), None)
    providers = []
    if claude_key:
        import anthropic
        providers.append(ClaudeProvider(anthropic.Anthropic(api_key=claude_key, max_retries=1,
                                                            timeout=30)))
    if groq_key:
        providers.append(GroqProvider(groq_key))
    return providers


def generate_sql(provider, system: str, question: str,
                 prev_sql: str | None = None, error: str | None = None) -> str:
    messages = [{"role": "user", "content": question}]
    if prev_sql and error:
        messages += [
            {"role": "assistant", "content": prev_sql},
            {"role": "user", "content": f"That query failed with this SQLite error: {error}\n"
                                        "Return a corrected query only."},
        ]
    return clean_sql(provider.complete(system, messages, 600))


def summarize(provider, question: str, sql: str, df: pd.DataFrame) -> str:
    prompt = (
        f"Question: {question}\n\nSQL used:\n{sql}\n\nResult (first rows, CSV):\n"
        f"{df.head(30).to_csv(index=False)}\n\n"
        "Answer the question in 2-3 plain sentences for a business manager. Quote the key numbers "
        "(use $ and thousands separators). Use only the data shown. Do not mention SQL.")
    return provider.complete(None, [{"role": "user", "content": prompt}], 300)


def local_summary(df: pd.DataFrame) -> str:
    """Used when the summary call fails or when answering offline."""
    if df.empty:
        return "The query returned no rows."
    if df.shape == (1, 1):
        return f"Result: {df.iat[0, 0]}"
    first = ", ".join(f"{c} = {v}" for c, v in df.iloc[0].items())
    return f"The query returned {len(df)} rows. Top row: {first}."


def run_query(sql: str) -> pd.DataFrame:
    conn = connect(read_only=True)          # the database itself refuses writes
    try:
        return pd.read_sql_query(sql, conn).head(MAX_ROWS)
    finally:
        conn.close()


# ------------------------------------------------------------ 1. Cache
def _norm(question: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", question.lower()))


def _cache_load() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _cache_put(question: str, sql: str, summary: str, source: str, model: str) -> None:
    try:
        data = _cache_load()
        data[_norm(question)] = {"sql": sql, "summary": summary, "source": source, "model": model}
        CACHE_PATH.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except Exception:
        pass                                  # cache is a bonus; never break the answer


def clear_cache() -> None:
    CACHE_PATH.unlink(missing_ok=True)


# ----------------------------------------------------------- 5. Offline
_STOP = {"the", "a", "an", "of", "in", "for", "by", "is", "are", "which", "what", "who", "to", "me"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOP


def match_fallback(question: str):
    best, best_score = None, 0.0
    for item in FALLBACK:
        a, b = _tokens(question), _tokens(item["question"])
        jaccard = len(a & b) / max(1, len(a | b))
        ratio = difflib.SequenceMatcher(None, question.lower(), item["question"].lower()).ratio()
        score = 0.6 * jaccard + 0.4 * ratio
        if score > best_score:
            best, best_score = item, score
    return best if best_score >= 0.35 else None


def _offline(result: dict, reason: str) -> dict:
    result["source"] = "offline backup"
    item = match_fallback(result["question"])
    if item is None:
        result["error"] = (f"{reason}. No matching pre-written question was found - "
                           "try one of the example questions.")
        return result
    df = run_query(item["sql"])
    result.update(sql=item["sql"], df=df, summary=local_summary(df),
                  note=f"{reason}. Showing the closest pre-written query: \"{item['question']}\"")
    return result


# ----------------------------------------------------------------- Main
def _answer_with(provider, question: str, system: str) -> dict:
    """Full pipeline for ONE provider. Raises only if the provider's API fails."""
    sql = generate_sql(provider, system, question)
    if sql.upper().startswith("NOT_AVAILABLE"):
        return {"summary": NOT_AVAILABLE_MSG}

    ok, cleaned = validate_sql(sql)
    if not ok:
        return {"sql": sql, "error": f"Blocked for safety: {cleaned}"}

    try:
        df = run_query(cleaned)
    except Exception as exc:                          # bad SQL -> one retry with the error
        retry = generate_sql(provider, system, question, prev_sql=cleaned, error=str(exc))
        ok, cleaned = validate_sql(retry)
        if not ok:
            return {"sql": retry, "error": f"Blocked for safety: {cleaned}"}
        try:
            df = run_query(cleaned)
        except Exception as exc2:
            return {"sql": cleaned, "error": f"The generated SQL failed twice: {exc2}"}

    try:
        summary = summarize(provider, question, cleaned, df)
    except Exception:
        summary = local_summary(df)
    return {"sql": cleaned, "df": df, "summary": summary}


def ask(question: str, client=None, providers: list | None = None, use_cache: bool = True) -> dict:
    """Answer a plain-English question. Never raises because of API/model problems."""
    result = {"question": question, "sql": None, "df": None, "summary": None, "error": None,
              "note": None, "source": None, "model": None, "cached": False}

    guard = guard_question(question)                                   # 0. rule check
    if guard:
        result.update(guard, source="rule check")
        return result

    if use_cache:                                                      # 1. cache
        hit = _cache_load().get(_norm(question))
        if hit:
            ok, sql = validate_sql(hit["sql"])
            if ok:
                try:
                    result.update(sql=sql, df=run_query(sql), summary=hit["summary"],
                                  source=hit["source"], model=hit["model"], cached=True,
                                  note="Served from cache - no API call used.")
                    return result
                except Exception:
                    pass

    if providers is None:                                              # 2. providers
        providers = [ClaudeProvider(client)] if client is not None else get_providers()
    if not providers:
        return _offline(result, "No API key found")

    system = build_system_prompt()
    failures = []
    for provider in providers:
        try:
            part = _answer_with(provider, question, system)
        except Exception as exc:
            failures.append(f"{provider.name}: {type(exc).__name__}")
            continue
        result.update(part, source=provider.name, model=provider.model)
        if failures:
            result["note"] = "Switched provider after: " + ", ".join(failures)
        if use_cache and result["df"] is not None and not result["error"]:
            _cache_put(question, result["sql"], result["summary"], provider.name, provider.model)
        return result

    return _offline(result, "All AI providers failed (" + ", ".join(failures) + ")")
