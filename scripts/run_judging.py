#!/usr/bin/env python
"""OpenRouter substitute-judging runner (reduced 50-anchor scope).

Judges each unique (anchor, candidate) pair in evaluation_api/pairs_50.csv
once by each of two models, independently:

  - DeepSeek V4 Pro 0423 -> deepseek/deepseek-v4-pro (pinned: DeepInfra fp8;
    no full-precision endpoint exists on OpenRouter)
  - Mistral Large 3 2512 -> mistralai/mistral-large-2512 (pinned: Mistral,
    first-party)

Key read from OPENROUTER_API_KEY only (last 4 chars ever printed);
temperature=0, max_tokens=50; DeepSeek reasoning disabled (the first-50 gate
verifies reasoning_tokens is zero and stops if not); provider pinned per
model with allow_fallbacks=false; append-only JSONL log per model, resumable
(skips pair_ids already logged); one retry on an invalid/failed call, then
logged invalid; concurrency default 6 with 429 back-off.

Flow: judge the first 50 pairs by both models, print a cost/token summary,
then continue to the rest unless (a) DeepSeek reasoning_tokens is nonzero, or
(b) the running cost-per-pair projects a full-run total above STOP_BUDGET.

Run:  OPENROUTER_API_KEY=... python run_judging.py           # full 50-anchor run
      python run_judging.py --pairs N                        # cap pairs (debug)
"""

import os
import sys
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
PAIRS_CSV = ROOT / "evaluation_api" / "pairs_50.csv"
OUT_DIR = ROOT / "judgments"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

GATE_N = 50            # judge the first this-many pairs, then check the two gates
STOP_BUDGET = 17.0     # abort before the rest if the projected full-run total exceeds this
MAX_TOKENS = 50
CONCURRENCY = 6
MAX_429_RETRIES = 6


# ---------------------------------------------------------------- API key (env only)
def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        # optional convenience: read .env (gitignored) without a dependency
        envf = ROOT / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        sys.exit("ERROR: OPENROUTER_API_KEY is not set (env var or .env). Aborting; no calls made.")
    return key


# ---------------------------------------------------------------- fixed prompt + schema
SYSTEM_PROMPT = """You are evaluating a grocery product substitute-detection system trained on
Instacart order data. You will be shown one anchor product and one candidate
product that the system proposed as a possible substitute for it.

Answer one question:

Would a typical shopper who came to buy the ANCHOR reasonably accept this
CANDIDATE as a replacement if the anchor were unavailable?

This is directional, anchor to candidate. Judge only that direction, do not
consider the reverse.

Rules, please follow these exactly:

1. SUBSTITUTE means serving the same need or occasion. Complements do not count:
peanut butter and jelly are bought together, not instead of each other, so
jelly is NOT a substitute for peanut butter.

2. Organic vs conventional versions of the same product ARE substitutes in both
directions. "Organic Whole Milk" and "Whole Milk" are substitutes. The same
goes for other certification or sourcing labels (cage-free, grass-fed,
non-GMO, fair trade) when the underlying product is the same.

3. If BOTH the anchor and the candidate share the same dietary restriction
(both gluten-free, both sugar-free, both vegan, both decaf), that shared
attribute is NOT a difference between them. Judge them on the underlying
product alone.

4. When the candidate has a restriction the anchor lacks, it usually still
counts as a substitute (someone buying regular cookies would generally accept
gluten-free cookies). When the anchor has a restriction the candidate lacks,
it usually does NOT (someone who specifically came for gluten-free cookies
cannot use regular ones). Apply judgement rather than treating this as
absolute.

5. Different flavours or varieties of the same product type ARE substitutes
(strawberry yogurt for blueberry yogurt). Different sizes or pack counts of
the same product ARE substitutes.

6. Same aisle or department alone is NOT sufficient. Milk and cheese are both
dairy but are not substitutes.

Along with your verdict, give exactly one reason code from the list below.
Each code belongs to one verdict. If more than one code fits, use the one that
appears first in its list.

Codes for YES:
- restriction_ok: the candidate has a dietary restriction the anchor lacks,
  and is otherwise the same kind of product (rule 4)
- certification: same product, differing only in organic, certification, or
  sourcing label (rule 2)
- variant: same product type, differing in flavour, variety, size, or pack
  count (rule 5)
- same_need: a different kind of product that serves the same need or
  occasion (rule 1)

Codes for NO:
- restriction_blocked: the anchor has a dietary restriction the candidate
  lacks (rule 4)
- complement: bought together with the anchor rather than instead of it
  (rule 1)
- category_only: shares the anchor's aisle or department but serves a
  different need (rule 6)
- different_need: serves a different need and is not a complement

If a pair is genuinely ambiguous, still commit to YES or NO.

Respond with a single JSON object and nothing else:
{"reason": "<code>", "verdict": "YES" or "NO"}"""

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string", "enum": [
            "restriction_ok", "certification", "variant", "same_need",
            "restriction_blocked", "complement", "category_only", "different_need"]},
        "verdict": {"type": "string", "enum": ["YES", "NO"]},
    },
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema",
                   "json_schema": {"name": "substitute_verdict", "strict": True, "schema": VERDICT_SCHEMA}}

YES_CODES = {"restriction_ok", "certification", "variant", "same_need"}
NO_CODES = {"restriction_blocked", "complement", "category_only", "different_need"}

# ---------------------------------------------------------------- the two models
MODELS = {
    "deepseek_v4pro": {
        "slug": "deepseek/deepseek-v4-pro",                     # name: "DeepSeek V4 Pro 0423"
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},  # fp8 (no full-precision offered)
        "reasoning": {"enabled": False},                        # disable reasoning tokens
    },
    "mistral_large3": {
        "slug": "mistralai/mistral-large-2512",                 # name: "Mistral Large 3 2512"
        "provider": {"only": ["mistral"], "allow_fallbacks": False},    # first-party
        "reasoning": None,                                      # not a reasoning model
    },
}


def user_message(row):
    return (f"ANCHOR: {row.anchor_name}\n"
            f"CANDIDATE: {row.candidate_name}\n"
            f"CANDIDATE AISLE: {row.candidate_aisle}\n"
            f"CANDIDATE DEPARTMENT: {row.candidate_department}")


def build_body(model_key, umsg):
    cfg = MODELS[model_key]
    body = {
        "model": cfg["slug"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": umsg}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "provider": cfg["provider"],
        "response_format": RESPONSE_FORMAT,
        "usage": {"include": True},          # ask OpenRouter to return cost + token details
    }
    if cfg["reasoning"] is not None:
        body["reasoning"] = cfg["reasoning"]
    return body


def parse_and_validate(content):
    """-> (verdict, reason, valid_bool). Invalid if not parseable, out-of-enum, or the
    verdict/reason pairing crosses the YES/NO code lists."""
    try:
        obj = json.loads(content)
    except Exception:
        return None, None, False
    v, r = obj.get("verdict"), obj.get("reason")
    if v not in ("YES", "NO") or r not in (YES_CODES | NO_CODES):
        return v, r, False
    if (v == "YES" and r not in YES_CODES) or (v == "NO" and r not in NO_CODES):
        return v, r, False
    return v, r, True


def one_call(api_key, model_key, umsg):
    """A single HTTP call with 429 back-off. Returns (content_or_None, usage_dict, raw_dict, err)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = build_body(model_key, umsg)
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=90)
        except Exception as e:
            return None, {}, {"error": f"request_exception: {e}"}, str(e)
        if resp.status_code == 429:
            wait = min(2 ** attempt, 30)
            time.sleep(wait)
            continue
        try:
            data = resp.json()
        except Exception:
            return None, {}, {"status": resp.status_code, "text": resp.text[:500]}, "non_json_http"
        if resp.status_code != 200 or "choices" not in data:
            return None, data.get("usage", {}) or {}, data, f"http_{resp.status_code}"
        content = data["choices"][0].get("message", {}).get("content")
        return content, data.get("usage", {}) or {}, data, None
    return None, {}, {"error": "429_exhausted"}, "429_exhausted"


def judge(api_key, model_key, row):
    """Judge one pair with one model: call + validate, one retry on invalid/failure.
    Always returns a JSONL-ready record (valid True/False)."""
    umsg = user_message(row)
    content, usage, raw, err = one_call(api_key, model_key, umsg)
    verdict, reason, valid = (None, None, False)
    if content is not None:
        verdict, reason, valid = parse_and_validate(content)
    if not valid:                                      # exactly one retry
        content2, usage2, raw2, err2 = one_call(api_key, model_key, umsg)
        if content2 is not None:
            v2, r2, ok2 = parse_and_validate(content2)
            if ok2 or content is None:                 # prefer the retry if the first had no content
                content, usage, raw, err = content2, usage2, raw2, err2
                verdict, reason, valid = v2, r2, ok2
    cdet = (usage or {}).get("completion_tokens_details", {}) or {}
    pdet = (usage or {}).get("prompt_tokens_details", {}) or {}
    return {
        "pair_id": row.pair_id,
        "model": model_key,
        "verdict": verdict if valid else None,
        "reason": reason if valid else None,
        "valid": bool(valid),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "reasoning_tokens": cdet.get("reasoning_tokens", 0),
        "cached_tokens": pdet.get("cached_tokens", 0),
        "cost": (usage or {}).get("cost"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": err,
        "raw_verdict": verdict, "raw_reason": reason,       # what came back even if inconsistent
        "raw": raw,
    }


# ---------------------------------------------------------------- logging / resume
_locks = {}

def out_path(model_key):
    return OUT_DIR / f"{model_key}.jsonl"

def done_ids(model_key):
    p = out_path(model_key)
    ids = set()
    if p.exists():
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["pair_id"])
                    except Exception:
                        pass
    return ids

def append_record(model_key, rec):
    p = out_path(model_key)
    _locks.setdefault(model_key, threading.Lock())
    with _locks[model_key]:
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")


def run_batch(api_key, rows, model_key, skip):
    """Judge `rows` for one model (skipping pair_ids in `skip`), concurrently, logging
    each as it completes. Returns the list of records written this call."""
    todo = [r for r in rows if r.pair_id not in skip]
    written = []
    if not todo:
        return written
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(judge, api_key, model_key, r) for r in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            append_record(model_key, rec)   # lock-guarded append
            written.append(rec)
    return written


def read_records(model_key, only_ids=None):
    p = out_path(model_key)
    recs = []
    if p.exists():
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if only_ids is None or r["pair_id"] in only_ids:
                    recs.append(r)
    return recs


def summarize(recs):
    n = len(recs)
    valid = sum(r["valid"] for r in recs)
    cost = sum((r.get("cost") or 0) for r in recs)
    pt = [r.get("prompt_tokens") or 0 for r in recs]
    ct = [r.get("completion_tokens") or 0 for r in recs]
    rt = [r.get("reasoning_tokens") or 0 for r in recs]
    cached = sum((r.get("cached_tokens") or 0) for r in recs)
    return {"n": n, "valid": valid, "invalid": n - valid, "cost": cost,
            "mean_prompt": (sum(pt) / n if n else 0), "mean_completion": (sum(ct) / n if n else 0),
            "mean_reasoning": (sum(rt) / n if n else 0), "max_reasoning": (max(rt) if rt else 0),
            "cached_tokens_total": cached}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=None, help="cap number of pairs (debug)")
    args = ap.parse_args()

    api_key = load_api_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OpenRouter key loaded (…{api_key[-4:]}).  models: {', '.join(MODELS)}", flush=True)

    df = pd.read_csv(PAIRS_CSV)
    if args.pairs:
        df = df.iloc[: args.pairs]
    rows = list(df.itertuples(index=False))
    total_pairs = len(rows)
    gate_rows = rows[:GATE_N]
    rest_rows = rows[GATE_N:]
    gate_ids = {r.pair_id for r in gate_rows}
    print(f"pairs to judge: {total_pairs} (gate on first {len(gate_rows)}); "
          f"outputs -> {OUT_DIR}/<model>.jsonl", flush=True)

    # ---- Phase 1: the first GATE_N pairs, both models (resume-aware) ----
    for mk in MODELS:
        run_batch(api_key, gate_rows, mk, done_ids(mk))

    # ---- Evaluate the two gates from the logged first-GATE_N records ----
    gate_recs = {mk: read_records(mk, only_ids=gate_ids) for mk in MODELS}
    s = {mk: summarize(gate_recs[mk]) for mk in MODELS}
    cost_gate = sum(s[mk]["cost"] for mk in MODELS)
    pairs_gate = min(s[mk]["n"] for mk in MODELS) or 1
    per_pair_all = cost_gate / pairs_gate                  # cost for one pair across BOTH models
    projected_full = per_pair_all * total_pairs
    ds = s["deepseek_v4pro"]
    print(f"[gate] pairs={pairs_gate} | cost so far ${cost_gate:.4f} | "
          f"per-pair(both) ${per_pair_all:.5f} | projected full ${projected_full:.2f} | "
          f"DeepSeek mean/max reasoning_tokens {ds['mean_reasoning']:.1f}/{ds['max_reasoning']} | "
          f"invalid ds={ds['invalid']} mistral={s['mistral_large3']['invalid']}", flush=True)

    if ds["max_reasoning"] > 0:
        print("STOP: DeepSeek returned nonzero reasoning_tokens — reasoning could not be "
              "disabled; this changes the cost projection. Not continuing.", flush=True)
        return
    if projected_full > STOP_BUDGET:
        print(f"STOP: projected full-run cost ${projected_full:.2f} exceeds "
              f"${STOP_BUDGET:.2f}. Not continuing.", flush=True)
        return

    # ---- Phase 2: the rest, both models ----
    for mk in MODELS:
        run_batch(api_key, rest_rows, mk, done_ids(mk))

    # ---- Final report ----
    print("\n" + "=" * 72, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 72, flush=True)
    all_recs = {mk: read_records(mk) for mk in MODELS}
    for mk in MODELS:
        fs = summarize(all_recs[mk])
        print(f"{mk:<16} judged {fs['n']}/{total_pairs} | valid {fs['valid']} invalid {fs['invalid']} "
              f"({fs['invalid']/max(fs['n'],1):.1%}) | spend ${fs['cost']:.4f} | "
              f"mean tok in/out {fs['mean_prompt']:.0f}/{fs['mean_completion']:.0f} | "
              f"reasoning mean {fs['mean_reasoning']:.1f} | cached {fs['cached_tokens_total']}", flush=True)

    # disagreement + completeness (only pairs both models judged validly)
    v = {}
    for mk in MODELS:
        v[mk] = {r["pair_id"]: r["verdict"] for r in all_recs[mk] if r["valid"]}
    common = set(v["deepseek_v4pro"]) & set(v["mistral_large3"])
    disagree = [pid for pid in common if v["deepseek_v4pro"][pid] != v["mistral_large3"][pid]]
    judged_both = {r["pair_id"] for r in all_recs["deepseek_v4pro"]} & {r["pair_id"] for r in all_recs["mistral_large3"]}
    gaps = total_pairs - len(judged_both)
    total_spend = sum(summarize(all_recs[mk])["cost"] for mk in MODELS)
    print(f"\nboth-model valid pairs: {len(common)} | disagreements: {len(disagree)} "
          f"({len(disagree)/max(len(common),1):.1%})", flush=True)
    print(f"pairs judged by BOTH (any status): {len(judged_both)}/{total_pairs} | gaps: {gaps}", flush=True)
    print(f"TOTAL actual spend (both models): ${total_spend:.4f}", flush=True)
    if gaps == 0:
        print("All pairs judged by both models — no gaps.", flush=True)


if __name__ == "__main__":
    main()
