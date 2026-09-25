#!/usr/bin/env python
"""Evaluation loop over pairs_master.csv -> judgments.jsonl.

Each line logs the exact strings sent to the model (sent_anchor /
sent_candidate / sent_aisle / sent_department), not just the response, so
pairs_master.csv is the sole source of truth needed to audit a judgment.
Each pair is judged by both models (DeepSeek, then Mistral) as one unit of
work, written atomically, so their completed-pair counts stay in lockstep.

Key from env only (never printed beyond last 4 chars); temperature=0,
max_tokens=50; DeepSeek reasoning disabled; provider pinned per model with
allow_fallbacks=false; one retry on a verdict/reason mismatch then log
invalid; concurrency across pairs (default 8).

Prints a running summary every 200 pairs. Stops cleanly (without losing
logged rows) if cumulative cost exceeds STOP_BUDGET ($16).

Run:  python run_evaluation.py            # reads OPENROUTER_API_KEY from env or .env
      python run_evaluation.py --pairs N  # cap number of pairs (debug)
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
PAIRS_CSV = ROOT / "pairs_master.csv"
OUT_PATH = ROOT / "judgments.jsonl"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

STOP_BUDGET = 16.0
SUMMARY_EVERY = 200
MAX_TOKENS = 50
CONCURRENCY = 8
MAX_429_RETRIES = 6


def load_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        envf = ROOT / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        sys.exit("ERROR: OPENROUTER_API_KEY not set (env or .env). Aborting; no calls made.")
    return key


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

# ---- the two models: pinned providers, no fallback; DeepSeek reasoning disabled ----
MODELS = ["deepseek_v4pro", "mistral_large3"]     # order per pair: DeepSeek, then Mistral
MODEL_CFG = {
    "deepseek_v4pro": {"slug": "deepseek/deepseek-v4-pro",
                       "provider": {"only": ["deepinfra"], "allow_fallbacks": False},   # fp8 (no full-precision offered)
                       "reasoning": {"enabled": False}},
    "mistral_large3": {"slug": "mistralai/mistral-large-2512",
                       "provider": {"only": ["mistral"], "allow_fallbacks": False},      # first-party
                       "reasoning": None},
}


def user_message(sent):
    return (f"ANCHOR: {sent['anchor']}\n"
            f"CANDIDATE: {sent['candidate']}\n"
            f"CANDIDATE AISLE: {sent['aisle']}\n"
            f"CANDIDATE DEPARTMENT: {sent['department']}")


def build_body(model_key, umsg):
    cfg = MODEL_CFG[model_key]
    body = {"model": cfg["slug"],
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": umsg}],
            "temperature": 0, "max_tokens": MAX_TOKENS,
            "provider": cfg["provider"], "response_format": RESPONSE_FORMAT,
            "usage": {"include": True}}
    if cfg["reasoning"] is not None:
        body["reasoning"] = cfg["reasoning"]
    return body


def parse_and_validate(content):
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = build_body(model_key, umsg)
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=90)
        except Exception as e:
            return None, {}, f"request_exception: {e}"
        if resp.status_code == 429:
            time.sleep(min(2 ** attempt, 30))
            continue
        try:
            data = resp.json()
        except Exception:
            return None, {}, f"non_json_http_{resp.status_code}"
        if resp.status_code != 200 or "choices" not in data:
            return None, data.get("usage", {}) or {}, f"http_{resp.status_code}"
        return data["choices"][0].get("message", {}).get("content"), data.get("usage", {}) or {}, None
    return None, {}, "429_exhausted"


def judge(api_key, model_key, sent):
    """One pair x one model: call + validate, one retry on invalid/failure. Returns a
    JSONL-ready record echoing the sent strings."""
    umsg = user_message(sent)
    content, usage, err = one_call(api_key, model_key, umsg)
    verdict, reason, valid = (None, None, False)
    if content is not None:
        verdict, reason, valid = parse_and_validate(content)
    if not valid:
        content2, usage2, err2 = one_call(api_key, model_key, umsg)
        if content2 is not None:
            v2, r2, ok2 = parse_and_validate(content2)
            if ok2 or content is None:
                content, usage, err = content2, usage2, err2
                verdict, reason, valid = v2, r2, ok2
    cdet = (usage or {}).get("completion_tokens_details", {}) or {}
    pdet = (usage or {}).get("prompt_tokens_details", {}) or {}
    return {
        "pair_id": sent["pair_id"], "model": model_key,
        "sent_anchor": sent["anchor"], "sent_candidate": sent["candidate"],
        "sent_aisle": sent["aisle"], "sent_department": sent["department"],
        "verdict": verdict if valid else None, "reason": reason if valid else None,
        "valid": bool(valid),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "reasoning_tokens": cdet.get("reasoning_tokens", 0),
        "cached_tokens": pdet.get("cached_tokens", 0),
        "cost": (usage or {}).get("cost"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


# ---------------------------------------------------------------- shared state
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.cost_by_model = {m: 0.0 for m in MODELS}
        self.n_by_model = {m: 0 for m in MODELS}
        self.pairs_done = 0
        self.stop = threading.Event()

    def total_cost(self):
        return sum(self.cost_by_model.values())


def done_combos():
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done.add((r["pair_id"], r["model"]))
                    except Exception:
                        pass
    return done


def process_pair(api_key, row, done, state, out_lock):
    """Judge one pair with BOTH models (DeepSeek then Mistral), then write both records
    atomically. Skips (pair,model) combos already logged. Respects the cost stop."""
    sent = {"pair_id": row.pair_id, "anchor": row.JUDGE_PROMPT_ANCHOR,
            "candidate": row.JUDGE_PROMPT_CANDIDATE, "aisle": row.JUDGE_PROMPT_AISLE,
            "department": row.JUDGE_PROMPT_DEPARTMENT}
    recs = []
    for model_key in MODELS:                       # DeepSeek, then Mistral — both before the next pair
        if (row.pair_id, model_key) in done:
            continue
        if state.stop.is_set():
            break
        recs.append(judge(api_key, model_key, sent))

    if not recs:
        return
    with out_lock:                                 # atomic append: a pair contributes 0 or its records
        with open(OUT_PATH, "a") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
    with state.lock:
        for rec in recs:
            state.cost_by_model[rec["model"]] += (rec.get("cost") or 0)
            state.n_by_model[rec["model"]] += 1
        state.pairs_done += 1
        if state.pairs_done % SUMMARY_EVERY == 0:
            print(f"[{state.pairs_done} pairs] "
                  + " | ".join(f"{m}: n={state.n_by_model[m]} ${state.cost_by_model[m]:.4f}" for m in MODELS)
                  + f" | total ${state.total_cost():.4f}", flush=True)
        if state.total_cost() > STOP_BUDGET:
            state.stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=None)
    args = ap.parse_args()

    api_key = load_api_key()
    df = pd.read_csv(PAIRS_CSV)
    if args.pairs:
        df = df.iloc[: args.pairs]
    rows = list(df.itertuples(index=False))
    total = len(rows)
    done = done_combos()
    print(f"key …{api_key[-4:]} | pairs_master rows: {total} | already-logged combos: {len(done)} | "
          f"models: {', '.join(MODELS)} | out -> {OUT_PATH.name}", flush=True)

    state = State()
    out_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(process_pair, api_key, r, done, state, out_lock) for r in rows]
        for fut in as_completed(futs):
            fut.result()

    stopped = state.stop.is_set()
    print("\n" + "=" * 72, flush=True)
    print("STOPPED at cost cap." if stopped else "RUN COMPLETE.", flush=True)
    print("=" * 72, flush=True)
    report(total)


def report(total_pairs):
    recs = []
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    per = {}
    for m in MODELS:
        mr = [r for r in recs if r["model"] == m]
        per[m] = {"n": len(mr), "valid": sum(r["valid"] for r in mr),
                  "cost": sum((r.get("cost") or 0) for r in mr)}
    for m in MODELS:
        p = per[m]
        print(f"{m:<16} judged {p['n']} | valid {p['valid']} invalid {p['n']-p['valid']} "
              f"({(p['n']-p['valid'])/max(p['n'],1):.2%}) | spend ${p['cost']:.4f}", flush=True)
    print(f"TOTAL spend (both models): ${sum(per[m]['cost'] for m in MODELS):.4f}", flush=True)

    # 12 original thesis anchors present among judged pairs (by candidate/anchor sent strings)
    TWELVE = {"Mini Babybel White Cheddar Cheese", "White Asparagus", "Crispy Chicken Strips",
              "Flour Burrito Caseras Tortillas", "Black Label Center Cut Bacon",
              "Ranch Original Topping & Dressing", "Mojo Peanut Butter Bar",
              "Cafe Domingo Coffee K-Cups", "Extra Strength Assorted Fruit Antacid",
              "Plastic Cups", "Grapefruit Sculpin IPA", "Poblano Pepper & Corn Chowder"}
    seen = {r["sent_anchor"] for r in recs}
    present = sorted(TWELVE & seen)
    print(f"12 original anchors present in judgments: {len(present)}/12"
          + ("" if len(present) == 12 else f"  MISSING: {sorted(TWELVE - seen)}"), flush=True)

    # disagreement across pairs both models judged validly
    dv = {r["pair_id"]: r["verdict"] for r in recs if r["model"] == "deepseek_v4pro" and r["valid"]}
    mv = {r["pair_id"]: r["verdict"] for r in recs if r["model"] == "mistral_large3" and r["valid"]}
    both = set(dv) & set(mv)
    disagree = [p for p in both if dv[p] != mv[p]]
    print(f"both-valid pairs: {len(both)} | disagreements: {len(disagree)} "
          f"({len(disagree)/max(len(both),1):.1%})", flush=True)
    judged_both = ({r["pair_id"] for r in recs if r["model"] == "deepseek_v4pro"}
                   & {r["pair_id"] for r in recs if r["model"] == "mistral_large3"})
    print(f"pairs judged by BOTH: {len(judged_both)}/{total_pairs} | "
          f"gaps: {total_pairs - len(judged_both)}", flush=True)


if __name__ == "__main__":
    main()
