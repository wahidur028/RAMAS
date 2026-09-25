#!/usr/bin/env python3
"""Smoke test of additional installed Ollama models against the frozen decision contract.

Not part of PROTOCOL.md.  Sends a few real requests (one exposed + one hidden state per
repeat) with the protocol's options and neutral prompt, disabling `think` on models that
advertise the thinking capability, and records validity, latency, stop reason and tokens.
Output: <output>/smoke_<model>.json and <output>/SMOKE_SUMMARY.json.  Use it to decide
whether a model is worth an extension run; it produces no scientific result by itself.
"""
import argparse, json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner as r

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="comma-separated Ollama model names")
    ap.add_argument("--output", required=True)
    ap.add_argument("--states", type=int, default=2, help="states per condition (default 2)")
    a = ap.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    bank, agent, prompt = r.load_inputs_cached()
    client = r.OllamaClient()
    idx = r.pilot_indices(bank)[: a.states]
    summary = {}
    for model in a.models.split(","):
        rec = {"model": model, "calls": [], "at": r.utcnow()}
        try:
            ident = r.serving_identity(client, (model,))
            think = r.think_flag(model, ident)
            rec["identity"] = ident["models"][model]; rec["think"] = think
        except (r.RunnerError, r.TransportFailure) as exc:
            rec["error"] = str(exc); summary[model] = {"status": "UNAVAILABLE", "detail": str(exc)}
            r.atomic(out / f"smoke_{model.replace(':','_').replace('/','_')}.json", rec); continue
        for i in idx:
            for condition in r.CONDITIONS:
                body, payload, visible = r.build_body(model, bank[i]["request"], condition, prompt, agent, think)
                started = r.utcnow()
                try:
                    outer, latency = client.complete(body)
                except r.TransportFailure as exc:
                    rec["calls"].append({"state_index": i, "condition": condition, "valid": False, "error": f"TRANSPORT {exc}"}); continue
                raw = (outer.get("message") or {}).get("content"); err = None; dec = None
                try:
                    if outer.get("error"): raise r.ContractError(str(outer["error"]))
                    if outer.get("done_reason") == "length": raise r.ContractError("TRUNCATED")
                    dec = r.validate_decision(raw if isinstance(raw, str) else "", agent, visible)
                except r.ContractError as exc:
                    err = f"{type(exc).__name__}: {exc}"
                rec["calls"].append({"state_index": i, "condition": condition, "started_at": started, "valid": err is None,
                                     "decision": dec, "error": err, "latency_seconds": latency,
                                     "done_reason": outer.get("done_reason"), "prompt_tokens": outer.get("prompt_eval_count"),
                                     "output_tokens": outer.get("eval_count"), "raw_content": raw,
                                     "thinking_present": bool((outer.get("message") or {}).get("thinking"))})
                print(f"{model:22s} s{i:04d} {condition:7s} valid={err is None} latency={latency:.1f}s action={dec['action'] if dec else '-'} {err or ''}", flush=True)
        client.unload(model)
        valid = sum(c.get("valid") for c in rec["calls"]); n = len(rec["calls"])
        lat = [c["latency_seconds"] for c in rec["calls"] if c.get("latency_seconds") is not None]
        summary[model] = {"status": "PASS" if valid == n and n else "FAIL", "valid": valid, "calls": n,
                          "mean_latency_seconds": sum(lat) / len(lat) if lat else None, "think": think,
                          "digest": rec["identity"]["digest"], "errors": sorted({c["error"] for c in rec["calls"] if c.get("error")})[:3],
                          "estimated_hours_for_7464_calls": (sum(lat) / len(lat) * 7464 / 3600) if lat else None}
        r.atomic(out / f"smoke_{model.replace(':','_').replace('/','_')}.json", rec)
    r.atomic(out / "SMOKE_SUMMARY.json", {"at": r.utcnow(), "states_per_condition": a.states, "models": summary})
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
