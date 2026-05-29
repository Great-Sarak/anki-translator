# Spike 001 — Skip `openclaw infer model run` CLI cold start

## Question

PR #37 swapped the default LLM dispatcher to `subprocess.run(["openclaw", "infer", "model", "run", ...])`. The smoke test got slow. Is there a way to grab model details/auth once and reuse them — i.e., bypass the per-call CLI cold start by talking to the long-lived gateway daemon (ws://127.0.0.1:18789) directly from Python?

## Measurements

All against `anthropic/claude-haiku-4-5`, prompt "Say only OK", with the gateway warm.

| Configuration | Wall time | Notes |
|---|---|---|
| `openclaw --version` | 0.04s | Node + minimal module load |
| `openclaw gateway call health` | 0.85s | Thin RPC: Node + WS handshake + one call |
| `openclaw infer model run` (current dispatcher, local SDK) | 7.5s | Node + provider SDK bootstrap + haiku call |
| `openclaw infer model run --gateway` | 7.2s | Same Node bootstrap, delegates to daemon |
| 3× sequential `openclaw infer model run` | 21.8s (≈7.3s each) | No amortization across calls |
| **3× parallel** | **7.6s** | Cold start parallelizes for free |
| **10× parallel** | **11.9s** | Sweet spot — ~5× throughput vs sequential |
| 20× parallel | 28.8s | OS contention dominates; net regression vs 10× |

## What the source says

`dist/capability-cli-DZgVO687.js:639` — when `--gateway` is set, the CLI calls `callGateway({method:"agent", params:{..., modelRun:true, model, provider, idempotencyKey}})` against the daemon. When `--model` is set (a "model override"), it elevates to `ADMIN_SCOPE`:

```js
clientName: hasModelOverride ? GATEWAY_CLIENT.GATEWAY_CLIENT : GATEWAY_CLIENT.CLI,
mode:       hasModelOverride ? CLIENT_MODES.BACKEND        : CLIENT_MODES.CLI,
...hasModelOverride ? { scopes: [ADMIN_SCOPE] } : {}
```

`openclaw gateway call agent --params '{...}'` runs in CLI scope and is rejected when we pass `provider`/`model`:

```
Gateway call failed: provider/model overrides are not authorized for this caller.
```

So bypassing the heavy `infer model run` CLI by calling the gateway directly works in principle, but the model-override scope check means we'd either:

1. Configure a dedicated agent in `openclaw.json` with `model.primary: anthropic/claude-haiku-4-5`, then call `agent` for that agent id (no override, no admin scope), OR
2. Mint an ADMIN-scoped token and authenticate with it (more invasive).

Neither is trivial — and the gateway client/auth code lives in mangled bundle chunks (`call-B2KojSjl.d.ts` → some `chunk-xxxxx.js`); `openclaw/plugin-sdk` does not re-export `callGateway`. A native Python WS client would have to reverse-engineer protocol negotiation + token resolution.

## The surprising win

CLI cold start (~7s) **parallelizes for free**. Three concurrent subprocesses finish in ~7.6s — basically the same as one. The OS schedules them onto separate cores and Anthropic's API serves them in parallel. The sweet spot is ~8–10 concurrent before disk/CPU contention starts pulling the line up.

For an article with 60 LLM calls (30 chunks × classifier + tagger):

| Approach | Estimated wall time | Implementation cost |
|---|---|---|
| Sequential subprocess (today) | ~7.2 × 60 = **432s (~7 min)** | already in PR #37 |
| Subprocess + `ThreadPoolExecutor(max_workers=8)` | ~11 × (60/8) = **~83s** | ~10 lines of Python |
| Native Python WS to gateway (no override, dedicated agent) | ~1.5 × (60/8) ≈ **15s** | days: protocol + auth + agent config |
| Native WS + batching (5 chunks/prompt) | ~1.5 × (12/8) ≈ **few seconds** | days + prompt-batching design |

The 5× win from concurrency lands today with trivial code. The further 5× from going native WS costs an order of magnitude more engineering and adds a fragile dep on openclaw's bundle layout.

## What failed / surprised

- `--gateway` flag does **not** skip CLI cold start. It only changes where inference runs (daemon vs in-process SDK). Node bootstrap is the cost, not SDK load.
- `openclaw gateway call agent` with `--model anthropic/claude-haiku-4-5` is rejected because model overrides require admin scope.
- `openclaw/plugin-sdk` does not expose `callGateway` — the deep import path that the CLI uses internally points at unstable bundle chunks.
- Concurrency scales until ~10 parallel processes, after which spawning + scheduling dominates and the run gets slower.

## Recommendation: SHIP concurrency, AVOID the gateway-direct rewrite

For PR #37 follow-up:

1. **Do this now.** Wrap the per-chunk classifier/tagger calls in `concurrent.futures.ThreadPoolExecutor(max_workers=8)` in `classifier.classify_chunks` and `tagger.tag_candidates`. The dispatcher (`_default_llm`) doesn't have to change — the subprocess is thread-safe by construction. Expected wall time: ~7 min → ~80s on a typical article. ~10 lines of code.

2. **Optional next step.** If 80s still hurts, batch 5–10 chunks per prompt and ask the model for a JSON array. Reduces call count proportionally with a modest hit to response stability (longer prompts, occasional malformed arrays to retry on). Easy to layer on top of (1).

3. **Don't do this.** Reverse-engineering the gateway WS protocol to skip CLI cold start saves at most 12% over a well-concurrent subprocess pool and creates an unstable dependency on openclaw bundle internals. The current trade isn't worth it.

## Verdict: PARTIAL

**Question:** Skip CLI cold start to make the dispatcher fast?

**Evidence:** CLI cold start is real (~7s/call) and is the dominant cost. Talking to the gateway daemon directly is *possible* but requires either a dedicated haiku agent in openclaw config (config-only) or reverse-engineered auth/scope handling (engineering-heavy). Either way the savings are bounded by what concurrency already captures.

**What worked:** Confirmed gateway is reachable; mapped the CLI's internal envelope to `gateway call agent`; identified the model-override scope gate.

**What surprised:** CLI invocations parallelize essentially for free up to ~10 concurrent — that gives ~5× throughput with zero protocol work, dwarfing the ~12% upper bound from skipping CLI bootstrap.

**Recommendation:** Adjust — add a small thread pool around the dispatcher in anki-translator, leave the subprocess shape alone, skip the gateway-direct rewrite. Revisit only if a 60-call article still feels slow after concurrency lands.
