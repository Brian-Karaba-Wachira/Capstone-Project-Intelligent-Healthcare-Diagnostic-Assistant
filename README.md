# AI Provider Proxy

A self-hosted gateway that lets Anthropic-speaking clients (Claude Code) and
OpenAI-speaking clients (Cursor, generic SDKs) talk to local llama.cpp-style
workers through one endpoint, with full tool-calling and streaming support in
both dialects.

## Components

| Directory | Crate | Role |
|---|---|---|
| [`src-ai-provider-converter/`](src-ai-provider-converter) | `ai-provider-converter` | Pure translation library: Anthropic ↔ OpenAI requests, responses, and SSE streams. No HTTP, no I/O policy — just shapes. |
| [`src_brain/`](src_brain) | `brain` | The proxy/gateway. Terminates client HTTP, auto-detects the API dialect, translates via the converter, routes to workers over a WS tunnel, streams results back. Also: auth/API keys, idempotency, metrics, hosted `web_search`/`web_fetch` tools. |
| [`src_lmodel/`](src_lmodel) | `lmodel` | Worker-side bridge that sits in front of llama.cpp (or any OpenAI-shaped server) and connects it to brain. |

Request flow for Claude Code:

```
Claude Code ──(Anthropic Messages API)──► brain ──(translate)──► internal TaskBundle
     ▲                                                                │
     └──(Anthropic SSE events)◄── brain ◄──(OpenAI SSE deltas)◄── worker (llama.cpp)
```

---

## Changelog — translation-layer overhaul (July 2026)

This round fixed the broken Anthropic ↔ OpenAI translation logic that was
corrupting Claude Code requests/responses, made both streaming directions
protocol-complete, and raised the per-request stream buffer.

> **Testing status:** these changes were verified by careful code review and
> ship with 12 new regression tests, but the development machine had no Rust
> toolchain, so they have **not been compiled or executed locally yet**. Run
> the [verification steps](#verification) on your build machine before
> deploying.

### Fixed — build

- **`src_brain` pointed at a converter copy that doesn't exist.**
  `src_brain/Cargo.toml` depended on `../../proxy/ai-provider-converter`, a
  path not present on disk, so the proxy could not build against the library
  at all (and any past builds used a different, stale copy). It now points at
  the in-repo [`src-ai-provider-converter`](src-ai-provider-converter).

### Fixed — request translation, Anthropic → OpenAI
(`src-ai-provider-converter/src/utils/translate.rs` — the path every Claude Code request takes)

- **`thinking` / `redacted_thinking` blocks no longer leak into the prompt.**
  Claude Code replays its thinking blocks on every turn; the old catch-all
  arm serialized them as raw JSON *into the assistant text*, polluting the
  worker's context. They are now dropped, and unknown block types contribute
  only their `text` field instead of raw JSON.
- **Tool results containing images no longer emit base64 JSON garbage.**
  Image blocks inside a `tool_result` become an `[image omitted]`
  placeholder; text blocks are extracted cleanly.
- **`stop_sequences` is now mapped to OpenAI `stop`** instead of being
  silently dropped by the library.
- **Assistant turns with no text and no tool calls** produced
  `content: null` without `tool_calls` — rejected by most OpenAI-compatible
  backends. They now carry an empty string.

### Fixed — request translation, OpenAI → Anthropic

- **`role: "tool"` messages were forwarded with the literal role `"tool"`**,
  which the Anthropic API rejects outright. They are now converted to
  `tool_result` blocks inside a user message.
- **Consecutive same-role messages are merged into one turn** (e.g. several
  tool results after a parallel tool call), satisfying Anthropic's
  user/assistant alternation requirement.
- `developer` role → system prompt; OpenAI `stop` → `stop_sequences`;
  non-data image URLs now use Anthropic's `source: {type: "url"}` instead of
  degrading to text.

### Fixed — response translation, OpenAI → Anthropic

- **`tool_use` blocks could lose their required `input` field entirely.**
  Empty or unparsable `arguments` produced `input: None`, which serde then
  *omitted from the JSON* — Claude Code rejects a `tool_use` block with no
  `input`. It now always serializes, defaulting to `{}`. Missing tool-call
  `id`s get a generated `toolu_*` fallback.

### Fixed — streaming, Anthropic → OpenAI
(`src-ai-provider-converter/src/stream/anthropic/parser.rs`)

- **Wrong stop-reason mapping:** a private, hand-copied map sent
  `stop_sequence` → `content_filter` (falsely claiming moderation fired on a
  clean stop) and contradicted the shared mapper in
  `utils/finish_reason.rs`. The shared mapper is now the single source of
  truth.
- **Tool-call indices are remapped to start at 0.** Anthropic block indices
  count *all* content blocks (a leading text block pushes the first tool to
  index 1); OpenAI `tool_calls[].index` counts tool calls only.
- **A stream that dies before `message_stop` now still terminates properly**
  with a final finish chunk and `data: [DONE]`, instead of leaving the
  client hanging until its own timeout.
- CRLF (`\r\n\r\n`) SSE event framing is handled, including a CR/LF pair
  split across two network chunks.
- Upstream `error` events are forwarded as OpenAI-style
  `{"error": {...}}` frames instead of being swallowed; `ping` events pass
  through as SSE comments so idle connections stay alive.

### Fixed — streaming, OpenAI → Anthropic
(`src-ai-provider-converter/src/stream/openai/mod.rs` — the Claude Code response shape)

- **Content-block framing is now strictly sequential**, as the Anthropic
  protocol requires: the text block is stopped *before* a tool block starts,
  and text arriving *after* a tool call opens a fresh block instead of being
  appended to one that conceptually ended. Previously the text block stayed
  open across tool blocks — invalid event ordering.
- **Streams that end without a `finish_reason` default to
  `stop_reason: "end_turn"`** instead of `null`, which Anthropic clients
  treat as malformed.
- Worker `{"error": ...}` frames become proper Anthropic `error` SSE events.
- `"usage": null` chunks (OpenAI `include_usage` mode sends one on every
  chunk) no longer zero out token counts already captured.
- Same CRLF framing fix as the other direction.

### Changed — stream buffer limit (`src_brain`)

The per-request worker→client stream channel was hardcoded at **256** chunks
in both the Anthropic and OpenAI handlers. It is now configurable and
defaults to **1024**:

```
BRAIN_STREAM_BUFFER_CHUNKS=1024   # default; any positive integer
```

The channel stays bounded on purpose — that is the memory protection that
keeps a disconnected client from making brain buffer an entire response —
it's just four times deeper, so a fast worker bursting tokens doesn't stall
on a briefly slow client socket. Set it higher for very fast workers on slow
links, lower for memory-constrained deployments.

### Added — regression tests

12 new tests in the converter covering each fix:

- `utils/translate.rs`: thinking-block exclusion, empty-assistant content,
  `stop_sequences` mapping, image-in-tool_result sanitization, tool-role →
  `tool_result` conversion + same-role merging, guaranteed `tool_use.input`.
- `stream/anthropic/parser.rs`: tool-index remapping, `stop_sequence`
  mapping, truncated-stream `[DONE]` guarantee.
- `stream/openai/mod.rs`: sequential block framing (text closed before tool
  opens), CRLF framing + default `end_turn`, error-frame passthrough.

The converter gained one **dev-only** dependency (`futures = "0.3"`) for the
stream tests; runtime dependencies are unchanged.

---

## Verification

On a machine with a Rust toolchain:

```sh
# Library: run the full test suite (unit + the 12 new regression tests)
cd src-ai-provider-converter
cargo test

# Proxy: confirm it compiles against the fixed dependency path
cd ../src_brain
cargo check
```

Note that `brain` uses monoio and is intended to build/run on Linux.

End-to-end smoke test with Claude Code:

```sh
ANTHROPIC_BASE_URL=http://<brain-host>:9999 \
ANTHROPIC_API_KEY=<your BRAIN_PSK or issued key> \
claude
```

Then exercise the paths that were broken: a multi-turn conversation with
tool use (Read/Write/Bash), a turn that returns text *after* a tool call,
and a long generation to confirm streaming stays smooth and terminates with
a proper `message_stop`.

## Key configuration (brain)

| Env var | Default | Purpose |
|---|---|---|
| `BRAIN_PSK` | *(required)* | Pre-shared key; usable as a bearer token by default |
| `BRAIN_ADDR` | `127.0.0.1:9999` | Listen address |
| `BRAIN_STREAM_BUFFER_CHUNKS` | `1024` | Per-request stream channel capacity *(new)* |
| `BRAIN_WORKER_TIMEOUT` | `120` | Seconds of worker silence before 504 |
| `BRAIN_MODEL_ALIASES` | — | `alias=real-model,...` routing map |
| `BRAIN_DEFAULT_CTX` | `32768` | Fallback context window for workers reporting 0 |
| `BRAIN_SEARXNG_URL` | — | Enables hosted `web_search`/`web_fetch` |
| `BRAIN_EGRESS_MODE` / `BRAIN_EGRESS_DOMAINS` | `deny_all` | `web_fetch` egress policy |
