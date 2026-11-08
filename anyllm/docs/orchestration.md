# Package A — model orchestration

This file is the feature note for team fan-out, saved targets, budgets, and the
opt-in tool protocol. It is not a README rewrite. Package B (attachments UI,
edits, undo, exports, Markdown rendering, CI) is intentionally not implemented
here.

## Commands

```text
anyllm targets add NAME --provider PROVIDER --model MODEL [--base-url URL]
anyllm targets list
anyllm targets remove NAME
anyllm ask --target NAME "Hello"
anyllm prices set NAME --input 0.20 --output 0.40 --currency USD
anyllm prices list
anyllm team ask --member openscale:model --member ollama:llama3.2 --view stacked "Question"
anyllm team ask --budget-usd 0.02 --max-tokens 512 --yes "Question"
anyllm capabilities probe --target NAME --tools

/target NAME
/targets
/team add NAME provider:model
/team add NAME alias
/team list
/team remove NAME
/team ask [--view stacked|columns] [--merge NAME] [--yes] PROMPT
/team show last
/team adopt NAME
/team adopt merged
/team cancel
/budget set --max-requests N --max-output-tokens N --estimated-usd N
/budget off
/usage --cost
/tools list
/tools enable read_attached_file
/tools disable all
```

`--member` and member specs split on the first colon only, so model ids may
contain slashes and further colons. A saved alias has no colon. Custom base
URLs come from `targets add --base-url`, never from the model string.

`--target` cannot be combined with `--provider`, `--model`, or `--base-url`.

A team ask needs two to four members. Default view is stacked. `columns` is
used only on an interactive terminal at least 120 columns wide; otherwise the
command says so and prints stacked text. `NO_COLOR` prints no ANSI codes.

## What is enforced, and what is only an estimate

- Request count and the `max_tokens` sent on each request are local gates.
- Prices are numbers you saved (per million tokens, with a timestamp). They are
  not downloaded, and they are not a bill.
- Context size before the call is a heuristic of about four characters per
  token. The output says that.
- A dollar gate refuses the run when any target has no price, prices use mixed
  currencies, or there is no output cap to multiply. Unknown is never shown as
  zero.
- After the call, cost uses reported input and output tokens only. Reasoning
  and cached counts are displayed separately and are not added on top.
- There is no hard dollar guarantee. A provider can ignore `max_tokens` or
  count tokens differently.

## Tools

Support is **unknown** until `capabilities probe --target NAME --tools` records
`accepted` or `rejected` for that exact base URL and model. The probe is one
small request and may cost money. It says nothing about other models, and it
does not claim JSON `response_format` support.

`read_attached_file` reads only paths already approved in the process-local
attachment store. It does not scan disk, run a shell, open a network
connection, or write files. A path outside that set returns a denial and asks
for a new approval. Tool rounds are capped at 4. There is no retry that drops
tools and sends a different request.

Anthropic and Gemini have no tool adapter. Those kinds are not probed and are
not sent a tools payload.

## Cancellation

Ctrl-C during `/team ask` stops scheduling further members and stops printing
new tokens. `/team cancel` sets the same flag when a run is active. Streaming
with a deadline or cancel flag reads the body on a daemon thread via the
public `read1` API and the request's own timeout. Bytes already read are
parsed before the deadline wins. A paused read is unblocked with a local
socket shutdown, not by timing out the buffered response and reading it
again. That shutdown does not cancel the provider request, which may still
finish and may still be billed. This is not a hard cancel. Team workers are daemon threads
started with `threading` and `queue`, so Ctrl-C is not stuck on a private
thread-pool implementation.

## Integration seams for Package B

Import `anyllm.seam` rather than copying provider calls or stuffing feature
payloads into `Session.messages`.

| Seam | Use |
| --- | --- |
| `RequestSnapshot` / `make_snapshot` / `snapshot_from_session` | Immutable history, system, sampling, and approved attachments. No API keys. |
| `ApprovedAttachment` and `attachment_store()` | In-memory files the user already approved. Package B should `approve()` after consent. Package A never walks the filesystem. |
| `run_request(provider, target, snapshot, options, *, credential, prompt=..., messages=..., cancel_event=..., event_queue=...)` | One completion. Returns `RunResult` (`text`, `usage`, `status`, `finish_reason`). Does not print. |
| `RunEvent` queue | Workers enqueue status. Only the main thread should render. |
| `Presentation.line` / `status_line` / `warn` / `block` | Main-thread renderer. A Markdown pass can wrap `block` later without changing scheduling. |
| `register_repl(name, handler, help_row)` | REPL dispatch. `handler` is `(repl, rest) -> None`. Help rows are appended by `install_repl_features`. |
| `Session.append_exchange(user, assistant, usage=...)` | Put one finished turn into the live chat without a provider call (`/team adopt`). |
| `load_feature` / `save_feature` | Versioned JSON under the config directory (`targets.json`, `prices.json`, `budget.json`, `tools.json`, `capabilities.json`, `team.json`, `team-runs/`). Not conversation transcripts. Version is `1`. |
| `Target(provider, model, base_url, alias)` | No secrets. Resolve credentials with `targets.resolve_named` or `resolve_member_spec` at send time. |

A target name cannot match a provider name, in any letter case, or normalize
to that provider's `ANYLLM_<PROVIDER>_API_KEY` name. A hand-edited
`targets.json` with such a name is rejected as a whole.

Target credentials live in a separate namespace: `ANYLLM_TARGET_<NAME>_API_KEY`
or `anyllm keys set target:<name>`. That command writes `target:<name>` and
rejects a colliding alias before it touches the key file, so it cannot
overwrite `keys set <provider>`. Target keys are never read from the
provider's saved key or `ANYLLM_<PROVIDER>_API_KEY`. A non-default host
without a target alias, including an old session file that only stored
`base_url`, does not inherit the builtin provider key. Sessions persist
`target` (the alias only, never the secret). `/load` and CLI resume use
`resolve_endpoint_key` and do not accept a provider key for a non-default host.

## Explicit `--base-url` compatibility

`anyllm ask`, `anyllm chat`, and `anyllm models` with `--provider NAME --base-url URL` still send NAME's usual key to that URL. That is the historical proxy and mirror behavior, kept on purpose. It is not host isolation. When URL is not NAME's configured endpoint, the command prints one stderr line and does not include the key. A normalized equivalent of the configured endpoint (host case, a trailing slash, or default port 80/443) is the same host and does not warn. Non-local plain HTTP adds that the key is not encrypted.

That explicit command also sets an in-memory `explicit_endpoint` on the live session, so the interactive REPL's first send and `/models` keep using the provider key for that exact URL. `/save` does not write the marker or the key. `/load`, CLI resume, and a saved target do not inherit it. Switching provider or `/target` clears it. The permission is not reused for a different URL or a different provider. `/info` prints the endpoint on one line and the key state from `describe_live_key`; it does not style or redact the URL (the UI branch wraps that line).

Saved targets, `/models` after a target, same-provider `/provider` on another host, `/load`, and CLI resume do not inherit that provider key. An old session file with a non-default host and no target alias is refused.

Owner question: should a later version require an explicit opt-in, such as a flag or a target key, before a provider key is sent to a different host? This branch does not make that change.

Quotation delimiters are a fresh `ANYLLM-QUOTE-...` token per synthesis, grown
until it does not occur in the quoted text. That avoids a member closing a
fixed marker. It does not guarantee the model will ignore instructions inside
the quotations.

The team budget gate is USD. Prices saved in another currency are not compared
to it and the run is refused. Cancellation stops scheduling. A deadline is
noticed between body chunks without retrying a timed-out buffered read. It
does not kill the provider request. Team workers are daemon threads.

Team runs record target identity, the snapshot id, outputs, usage, and status.
They do not record credentials. Attachment bodies are not stored. Path and
hash are stored only when a caller passes `record_refs=True`.

Package B can run an edit proposal through `run_request` on an isolated
snapshot. It should not depend on tool calling. Exports can read `Session`
and `team-runs/*.json` without the raw attachment text, which is absent by
default.
