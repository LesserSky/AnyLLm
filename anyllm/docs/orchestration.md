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
new tokens. `/team cancel` sets the same flag when a run is active. A socket
read that has already started cannot be killed, so that provider call may
still finish and may still be billed. Per-member timeouts and the overall
deadline bound the wait. This is not a hard cancel.

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

A target whose URL is not the provider's configured host does not receive that
provider's usual key. Set `ANYLLM_<ALIAS>_API_KEY` or `anyllm keys set ALIAS`.
The built-in `custom` provider (empty default URL) still uses the normal custom
key rules, including `ANYLLM_API_KEY`.

Team runs record target identity, the snapshot id, outputs, usage, and status.
They do not record credentials. Attachment bodies are not stored. Path and
hash are stored only when a caller passes `record_refs=True`.

Package B can run an edit proposal through `run_request` on an isolated
snapshot. It should not depend on tool calling. Exports can read `Session`
and `team-runs/*.json` without the raw attachment text, which is absent by
default.
