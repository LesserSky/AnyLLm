# AnyLLM

One small terminal client for any LLM API key.

Save a key once, then chat, pipe prompts, switch models, and keep conversations
without remembering each provider's API shape. It speaks OpenAI, Anthropic
Claude, Google Gemini, and any OpenAI-compatible endpoint.

The client lives in [`anyllm/`](anyllm/) and uses the Python 3.10+ standard
library only. Run the commands below from that directory.

## Features

- Streaming answers, with a spinner until the first token on a real terminal.
- Reasoning text is dim and labeled, so it stays distinct from the answer.
- A compact status line shows elapsed time and, when the provider reports
  usage, input, output, reasoning, and cached tokens.
- Startup banner shows the provider, model, and endpoint. Secrets in the URL
  are redacted.
- Saved conversations, a provider catalog, and per-provider API keys.
- Keys are encrypted with Windows DPAPI, or stored in a `0600` file elsewhere.
- Plain, parseable output when stdout is not a terminal, when `NO_COLOR` is
  set, with `--no-color`, or on an older Windows console without ANSI support.
- Custom providers for vLLM, LiteLLM, Azure, a company gateway, or anything
  else that implements a supported chat API.

## Quick start

### Windows

From the `anyllm/` directory:

```powershell
.\anyllm.bat keys set openai
.\anyllm.bat chat
```

Or with Python directly:

```powershell
python -m anyllm keys set openai
python -m anyllm chat
```

### Linux / macOS

```sh
chmod +x anyllm.sh
./anyllm.sh keys set openai
./anyllm.sh chat
```

Or with Python directly:

```sh
python3 -m anyllm keys set openai
python3 -m anyllm chat
```

`anyllm` with no arguments opens the chat. The first time no key is configured,
a terminal session asks for a provider and a key, saves them, and starts the
chat. Pipes and scripts skip that prompt and exit with the same error text as
before, including a hint for `anyllm keys set`.

If the program exits with an error, the Windows launcher keeps the window open
so the message does not disappear.

### Optional: make `anyllm` a command

`anyllm.bat` is in `anyllm/`. Adding the directory to PATH gives you an
`anyllm` command in PowerShell and cmd. Open a new terminal after changing
PATH, then run `anyllm --version`.

`python -m pip install --user .` also installs a command. The Python scripts
directory must be on PATH. The launcher works without installing anything.

## A session

On a terminal the banner is colored and a spinner runs until the first token.
`NO_COLOR=1`, `--no-color`, a pipe, or an old Windows console shows this plain
layout instead:

```text
$ anyllm chat

anyllm 1.0.0
openai/gpt-4o-mini
https://api.openai.com/v1
Type /help for commands, /exit to quit.

you> why is the sky blue?
reasoning:
Sunlight scatters more at short wavelengths.
Blue light is scattered toward your eyes.
  1.1s | 20 in | 14 out | 6 reasoning

you> /exit
bye
```

The prompt is `you>`. A color terminal paints that label and draws the status
line with middle dots (`1.1s · 20 in · 14 out`). The answer itself stays
normal text. Reasoning, when you ask for it, is a `reasoning:` block in dim
italic.

End a line with `\` to continue it. Send a message that starts with `/` as
`//text`. Ctrl+C stops a streaming answer and keeps the partial reply.
Ctrl+C at the prompt clears the line. Ctrl+D exits.

## Chat commands

Inside `anyllm chat`:

| Command | Action |
| --- | --- |
| `/help` | Show the command list |
| `/model NAME` | Switch model for this conversation |
| `/models [TEXT]` | List models available to the current key |
| `/provider [NAME]` | Switch provider; history is kept |
| `/system [TEXT]` | Show or set the system prompt (`-` clears it) |
| `/temp [VALUE]` | Show or set temperature (`off` clears it) |
| `/max [N]` | Show or set the max response tokens (`off` clears it) |
| `/reasoning on\|off` | Show or hide a model's thinking stream |
| `/clear` | Start a new conversation |
| `/usage` | Session token totals |
| `/info` | Connection details, without showing the key |
| `/save [NAME]` | Save the conversation |
| `/load [NAME]` | Load a saved conversation (latest if no name) |
| `/sessions` | List saved conversations |
| `/keys` | Show where the current key comes from |
| `/exit` | Quit |

`/models`, `/provider`, `/sessions`, and `/help` print aligned tables. In a
color terminal the name column is highlighted and a dim rule sits under the
header. Plain mode is ASCII columns with the same text.

## One-shot prompts and pipes

`anyllm ask` prints the answer and nothing else when stdout is not a terminal:
no banner, spinner, color, or status line. Timing is shown on the terminal
only, and on stderr, so a redirect captures the answer alone.

```powershell
anyllm ask "explain git rebase in three lines"
```

```sh
cat error.log | anyllm ask "what is failing here?"
git diff | anyllm ask -m claude-haiku-4-5 "review this change"
```

`anyllm chat` with piped input answers once and exits:

```sh
echo "summarize this" | anyllm
```

Machine-readable output:

```powershell
anyllm ask --json "hi"
```

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "text": "Hi! How can I help?",
  "usage": {"input": 8, "output": 9, "total": 17, "reasoning": null, "cached": null},
  "finish_reason": "stop",
  "elapsed_seconds": 1.2,
  "base_url": "https://api.openai.com/v1"
}
```

`--no-stream` waits for the whole answer before printing. `--no-color` and
`NO_COLOR=1` force plain text even on a terminal. `FORCE_COLOR=1` requests
color, but piped `ask` / `chat` stdout stays plain so it can be parsed.
`NO_COLOR` wins over `FORCE_COLOR`.

## Save a key

```powershell
anyllm keys set openai
anyllm keys set anthropic
anyllm keys set lab
```

AnyLLM checks, in order:

1. `ANYLLM_<PROVIDER>_API_KEY` (for example `ANYLLM_OPENAI_API_KEY`)
2. The provider's usual variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...)
3. `ANYLLM_API_KEY`
4. The key saved with `anyllm keys set`

Environment variables win over saved keys. Set one in the current shell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

```sh
export OPENAI_API_KEY="sk-..."
```

`anyllm keys list` and `anyllm doctor` show where a key comes from. They never
print the key itself. Saved keys are masked.

## Providers

```powershell
anyllm providers list
anyllm models openai
```

| Name | Service | Default model |
| --- | --- | --- |
| `openai` | OpenAI | `gpt-4o-mini` |
| `anthropic` | Anthropic Claude | `claude-sonnet-4-5` |
| `google` | Google Gemini | `gemini-2.5-flash` |
| `deepseek` | DeepSeek | `deepseek-chat` |
| `groq` | Groq | `llama-3.3-70b-versatile` |
| `mistral` | Mistral | `mistral-large-latest` |
| `xai` | xAI Grok | `grok-3` |
| `openrouter` | OpenRouter | `openrouter/auto` |
| `openscale` | OpenScale | `deepseek/deepseek-v4.1-flash` |
| `together` | Together AI | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `ollama` | Ollama on this machine | `llama3.2` |
| `lmstudio` | LM Studio on this machine | set with `-m` |
| `custom` | Any OpenAI-compatible URL | set with `-m` |

Starter defaults are suggestions. Model names change; `anyllm models <provider>`
lists what your key can use. Save a default with:

```powershell
anyllm config set provider openai
anyllm config set model gpt-4o-mini
```

Ollama and LM Studio need no key. Start their servers, then use
`-p ollama -m llama3.2`.

## Custom endpoints

```powershell
anyllm ask -p openai --base-url http://localhost:8000/v1 -m my-model "hello"
```

A named provider:

```powershell
anyllm providers add lab `
  --base-url http://localhost:8000/v1 `
  --model my-model `
  --env LAB_API_KEY

anyllm keys set lab
anyllm chat -p lab
```

| Option | Meaning |
| --- | --- |
| `--kind openai\|anthropic\|gemini` | Wire protocol (default `openai`) |
| `--base-url URL` | Endpoint root, usually ending in `/v1` |
| `--model NAME` | Suggested model; repeatable |
| `--env VAR` | Extra environment variable; repeatable |
| `--no-key` | Endpoint needs no key |
| `--auth-header HEADER` | Header carrying the key |
| `--auth-prefix TEXT` | Text before the key, usually `"Bearer "` |
| `--label TEXT` | Display name |

Usernames, passwords, and query values such as `api_key` or `token` are
redacted in the banner, `/info`, and `providers show`. `api-version` is kept.
The banner and those screens never print the API key.

Azure-style endpoints put the deployment and API version in the base URL and
use the `api-key` header:

```powershell
anyllm providers add azure `
  --base-url "https://RESOURCE.openai.azure.com/openai/deployments/DEPLOYMENT?api-version=2024-10-21" `
  --auth-header api-key --auth-prefix "" `
  --model DEPLOYMENT
anyllm keys set azure
anyllm chat -p azure
```

## Command reference

```text
anyllm [command]

  chat       interactive conversation (default)
  ask        one prompt, printed answer
  models     list models available to a key
  providers  list, add, show, or remove providers
  keys       set, list, remove keys; print the key file path
  config     show, get, set, or unset defaults; print the config path
  doctor     show key sources and optionally test providers
```

Common options for `chat` and `ask`:

```text
-p, --provider NAME      provider from `anyllm providers list`
-m, --model NAME         model name
-s, --system TEXT        system prompt
--base-url URL           override the endpoint
--temperature N          0 to 2
--max-tokens N           limit the response length
--timeout SECONDS        network timeout (default 120)
--reasoning              show thinking tokens when the provider sends them
--no-color               disable ANSI colors
```

`ask` also accepts `--json` and `--no-stream`.

```powershell
anyllm chat -p anthropic -m claude-haiku-4-5
anyllm ask -m deepseek-reasoner --reasoning "why is the sky blue?"
anyllm chat --continue
anyllm chat --load notes
anyllm doctor --test
```

## Where things are stored

| Platform | Directory |
| --- | --- |
| Windows | `%APPDATA%\anyllm` |
| Linux / macOS | `~/.config/anyllm` |

`ANYLLM_HOME` selects another directory (portable copies and tests).

```text
config.json       defaults and custom providers (no API keys)
keys.json         API keys, DPAPI on Windows, mode 0600 elsewhere
sessions/*.json   saved conversations, mode 0600
input-history     readline prompt history
```

`anyllm config path` and `anyllm keys path` print the exact locations.

## Security

- On Windows, saved keys are encrypted with DPAPI for the current user.
  Copying `keys.json` to another machine or user does not reveal them.
- On Linux and macOS the key file is mode `0600` in a `0700` directory. It is
  not encrypted there; only your user can read it.
- Keys are not shown by `keys list`, `doctor`, logs, the banner, or `/info`.
  Only `keys set` writes them, from a hidden prompt or one line of stdin.
- Prompts and replies go to the provider you select. AnyLLM has no telemetry
  and no server of its own.
- Saved conversations contain your prompts. Delete `sessions/` to drop them.

## Troubleshooting

Errors look like `error: ...` followed by an indented hint. A color terminal
prints the summary in red, or yellow for rate limits, network failures, and
usage mistakes. The hint text is unchanged.

**The window opens and closes immediately.** Run it from PowerShell, or use
`anyllm.bat`, which waits after an error. The usual cause is a missing key:
`anyllm.bat keys set openai`. If Python is missing, the launcher says so.

**`No API key found for ...`.** `anyllm keys set <provider>`, or set the
environment variable named in the hint. `anyllm doctor` shows which providers
are ready.

**`authentication failed (HTTP 401)`.** The key was rejected. Watch for a
trailing space, or an old value in the environment. Environment variables
override saved keys.

**`endpoint or model not found (HTTP 404)`.** Check the model and base URL.
`anyllm models <provider>` lists names. Custom base URLs often need `/v1`.

**`rate limited (HTTP 429)`.** Wait, check the quota, or switch provider.

**`could not reach ...`.** Check the network, VPN, proxy, and base URL.
`HTTPS_PROXY` and `NO_PROXY` are honored. Raise `--timeout` on a slow link.

**Colors look wrong, or a pipe contains escape codes.** Use `--no-color` or
`NO_COLOR=1`. Piped `ask` output is plain text even if `FORCE_COLOR` is set.
On very old Windows consoles, AnyLLM detects that virtual terminal processing
is unavailable and stays plain. Windows Terminal has working color.

**The default model is gone.** Pass `-m` and save it with
`anyllm config set model NAME`.

**A model rejects `temperature` or `max_tokens`.** Leave those unset. The
request omits them.

## What it does not do

- Image, audio, or file uploads; embeddings; tool calling.
- Token price estimates.
- Starting Ollama or LM Studio. Those servers must already be running.
- Multi-user key sharing. Keys are per user.

## Development

From the `anyllm/` directory:

```sh
python3 -m unittest discover -s tests -v
```

The suite starts a local mock provider for the OpenAI, Anthropic, and Gemini
streaming formats. It needs no network access and no real API keys.

```text
anyllm/            package (CLI, providers, transport, REPL, terminal theme)
tests/             offline tests, mock API, and tests/render_ui_demo.py
anyllm.bat         Windows launcher
anyllm.sh          Linux/macOS launcher
pyproject.toml     optional pip installation
```

`python3 tests/render_ui_demo.py` prints a plain and ANSI transcript of the
banner, a reply with reasoning and token status, the tables, and an error.
