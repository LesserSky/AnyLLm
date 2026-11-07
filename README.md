# AnyLLM

One small terminal client for **any LLM API key**.

Save a key once, then chat, Type prompts, switch models, and keep conversations
without remembering each provider's API shape. It speaks OpenAI, Anthropic
Claude, Google Gemini, and any OpenAI-compatible endpoint.

- No dependencies. Python 3.10+ standard library only.
- Streaming answers, token counts, and saved conversations.
- Keys encrypted with Windows DPAPI, or stored in a `0600` file elsewhere.
- Custom providers for vLLM, LiteLLM, Azure, a company gateway, or anything
  else that implements the OpenAI chat API.

> This folder is self-contained. It does not use the Pulse monitor project that
> happens to sit next to it.

## Quick start

### Windows

Run from this folder:

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
it asks for a provider and a key, saves them, and starts the chat. The assistant
streams its answer, then shows elapsed time and token usage.

If the program exits with an error, the Windows launcher keeps the window open
so the message does not disappear.

### Optional: make `anyllm` a command

The folder already contains `anyllm.bat`, so adding this folder to your PATH
gives you an `anyllm` command in both PowerShell and cmd. After changing PATH,
open a new terminal and run:

```powershell
anyllm --version
```

You can also install the package (`python -m pip install --user .`), but the
Python scripts folder must then be on PATH for the installed command to be
found. The launcher works without installing anything.

## Save a key

```powershell
anyllm keys set openai          # prompts with hidden input
anyllm keys set anthropic
anyllm keys set lab             # a custom provider
```

You can also use environment variables. AnyLLM checks, in order:

1. `ANYLLM_<PROVIDER>_API_KEY` (for example `ANYLLM_OPENAI_API_KEY`)
2. The provider's usual variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...)
3. `ANYLLM_API_KEY` (useful for a custom endpoint)
4. The key saved with `anyllm keys set`

Environment variables win over saved keys, which keeps CI predictable. Set
them in the current shell like this:

```powershell
# PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

```sh
# bash / zsh
export OPENAI_API_KEY="sk-..."
```

Inspect what the tool sees with:

```powershell
anyllm keys list
anyllm doctor
```

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
| `/reasoning on\|off` | Show or hide a model's thinking stream when available |
| `/clear` | Start a new conversation |
| `/usage` | Session token totals |
| `/info` | Connection details, without showing the key |
| `/save [NAME]` | Save the conversation |
| `/load [NAME]` | Load a saved conversation (latest if no name) |
| `/sessions` | List saved conversations |
| `/keys` | Show where the current key comes from |
| `/exit` | Quit |

End a line with `\` to continue it on the next line. A message that must start
with `/` can be sent as `//text`.

Ctrl+C stops a streaming answer and keeps the partial reply in the
conversation. Ctrl+C at the prompt clears the line; Ctrl+D exits.

## One-shot prompts and pipes

Print just the answer, with no headers:

```powershell
anyllm ask "explain git rebase in three lines"
```

```sh
cat error.log | anyllm ask "what is failing here?"
git diff | anyllm ask -m claude-haiku-4-5 "review this change"
```

`anyllm chat` with piped input also answers once and exits, so this works:

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

Use `--no-stream` to wait for the whole answer before printing. Use
`--no-color` (or set `NO_COLOR=1`) for plain output.

## Providers

```powershell
anyllm providers list
anyllm models openai           # ask the API which models the key can use
```

| Name | Service | Default model* |
| --- | --- | --- |
| `openai` | OpenAI | `gpt-4o-mini` |
| `anthropic` | Anthropic Claude | `claude-sonnet-4-5` |
| `google` | Google Gemini | `gemini-2.5-flash` |
| `deepseek` | DeepSeek | `deepseek-chat` |
| `groq` | Groq | `llama-3.3-70b-versatile` |
| `mistral` | Mistral | `mistral-large-latest` |
| `xai` | xAI Grok | `grok-3` |
| `openrouter` | OpenRouter | `openrouter/auto` |
| `openscale` | OpenScale (DeepSeek, GLM) | `deepseek/deepseek-v4.1-flash` |
| `together` | Together AI | `Llama-3.3-70B-Instruct-Turbo` |
| `ollama` | Ollama on this machine | `llama3.2` |
| `lmstudio` | LM Studio on this machine | set with `-m` |
| `custom` | Any OpenAI-compatible URL | set with `-m` |

\* Starter defaults are only suggestions. Model names change; run
`anyllm models <provider>` to see what your key can actually use. If a default
disappears, pass a current one with `-m` or save it:

```powershell
anyllm config set provider openai
anyllm config set model gpt-4o-mini
```

Ollama and LM Studio need no key. Use `-p ollama -m llama3.2` (or set it as the
default) while their local servers are running.

## Custom endpoints

Any OpenAI-compatible base URL works directly:

```powershell
anyllm ask -p openai --base-url http://localhost:8000/v1 -m my-model "hello"
```

For a permanent, named provider:

```powershell
anyllm providers add lab `
  --base-url http://localhost:8000/v1 `
  --model my-model `
  --env LAB_API_KEY

anyllm keys set lab
anyllm chat -p lab
```

`providers add` options:

| Option | Meaning |
| --- | --- |
| `--kind openai\|anthropic\|gemini` | Wire protocol (default `openai`) |
| `--base-url URL` | Endpoint root, usually ending in `/v1` |
| `--model NAME` | Suggested/default model; repeatable |
| `--env VAR` | Extra environment variable to check; repeatable |
| `--no-key` | Endpoint needs no key |
| `--auth-header HEADER` | Header carrying the key (default per kind) |
| `--auth-prefix TEXT` | Text before the key, usually `"Bearer "` |
| `--label TEXT` | Display name |

Azure-style endpoints work by putting the deployment and API version in the
base URL and using the `api-key` header:

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
  config     show/set/unset defaults; print the config path
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
--reasoning              show thinking tokens when supported
--no-color               disable ANSI colors
```

Examples:

```powershell
anyllm chat -p anthropic -m claude-haiku-4-5
anyllm ask -m deepseek-reasoner --reasoning "why is the sky blue?"
anyllm chat --continue                 # resume the latest saved conversation
anyllm chat --load notes               # resume a named conversation
anyllm doctor --test                   # send a tiny request to each configured provider
```

## Where things are stored

| Platform | Directory |
| --- | --- |
| Windows | `%APPDATA%\anyllm` |
| Linux / macOS | `~/.config/anyllm` |

Set `ANYLLM_HOME` to use another directory (portable installs, tests).

```text
config.json       defaults and custom providers (no secrets)
keys.json         API keys, DPAPI-encrypted on Windows / mode 0600 elsewhere
sessions/*.json   saved conversations, mode 0600
input-history     readline prompt history (Linux/macOS)
```

`anyllm config path` and `anyllm keys path` print the exact locations.

## Security notes

- Saved keys are encrypted with **Windows DPAPI** for the current Windows user.
  Copying `keys.json` to another machine or user makes it unreadable; set the
  key again there.
- On Linux/macOS the key file uses permissions `0600` in a `0700` directory.
  The key is not encrypted there; the file is simply readable only by your
  user.
- Keys are never shown in `keys list`, `doctor`, logs, or JSON output. Only
  `keys set` writes them, from a hidden prompt or one line of stdin.
- Prompts and replies go directly to the provider you select. AnyLLM has no
  telemetry, no account, and no server of its own.
- Saved conversations contain your prompts. They are written with owner-only
  permissions on POSIX; delete `sessions/` if you do not want them kept.

## Troubleshooting

**The window opens and closes immediately** — the app exited before you could
read the message. Run it from PowerShell to see the text, or launch
`anyllm.bat` again: it now pauses on errors. The usual cause is that no API key
is configured yet; run `anyllm.bat keys set openai`, or launch it and answer the
setup prompts. If Python itself is missing, the launcher now says so and waits
for a key press.

**`No API key found for ...`** — run `anyllm keys set <provider>`, or set the
environment variable shown. `anyllm doctor` shows which providers are ready.

**`authentication failed (HTTP 401)`** — the key was rejected. Check for a
trailing space or an old value in the environment; environment variables take
precedence over saved keys.

**`endpoint or model not found (HTTP 404)`** — the model name or base URL is
wrong. `anyllm models <provider>` lists valid model names. For a custom
endpoint, make sure `--base-url` includes the right prefix, often `/v1`.

**`rate limited (HTTP 429)`** — wait, check your quota, or switch provider or
model.

**`could not reach ...`** — check the connection, VPN, proxy, and base URL.
Proxy environment variables (`HTTPS_PROXY`, `NO_PROXY`) are honored. Use
`--timeout` for slow connections.

**Colors look wrong** — use `--no-color` or set `NO_COLOR=1`. On very old
Windows consoles, enable Virtual Terminal Processing by using Windows Terminal.

**The default model is no longer available** — starter defaults are hints, not
guarantees. Pick a current model with `-m` and save it with
`anyllm config set model NAME`.

**A model rejects `temperature` or `max_tokens`** — some reasoning models do.
Leave those options unset and the request omits them.

## What it does not do

- Image, audio, or file uploads; embeddings; tool/function calling.
- Token cost estimates (provider pricing and plans vary too much).
- Local model management (Ollama and LM Studio must already be running).
- Multi-user or shared-server operation. Keys are per-user.

## Development

Run the tests from this folder:

```powershell
python -m unittest discover -s tests -v
```

The suite starts a local mock provider that speaks the OpenAI, Anthropic, and
Gemini streaming formats, so it needs no network access and no real API keys.

Files:

```text
anyllm/            package (CLI, providers, transport, REPL)
tests/             offline tests + mock API server
anyllm.bat         Windows launcher
anyllm.sh          Linux/macOS launcher
pyproject.toml     optional pip installation
```
