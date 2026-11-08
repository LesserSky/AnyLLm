"""Opt-in tool calls for one read-only attachment tool.

Tool support is unknown until ``capabilities probe --tools`` records evidence
for that exact base URL and model. A probe is a small request that may be
billed. It does not say anything about other models on the same provider.

The only built-in tool reads attachment text already approved in this process.
It does not open arbitrary paths, and it does not touch the shell, the network,
or the disk. There is no silent retry that drops tools and changes the request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .catalog import OPENAI
from .errors import ConfigError, UsageError
from .seam import (
    RunOptions,
    RunResult,
    feature_path,
    load_feature,
    messages_for,
    register_repl,
    run_request,
    save_feature,
)

READ_ATTACHED_FILE = "read_attached_file"
MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS_PER_ROUND = 8
ARGUMENT_BYTE_CAP = 8 * 1024
RESULT_BYTE_CAP = 32 * 1024

_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": READ_ATTACHED_FILE,
        "description": (
            "Read one file the user already approved for this request. "
            "Any other path is denied and nothing is read from disk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Exact path of an approved attachment.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


class ToolInputError(UsageError):
    """The model supplied a tool call this client will not run."""


@dataclass
class AssembledCall:
    index: int
    id: str
    name: str
    arguments: str


def openai_tool_payload(names: list[str]) -> list[dict]:
    payload = []
    for name in names:
        if name != READ_ATTACHED_FILE:
            raise UsageError(
                f"Unknown tool {name!r}.",
                hint=f"The only tool is {READ_ATTACHED_FILE}.",
            )
        payload.append(_TOOL_SCHEMA)
    return payload


def tools_path():
    return feature_path("tools.json")


def capabilities_path():
    return feature_path("capabilities.json")


def enabled_tools() -> list[str]:
    data = load_feature(tools_path(), what="tools")
    enabled = data.get("enabled") or []
    if not isinstance(enabled, list) or not all(isinstance(item, str) for item in enabled):
        raise ConfigError("tools.json is malformed.", hint="The enabled list must be an array of names.")
    return list(enabled)


def enable_tool(name: str) -> None:
    if name != READ_ATTACHED_FILE:
        raise UsageError(
            f"Unknown tool {name!r}.",
            hint=f"Enable {READ_ATTACHED_FILE} only. Other tools are not implemented.",
        )
    current = enabled_tools()
    if name not in current:
        current.append(name)
    save_feature(tools_path(), {"enabled": current})


def disable_all_tools() -> None:
    save_feature(tools_path(), {"enabled": []})


def _capability_id(base_url: str, model: str) -> str:
    return f"{base_url}\n{model}"


def capability_record(base_url: str, model: str) -> dict | None:
    data = load_feature(capabilities_path(), what="capabilities")
    probes = data.get("probes") or {}
    if not isinstance(probes, dict):
        raise ConfigError("capabilities.json is malformed.")
    record = probes.get(_capability_id(base_url, model))
    return record if isinstance(record, dict) else None


def _store_capability(record: dict) -> None:
    data = load_feature(capabilities_path(), what="capabilities")
    probes = data.get("probes")
    if not isinstance(probes, dict):
        probes = {}
    probes[_capability_id(record["base_url"], record["model"])] = record
    save_feature(capabilities_path(), {"probes": probes})


def require_tool_capability(base_url: str, model: str) -> dict:
    record = capability_record(base_url, model)
    if not record or record.get("tools") != "accepted":
        state = "unknown" if not record else record.get("tools")
        raise UsageError(
            f"Tool support is {state} for {model} at {base_url}.",
            hint="Run `anyllm capabilities probe --target NAME --tools`. "
            "That is one small request which may be billed, and it applies only "
            "to that exact endpoint and model. Nothing is assumed from the provider name.",
        )
    return record


def assemble_tool_calls(deltas) -> list[AssembledCall]:
    """Join streamed fragments. Duplicate ids are rejected, not merged."""

    order: list[int] = []
    slots: dict[int, dict] = {}
    for delta in deltas or []:
        index = int(delta.index)
        if index not in slots:
            slots[index] = {"id": "", "name": "", "arguments": ""}
            order.append(index)
        slot = slots[index]
        incoming_id = getattr(delta, "id", "") or ""
        if incoming_id:
            if slot["id"] and slot["id"] != incoming_id:
                raise ToolInputError(
                    f"Tool call index {index} changed id from {slot['id']!r} to {incoming_id!r}."
                )
            slot["id"] = incoming_id
        name = getattr(delta, "name", "") or ""
        if name:
            slot["name"] += name
        arguments = getattr(delta, "arguments", "") or ""
        if arguments:
            slot["arguments"] += arguments
            if len(slot["arguments"].encode("utf-8")) > ARGUMENT_BYTE_CAP:
                raise ToolInputError(
                    "Tool arguments exceeded 8192 bytes while they were still streaming."
                )
        if len(slots) > MAX_TOOL_CALLS_PER_ROUND:
            raise ToolInputError(
                f"More than {MAX_TOOL_CALLS_PER_ROUND} tool calls arrived in one round."
            )
    calls = []
    seen: set[str] = set()
    for index in order:
        slot = slots[index]
        if not slot["id"] or not slot["name"]:
            raise ToolInputError(f"Tool call {index} is missing an id or a name.")
        if slot["id"] in seen:
            raise ToolInputError(f"Duplicate tool call id {slot['id']!r}.")
        seen.add(slot["id"])
        calls.append(
            AssembledCall(
                index=index,
                id=slot["id"],
                name=slot["name"],
                arguments=slot["arguments"],
            )
        )
    return calls


def parse_path_argument(raw: str) -> str:
    if raw is None:
        raise ToolInputError("Tool arguments are missing.")
    encoded = raw.encode("utf-8")
    if len(encoded) > ARGUMENT_BYTE_CAP:
        raise ToolInputError("Tool arguments are larger than the local limit.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolInputError(f"Tool arguments are not valid JSON ({exc.msg}).") from None
    if not isinstance(value, dict):
        raise ToolInputError("Tool arguments must be a JSON object.")
    extra = [key for key in value if key != "path"]
    if extra:
        raise ToolInputError(
            "Tool arguments contain unknown fields: " + ", ".join(sorted(str(key) for key in extra)) + "."
        )
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise ToolInputError("Tool arguments need a non-empty path string.")
    return path


def execute_read_attached_file(path: str, snapshot) -> dict:
    """Return approved text, or a denial. This function does not touch the filesystem."""

    attachment = snapshot.attachment_map().get(path)
    if attachment is None:
        return {
            "ok": False,
            "path": path,
            "error": "That path is not an approved attachment. A new user approval is required.",
        }
    encoded = attachment.content.encode("utf-8")
    truncated = len(encoded) > RESULT_BYTE_CAP
    if truncated:
        content = encoded[:RESULT_BYTE_CAP].decode("utf-8", "ignore")
    else:
        content = attachment.content
    return {
        "ok": True,
        "path": attachment.path,
        "sha256": attachment.sha256,
        "truncated": truncated,
        "content": content,
    }


def _tool_message(call: AssembledCall, payload: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def run_tool_conversation(
    provider,
    target,
    snapshot,
    options: RunOptions,
    *,
    credential: str | None,
    prompt: str,
    tool_names: list[str],
    max_rounds: int,
    cancel_event=None,
    on_activity=None,
) -> RunResult:
    """Call the model, then run only approved tools, up to ``max_rounds``."""

    if max_rounds < 1 or max_rounds > MAX_TOOL_ROUNDS:
        raise UsageError(
            f"--max-tool-rounds must be from 1 to {MAX_TOOL_ROUNDS}.",
        )
    if not tool_names:
        raise UsageError("Name at least one tool.", hint=READ_ATTACHED_FILE)
    enabled = set(enabled_tools())
    for name in tool_names:
        if name not in enabled:
            raise UsageError(
                f"Tool {name!r} is not enabled.",
                hint=f"Run `/tools enable {name}` first. Tools stay off until you do.",
            )
    require_tool_capability(target.base_url, target.model)
    if getattr(provider, "kind", None) != OPENAI:
        raise UsageError(
            f"No tool adapter is implemented for provider kind {getattr(provider, 'kind', None)!r}.",
            hint="Support stays unknown. This client will not send a tools payload to that API.",
        )
    payload = openai_tool_payload(tool_names)
    working = messages_for(snapshot, prompt)
    trace: list[dict] = []
    combined = None
    requests = 0
    last = RunResult(status="error", error="No tool round ran.")
    round_options = RunOptions(
        timeout=options.timeout,
        deadline=options.deadline,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        show_reasoning=options.show_reasoning,
        tools_payload=payload,
    )
    for round_index in range(max_rounds + 1):
        if cancel_event is not None and cancel_event.is_set():
            last.status = "cancelled"
            last.incomplete = True
            last.error = "Cancelled before the next tool round."
            break
        last = run_request(
            provider,
            target,
            snapshot,
            round_options,
            credential=credential,
            messages=working,
            cancel_event=cancel_event,
        )
        requests += last.request_count
        combined = _merge_usage(combined, last.usage)
        try:
            calls = assemble_tool_calls(last.tool_deltas)
        except ToolInputError as exc:
            trace.append({"round": round_index, "error": str(exc), "executed": False})
            last.status = "error"
            last.error = str(exc)
            last.incomplete = True
            break
        if not calls:
            break
        if round_index >= max_rounds:
            last.status = "incomplete"
            last.incomplete = True
            last.error = "Stopped because the tool round limit was reached."
            trace.append({"round": round_index, "error": last.error, "executed": False})
            break
        assistant_calls = []
        tool_messages = []
        for call in calls:
            if call.name != READ_ATTACHED_FILE:
                body = {"ok": False, "error": f"Unknown tool {call.name!r}. It was not run."}
                executed = False
            else:
                try:
                    path = parse_path_argument(call.arguments)
                except ToolInputError as exc:
                    body = {"ok": False, "error": str(exc)}
                    executed = False
                else:
                    body = execute_read_attached_file(path, snapshot)
                    executed = bool(body.get("ok"))
            trace.append(
                {
                    "round": round_index,
                    "id": call.id,
                    "name": call.name,
                    "executed": executed,
                    "ok": bool(body.get("ok")),
                    "error": body.get("error"),
                }
            )
            if on_activity is not None:
                state = "ok" if body.get("ok") else "denied"
                on_activity(f"tool {call.name} {state}")
            assistant_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
            )
            tool_messages.append(_tool_message(call, body))
        working.append(
            {
                "role": "assistant",
                "content": last.text or "",
                "tool_calls": assistant_calls,
            }
        )
        working.extend(tool_messages)
    last.usage = combined
    last.request_count = requests
    last.tool_trace = tuple(trace)  # type: ignore[attr-defined]
    return last


def _merge_usage(left, right):
    from .adapters import Usage

    if right is None:
        return left
    if left is None:
        return right
    return left.combined(right)


def _probe_state(result: RunResult) -> str:
    """Classify a probe without treating an auth or server failure as 'unsupported'."""

    if result.status in {"ok", "incomplete"} and result.error is None:
        return "accepted"
    text = (result.error or "").lower()
    if any(token in text for token in ("http 401", "http 403", "http 404", "http 429", "http 500", "http 502", "http 503")):
        return "unknown"
    if ("http 400" in text or "http 422" in text) and "tool" in text:
        return "rejected"
    return "unknown"


def probe_tools(provider, target, *, credential: str | None, timeout: float) -> dict:
    """One explicit probe. Does not claim support for any other model or host."""

    from datetime import datetime, timezone

    if getattr(provider, "kind", None) != OPENAI:
        return {
            "provider": provider.name,
            "model": target.model,
            "base_url": target.base_url,
            "tools": "unknown",
            "probed_at": None,
            "detail": (
                "No tool adapter is implemented for this provider kind. "
                "Support stays unknown, and no request was sent."
            ),
            "sent": False,
        }
    from .seam import make_snapshot

    snapshot = make_snapshot(
        [],
        system="You are a capability probe. Reply with the single word ok. Do not call tools.",
    )
    options = RunOptions(
        timeout=min(float(timeout), 30.0),
        max_tokens=16,
        tools_payload=openai_tool_payload([READ_ATTACHED_FILE]),
    )
    result = run_request(
        provider,
        target,
        snapshot,
        options,
        credential=credential,
        prompt="ok",
    )
    state = _probe_state(result)
    detail = {
        "accepted": (
            "This endpoint accepted a tools parameter for this model at this URL. "
            "That is not evidence for other models, hosts, or JSON response_format."
        ),
        "rejected": "This endpoint rejected the tools parameter for this model.",
        "unknown": result.error or "The probe did not show whether tools are accepted.",
    }[state]
    record = {
        "provider": provider.name,
        "model": target.model,
        "base_url": target.base_url,
        "tools": state,
        "probed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "detail": detail,
        "sent": True,
    }
    if state in {"accepted", "rejected"}:
        _store_capability(record)
    return record


def register(register_command=register_repl) -> None:
    register_command("tools", repl_tools, ("/tools ...", "opt-in tool protocol"))


def repl_tools(repl, rest: str) -> None:
    import shlex

    try:
        tokens = shlex.split(rest or "")
    except ValueError as exc:
        raise UsageError(f"Could not parse /tools: {exc}") from None
    if not tokens or tokens[0] == "list":
        _print_tools(repl)
        return
    if tokens[0] == "enable" and len(tokens) == 2:
        enable_tool(tokens[1])
        repl.say(repl.style.dim(f"enabled {tokens[1]} (still requires a successful probe for the endpoint)"))
        return
    if tokens[0] == "disable" and len(tokens) == 2 and tokens[1] == "all":
        disable_all_tools()
        repl.say(repl.style.dim("all tools disabled"))
        return
    raise UsageError("Usage: /tools list, /tools enable read_attached_file, or /tools disable all.")


def _print_tools(repl) -> None:
    enabled = enabled_tools()
    repl.say(repl.style.bold("Tools"))
    if not enabled:
        repl.say(repl.style.dim("  none enabled"))
    for name in enabled:
        repl.say(f"  {name}")
    repl.say(repl.style.dim("  Support is unknown until a probe succeeds for the exact endpoint and model."))
    repl.say("")
