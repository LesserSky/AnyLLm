//! Opt-in, attachment-only tool protocol. Port of `anyllm/tool_protocol.py`.
//!
//! Tool calls are accepted only after an explicit probe for the exact URL and
//! model. The only implemented function reads text already approved in the
//! process attachment store; this module never opens a model-supplied path.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::seam::{
    feature_path, handler, handler_words, load_feature, make_snapshot, messages_for, run_request,
    save_feature, ApprovedAttachment, Repl, ReplRegistry, RequestSnapshot, RunCall, RunOptions,
    RunResult,
};
use crate::sys::{ChatBackend, Error, Message, Provider, Usage, OPENAI};
use crate::targets::Target;

pub const READ_ATTACHED_FILE: &str = "read_attached_file";
pub const MAX_TOOL_ROUNDS: usize = 4;
pub const MAX_TOOL_CALLS_PER_ROUND: usize = 8;
pub const ARGUMENT_BYTE_CAP: usize = 8 * 1024;
pub const RESULT_BYTE_CAP: usize = 32 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssembledCall {
    pub index: i64,
    pub id: String,
    pub name: String,
    pub arguments: String,
}

fn tool_schema() -> Value {
    json!({"type":"function","function":{
        "name":READ_ATTACHED_FILE,
        "description":"Read one file the user already approved for this request. Any other path is denied and nothing is read from disk.",
        "parameters":{"type":"object","properties":{"path":{"type":"string","description":"Exact path of an approved attachment."}},"required":["path"],"additionalProperties":false}
    }})
}

pub fn openai_tool_payload(names: &[String]) -> Result<Value, Error> {
    let mut payload = Vec::new();
    for name in names {
        if name != READ_ATTACHED_FILE {
            return Err(Error::usage_hint(
                format!("Unknown tool {:?}.", name),
                format!("The only tool is {}.", READ_ATTACHED_FILE),
            ));
        }
        payload.push(tool_schema());
    }
    Ok(Value::Array(payload))
}

pub fn tools_path() -> std::path::PathBuf {
    feature_path("tools.json")
}
pub fn capabilities_path() -> std::path::PathBuf {
    feature_path("capabilities.json")
}

pub fn enabled_tools() -> Result<Vec<String>, Error> {
    let data = load_feature(&tools_path(), "tools")?;
    let enabled = data
        .get("enabled")
        .filter(|value| !value.is_null() && *value != &Value::Bool(false))
        .cloned()
        .unwrap_or_else(|| json!([]));
    let Some(items) = enabled.as_array() else {
        return Err(Error::config_hint(
            "tools.json is malformed.",
            "The enabled list must be an array of names.",
        ));
    };
    items
        .iter()
        .map(|item| {
            item.as_str().map(str::to_owned).ok_or_else(|| {
                Error::config_hint(
                    "tools.json is malformed.",
                    "The enabled list must be an array of names.",
                )
            })
        })
        .collect()
}

pub fn enable_tool(name: &str) -> Result<(), Error> {
    if name != READ_ATTACHED_FILE {
        return Err(Error::usage_hint(
            format!("Unknown tool {:?}.", name),
            format!(
                "Enable {} only. Other tools are not implemented.",
                READ_ATTACHED_FILE
            ),
        ));
    }
    let mut current = enabled_tools()?;
    if !current.iter().any(|item| item == name) {
        current.push(name.to_owned());
    }
    save_feature(
        &tools_path(),
        Map::from_iter([("enabled".to_owned(), json!(current))]),
    )
}

pub fn disable_all_tools() -> Result<(), Error> {
    save_feature(
        &tools_path(),
        Map::from_iter([("enabled".to_owned(), json!([]))]),
    )
}

fn capability_id(base_url: &str, model: &str) -> String {
    format!("{}\n{}", base_url, model)
}

pub fn capability_record(base_url: &str, model: &str) -> Result<Option<Value>, Error> {
    let data = load_feature(&capabilities_path(), "capabilities")?;
    let probes = data
        .get("probes")
        .filter(|value| !value.is_null() && *value != &Value::Bool(false))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let Some(probes) = probes.as_object() else {
        return Err(Error::config("capabilities.json is malformed.", None));
    };
    Ok(probes
        .get(&capability_id(base_url, model))
        .filter(|item| item.is_object())
        .cloned())
}

fn store_capability(record: &Value) -> Result<(), Error> {
    let data = load_feature(&capabilities_path(), "capabilities")?;
    let mut probes = data
        .get("probes")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let base_url = record.get("base_url").and_then(Value::as_str).unwrap_or("");
    let model = record.get("model").and_then(Value::as_str).unwrap_or("");
    probes.insert(capability_id(base_url, model), record.clone());
    save_feature(
        &capabilities_path(),
        Map::from_iter([("probes".to_owned(), Value::Object(probes))]),
    )
}

pub fn require_tool_capability(base_url: &str, model: &str) -> Result<Value, Error> {
    let record = capability_record(base_url, model)?;
    if let Some(record) = record
        .as_ref()
        .filter(|r| r.get("tools").and_then(Value::as_str) == Some("accepted"))
    {
        return Ok(record.clone());
    }
    let state = record
        .as_ref()
        .and_then(|r| r.get("tools"))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    Err(Error::usage_hint(format!("Tool support is {} for {} at {}.", state, model, base_url), "Run `anyllm capabilities probe --target NAME --tools`. That is one small request which may be billed, and it applies only to that exact endpoint and model. Nothing is assumed from the provider name."))
}

pub fn assemble_tool_calls(
    deltas: &[crate::sys::ToolCallDelta],
) -> Result<Vec<AssembledCall>, Error> {
    let mut order = Vec::new();
    let mut slots: BTreeMap<i64, (String, String, String)> = BTreeMap::new();
    for delta in deltas {
        let slot = slots.entry(delta.index).or_insert_with(|| {
            order.push(delta.index);
            (String::new(), String::new(), String::new())
        });
        if !delta.id.is_empty() {
            if !slot.0.is_empty() && slot.0 != delta.id {
                return Err(Error::usage(
                    format!(
                        "Tool call index {} changed id from {:?} to {:?}.",
                        delta.index, slot.0, delta.id
                    ),
                    None,
                ));
            }
            slot.0 = delta.id.clone();
        }
        slot.1.push_str(&delta.name);
        slot.2.push_str(&delta.arguments);
        if slot.2.len() > ARGUMENT_BYTE_CAP {
            return Err(Error::usage(
                "Tool arguments exceeded 8192 bytes while they were still streaming.",
                None,
            ));
        }
        if slots.len() > MAX_TOOL_CALLS_PER_ROUND {
            return Err(Error::usage(
                format!(
                    "More than {} tool calls arrived in one round.",
                    MAX_TOOL_CALLS_PER_ROUND
                ),
                None,
            ));
        }
    }
    let mut seen = BTreeSet::new();
    order
        .into_iter()
        .map(|index| {
            let (id, name, arguments) = slots.remove(&index).unwrap();
            if id.is_empty() || name.is_empty() {
                return Err(Error::usage(
                    format!("Tool call {} is missing an id or a name.", index),
                    None,
                ));
            }
            if !seen.insert(id.clone()) {
                return Err(Error::usage(
                    format!("Duplicate tool call id {:?}.", id),
                    None,
                ));
            }
            Ok(AssembledCall {
                index,
                id,
                name,
                arguments,
            })
        })
        .collect()
}

pub fn parse_path_argument(raw: Option<&str>) -> Result<String, Error> {
    let Some(raw) = raw else {
        return Err(Error::usage("Tool arguments are missing.", None));
    };
    if raw.len() > ARGUMENT_BYTE_CAP {
        return Err(Error::usage(
            "Tool arguments are larger than the local limit.",
            None,
        ));
    }
    let value: Value = serde_json::from_str(raw).map_err(|e| {
        Error::usage(
            format!(
                "Tool arguments are not valid JSON ({}).",
                e.to_string()
                    .split(" at line")
                    .next()
                    .unwrap_or("expected value")
            ),
            None,
        )
    })?;
    let Some(object) = value.as_object() else {
        return Err(Error::usage("Tool arguments must be a JSON object.", None));
    };
    let extra: Vec<String> = object
        .keys()
        .filter(|key| key.as_str() != "path")
        .cloned()
        .collect();
    if !extra.is_empty() {
        return Err(Error::usage(
            format!(
                "Tool arguments contain unknown fields: {}.",
                extra.join(", ")
            ),
            None,
        ));
    }
    match object
        .get("path")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        Some(path) => Ok(path.to_owned()),
        None => Err(Error::usage(
            "Tool arguments need a non-empty path string.",
            None,
        )),
    }
}

pub fn execute_read_attached_file(path: &str, snapshot: &RequestSnapshot) -> Value {
    let Some(attachment) = snapshot.attachment_map().remove(path) else {
        return json!({"ok":false,"path":path,"error":"That path is not an approved attachment. A new user approval is required."});
    };
    let bytes = attachment.content.as_bytes();
    let truncated = bytes.len() > RESULT_BYTE_CAP;
    // Keep a valid UTF-8 prefix at the 32 KiB boundary.
    let mut end = bytes.len().min(RESULT_BYTE_CAP);
    while !attachment.content.is_char_boundary(end) {
        end -= 1;
    }
    json!({"ok":true,"path":attachment.path,"sha256":attachment.sha256,"truncated":truncated,"content":&attachment.content[..end]})
}

fn tool_message(call: &AssembledCall, body: &Value) -> Message {
    let mut message = Message::new(
        "tool",
        &serde_json::to_string(body).unwrap_or_else(|_| "{}".into()),
    );
    message.tool_call_id = Some(call.id.clone());
    message
}

pub fn run_tool_conversation(
    backend: &dyn ChatBackend,
    provider: &Provider,
    target: &Target,
    snapshot: &RequestSnapshot,
    options: &RunOptions,
    credential: Option<&str>,
    prompt: &str,
    tool_names: &[String],
    max_rounds: usize,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<RunResult, Error> {
    if !(1..=MAX_TOOL_ROUNDS).contains(&max_rounds) {
        return Err(Error::usage(
            format!("--max-tool-rounds must be from 1 to {}.", MAX_TOOL_ROUNDS),
            None,
        ));
    }
    if tool_names.is_empty() {
        return Err(Error::usage_hint(
            "Name at least one tool.",
            READ_ATTACHED_FILE,
        ));
    }
    let enabled: BTreeSet<String> = enabled_tools()?.into_iter().collect();
    for name in tool_names {
        if !enabled.contains(name) {
            return Err(Error::usage_hint(
                format!("Tool {:?} is not enabled.", name),
                format!(
                    "Run `/tools enable {}` first. Tools stay off until you do.",
                    name
                ),
            ));
        }
    }
    require_tool_capability(&target.base_url, &target.model)?;
    if provider.kind != OPENAI {
        return Err(Error::usage_hint(
            format!(
                "No tool adapter is implemented for provider kind {:?}.",
                provider.kind
            ),
            "Support stays unknown. This client will not send a tools payload to that API.",
        ));
    }
    let payload = openai_tool_payload(tool_names)?;
    let mut working = messages_for(snapshot, prompt, None);
    let mut trace = Vec::new();
    let mut usage: Option<Usage> = None;
    let mut requests = 0;
    let mut last = RunResult {
        status: "error".into(),
        error: Some("No tool round ran.".into()),
        ..Default::default()
    };
    let round_options = RunOptions {
        tools_payload: Some(payload),
        ..options.clone()
    };
    for round_index in 0..=max_rounds {
        if cancel
            .as_ref()
            .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::SeqCst))
        {
            last.status = "cancelled".into();
            last.incomplete = true;
            last.error = Some("Cancelled before the next tool round.".into());
            break;
        }
        last = run_request(
            backend,
            provider,
            target,
            snapshot,
            &round_options,
            RunCall {
                credential,
                prompt: None,
                messages: Some(working.clone()),
                cancel: cancel.clone(),
                events: None,
            },
        );
        requests += last.request_count;
        if let Some(next) = &last.usage {
            usage = Some(
                usage
                    .as_ref()
                    .map(|u| u.combined(next))
                    .unwrap_or_else(|| next.clone()),
            );
        }
        let calls = match assemble_tool_calls(&last.tool_deltas) {
            Ok(calls) => calls,
            Err(error) => {
                trace.push(json!({"round":round_index,"error":error.message,"executed":false}));
                last.status = "error".into();
                last.error = Some(error.message);
                last.incomplete = true;
                break;
            }
        };
        if calls.is_empty() {
            break;
        }
        if round_index >= max_rounds {
            last.status = "incomplete".into();
            last.incomplete = true;
            last.error = Some("Stopped because the tool round limit was reached.".into());
            trace.push(json!({"round":round_index,"error":last.error,"executed":false}));
            break;
        }
        let mut assistant_calls = Vec::new();
        let mut tool_messages = Vec::new();
        for call in &calls {
            let (body, executed) = if call.name != READ_ATTACHED_FILE {
                (
                    json!({"ok":false,"error":format!("Unknown tool {:?}. It was not run.",call.name)}),
                    false,
                )
            } else {
                match parse_path_argument(Some(&call.arguments)) {
                    Err(error) => (json!({"ok":false,"error":error.message}), false),
                    Ok(path) => {
                        let body = execute_read_attached_file(&path, snapshot);
                        let ok = body["ok"].as_bool().unwrap_or(false);
                        (body, ok)
                    }
                }
            };
            trace.push(json!({"round":round_index,"id":call.id,"name":call.name,"executed":executed,"ok":body["ok"].as_bool().unwrap_or(false),"error":body.get("error")}));
            assistant_calls.push(json!({"id":call.id,"type":"function","function":{"name":call.name,"arguments":call.arguments}}));
            tool_messages.push(tool_message(call, &body));
        }
        let mut assistant = Message::new("assistant", &last.text);
        assistant.tool_calls = Some(Value::Array(assistant_calls));
        working.push(assistant);
        working.extend(tool_messages);
    }
    last.usage = usage;
    last.request_count = requests;
    last.tool_trace = trace;
    Ok(last)
}

fn probe_state(result: &RunResult) -> &'static str {
    if (result.status == "ok" || result.status == "incomplete") && result.error.is_none() {
        return "accepted";
    }
    let text = result.error.as_deref().unwrap_or("").to_lowercase();
    if [
        "http 401", "http 403", "http 404", "http 429", "http 500", "http 502", "http 503",
    ]
    .iter()
    .any(|s| text.contains(s))
    {
        return "unknown";
    }
    if (text.contains("http 400") || text.contains("http 422")) && text.contains("tool") {
        return "rejected";
    }
    "unknown"
}

pub fn probe_tools(
    backend: &dyn ChatBackend,
    provider: &Provider,
    target: &Target,
    credential: Option<&str>,
    timeout: f64,
) -> Result<Value, Error> {
    if provider.kind != OPENAI {
        return Ok(
            json!({"provider":provider.name,"model":target.model,"base_url":target.base_url,"tools":"unknown","probed_at":null,"detail":"No tool adapter is implemented for this provider kind. Support stays unknown, and no request was sent.","sent":false}),
        );
    }
    let snapshot = make_snapshot(
        &[],
        Some("You are a capability probe. Reply with the single word ok. Do not call tools."),
        None,
        None,
        None,
        false,
        None,
    );
    let names = vec![READ_ATTACHED_FILE.to_owned()];
    let options = RunOptions {
        timeout: timeout.min(30.0),
        max_tokens: Some(16),
        tools_payload: Some(openai_tool_payload(&names)?),
        ..Default::default()
    };
    let result = run_request(
        backend,
        provider,
        target,
        &snapshot,
        &options,
        RunCall {
            credential,
            prompt: Some("ok"),
            messages: None,
            cancel: None,
            events: None,
        },
    );
    let state = probe_state(&result);
    let detail = match state {
        "accepted" => "This endpoint accepted a tools parameter for this model at this URL. That is not evidence for other models, hosts, or JSON response_format.",
        "rejected" => "This endpoint rejected the tools parameter for this model.",
        _ => result.error.as_deref().unwrap_or("The probe did not show whether tools are accepted."),
    };
    let record = json!({"provider":provider.name,"model":target.model,"base_url":target.base_url,"tools":state,"probed_at":chrono_free_timestamp(),"detail":detail,"sent":true});
    if state == "accepted" || state == "rejected" {
        store_capability(&record)?;
    }
    Ok(record)
}

fn chrono_free_timestamp() -> String {
    // UNIX time in UTC is stable and dependency-free; format ISO-8601 via the
    // platform's `date` would add an unwanted shell dependency, so use a small
    // civil-date conversion.
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let days = seconds.div_euclid(86400);
    let rem = seconds.rem_euclid(86400);
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 }.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096).div_euclid(365);
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2).div_euclid(153);
    let day = doy - (153 * mp + 2).div_euclid(5) + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += if month <= 2 { 1 } else { 0 };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}+00:00",
        year,
        month,
        day,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

pub fn register(registry: &mut ReplRegistry) {
    registry.register(
        "tools",
        handler(|repl, rest| repl_tools(repl, rest)),
        Some(("/tools ...", "opt-in tool protocol")),
    );
}

pub fn repl_tools(repl: &mut dyn Repl, rest: &str) -> Result<(), Error> {
    let tokens = handler_words(rest)
        .map_err(|exc| Error::usage(format!("Could not parse /tools: {}", exc.message), None))?;
    if tokens.is_empty() || tokens[0] == "list" {
        return print_tools(repl);
    }
    if tokens.len() == 2 && tokens[0] == "enable" {
        enable_tool(&tokens[1])?;
        repl.say(&repl.style().dim(&format!(
            "enabled {} (still requires a successful probe for the endpoint)",
            tokens[1]
        )));
        return Ok(());
    }
    if tokens.len() == 2 && tokens[0] == "disable" && tokens[1] == "all" {
        disable_all_tools()?;
        repl.say(&repl.style().dim("all tools disabled"));
        return Ok(());
    }
    Err(Error::usage(
        "Usage: /tools list, /tools enable read_attached_file, or /tools disable all.",
        None,
    ))
}

fn print_tools(repl: &mut dyn Repl) -> Result<(), Error> {
    let enabled = enabled_tools()?;
    repl.say(&repl.style().bold("Tools"));
    if enabled.is_empty() {
        repl.say(&repl.style().dim("  none enabled"));
    }
    for name in enabled {
        repl.say(&format!("  {}", name));
    }
    repl.say(
        &repl
            .style()
            .dim("  Support is unknown until a probe succeeds for the exact endpoint and model."),
    );
    repl.say("");
    Ok(())
}

/// Helpers for hosts that expose explicitly approved local file edits. The
/// model tool protocol itself intentionally has no write/edit/undo operation.
pub fn read_approved_attachment(
    path: &str,
    attachments: &[ApprovedAttachment],
) -> Option<ApprovedAttachment> {
    attachments.iter().find(|item| item.path == path).cloned()
}
