//! CLI dispatch and user-facing commands.

use std::collections::BTreeMap;
use std::io::{self, IsTerminal, Read, Write};

use anyllm_core as core;
use anyllm_core::{AnyLLMError, Config, KeyStore, Provider, Session, Style};
use serde_json::{json, Value};

#[derive(Clone)]
struct Parsed {
    pos: Vec<String>,
    opts: BTreeMap<String, Vec<String>>,
    flags: Vec<String>,
}

fn parse_options(args: &[String]) -> Result<Parsed, AnyLLMError> {
    let mut parsed = Parsed {
        pos: vec![],
        opts: BTreeMap::new(),
        flags: vec![],
    };
    let mut i = 0;
    let mut positional = false;
    while i < args.len() {
        let a = &args[i];
        if a == "--" {
            positional = true;
            i += 1;
            continue;
        }
        if !positional && a.starts_with('-') {
            let (raw_key, inline) = a
                .split_once('=')
                .map(|(k, v)| (k.to_string(), Some(v.to_string())))
                .unwrap_or((a.clone(), None));
            let key = match raw_key.as_str() {
                "-p" => "--provider".to_string(),
                "-m" => "--model".to_string(),
                "-s" => "--system".to_string(),
                _ => raw_key,
            };
            if key == "--tools"
                && inline.is_none()
                && i + 1 < args.len()
                && !args[i + 1].starts_with('-')
            {
                i += 1;
                parsed.opts.entry(key).or_default().push(args[i].clone());
            } else if matches!(
                key.as_str(),
                "-h" | "--help"
                    | "--no-color"
                    | "--json"
                    | "--no-stream"
                    | "--reasoning"
                    | "--stdin"
                    | "--test"
                    | "--no-key"
                    | "--yes"
                    | "--continue"
            ) {
                parsed.flags.push(key);
            } else if let Some(value) = inline {
                parsed.opts.entry(key).or_default().push(value);
            } else if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                i += 1;
                parsed.opts.entry(key).or_default().push(args[i].clone());
            } else {
                return Err(AnyLLMError::usage(format!(
                    "argument {}: expected one argument",
                    key
                )));
            }
        } else {
            parsed.pos.push(a.clone());
        }
        i += 1;
    }
    Ok(parsed)
}
impl Parsed {
    fn flag(&self, k: &str) -> bool {
        self.flags.iter().any(|x| x == k)
    }
    fn get(&self, k: &str) -> Option<&str> {
        self.opts.get(k).and_then(|v| v.last()).map(String::as_str)
    }
    fn all(&self, k: &str) -> Vec<String> {
        self.opts.get(k).cloned().unwrap_or_default()
    }
}
fn load() -> Result<(Config, BTreeMap<String, Provider>), AnyLLMError> {
    let c = core::load_config()?;
    let p = core::load_providers(&c)?;
    Ok((c, p))
}
fn style(p: &Parsed) -> Style {
    if p.flag("--no-color") || p.flag("--no-color-global") {
        Style::plain()
    } else {
        Style::new(core::color_enabled(false, io::stdout().is_terminal()))
    }
}
fn err(e: AnyLLMError) -> i32 {
    let s = Style::new(core::color_enabled(false, io::stderr().is_terminal()));
    for line in core::format_error(&s, &e) {
        eprintln!("{}", line);
    }
    if e.kind == core::ErrorKind::Usage {
        2
    } else {
        1
    }
}
fn require<'a>(p: &'a Parsed, k: &str) -> Result<&'a str, AnyLLMError> {
    let positional = match k {
        "name" | "key" => p.pos.get(1).map(String::as_str),
        "value" => p.pos.get(2).map(String::as_str),
        _ => None,
    };
    positional
        .or_else(|| p.get(k))
        .ok_or_else(|| AnyLLMError::usage(format!("argument {} is required", k)))
}

pub fn run(args: Vec<String>) -> i32 {
    core::setup();
    let args = crate::args::parse(args).args;
    let args = if args.is_empty() {
        vec!["chat".to_string()]
    } else {
        args
    };
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("anyllm {}", core::VERSION);
        return 0;
    }
    if args.iter().any(|a| a == "--help" || a == "-h")
        && (args.len() == 1 || args[0] == "--help" || args[0] == "-h")
    {
        print!("{}", crate::args::HELP);
        return 0;
    }
    let mut args = args;
    let global_no_color = args.first().map(|a| a == "--no-color").unwrap_or(false);
    if global_no_color {
        args.remove(0);
    }
    if args.is_empty() {
        args.push("chat".to_string());
    }
    let command = args[0].as_str();
    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        if let Some(h) = crate::args::command_help(command) {
            print!("{}", h);
            return 0;
        }
    }
    let mut p = match parse_options(&args[1..]) {
        Ok(x) => x,
        Err(e) => return err(e),
    };
    if global_no_color {
        p.flags.push("--no-color".to_string());
    }
    let st = style(&p);
    let result = match command {
        "ask" => cmd_ask(&p, &st),
        "chat" => cmd_chat(&p, &st),
        "models" => cmd_models(&p, &st),
        "providers" => cmd_providers(&p, &st),
        "keys" => cmd_keys(&p, &st),
        "config" => cmd_config(&p, &st),
        "doctor" => cmd_doctor(&p, &st),
        "targets" => cmd_targets(&p, &st),
        "prices" => cmd_prices(&p, &st),
        "team" | "tools" | "capabilities" => Err(AnyLLMError::usage(format!(
            "{} command is not available in this Rust build yet.",
            command
        ))),
        _ => {
            print!("{}", crate::args::HELP);
            return 2;
        }
    };
    match result {
        Ok(code) => code,
        Err(e) => err(e),
    }
}

fn choose_provider<'a>(
    p: &Parsed,
    c: &Config,
    providers: &'a BTreeMap<String, Provider>,
    store: &KeyStore,
) -> Result<Provider, AnyLLMError> {
    if let Some(n) = p.get("--provider") {
        return core::get_provider(n, providers);
    }
    if let Some(n) = c.get("provider").as_str() {
        if !n.is_empty() {
            return core::get_provider(n, providers);
        }
    }
    let ready: Vec<_> = providers
        .values()
        .filter(|x| {
            core::resolve_key(x, store).present()
                && core::resolve_key(x, store).env_var.as_deref() != Some("ANYLLM_API_KEY")
        })
        .collect();
    if ready.len() == 1 {
        return Ok(ready[0].clone());
    }
    if ready.len() > 1 {
        return Err(AnyLLMError::config(format!(
            "No provider selected, and several have keys: {}",
            ready
                .iter()
                .map(|x| x.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .with_hint("Choose one with --provider NAME, or run `anyllm config set provider NAME`."));
    }
    Err(AnyLLMError::config("No provider selected.").with_hint(format!("Pass --provider NAME, or run `anyllm config set provider NAME`. Available: {}. Start with `anyllm keys set openai`.",providers.keys().cloned().collect::<Vec<_>>().join(", "))))
}

fn make_session(
    p: &Parsed,
    c: &Config,
    providers: &BTreeMap<String, Provider>,
    store: &KeyStore,
) -> Result<Session, AnyLLMError> {
    let provider = choose_provider(p, c, providers, store)?;
    let base = if let Some(u) = p.get("--base-url") {
        core::validate_base_url(u)?
    } else {
        provider.base_url.clone()
    };
    if base.is_empty() {
        return Err(AnyLLMError::config("The 'custom' provider needs an endpoint.").with_hint("Pass --base-url https://..., or add a named provider with `anyllm providers add NAME --base-url URL`."));
    }
    let key = core::resolve_key(&provider, store);
    if provider.key_required && !key.present() {
        let variable = provider
            .env_vars
            .first()
            .map(String::as_str)
            .unwrap_or("ANYLLM_API_KEY");
        return Err(
            AnyLLMError::config(format!("No API key found for {}.", provider.name)).with_hint(
                format!(
                    "Run `anyllm keys set {}`, or set {} (for example: export {}=\"YOUR_KEY\").",
                    provider.name, variable, variable
                ),
            ),
        );
    }
    let configured = c.get("model");
    let configured_model = if c.get("provider").as_str() == Some(provider.name.as_str()) {
        configured.as_str()
    } else {
        None
    };
    let model = p
        .get("--model")
        .or(configured_model)
        .unwrap_or(provider.default_model())
        .to_string();
    if model.is_empty() {
        return Err(
            AnyLLMError::config(format!("No model selected for {}.", provider.name)).with_hint(
                format!(
                    "Pass --model NAME, or run `anyllm models {}` to see available models.",
                    provider.name
                ),
            ),
        );
    }
    let configured_system = c.get("system");
    let system = p
        .get("--system")
        .map(str::to_string)
        .or_else(|| configured_system.as_str().map(str::to_string));
    let temp = p
        .get("--temperature")
        .and_then(|v| v.parse::<f64>().ok())
        .map(|v| json!(v))
        .unwrap_or_else(|| c.get("temperature"));
    let max = p
        .get("--max-tokens")
        .and_then(|v| v.parse::<i64>().ok())
        .map(|v| json!(v))
        .unwrap_or_else(|| c.get("max_tokens"));
    Ok(Session::new(
        provider,
        &model,
        Some(&base),
        key.value.as_deref(),
        &key.source,
        system.as_deref(),
        core::validate_temperature(&temp)?,
        core::validate_max_tokens(&max)?,
        p.get("--timeout")
            .and_then(|v| v.parse().ok())
            .unwrap_or(60.0),
        p.flag("--reasoning") || c.get("show_reasoning").as_bool() == Some(true),
    ))
}

fn cmd_ask(p: &Parsed, _st: &Style) -> Result<i32, AnyLLMError> {
    let mut prompt = p.pos.join(" ").trim().to_string();
    if prompt.is_empty() {
        if io::stdin().is_terminal() {
            return Err(AnyLLMError::usage("No prompt given.")
                .with_hint("Try: anyllm ask \"explain git rebase in three lines\""));
        }
        io::stdin()
            .read_to_string(&mut prompt)
            .map_err(|e| AnyLLMError::config(e.to_string()))?;
        prompt = prompt.trim().to_string();
    }
    if prompt.is_empty() {
        return Err(AnyLLMError::usage("No prompt given."));
    }
    let (c, providers) = load()?;
    let store = KeyStore::new();
    let mut session = make_session(p, &c, &providers, &store)?;
    let mut observer = core::NoObserver;
    let turn = session.send(&prompt, &mut observer)?;
    if p.flag("--json") {
        let mut value = json!({"provider":session.provider.name,"model":session.model,"text":turn.text,"usage":turn.usage.to_dict(),"finish_reason":turn.finish_reason,"elapsed_seconds":(turn.elapsed*1000.0).round()/1000.0,"base_url":session.base_url});
        if session.show_reasoning && !turn.reasoning.is_empty() {
            value["reasoning"] = json!(turn.reasoning)
        }
        println!("{}", serde_json::to_string_pretty(&value).unwrap());
    } else {
        print!("{}", turn.text);
        if !turn.text.ends_with('\n') {
            println!()
        }
        io::stdout().flush().ok();
    }
    Ok(0)
}
fn cmd_chat(p: &Parsed, st: &Style) -> Result<i32, AnyLLMError> {
    if io::stdin().is_terminal() {
        return Err(AnyLLMError::usage(
            "Interactive chat is not implemented yet.",
        ));
    }
    let mut prompt = String::new();
    io::stdin()
        .read_to_string(&mut prompt)
        .map_err(|e| AnyLLMError::config(e.to_string()))?;
    let mut q = p.clone();
    q.pos = vec![prompt];
    cmd_ask(&q, st)
}

fn cmd_models(p: &Parsed, st: &Style) -> Result<i32, AnyLLMError> {
    let (c, providers) = load()?;
    let store = KeyStore::new();
    let mut model_args = p.clone();
    if let Some(name) = p.pos.first() {
        model_args
            .opts
            .entry("--provider".into())
            .or_default()
            .push(name.clone());
    }
    let provider = choose_provider(&model_args, &c, &providers, &store)?;
    let key = core::resolve_key(&provider, &store);
    if provider.key_required && !key.present() {
        return Err(AnyLLMError::config(format!(
            "No API key found for {}.",
            provider.name
        )));
    }
    let models = core::list_models(
        &provider,
        key.value.as_deref(),
        p.get("--base-url"),
        p.get("--timeout")
            .and_then(|v| v.parse().ok())
            .unwrap_or(60.0),
    )?;
    let models: Vec<_> = models
        .into_iter()
        .filter(|m| {
            p.get("--filter")
                .map(|f| m.to_lowercase().contains(&f.to_lowercase()))
                .unwrap_or(true)
        })
        .collect();
    if p.flag("--json") {
        println!("{}", serde_json::to_string_pretty(&models).unwrap());
    } else if models.is_empty() {
        eprintln!("{}", "no models returned");
    } else {
        for m in models {
            println!("{}", m)
        }
    }
    let _ = st;
    Ok(0)
}

fn cmd_providers(p: &Parsed, st: &Style) -> Result<i32, AnyLLMError> {
    let (c, providers) = load()?;
    let sub = p.pos.first().map(String::as_str).unwrap_or("list");
    match sub {
        "list" => {
            let store = KeyStore::new();
            let rows: Vec<Vec<String>> = providers
                .values()
                .map(|x| {
                    let k = core::resolve_key(x, &store);
                    vec![
                        x.name.clone(),
                        x.kind.clone(),
                        if !x.key_required {
                            "not needed".into()
                        } else {
                            k.display()
                        },
                        if x.default_model().is_empty() {
                            "-".into()
                        } else {
                            x.default_model().into()
                        },
                    ]
                })
                .collect();
            for l in core::format_table(st, &["PROVIDER", "KIND", "KEY", "DEFAULT MODEL"], &rows) {
                println!("{}", l)
            }
            println!();
            println!(
                "{}",
                st.dim("* = custom provider; see `anyllm providers show NAME`")
            );
        }
        "show" => {
            let n = p
                .pos
                .get(1)
                .ok_or_else(|| AnyLLMError::usage("provider name is required"))?;
            let x = core::get_provider(n, &providers)?;
            println!("{}", st.bold(&x.name));
            println!("  label:      {}\n  kind:       {}\n  base URL:   {}\n  key needed: {}\n  models:     {}",x.label,x.kind,core::redact_endpoint(&x.base_url,&[]),if x.key_required{"yes"}else{"no"},if x.models.is_empty(){"(none suggested)".into()}else{x.models.join(", ")});
        }
        "add" => {
            let n = require(p, "name")?;
            let url = require(p, "--base-url")?;
            let spec = core::catalog::provider_spec_from_args(
                n,
                &core::ProviderSpecArgs {
                    kind: p.get("--kind").unwrap_or("openai").to_string(),
                    base_url: url.into(),
                    env: p.all("--env"),
                    models: p.all("--model"),
                    no_key: p.flag("--no-key"),
                    auth_header: p.get("--auth-header").map(str::to_string),
                    auth_prefix: p.get("--auth-prefix").map(str::to_string),
                    label: p.get("--label").map(str::to_string),
                },
            )?;
            let mut custom = c
                .get("custom_providers")
                .as_object()
                .cloned()
                .unwrap_or_default();
            custom.insert(core::normalize_name(n), spec);
            let mut c = c;
            c.set("custom_providers", Value::Object(custom))?;
            c.save()?;
            println!("added provider {}", n);
        }
        "remove" => {
            let n = p
                .pos
                .get(1)
                .ok_or_else(|| AnyLLMError::usage("provider name is required"))?;
            let mut custom = c
                .get("custom_providers")
                .as_object()
                .cloned()
                .unwrap_or_default();
            if custom.remove(&core::normalize_name(n)).is_none() {
                return Err(
                    AnyLLMError::config(format!("{:?} is not a custom provider.", n))
                        .with_hint("Built-in providers cannot be removed."),
                );
            }
            let mut c = c;
            c.set("custom_providers", Value::Object(custom))?;
            c.save()?;
            println!("removed provider {}", n);
        }
        _ => {
            return Err(AnyLLMError::usage(
                "expected providers list, show, add, or remove",
            ))
        }
    }
    Ok(0)
}

fn cmd_keys(p: &Parsed, st: &Style) -> Result<i32, AnyLLMError> {
    let (c, providers) = load()?;
    let store = KeyStore::new();
    match p.pos.first().map(String::as_str).unwrap_or("") {
        "path" => println!("{}", core::keys_path().display()),
        "list" => {
            let infos = store.info()?;
            let rows: Vec<Vec<String>> = providers
                .values()
                .map(|x| {
                    let k = core::resolve_key(x, &store);
                    vec![
                        x.name.clone(),
                        infos
                            .get(&x.name)
                            .map(|_| "saved".to_string())
                            .unwrap_or_else(|| "-".into()),
                        k.env_var.unwrap_or_else(|| "-".into()),
                    ]
                })
                .collect();
            for l in core::format_table(st, &["PROVIDER", "SAVED", "ENVIRONMENT"], &rows) {
                println!("{}", l)
            }
            println!();
            println!(
                "{}",
                st.dim(&format!("key file: {}", core::keys_path().display()))
            );
        }
        "set" => {
            let configured_name = c.get("provider");
            let name = p
                .pos
                .get(1)
                .cloned()
                .or_else(|| configured_name.as_str().map(str::to_string))
                .ok_or_else(|| AnyLLMError::usage("Choose a provider for the key."))?;
            let key = if p.flag("--stdin") {
                let mut s = String::new();
                io::stdin()
                    .read_line(&mut s)
                    .map_err(|e| AnyLLMError::config(e.to_string()))?;
                s.trim().to_string()
            } else {
                return Err(AnyLLMError::usage(
                    "Use --stdin to provide a key non-interactively.",
                ));
            };
            let scheme = store.set(&name, &key)?;
            println!(
                "saved {} key → {} ({})",
                &name,
                core::keys_path().display(),
                if scheme == "dpapi" {
                    "encrypted with Windows DPAPI"
                } else {
                    "stored in a file with permissions 600"
                }
            );
        }
        "remove" => {
            let name = p
                .pos
                .get(1)
                .map(String::as_str)
                .ok_or_else(|| AnyLLMError::usage("Choose a provider to remove."))?;
            println!(
                "{}",
                if store.remove(name)? {
                    format!("removed the saved {} key", name)
                } else {
                    format!("no saved key for {}", name)
                }
            );
        }
        _ => {
            return Err(AnyLLMError::usage(
                "expected keys set, list, remove, or path",
            ))
        }
    }
    Ok(0)
}

fn cmd_config(p: &Parsed, st: &Style) -> Result<i32, AnyLLMError> {
    let sub = p.pos.first().map(String::as_str).unwrap_or("");
    let mut c = core::load_config()?;
    match sub {
        "path" => println!("{}", core::config_path().display()),
        "show" => {
            println!("{}", serde_json::to_string_pretty(&c.data).unwrap());
            eprintln!(
                "{}",
                st.dim(&format!("# {}", core::config_path().display()))
            );
        }
        "get" => {
            let k = require(p, "key")?;
            if !core::ALLOWED_KEYS.contains(&k) {
                return Err(AnyLLMError::usage(format!("Unknown config key {:?}.", k))
                    .with_hint(format!("Known keys: {}", core::ALLOWED_KEYS.join(", "))));
            }
            let v = c.get(k);
            println!(
                "{}",
                if v.is_string() {
                    v.as_str().unwrap().to_string()
                } else if v.is_null() {
                    "null".into()
                } else {
                    v.to_string()
                }
            );
        }
        "set" => {
            let k = require(p, "key")?;
            let v = require(p, "value")?;
            let value = core::parse_config_value(v);
            if k == "provider" && !value.is_null() {
                let providers = core::load_providers(&c)?;
                let provider =
                    core::get_provider(value.as_str().unwrap_or(&value.to_string()), &providers)?;
                c.set(k, json!(provider.name))?;
            } else if k == "temperature" {
                c.set(k, json!(core::validate_temperature(&value)?))?;
            } else if k == "max_tokens" {
                c.set(k, json!(core::validate_max_tokens(&value)?))?;
            } else if k == "show_reasoning" {
                let b = match value {
                    Value::Bool(v) => v,
                    Value::String(s) => {
                        matches!(s.to_lowercase().as_str(), "true" | "yes" | "on" | "1")
                    }
                    _ => false,
                };
                c.set(k, json!(b))?;
            } else {
                c.set(k, value)?;
            }
            c.save()?;
            println!("{} = {}", k, c.get(k));
        }
        "unset" => {
            let k = require(p, "key")?;
            c.unset(k)?;
            c.save()?;
            println!("{} reset to default", k);
        }
        _ => {
            return Err(AnyLLMError::usage(
                "expected config show, get, set, unset, or path",
            ))
        }
    }
    Ok(0)
}

fn cmd_doctor(p: &Parsed, st: &Style) -> Result<i32, AnyLLMError> {
    let (c, providers) = load()?;
    let store = KeyStore::new();
    let selected = p.all("--provider");
    let mut failed = false;
    for provider in providers
        .values()
        .filter(|x| selected.is_empty() || selected.contains(&x.name))
    {
        let key = core::resolve_key(provider, &store);
        let state = if !provider.key_required {
            "not needed".into()
        } else {
            key.display()
        };
        println!(
            "{}  {}  {}",
            provider.name,
            core::redact_endpoint(&provider.base_url, &[]),
            state
        );
        if p.flag("--test") && key.present() {
            let model = provider.default_model();
            if !model.is_empty() {
                let mut s = Session::new(
                    provider.clone(),
                    model,
                    None,
                    key.value.as_deref(),
                    &key.source,
                    None,
                    None,
                    Some(1),
                    p.get("--timeout")
                        .and_then(|x| x.parse().ok())
                        .unwrap_or(30.0),
                    false,
                );
                let mut o = core::NoObserver;
                if let Err(e) = s.send("Reply with OK.", &mut o) {
                    failed = true;
                    eprintln!("{}", core::format_error(st, &e).join("\n"));
                }
            }
        } else if provider.key_required && !key.present() {
            failed = true;
        }
    }
    let _ = c;
    Ok(if failed { 1 } else { 0 })
}

fn cmd_targets(p: &Parsed, st: &Style) -> Result<i32, AnyLLMError> {
    let sub = p.pos.first().map(String::as_str).unwrap_or("");
    let (c, cp) = load()?;
    let providers = to_orch_providers(&cp);
    let store = OrchStore(KeyStore::new());
    match sub {
        "add" => {
            let t = anyllm_orch::targets::add_target(
                require(p, "name")?,
                require(p, "--provider")?,
                require(p, "--model")?,
                p.get("--base-url"),
                &providers,
                &store,
            )
            .map_err(from_orch)?;
            println!(
                "added target {} → {}/{}",
                t.alias.unwrap_or_default(),
                t.provider,
                t.model
            );
        }
        "list" => {
            for line in anyllm_orch::targets::render_target_list(
                &anyllm_orch::sys::Style::new(st.enabled),
                &providers,
            )
            .map_err(from_orch)?
            {
                println!("{}", line)
            }
        }
        "remove" => {
            anyllm_orch::targets::remove_target(require(p, "name")?, &providers)
                .map_err(from_orch)?;
            println!("removed target {}", p.pos[1]);
        }
        _ => return Err(AnyLLMError::usage("expected targets add, list, or remove")),
    }
    let _ = c;
    Ok(0)
}
fn cmd_prices(p: &Parsed, st: &Style) -> Result<i32, AnyLLMError> {
    match p.pos.first().map(String::as_str).unwrap_or("") {
        "set" => {
            let name = require(p, "name")?;
            let input = json!(require(p, "--input")?);
            let output = json!(require(p, "--output")?);
            let price = anyllm_orch::budgets::set_price(
                name,
                &input,
                &output,
                p.get("--currency"),
                &core::keystore::utc_now_iso(),
            )
            .map_err(from_orch)?;
            println!(
                "{}: {} / {} {} per million tokens ({})",
                price.name,
                price.input_per_million,
                price.output_per_million,
                price.currency,
                price.updated
            );
        }
        "list" => {
            let prices = anyllm_orch::budgets::load_prices().map_err(from_orch)?;
            if prices.is_empty() {
                println!("{}", st.dim("no saved prices"))
            }
            for x in prices.values() {
                println!(
                    "{}: in {} / out {} {} per million ({})",
                    x.name, x.input_per_million, x.output_per_million, x.currency, x.updated
                )
            }
        }
        _ => return Err(AnyLLMError::usage("expected prices set or list")),
    }
    Ok(0)
}
fn to_orch_providers(src: &BTreeMap<String, Provider>) -> anyllm_orch::sys::ProviderMap {
    src.iter()
        .map(|(k, p)| {
            let mut x = anyllm_orch::sys::Provider::new(&p.name, &p.label, &p.kind, &p.base_url);
            x.env_vars = p.env_vars.clone();
            x.models = p.models.clone();
            x.key_required = p.key_required;
            x.local = p.local;
            x.docs = p.docs.clone();
            x.auth_header = p.auth_header.clone();
            x.auth_prefix = p.auth_prefix.clone();
            (k.clone(), x)
        })
        .collect()
}
struct OrchStore(KeyStore);
impl anyllm_orch::sys::KeyStore for OrchStore {
    fn get(&self, n: &str) -> anyllm_orch::sys::Result<Option<String>> {
        self.0
            .get(n)
            .map_err(|e| anyllm_orch::sys::Error::config(e.message, e.hint))
    }
}
fn from_orch(e: anyllm_orch::sys::Error) -> AnyLLMError {
    AnyLLMError::config(e.message).with_hint_opt(e.hint)
}
