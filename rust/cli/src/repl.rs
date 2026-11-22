//! Interactive chat REPL.
//!
//! The Python implementation remains the behavior spec. This module keeps the
//! input loop, history persistence, streaming display, and slash commands in
//! the CLI crate so the lower-level session and terminal APIs stay reusable.

use std::fs;
use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyllm_core as core;
use core::{AnyLLMError, Config, KeyStore, Provider, Session, SessionObserver, Style, Turn, Usage};
use serde_json::{Map, Value};

/// Print streamed answer text and reasoning while keeping code fences muted.
pub struct StreamPrinter {
    style: Style,
    show_reasoning: bool,
    spinner: bool,
    in_code: bool,
    wrote_text: bool,
    wrote_reasoning: bool,
    last: char,
    spinner_stop: Arc<AtomicBool>,
    output_lock: Arc<Mutex<()>>,
    spinner_thread: Option<thread::JoinHandle<()>>,
}

impl StreamPrinter {
    pub fn new(style: Style, show_reasoning: bool, spinner: bool) -> Self {
        StreamPrinter {
            style,
            show_reasoning,
            spinner,
            in_code: false,
            wrote_text: false,
            wrote_reasoning: false,
            last: '\0',
            spinner_stop: Arc::new(AtomicBool::new(false)),
            output_lock: Arc::new(Mutex::new(())),
            spinner_thread: None,
        }
    }

    pub fn begin(&mut self) {
        if !self.spinner || self.spinner_thread.is_some() {
            return;
        }
        self.spinner_stop.store(false, Ordering::SeqCst);
        let stop = Arc::clone(&self.spinner_stop);
        let lock = Arc::clone(&self.output_lock);
        let style = self.style;
        self.spinner_thread = Some(thread::spawn(move || {
            let frames = if style.enabled {
                "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            } else {
                "|/-\\"
            };
            let frames: Vec<char> = frames.chars().collect();
            let mut i = 0;
            while !stop.load(Ordering::SeqCst) {
                if let Ok(_guard) = lock.lock() {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let frame = frames[i % frames.len()].to_string();
                    if style.enabled {
                        print!("\r{}\x1b[K", style.accent(&frame));
                    } else {
                        print!("\r{}", frame);
                    }
                    let _ = io::stdout().flush();
                }
                i += 1;
                thread::sleep(Duration::from_millis(80));
            }
        }));
    }

    fn clear_indicator(&mut self) {
        self.spinner_stop.store(true, Ordering::SeqCst);
        if let Some(handle) = self.spinner_thread.take() {
            if handle.thread().id() != thread::current().id() {
                let _ = handle.join();
            }
            if let Ok(_guard) = self.output_lock.lock() {
                if self.style.enabled {
                    print!("\r\x1b[2K");
                } else {
                    print!("\r \r");
                }
                let _ = io::stdout().flush();
            }
        }
    }

    fn write(&self, text: &str) {
        if let Ok(_guard) = self.output_lock.lock() {
            print!("{}", text);
            let _ = io::stdout().flush();
        }
    }

    pub fn text(&mut self, value: &str) {
        if value.is_empty() {
            return;
        }
        self.clear_indicator();
        if self.show_reasoning
            && self.wrote_reasoning
            && !self.wrote_text
            && self.last != '\n'
            && self.last != '\0'
        {
            self.write("\n");
        }
        let mut rest = value;
        while let Some(index) = rest.find("```") {
            self.emit(&rest[..index], None);
            self.emit("```", Some(true));
            self.in_code = !self.in_code;
            rest = &rest[index + 3..];
        }
        self.emit(rest, None);
        self.wrote_text = true;
        self.last = value.chars().last().unwrap_or('\0');
    }

    fn emit(&self, value: &str, code: Option<bool>) {
        if value.is_empty() {
            return;
        }
        let code = code.unwrap_or(self.in_code);
        if self.style.enabled && code {
            self.write(&self.style.code(value));
        } else {
            self.write(value);
        }
    }

    pub fn reasoning(&mut self, value: &str) {
        if value.is_empty() {
            return;
        }
        self.clear_indicator();
        if !self.show_reasoning {
            self.wrote_reasoning = true;
            return;
        }
        if !self.wrote_reasoning {
            self.write(&self.style.muted("reasoning:\n"));
        }
        self.write(&self.style.reasoning(value));
        self.wrote_reasoning = true;
        self.last = value.chars().last().unwrap_or('\0');
    }

    pub fn end(&mut self) {
        self.clear_indicator();
        if self.last != '\0' && self.last != '\n' {
            self.write("\n");
        }
    }
}

impl SessionObserver for StreamPrinter {
    fn on_text(&mut self, text: &str) {
        self.text(text);
    }

    fn on_reasoning(&mut self, text: &str) {
        self.reasoning(text);
    }
}

pub struct Repl {
    pub session: Session,
    pub style: Style,
    pub providers: std::collections::BTreeMap<String, Provider>,
    pub store: KeyStore,
    pub config: Config,
    running: bool,
    history: Vec<String>,
}

impl Repl {
    pub fn new(
        session: Session,
        style: Style,
        providers: std::collections::BTreeMap<String, Provider>,
        store: KeyStore,
        config: Config,
    ) -> Self {
        let history = fs::read_to_string(core::config::history_path())
            .map(|text| text.lines().map(str::to_string).collect())
            .unwrap_or_default();
        let mut repl = Repl {
            session,
            style,
            providers,
            store,
            config,
            running: true,
            history,
        };
        repl.history.truncate(500);
        repl
    }

    fn prompt(&self, suffix: &str) -> String {
        self.style.prompt(suffix)
    }

    fn read_line(&self, suffix: &str) -> io::Result<Option<String>> {
        print!("{}", self.prompt(suffix));
        io::stdout().flush()?;
        let mut line = String::new();
        match io::stdin().lock().read_line(&mut line) {
            Ok(0) => Ok(None),
            Ok(_) => {
                while line.ends_with(['\n', '\r']) {
                    line.pop();
                }
                Ok(Some(line))
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(Some(String::new())),
            Err(error) => Err(error),
        }
    }

    fn read_input(&self) -> io::Result<Option<String>> {
        let Some(mut line) = self.read_line("you> ")? else {
            return Ok(None);
        };
        while line.trim_end().ends_with('\\') {
            let trimmed = line.trim_end().trim_end_matches('\\');
            line = format!(
                "{}\n{}",
                trimmed,
                self.read_line("... ")?.unwrap_or_default()
            );
        }
        Ok(Some(line))
    }

    fn save_history(&self) {
        let path = core::config::history_path();
        if let Some(parent) = path.parent() {
            if fs::create_dir_all(parent).is_err() {
                return;
            }
        }
        let body = self
            .history
            .iter()
            .rev()
            .take(500)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if fs::write(
            &path,
            format!("{}{}", body, if body.is_empty() { "" } else { "\n" }),
        )
        .is_ok()
        {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
            }
        }
    }

    fn banner(&self) {
        let endpoint = core::redact_endpoint(
            &self.session.base_url,
            &[self.session.api_key.as_deref().unwrap_or("")],
        );
        for line in core::startup_banner(
            &self.style,
            core::VERSION,
            &self.session.provider.name,
            &self.session.model,
            &endpoint,
        ) {
            println!("{}", line);
        }
    }

    pub fn run(&mut self) -> i32 {
        self.banner();
        while self.running {
            let line = match self.read_input() {
                Ok(Some(line)) => line,
                Ok(None) => {
                    println!();
                    break;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {
                    println!();
                    continue;
                }
                Err(error) => {
                    self.error(&AnyLLMError::config(format!(
                        "Could not read input: {}",
                        error
                    )));
                    break;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            self.history.push(line.clone());
            if line.starts_with("//") {
                self.send(&line[1..]);
            } else if line.starts_with('/') {
                self.command(&line);
            } else {
                self.send(&line);
            }
        }
        self.save_history();
        0
    }

    fn send(&mut self, text: &str) {
        let mut printer =
            StreamPrinter::new(self.style, self.session.show_reasoning, core::is_tty());
        printer.begin();
        match self.session.send(text, &mut printer) {
            Ok(turn) => {
                printer.end();
                println!(
                    "{}",
                    core::format_reply_status(&self.style, turn.elapsed, Some(&turn.usage))
                );
                if let Some(notice) = core::finish_notice(turn.finish_reason.as_deref()) {
                    println!("{}", self.style.yellow(&format!("! {}", notice)));
                }
                println!();
            }
            Err(error) => {
                printer.end();
                self.error(&error);
            }
        }
    }

    fn error(&self, error: &AnyLLMError) {
        for line in core::format_error(&self.style, error) {
            println!("{}", line);
        }
    }

    fn command(&mut self, line: &str) {
        let raw = line[1..].trim();
        if raw.is_empty() {
            println!("{}", self.style.dim("  type /help for the command list"));
            return;
        }
        let (name, rest) = raw
            .split_once(char::is_whitespace)
            .map(|(n, r)| (n.to_lowercase(), r.trim()))
            .unwrap_or((raw.to_lowercase(), ""));
        let result = match name.as_str() {
            "help" => self.cmd_help(),
            "model" => self.cmd_model(rest),
            "models" => self.cmd_models(rest),
            "provider" => self.cmd_provider(rest),
            "system" => self.cmd_system(rest),
            "temp" => self.cmd_temp(rest),
            "max" => self.cmd_max(rest),
            "reasoning" => self.cmd_reasoning(rest),
            "clear" => {
                self.session.messages.clear();
                self.session.usage = Usage::default();
                self.session.turns = 0;
                println!("{}", self.style.dim("conversation cleared"));
                Ok(())
            }
            "usage" => self.cmd_usage(rest),
            "info" => self.cmd_info(),
            "keys" => self.cmd_keys(),
            "save" => self.cmd_save(rest),
            "load" => self.cmd_load(rest),
            "sessions" => self.cmd_sessions(),
            "exit" | "quit" => {
                self.running = false;
                println!("{}", self.style.dim("bye"));
                Ok(())
            }
            _ => {
                println!(
                    "{}",
                    self.style.yellow(&format!("unknown command /{}", name))
                );
                println!(
                    "{}",
                    self.style.dim(
                        "  /help lists commands; //text sends a message starting with a slash"
                    )
                );
                Ok(())
            }
        };
        if let Err(error) = result {
            self.error(&error);
        }
    }

    fn cmd_help(&self) -> Result<(), AnyLLMError> {
        let rows = vec![
            ("/help", "show this list"),
            ("/model NAME", "switch model"),
            ("/models [TEXT]", "list models available to this key"),
            (
                "/provider [NAME]",
                "switch provider, keeping the conversation",
            ),
            (
                "/system [TEXT]",
                "show or set the system prompt ('-' clears)",
            ),
            ("/temp [VALUE]", "show or set temperature ('off' clears)"),
            (
                "/max [N]",
                "show or set maximum response tokens ('off' clears)",
            ),
            (
                "/reasoning on|off",
                "show or hide the model's thinking stream",
            ),
            ("/clear", "start a new conversation"),
            (
                "/usage [--cost]",
                "session totals; --cost estimates from saved rates",
            ),
            ("/info", "connection details"),
            ("/save [NAME]", "save this conversation"),
            (
                "/load [NAME]",
                "load a saved conversation (latest by default)",
            ),
            ("/sessions", "list saved conversations"),
            ("/keys", "show which key this provider uses"),
            ("/exit", "quit"),
        ];
        println!("{}", self.style.bold("Commands"));
        let rows = rows
            .iter()
            .map(|(a, b)| vec![a.to_string(), b.to_string()])
            .collect::<Vec<_>>();
        for line in core::format_table(&self.style, &["COMMAND", "ACTION"], &rows) {
            println!("  {}", line);
        }
        println!();
        println!(
            "{}",
            self.style
                .muted("  End a line with \\ to continue on the next line.")
        );
        println!(
            "{}",
            self.style
                .muted("  //text sends a message that starts with a slash.")
        );
        println!();
        Ok(())
    }

    fn cmd_model(&mut self, rest: &str) -> Result<(), AnyLLMError> {
        if rest.is_empty() {
            println!("model: {}", self.style.bold(&self.session.model));
            println!(
                "{}",
                self.style
                    .dim("  change it with /model NAME, or list with /models")
            );
        } else {
            self.session.model = rest.to_string();
            println!(
                "{}",
                self.style
                    .dim(&format!("model → {} (this conversation only)", rest))
            );
        }
        Ok(())
    }

    fn cmd_models(&mut self, rest: &str) -> Result<(), AnyLLMError> {
        println!("{}", self.style.dim("fetching models…"));
        let mut models = core::list_models(
            &self.session.provider,
            self.session.api_key.as_deref(),
            Some(&self.session.base_url),
            self.session.timeout.min(60.0),
        )?;
        if !rest.is_empty() {
            let needle = rest.to_lowercase();
            models.retain(|name| name.to_lowercase().contains(&needle));
        }
        if models.is_empty() {
            println!("{}", self.style.yellow("no matching models were returned"));
            return Ok(());
        }
        println!("{}", self.style.bold(&format!("Models ({})", models.len())));
        for line in core::format_table(
            &self.style,
            &["MODEL"],
            &models
                .iter()
                .take(200)
                .map(|m| vec![m.clone()])
                .collect::<Vec<_>>(),
        ) {
            println!("{}", line);
        }
        if models.len() > 200 {
            println!(
                "{}",
                self.style
                    .dim(&format!("  … and {} more", models.len() - 200))
            );
        }
        println!();
        Ok(())
    }

    fn cmd_provider(&mut self, rest: &str) -> Result<(), AnyLLMError> {
        if rest.is_empty() {
            println!("provider: {}", self.style.bold(&self.session.provider.name));
            println!("{}", self.style.dim("  available:"));
            let rows = self
                .providers
                .iter()
                .map(|(name, p)| {
                    let key = core::resolve_key(p, &self.store);
                    let status = if !p.key_required {
                        "no key needed".to_string()
                    } else if key.present() {
                        key.display()
                    } else {
                        "no key".to_string()
                    };
                    vec![
                        if name == &self.session.provider.name {
                            format!("{} *", name)
                        } else {
                            name.clone()
                        },
                        p.kind.clone(),
                        status,
                    ]
                })
                .collect::<Vec<_>>();
            for line in core::format_table(&self.style, &["PROVIDER", "KIND", "KEY"], &rows) {
                println!("  {}", line);
            }
            println!("{}", self.style.dim("  * current provider"));
            println!();
            return Ok(());
        }
        let provider = core::get_provider(rest, &self.providers)?;
        let switching = provider.name != self.session.provider.name;
        if provider.base_url.is_empty() && (switching || self.session.base_url.is_empty()) {
            return Err(AnyLLMError::config(format!("Provider {:?} has no base URL.", provider.name)).with_hint("Add a named endpoint with `anyllm providers add NAME --base-url URL`, or start the chat with --base-url. The current provider was left unchanged."));
        }
        let old_model = self.session.model.clone();
        if switching {
            let key = core::resolve_key(&provider, &self.store);
            if provider.key_required && !key.present() {
                return Err(
                    AnyLLMError::config(format!("No API key for {}.", provider.name)).with_hint(
                        format!(
                            "Run `anyllm keys set {}`, then /provider {} again.",
                            provider.name, provider.name
                        ),
                    ),
                );
            }
            self.session.provider = provider.clone();
            self.session.api_key = key.value;
            self.session.key_source = key.source;
            self.session.target_alias = None;
            self.session.base_url = provider.base_url.clone();
            self.session.model = provider.default_model().to_string();
        } else {
            self.session.provider = provider.clone();
        }
        let shown = if self.session.model.is_empty() {
            "(unset)"
        } else {
            &self.session.model
        };
        println!(
            "{}",
            self.style.dim(&format!(
                "provider → {} · model {} (history kept; /clear to reset)",
                provider.name, shown
            ))
        );
        if self.session.model.is_empty() {
            println!(
                "{}",
                self.style.dim(&format!(
                    "  no default model for {}; set one with /model NAME",
                    provider.name
                ))
            );
            if switching && !old_model.is_empty() {
                println!(
                    "{}",
                    self.style
                        .dim(&format!("  cleared the previous model {}", old_model))
                );
            }
        } else if self.session.model != old_model {
            println!(
                "{}",
                self.style.dim(&format!(
                    "  model reset from {} to {}",
                    old_model, self.session.model
                ))
            );
        }
        Ok(())
    }

    fn cmd_system(&mut self, rest: &str) -> Result<(), AnyLLMError> {
        if rest.is_empty() {
            if let Some(system) = &self.session.system {
                println!("{}", self.style.bold("system prompt:"));
                println!("{}", system);
            } else {
                println!("{}", self.style.dim("no system prompt set"));
            }
        } else if rest == "-" {
            self.session.system = None;
            println!("{}", self.style.dim("system prompt cleared"));
        } else {
            self.session.system = Some(rest.to_string());
            println!(
                "{}",
                self.style.dim("system prompt set (this conversation only)")
            );
        }
        Ok(())
    }

    fn cmd_temp(&mut self, rest: &str) -> Result<(), AnyLLMError> {
        if rest.is_empty() {
            println!(
                "temperature: {}",
                self.session
                    .temperature
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "provider default".into())
            );
            return Ok(());
        }
        if ["off", "none", "-", "default"].contains(&rest.to_lowercase().as_str()) {
            self.session.temperature = None;
            println!("{}", self.style.dim("temperature → provider default"));
            return Ok(());
        }
        let value = rest
            .parse::<f64>()
            .map_err(|_| AnyLLMError::config("invalid float literal"))?;
        if !(0.0..=2.0).contains(&value) {
            return Err(AnyLLMError::config("temperature must be between 0 and 2"));
        }
        self.session.temperature = Some(value);
        println!("{}", self.style.dim(&format!("temperature → {}", value)));
        Ok(())
    }

    fn cmd_max(&mut self, rest: &str) -> Result<(), AnyLLMError> {
        if rest.is_empty() {
            println!(
                "max tokens: {}",
                self.session
                    .max_tokens
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "provider default".into())
            );
            return Ok(());
        }
        if ["off", "none", "-", "default"].contains(&rest.to_lowercase().as_str()) {
            self.session.max_tokens = None;
            println!("{}", self.style.dim("max tokens → provider default"));
            return Ok(());
        }
        let value = rest
            .parse::<i64>()
            .map_err(|_| AnyLLMError::config("invalid integer literal"))?;
        if value <= 0 {
            return Err(AnyLLMError::config("max tokens must be a positive number"));
        }
        self.session.max_tokens = Some(value);
        println!("{}", self.style.dim(&format!("max tokens → {}", value)));
        Ok(())
    }

    fn cmd_reasoning(&mut self, rest: &str) -> Result<(), AnyLLMError> {
        if rest.is_empty() {
            println!(
                "reasoning display: {}",
                if self.session.show_reasoning {
                    "on"
                } else {
                    "off"
                }
            );
            return Ok(());
        }
        match rest.to_lowercase().as_str() {
            "on" | "true" | "1" | "yes" => self.session.show_reasoning = true,
            "off" | "false" | "0" | "no" => self.session.show_reasoning = false,
            _ => return Err(AnyLLMError::config("use /reasoning on or /reasoning off")),
        }
        println!(
            "{}",
            self.style.dim(&format!(
                "reasoning display → {}",
                if self.session.show_reasoning {
                    "on"
                } else {
                    "off"
                }
            ))
        );
        Ok(())
    }

    fn cmd_usage(&self, rest: &str) -> Result<(), AnyLLMError> {
        if rest == "--cost" {
            return Err(AnyLLMError::config(
                "/usage --cost is unavailable until saved-rate pricing is wired into the CLI.",
            ));
        }
        if !rest.is_empty() {
            return Err(AnyLLMError::config("Usage: /usage or /usage --cost"));
        }
        let usage = &self.session.usage;
        if usage.total().is_none() && self.session.turns == 0 {
            println!("{}", self.style.dim("no requests in this session yet"));
            return Ok(());
        }
        println!("{}", self.style.bold("Session usage"));
        println!("  turns:   {}", self.session.turns);
        println!("  input:   {}", opt_count(usage.input_tokens));
        println!("  output:  {}", opt_count(usage.output_tokens));
        if let Some(n) = usage.reasoning_tokens.filter(|n| *n != 0) {
            println!("  reasoning: {}", n);
        }
        if let Some(n) = usage.cached_tokens.filter(|n| *n != 0) {
            println!("  cached:  {}", n);
        }
        if let Some(n) = usage.total() {
            println!("  total:   {}", n);
        }
        println!();
        Ok(())
    }

    fn cmd_info(&self) -> Result<(), AnyLLMError> {
        let p = &self.session.provider;
        println!("{}", self.style.bold("Connection"));
        println!("  provider:  {} ({})", self.style.bold(&p.name), p.label);
        println!("  model:     {}", self.style.bold(&self.session.model));
        println!(
            "  endpoint:  {}",
            self.style.muted(&core::redact_endpoint(
                &self.session.base_url,
                &[self.session.api_key.as_deref().unwrap_or("")]
            ))
        );
        let key = if !p.key_required {
            "not needed".into()
        } else if self.session.key_source.starts_with("env:") {
            self.session.key_source.clone()
        } else if self.session.api_key.is_some() {
            self.session.key_source.clone()
        } else {
            "missing".into()
        };
        println!("  key:       {}", key);
        println!(
            "  system:    {}",
            self.session.system.as_deref().unwrap_or("(none)")
        );
        println!("  messages:  {}", self.session.messages.len());
        println!("  timeout:   {}s", self.session.timeout);
        println!();
        Ok(())
    }

    fn cmd_keys(&self) -> Result<(), AnyLLMError> {
        if !self.session.provider.key_required {
            println!(
                "{}",
                self.style.dim(&format!(
                    "{} does not need an API key",
                    self.session.provider.name
                ))
            );
        } else if self.session.api_key.is_none() {
            println!(
                "{}",
                self.style
                    .yellow(&format!("{}: no API key", self.session.provider.name))
            );
        } else {
            println!(
                "{}: {}",
                self.session.provider.name,
                if self.session.key_source.is_empty() {
                    "configured"
                } else {
                    &self.session.key_source
                }
            );
        }
        Ok(())
    }

    fn cmd_save(&self, rest: &str) -> Result<(), AnyLLMError> {
        if self.session.messages.is_empty() {
            return Err(AnyLLMError::config("Nothing to save yet."));
        }
        let path = core::save_session(
            &self.session,
            if rest.is_empty() { None } else { Some(rest) },
        )?;
        println!("{}", self.style.dim(&format!("saved → {}", path.display())));
        Ok(())
    }

    fn cmd_load(&mut self, rest: &str) -> Result<(), AnyLLMError> {
        let name = if rest.is_empty() {
            core::latest_session_name().ok_or_else(|| {
                AnyLLMError::config("No saved sessions yet.")
                    .with_hint("Use /save NAME to save this conversation first.")
            })?
        } else {
            rest.to_string()
        };
        let data = core::load_session_data(&name)?;
        let provider_name = data.get("provider").and_then(Value::as_str).unwrap_or("");
        let provider = core::get_provider(provider_name, &self.providers)?;
        let model = data
            .get("model")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| provider.default_model().to_string());
        if model.trim().is_empty() {
            return Err(AnyLLMError::config(format!(
                "Session {:?} has no model.",
                name
            )));
        }
        let temperature =
            core::validate_temperature(data.get("temperature").unwrap_or(&Value::Null))?;
        let max_tokens = core::validate_max_tokens(data.get("max_tokens").unwrap_or(&Value::Null))?;
        let object = Map::from_iter(data.iter().map(|(k, v)| (k.clone(), v.clone())));
        let base_url = core::session_base_url(&object, &provider, None)?;
        let key = core::resolve_key(&provider, &self.store);
        if provider.key_required && !key.present() {
            return Err(AnyLLMError::config(format!(
                "No API key for {}.",
                provider.name
            )));
        }
        let mut replacement = Session::new(
            provider.clone(),
            &model,
            Some(&base_url),
            key.value.as_deref(),
            &key.source,
            data.get("system").and_then(Value::as_str),
            temperature,
            max_tokens,
            self.session.timeout,
            data.get("show_reasoning")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        );
        let messages = data
            .get("messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|v| {
                Some(core::Message::new(
                    v.get("role")?.as_str()?,
                    v.get("content")?.as_str()?,
                ))
            })
            .collect::<Vec<_>>();
        replacement.restore(
            messages,
            Usage::from_dict(data.get("usage").unwrap_or(&Value::Null)),
        );
        replacement.target_alias = data
            .get("target")
            .and_then(Value::as_str)
            .map(str::to_string);
        let count = replacement.messages.len();
        self.session = replacement;
        println!(
            "{}",
            self.style.dim(&format!(
                "loaded {} · {}/{} · {} messages",
                name, provider.name, model, count
            ))
        );
        Ok(())
    }

    fn cmd_sessions(&self) -> Result<(), AnyLLMError> {
        let entries = core::list_sessions();
        if entries.is_empty() {
            println!("{}", self.style.dim("no saved sessions"));
            return Ok(());
        }
        println!("{}", self.style.bold("Saved sessions"));
        let rows = entries
            .iter()
            .take(20)
            .map(|e| {
                vec![
                    e.name.clone(),
                    format!("{}/{}", e.provider, e.model),
                    format!("{} msg", e.messages),
                    format_when(&e.updated),
                ]
            })
            .collect::<Vec<_>>();
        for line in core::format_table(&self.style, &["NAME", "TARGET", "SIZE", "UPDATED"], &rows) {
            println!("{}", line);
        }
        println!();
        Ok(())
    }
}

fn opt_count(value: Option<i64>) -> String {
    value.map(|n| n.to_string()).unwrap_or_else(|| "?".into())
}

fn format_when(value: &str) -> String {
    if value.is_empty() {
        return "?".into();
    }
    if value.len() >= 16 {
        value[..16].replace('T', " ")
    } else {
        value.to_string()
    }
}

#[allow(dead_code)]
fn _turn_status(style: &Style, turn: &Turn) -> String {
    core::format_reply_status(style, turn.elapsed, Some(&turn.usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_session_timestamps_for_the_list() {
        assert_eq!(format_when("2026-11-22T12:34:56+00:00"), "2026-11-22 12:34");
        assert_eq!(format_when(""), "?");
        assert_eq!(format_when("bad"), "bad");
    }

    #[test]
    fn missing_usage_is_question_mark_not_zero() {
        assert_eq!(opt_count(None), "?");
        assert_eq!(opt_count(Some(0)), "0");
    }
}
