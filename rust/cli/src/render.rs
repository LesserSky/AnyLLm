//! CLI rendering helpers and streamed response printer.
//!
//! Terminal primitives are implemented by `anyllm-core`; this module adds the
//! REPL's live spinner and its code/reasoning presentation.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub use anyllm_core::term::{
    color_enabled, format_error, format_reply_status, format_table, is_tty, redact_endpoint, setup,
    startup_banner, Style,
};

static OUTPUT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn output_lock() -> &'static Mutex<()> {
    OUTPUT_LOCK.get_or_init(|| Mutex::new(()))
}

fn write_stdout(value: &str) {
    let _guard = output_lock().lock().unwrap_or_else(|e| e.into_inner());
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_all(value.as_bytes());
    let _ = stdout.flush();
}

/// Print a streamed response, showing a spinner until the first content.
/// Callers should disable the spinner for piped output.
pub struct StreamPrinter {
    style: Style,
    show_reasoning: bool,
    spinner: bool,
    in_code: bool,
    wrote_text: bool,
    wrote_reasoning: bool,
    last: Option<char>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    indicator: bool,
}

impl StreamPrinter {
    pub fn new(style: Style, show_reasoning: bool, spinner: bool) -> Self {
        Self {
            style,
            show_reasoning,
            spinner,
            in_code: false,
            wrote_text: false,
            wrote_reasoning: false,
            last: None,
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            indicator: false,
        }
    }

    pub fn begin(&mut self) {
        if !self.spinner || self.thread.is_some() {
            return;
        }
        self.stop.store(false, Ordering::SeqCst);
        self.indicator = true;
        let stop = Arc::clone(&self.stop);
        let style = self.style;
        let ansi = style.enabled;
        self.thread = thread::Builder::new()
            .name("anyllm-spinner".to_string())
            .spawn(move || {
                let frames: &[&str] = if ansi {
                    &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                } else {
                    &["|", "/", "-", "\\"]
                };
                let mut index = 0;
                while !stop.load(Ordering::SeqCst) {
                    let frame = frames[index % frames.len()];
                    index += 1;
                    write_stdout(&if ansi {
                        format!("\r{}\x1b[K", style.accent(frame))
                    } else {
                        format!("\r{}", frame)
                    });
                    thread::sleep(Duration::from_millis(80));
                }
            })
            .ok();
    }

    fn clear_indicator(&mut self) {
        if !self.indicator && self.thread.is_none() {
            return;
        }
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            if thread.thread().id() != thread::current().id() {
                let _ = thread.join();
            }
        }
        if self.indicator {
            write_stdout(if self.style.enabled {
                "\r\x1b[2K"
            } else {
                "\r \r"
            });
            self.indicator = false;
        }
    }

    pub fn text(&mut self, value: &str) {
        if value.is_empty() {
            return;
        }
        self.clear_indicator();
        if self.show_reasoning && self.wrote_reasoning && !self.wrote_text {
            if self.last.is_some_and(|c| c != '\n') {
                write_stdout("\n");
            }
        }
        if self.style.enabled {
            self.write_styled(value);
        } else {
            write_stdout(value);
        }
        self.wrote_text = true;
        self.last = value.chars().last();
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
            write_stdout(&self.style.muted("reasoning:\n"));
        }
        write_stdout(&self.style.reasoning(value));
        self.wrote_reasoning = true;
        self.last = value.chars().last();
    }

    fn write_styled(&mut self, mut value: &str) {
        while !value.is_empty() {
            if let Some(index) = value.find("```") {
                self.emit(&value[..index], None);
                self.emit("```", Some(false));
                value = &value[index + 3..];
                self.in_code = !self.in_code;
            } else {
                self.emit(value, None);
                return;
            }
        }
    }

    fn emit(&self, value: &str, code: Option<bool>) {
        if value.is_empty() {
            return;
        }
        let in_code = code.unwrap_or(self.in_code);
        write_stdout(&if in_code {
            self.style.code(value)
        } else {
            value.to_string()
        });
    }

    pub fn end(&mut self) {
        self.clear_indicator();
        if self.last.is_some_and(|c| c != '\n') {
            write_stdout("\n");
            self.last = Some('\n');
        }
    }
}

impl Drop for StreamPrinter {
    fn drop(&mut self) {
        self.clear_indicator();
    }
}
