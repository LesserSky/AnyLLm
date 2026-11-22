//! CLI argument surface. Kept dependency-free so parsing and help work before
//! any provider or configuration is loaded.

#[derive(Debug, Default)]
pub struct Cli {
    pub args: Vec<String>,
}

pub fn parse(args: Vec<String>) -> Cli {
    Cli { args }
}

pub const HELP: &str = "Chat with any LLM API key from your terminal.\n\nUsage: anyllm [--no-color] <command> [options]\n\nCommands:\n  chat          start an interactive conversation\n  ask           send one prompt and print the answer\n  models        list the models available to your key\n  providers     list or define providers\n  keys          save, list, or remove API keys\n  config        read or change saved defaults\n  doctor        check keys, endpoints, and saved settings\n  targets       save named provider/model targets\n  prices        save per-million token rates you supply\n  team          ask several models the same question\n  tools         inspect or run opt-in tools\n  capabilities  probe an endpoint capability\n\nOptions:\n  -h, --help    show this help message and exit\n  --version     show program's version number and exit\n  --no-color    disable ANSI colors\n\nExamples:\n  anyllm keys set openai\n  anyllm ask --provider openai --model gpt-4o-mini \"What is Rust?\"\n  anyllm chat\n";

pub fn command_help(command: &str) -> Option<&'static str> {
    match command {
        "ask" => Some("Send one prompt and print the answer. Reads stdin when no prompt is given.\n\nUsage: anyllm ask [options] [prompt ...]\n\nOptions:\n  --provider NAME\n  --model NAME\n  --base-url URL\n  --system TEXT\n  --temperature N\n  --max-tokens N\n  --timeout SECONDS\n  --reasoning\n  --json\n  --no-stream\n  --no-color\n"),
        "chat" => Some("Start an interactive conversation. Piped stdin is answered once.\n\nUsage: anyllm chat [options]\n"),
        "models" => Some("List model names returned by the provider.\n\nUsage: anyllm models [provider] [--filter TEXT] [--base-url URL] [--json]\n"),
        _ => None,
    }
}
