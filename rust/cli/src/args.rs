//! Command-line parsing. Port `anyllm/cli.py::build_parser` and `main`.

/// Parsed command line. Replace this stub with the real parser.
#[derive(Debug, Default)]
pub struct Cli {
    pub args: Vec<String>,
}

pub fn parse(args: Vec<String>) -> Cli {
    Cli { args }
}
