//! Command dispatch. Port the `cmd_*` functions from `anyllm/cli.py`.

pub fn run(args: Vec<String>) -> i32 {
    let cli = crate::args::parse(args);
    if cli.args.iter().any(|a| a == "--version" || a == "-V") {
        println!("anyllm {}", anyllm_core::VERSION);
        return 0;
    }
    eprintln!("anyllm: Rust port in progress");
    2
}
