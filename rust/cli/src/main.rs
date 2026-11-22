fn main() {
    std::process::exit(anyllm_cli::run(std::env::args().skip(1).collect()));
}
