//! `anyllm` binary: argument parsing, command dispatch, REPL, and rendering.

pub mod args;
pub mod bridge;
pub mod commands;
pub mod render;
pub mod repl;

pub fn run(args: Vec<String>) -> i32 {
    commands::run(args)
}
