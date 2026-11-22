//! AnyLLM orchestration layer in Rust: named targets, price/budget gates,
//! and the integration seam.
//!
//! Port of `anyllm.targets`, `anyllm.budgets`, and `anyllm.seam`. On-disk JSON
//! shapes match the Python version exactly, so `targets.json`, `prices.json`,
//! and `budget.json` stay compatible between the two implementations.
//!
//! `anyllm.team` and `anyllm.tool_protocol` are ported by a sibling worker
//! into `src/team.rs` and `src/tool_protocol.rs`; they code against the
//! `sys`, `seam`, `targets`, and `budgets` APIs in this crate.

pub mod budgets;
pub mod decimal;
pub mod seam;
pub mod sys;
pub mod targets;
