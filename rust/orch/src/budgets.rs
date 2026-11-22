//! User-supplied price notes and local request gates. Port of
//! `anyllm.budgets`.
//!
//! Prices are rates the user typed, in currency units per million tokens,
//! with the date they were saved. They are not fetched from providers and
//! they are not a bill. A dollar figure is an estimate, not a hard spending
//! cap. The gates that can actually be enforced here are the number of
//! requests and the `max_tokens` value sent on each request.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::decimal::Decimal;
use crate::seam::{feature_path, load_feature, save_feature};
use crate::sys::{split_words, Error, Result, Style, Usage};

pub const HEURISTIC_NOTE: &str = "Heuristic context estimate only (about 4 characters per \
token). This is not a provider tokenizer and not a bill.";
pub const DOLLAR_NOTE: &str = "Dollar figures are estimates from your saved per-million \
rates. They are not a hard spending cap and can be wrong if prices are stale or the provider \
counts tokens differently.";

/// Mirrors `anyllm.budgets.Price`.
#[derive(Clone, Debug, PartialEq)]
pub struct Price {
    pub name: String,
    pub input_per_million: Decimal,
    pub output_per_million: Decimal,
    pub currency: String,
    pub updated: String,
}

/// Mirrors `anyllm.budgets.Budget`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Budget {
    pub max_requests: Option<i64>,
    pub max_output_tokens: Option<i64>,
    pub estimated_usd: Option<Decimal>,
    pub confirm_above_requests: i64,
    pub enabled: bool,
}

impl Budget {
    /// Python's dataclass default: enabled with the 4-request threshold.
    pub fn new() -> Budget {
        Budget {
            max_requests: None,
            max_output_tokens: None,
            estimated_usd: None,
            confirm_above_requests: 4,
            enabled: true,
        }
    }
}

/// Mirrors `anyllm.budgets.PlannedCall`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedCall {
    pub key: String,
    pub label: String,
    pub input_tokens_estimate: i64,
    pub output_token_cap: Option<i64>,
}

impl PlannedCall {
    pub fn new(key: &str, label: &str, input_tokens_estimate: i64, output_token_cap: Option<i64>) -> PlannedCall {
        PlannedCall {
            key: key.to_string(),
            label: label.to_string(),
            input_tokens_estimate,
            output_token_cap,
        }
    }
}

/// Mirrors `anyllm.budgets.Preflight`.
#[derive(Clone, Debug)]
pub struct Preflight {
    pub allowed: bool,
    pub reason: Option<String>,
    pub currency: Option<String>,
    pub estimate: Option<Decimal>,
    pub unknown: Vec<String>,
    pub request_count: usize,
    pub output_caps: BTreeMap<String, Option<i64>>,
    pub disclosure: String,
}

pub fn prices_path() -> std::path::PathBuf {
    feature_path("prices.json")
}

pub fn budget_path() -> std::path::PathBuf {
    feature_path("budget.json")
}

/// Parse and validate a money value. Mirrors `budgets._money`.
fn money(value: &Value, name: &str) -> Result<Decimal> {
    let text = match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        _ => String::new(),
    };
    let number = Decimal::parse(&text).map_err(|_| {
        Error::config(
            format!("Invalid {} {:?}", name, value_text(value)),
            Some("Use a decimal such as 0.20 (currency units per million tokens).".to_string()),
        )
    })?;
    if number.cmp_decimal(&Decimal::from_int(0)) == std::cmp::Ordering::Less {
        // Decimal rejects negatives at parse time; this arm is unreachable
        // but keeps the error text of the Python guard.
        return Err(Error::config(
            format!("Invalid {} {:?}", name, value_text(value)),
            Some("Use a non-negative finite number.".to_string()),
        ));
    }
    if number.cmp_decimal(&Decimal::from_int(1_000_000)) == std::cmp::Ordering::Greater {
        return Err(Error::config(
            format!("{} is unreasonably large.", name),
            Some("Rates are per million tokens.".to_string()),
        ));
    }
    Ok(number)
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => format!("'{}'", text),
        other => other.to_string(),
    }
}

/// Parse and validate a currency code. Mirrors `budgets._currency`.
fn currency(value: Option<&str>) -> Result<String> {
    let text = value.unwrap_or("USD").trim().to_uppercase();
    let valid = text.len() >= 3 && text.len() <= 8 && text.chars().all(|c| c.is_ascii_alphabetic());
    if !valid {
        return Err(Error::config(
            format!("Invalid currency {:?}", value),
            Some("Use a short alphabetic code such as USD.".to_string()),
        ));
    }
    Ok(text)
}

/// Rough upper-leaning character count. Not a tokenizer.
/// Mirrors `budgets.estimate_tokens`.
pub fn estimate_tokens(text: &str) -> i64 {
    if text.is_empty() {
        return 0;
    }
    let length = text.chars().count() as i64;
    (length + 3) / 4
}

/// Mirrors `budgets.estimate_messages`.
pub fn estimate_messages(messages: &[crate::sys::Message], system: Option<&str>) -> i64 {
    let mut parts: Vec<String> = Vec::new();
    if let Some(text) = system {
        parts.push(text.to_string());
    }
    for message in messages {
        parts.push(message.content.clone());
    }
    estimate_tokens(&parts.join("\n"))
}

/// Mirrors `budgets.load_prices`.
pub fn load_prices() -> Result<BTreeMap<String, Price>> {
    let data = load_feature(&prices_path(), "prices")?;
    let raw = match data.get("prices") {
        Some(Value::Object(map)) => map.clone(),
        Some(Value::Null) | None => Map::new(),
        _ => {
            return Err(Error::config_hint(
                "prices.json is malformed.",
                "The prices object must be a map of names.".to_string(),
            ))
        }
    };
    let mut prices = BTreeMap::new();
    for (name, spec) in &raw {
        if !spec.is_object() {
            return Err(Error::config(format!("Price {:?} is malformed.", name), None));
        }
        let input_per_million = money(
            spec.get("input_per_million").unwrap_or(&Value::Null),
            "input rate",
        )
        .map_err(|exc| Error::config(format!("Price {:?} is malformed.", name), exc.hint))?;
        let output_per_million = money(
            spec.get("output_per_million").unwrap_or(&Value::Null),
            "output rate",
        )
        .map_err(|exc| Error::config(format!("Price {:?} is malformed.", name), exc.hint))?;
        let currency_text = currency(
            spec.get("currency").and_then(Value::as_str),
        )
        .map_err(|exc| Error::config(format!("Price {:?} is malformed.", name), exc.hint))?;
        prices.insert(
            name.clone(),
            Price {
                name: name.clone(),
                input_per_million,
                output_per_million,
                currency: currency_text,
                updated: spec
                    .get("updated")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            },
        );
    }
    Ok(prices)
}

/// Mirrors `budgets.save_prices`. Rates are written with `format(x, "f")`.
pub fn save_prices(prices: &BTreeMap<String, Price>) -> Result<()> {
    let mut map = Map::new();
    for (name, price) in prices {
        map.insert(
            name.clone(),
            json!({
                "input_per_million": price.input_per_million.format_f(),
                "output_per_million": price.output_per_million.format_f(),
                "currency": price.currency,
                "updated": price.updated,
            }),
        );
    }
    let mut payload = Map::new();
    payload.insert("prices".to_string(), Value::Object(map));
    save_feature(&prices_path(), payload)
}

/// Mirrors `budgets.set_price`.
pub fn set_price(
    name: &str,
    input_rate: &Value,
    output_rate: &Value,
    currency_text: Option<&str>,
    updated: &str,
) -> Result<Price> {
    let cleaned = name.trim();
    if cleaned.is_empty() || cleaned.chars().count() > 80 {
        return Err(Error::config(
            "A price needs a target name or provider:model key.",
            None,
        ));
    }
    let price = Price {
        name: cleaned.to_string(),
        input_per_million: money(input_rate, "input rate")?,
        output_per_million: money(output_rate, "output rate")?,
        currency: currency(currency_text)?,
        updated: updated.to_string(),
    };
    let mut prices = load_prices()?;
    prices.insert(cleaned.to_string(), price.clone());
    save_prices(&prices)?;
    Ok(price)
}

fn positive_int(value: Option<&Value>, name: &str) -> Result<Option<i64>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => match number.as_i64() {
            Some(parsed) if parsed > 0 => Ok(Some(parsed)),
            _ => Err(Error::config(
                format!("Invalid {} {}", name, value_text(value.unwrap_or(&Value::Null))),
                Some("Use a positive whole number.".to_string()),
            )),
        },
        Some(Value::String(text)) => match text.parse::<i64>() {
            Ok(parsed) if parsed > 0 => Ok(Some(parsed)),
            _ => Err(Error::config(
                format!("Invalid {} {:?}", name, text),
                Some("Use a positive whole number.".to_string()),
            )),
        },
        _ => Err(Error::config(
            format!("Invalid {} {}", name, value_text(value.unwrap_or(&Value::Null))),
            Some("Use a positive whole number.".to_string()),
        )),
    }
}

fn optional_money(value: Option<&Value>, name: &str) -> Result<Option<Decimal>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text.is_empty() => Ok(None),
        Some(other) => money(other, name).map(Some),
    }
}

fn confirm_value(value: Option<&Value>) -> Result<i64> {
    match value {
        None => Ok(4),
        Some(other) => {
            let number = positive_int(Some(other), "confirm_above_requests")?;
            Ok(number.unwrap_or(4))
        }
    }
}

/// Mirrors `budgets.load_budget`.
pub fn load_budget() -> Result<Budget> {
    let data = load_feature(&budget_path(), "budget")?;
    if data.get("enabled") == Some(&Value::Bool(false)) {
        return Ok(Budget {
            enabled: false,
            confirm_above_requests: confirm_value(data.get("confirm_above_requests"))?,
            ..Budget::default()
        });
    }
    Ok(Budget {
        max_requests: positive_int(data.get("max_requests"), "max_requests")?,
        max_output_tokens: positive_int(data.get("max_output_tokens"), "max_output_tokens")?,
        estimated_usd: optional_money(data.get("estimated_usd"), "estimated_usd")?,
        confirm_above_requests: confirm_value(data.get("confirm_above_requests"))?,
        enabled: true,
    })
}

/// Mirrors `budgets.save_budget`.
pub fn save_budget(budget: &Budget) -> Result<()> {
    let payload = json!({
        "enabled": budget.enabled,
        "max_requests": budget.max_requests,
        "max_output_tokens": budget.max_output_tokens,
        "estimated_usd": budget.estimated_usd.map(|value| value.format_f()),
        "confirm_above_requests": budget.confirm_above_requests,
    });
    let object = payload.as_object().cloned().unwrap_or_default();
    save_feature(&budget_path(), object)
}

/// Mirrors `budgets.budget_off`.
pub fn budget_off() -> Result<Budget> {
    let current = load_budget()?;
    let cleared = Budget {
        enabled: false,
        confirm_above_requests: current.confirm_above_requests,
        ..Budget::default()
    };
    save_budget(&cleared)?;
    Ok(cleared)
}

/// Mirrors `budgets.lookup_price`.
pub fn lookup_price<'a>(
    prices: &'a BTreeMap<String, Price>,
    alias: Option<&str>,
    provider: &str,
    model: &str,
) -> Option<&'a Price> {
    if let Some(alias) = alias {
        if let Some(price) = prices.get(alias) {
            return Some(price);
        }
    }
    prices.get(&format!("{}:{}", provider, model))
}

/// Mirrors `budgets.estimate_cost`.
pub fn estimate_cost(price: &Price, input_tokens: i64, output_tokens: i64) -> Decimal {
    let cost = price
        .input_per_million
        .mul_int(input_tokens.max(0) as u128)
        .add(price.output_per_million.mul_int(output_tokens.max(0) as u128))
        .div_million();
    cost.quantize6()
}

/// Decide whether a fan-out, including synthesis, may start.
/// Mirrors `budgets.preflight`.
pub fn preflight(
    calls: &[PlannedCall],
    prices: &BTreeMap<String, Price>,
    budget: &Budget,
    dollar_gate: Option<Decimal>,
) -> Preflight {
    let request_count = calls.len();
    let mut reasons: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    let mut caps: BTreeMap<String, Option<i64>> = BTreeMap::new();
    let mut currencies: Vec<String> = Vec::new();
    let mut total: Option<Decimal> = Some(Decimal::ZERO);
    let mut gate = dollar_gate;
    let saved_gate = if budget.enabled { budget.estimated_usd } else { None };
    if let (Some(dollar), Some(saved)) = (gate, saved_gate) {
        gate = Some(if dollar.cmp_decimal(&saved) == std::cmp::Ordering::Less {
            dollar
        } else {
            saved
        });
    } else if gate.is_none() {
        gate = saved_gate;
    }

    let max_requests = if budget.enabled { budget.max_requests } else { None };
    let max_output = if budget.enabled { budget.max_output_tokens } else { None };
    if let Some(max) = max_requests {
        if request_count > max as usize {
            reasons.push(format!(
                "This would send {} requests, above the budget of {}.",
                request_count, max
            ));
        }
    }

    for call in calls {
        let mut cap = call.output_token_cap;
        if let Some(max_output) = max_output {
            match cap {
                None => cap = Some(max_output),
                Some(value) if value > max_output => reasons.push(format!(
                    "{} asks for {} output tokens, above the budget of {}.",
                    call.label, value, max_output
                )),
                _ => {}
            }
        }
        caps.insert(call.label.clone(), cap);
        let price = prices.get(&call.key);
        let price = match price {
            Some(price) => price,
            None => {
                unknown.push(call.label.clone());
                total = None;
                continue;
            }
        };
        if !currencies.contains(&price.currency) {
            currencies.push(price.currency.clone());
        }
        match cap {
            None => {
                unknown.push(call.label.clone());
                total = None;
            }
            Some(cap) => {
                if let Some(running) = total {
                    total = Some(
                        running.add(estimate_cost(
                            price,
                            call.input_tokens_estimate,
                            cap,
                        )),
                    );
                }
            }
        }
    }

    if currencies.len() > 1 {
        total = None;
        currencies.sort();
        reasons.push(format!(
            "Saved prices use more than one currency ({}).",
            currencies.join(", ")
        ));
    }
    let currency = if currencies.len() == 1 {
        currencies.first().cloned()
    } else {
        None
    };

    if let Some(gate) = gate {
        // Python is a three-way elif chain: a non-USD currency only raises
        // the currency reason; an unknown or missing estimate only raises
        // the coverage reason; otherwise the total is compared with the gate.
        if currency.as_deref().map(|value| value != "USD").unwrap_or(false) {
            reasons.push(format!(
                "The budget gate is USD ({}), but the saved rates are {}. \
Refusing rather than treating those rates as US dollars.",
                gate.format_f(),
                currency.as_deref().unwrap_or_default()
            ));
            total = None;
        } else if !unknown.is_empty() || total.is_none() {
            let names = if unknown.is_empty() {
                "one or more targets".to_string()
            } else {
                unknown.join(", ")
            };
            reasons.push(format!(
                "A USD gate is set ({}), but the estimate does not cover {}. \
Refusing rather than treating an unknown price as zero.",
                gate.format_f(),
                names
            ));
        } else if let Some(total) = total {
            if total.cmp_decimal(&gate) == std::cmp::Ordering::Greater {
                reasons.push(format!(
                    "Heuristic estimate {} is above the USD gate {}.",
                    format_money(&total, "USD"),
                    format_money(&gate, "USD")
                ));
            }
        }
    }

    let estimate_line = if total.is_some() && currency.is_some() {
        let money = format_money(&total.unwrap().quantize6(), currency.as_deref().unwrap_or(""));
        format!("Heuristic estimate {}. {}", money, DOLLAR_NOTE)
    } else if !unknown.is_empty() {
        format!(
            "Estimate unknown for: {}. Unknown is not zero. {}",
            unknown.join(", "),
            DOLLAR_NOTE
        )
    } else {
        format!("Estimate unknown. {}", DOLLAR_NOTE)
    };

    let disclosure = format!(
        "{} request(s). {} {}",
        request_count, HEURISTIC_NOTE, estimate_line
    );
    let allowed = reasons.is_empty();
    Preflight {
        allowed,
        reason: if reasons.is_empty() { None } else { Some(reasons.join("; ")) },
        currency,
        estimate: total.map(|value| value.quantize6()),
        unknown,
        request_count,
        output_caps: caps,
        disclosure,
    }
}

/// Separate reported provider tokens from what was only estimated.
/// Mirrors `budgets.receipt`, including the exact JSON shape.
pub fn receipt(usage: Option<&Usage>, price: Option<&Price>) -> Value {
    let reported = |value: Option<i64>| -> Value {
        match value {
            Some(number) => Value::from(number),
            None => Value::from("unknown"),
        }
    };
    let (input, output, reasoning, cached) = match usage {
        None => (None, None, None, None),
        Some(usage) => (
            usage.input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            usage.cached_tokens,
        ),
    };
    let mut cost: Option<Value> = None;
    let mut currency: Option<String> = None;
    if let (Some(price), Some(input), Some(output)) = (price, input, output) {
        cost = Some(Value::from(estimate_cost(price, input, output).format_f()));
        currency = Some(price.currency.clone());
    }
    json!({
        "input_tokens": reported(input),
        "output_tokens": reported(output),
        "reasoning_tokens": reported(reasoning),
        "cached_tokens": reported(cached),
        "estimated_cost": cost.clone().unwrap_or(Value::from("unknown")),
        "currency": currency,
        "cost_basis": if cost.is_some() { "reported" } else { "unknown" },
        "note": if cost.is_some() {
            DOLLAR_NOTE
        } else {
            "Cost unknown. Missing usage or price is not shown as zero."
        },
    })
}

fn format_money(amount: &Decimal, currency: &str) -> String {
    format!("{} {}", amount.format_f(), currency)
        .trim()
        .to_string()
}

/// Mirrors `budgets.tighter_dollar_gate`.
pub fn tighter_dollar_gate(text: Option<&str>, _budget: &Budget) -> Result<Option<Decimal>> {
    match text {
        None => Ok(None),
        Some(text) if text.trim().is_empty() => Ok(None),
        Some(text) => money(&Value::from(text), "budget").map(Some),
    }
}

// -------------------------------------------------------------- REPL layer

/// Mirrors `anyllm.budgets.repl_budget`.
pub fn repl_budget(repl: &mut dyn crate::seam::Repl, rest: &str) -> Result<()> {
    let tokens = split_words(rest).map_err(|exc| Error::usage(format!("Could not parse /budget: {}", exc), None))?;
    if tokens.is_empty() || tokens[0] == "show" || tokens[0] == "status" {
        let budget = load_budget()?;
        let style = *repl.style();
        for line in budget_lines(&style, &budget) {
            repl.say(&line);
        }
        return Ok(());
    }
    if tokens[0] == "off" {
        budget_off()?;
        repl.say(&repl
            .style()
            .dim("budget gates off (partial-merge confirmation still applies)"));
        return Ok(());
    }
    if tokens[0] != "set" {
        return Err(Error::usage(
            "Usage: /budget set --max-requests N --max-output-tokens N --estimated-usd N, \
or /budget off.",
            None,
        ));
    }
    let current = load_budget()?;
    let mut max_requests = current.max_requests;
    let mut max_output_tokens = current.max_output_tokens;
    let mut estimated_usd = current.estimated_usd;
    let mut confirm_above_requests = current.confirm_above_requests;
    let flags: [(&str, &str); 4] = [
        ("--max-requests", "max_requests"),
        ("--max-output-tokens", "max_output_tokens"),
        ("--estimated-usd", "estimated_usd"),
        ("--confirm-above-requests", "confirm_above_requests"),
    ];
    let mut index = 1;
    while index < tokens.len() {
        let flag = &tokens[index];
        let key = match flags.iter().find(|(name, _)| name == flag) {
            Some((_, key)) => *key,
            None => {
                return Err(Error::usage(format!("Unknown /budget flag {:?}.", flag), None));
            }
        };
        if index + 1 >= tokens.len() {
            return Err(Error::usage(format!("{} needs a value.", flag), None));
        }
        let raw = tokens[index + 1].clone();
        if key == "estimated_usd" {
            estimated_usd = Some(money(&Value::from(raw), "estimated_usd")?);
        } else {
            let parsed = positive_int(Some(&Value::from(raw)), key)?;
            match key {
                "max_requests" => max_requests = parsed,
                "max_output_tokens" => max_output_tokens = parsed,
                "confirm_above_requests" => confirm_above_requests = parsed.unwrap_or(4),
                _ => {}
            }
        }
        index += 2;
    }
    let updated = Budget {
        max_requests,
        max_output_tokens,
        estimated_usd,
        confirm_above_requests: if confirm_above_requests == 0 { 4 } else { confirm_above_requests },
        enabled: true,
    };
    save_budget(&updated)?;
    let style = *repl.style();
    for line in budget_lines(&style, &updated) {
        repl.say(&line);
    }
    Ok(())
}

/// Mirrors `anyllm.budgets._print_budget`, rendered into lines.
pub fn budget_lines(style: &Style, budget: &Budget) -> Vec<String> {
    let mut lines = Vec::new();
    if !budget.enabled {
        lines.push("budget: off".to_string());
        return lines;
    }
    lines.push("budget:".to_string());
    lines.push(format!(
        "  max requests: {}",
        budget
            .max_requests
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unset".to_string())
    ));
    lines.push(format!(
        "  max output tokens: {}",
        budget
            .max_output_tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unset".to_string())
    ));
    let money = budget
        .estimated_usd
        .as_ref()
        .map(|value| value.format_f())
        .unwrap_or_else(|| "unset".to_string());
    lines.push(format!("  estimated usd gate: {}", money));
    lines.push(format!("  confirm above requests: {}", budget.confirm_above_requests));
    lines.push(style.dim(&format!("  {}", DOLLAR_NOTE)));
    lines
}

/// Mirrors `anyllm.budgets.repl_usage_cost`, rendered into lines.
pub fn usage_cost_lines(
    style: &Style,
    prices: &BTreeMap<String, Price>,
    alias: Option<&str>,
    provider_name: &str,
    model: &str,
    usage: Option<&Usage>,
) -> Vec<String> {
    let price = lookup_price(prices, alias, provider_name, model);
    let info = receipt(usage, price);
    let get = |key: &str| info.get(key).cloned().unwrap_or(Value::Null);
    let text = |key: &str| match get(key) {
        Value::String(text) => text,
        Value::Number(number) => number.to_string(),
        _ => "unknown".to_string(),
    };
    let mut lines = Vec::new();
    lines.push(style.bold("Usage cost"));
    lines.push(format!("  input:     {}", text("input_tokens")));
    lines.push(format!("  output:    {}", text("output_tokens")));
    lines.push(format!("  reasoning: {} (not added on top of output)", text("reasoning_tokens")));
    lines.push(format!("  cached:    {} (not added on top of input)", text("cached_tokens")));
    let currency_value = get("currency");
    let currency = currency_value.as_str().unwrap_or_default();
    lines.push(format!(
        "  estimate:  {} {}",
        text("estimated_cost"),
        currency
    )
    .trim_end()
    .to_string());
    lines.push(style.dim(&format!("  {}", info.get("note").and_then(Value::as_str).unwrap_or(""))));
    lines
}

/// Mirrors `anyllm.budgets.register`.
pub fn register(registry: &mut crate::seam::ReplRegistry) {
    registry.register(
        "budget",
        crate::seam::handler(|repl, rest| repl_budget(repl, rest)),
        Some(("/budget ...", "request and estimate gates")),
    );
}
