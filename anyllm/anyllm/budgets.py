"""User-supplied price notes and local request gates.

Prices are rates the user typed, in currency units per million tokens, with the
date they were saved. They are not fetched from providers and they are not a
bill. Token counts used before a request are a character heuristic (about four
characters per token). Reported usage after a request is whatever the provider
sent; missing numbers stay unknown and are never shown as zero.

A dollar figure is an estimate. It is not a hard spending cap. The gates that
can actually be enforced here are the number of requests and the max_tokens
value sent on each request. If a dollar gate is set and any price or output
cap is missing, the run is refused instead of pretending the estimate covers it.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from .errors import ConfigError, UsageError
from .seam import feature_path, load_feature, register_repl, save_feature

HEURISTIC_NOTE = (
    "Heuristic context estimate only (about 4 characters per token). "
    "This is not a provider tokenizer and not a bill."
)
DOLLAR_NOTE = (
    "Dollar figures are estimates from your saved per-million rates. "
    "They are not a hard spending cap and can be wrong if prices are stale "
    "or the provider counts tokens differently."
)


@dataclass(frozen=True)
class Price:
    name: str
    input_per_million: Decimal
    output_per_million: Decimal
    currency: str
    updated: str


@dataclass(frozen=True)
class Budget:
    max_requests: int | None = None
    max_output_tokens: int | None = None
    estimated_usd: Decimal | None = None
    confirm_above_requests: int = 4
    enabled: bool = True


@dataclass(frozen=True)
class PlannedCall:
    key: str
    label: str
    input_tokens_estimate: int
    output_token_cap: int | None


@dataclass
class Preflight:
    allowed: bool
    reason: str | None
    currency: str | None
    estimate: Decimal | None
    unknown: tuple[str, ...]
    request_count: int
    output_caps: dict[str, int | None]
    disclosure: str


def prices_path():
    return feature_path("prices.json")


def budget_path():
    return feature_path("budget.json")


def _money(value, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ConfigError(
            f"Invalid {name} {value!r}.",
            hint="Use a decimal such as 0.20 (currency units per million tokens).",
        ) from None
    if not number.is_finite() or number < 0:
        raise ConfigError(f"Invalid {name} {value!r}.", hint="Use a non-negative finite number.")
    if number > Decimal("1000000"):
        raise ConfigError(f"{name} is unreasonably large.", hint="Rates are per million tokens.")
    return number


def _currency(value: str | None) -> str:
    text = (value or "USD").strip().upper()
    if len(text) < 3 or len(text) > 8 or not text.isalpha():
        raise ConfigError(
            f"Invalid currency {value!r}.",
            hint="Use a short alphabetic code such as USD.",
        )
    return text


def estimate_tokens(text: str) -> int:
    """Rough upper-leaning character count. Not a tokenizer."""

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_messages(messages, system: str | None = None) -> int:
    parts = []
    if system:
        parts.append(system)
    for item in messages or []:
        if isinstance(item, dict):
            content = item.get("content")
        else:
            content = item[1] if len(item) > 1 else ""
        if isinstance(content, str):
            parts.append(content)
    return estimate_tokens("\n".join(parts))


def load_prices() -> dict[str, Price]:
    data = load_feature(prices_path(), what="prices")
    raw = data.get("prices", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("prices.json is malformed.", hint="The prices object must be a map of names.")
    prices: dict[str, Price] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"Price {name!r} is malformed.")
        try:
            prices[name] = Price(
                name=str(name),
                input_per_million=_money(spec.get("input_per_million"), "input rate"),
                output_per_million=_money(spec.get("output_per_million"), "output rate"),
                currency=_currency(spec.get("currency")),
                updated=str(spec.get("updated") or ""),
            )
        except ConfigError as exc:
            raise ConfigError(f"Price {name!r} is malformed.", hint=exc.hint) from None
    return prices


def save_prices(prices: dict[str, Price]) -> None:
    payload = {
        "prices": {
            name: {
                "input_per_million": format(price.input_per_million, "f"),
                "output_per_million": format(price.output_per_million, "f"),
                "currency": price.currency,
                "updated": price.updated,
            }
            for name, price in prices.items()
        }
    }
    save_feature(prices_path(), payload)


def set_price(name: str, *, input_rate, output_rate, currency: str | None = None) -> Price:
    cleaned = (name or "").strip()
    if not cleaned or len(cleaned) > 80:
        raise ConfigError("A price needs a target name or provider:model key.")
    price = Price(
        name=cleaned,
        input_per_million=_money(input_rate, "input rate"),
        output_per_million=_money(output_rate, "output rate"),
        currency=_currency(currency),
        updated=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    prices = load_prices()
    prices[cleaned] = price
    save_prices(prices)
    return price


def load_budget() -> Budget:
    data = load_feature(budget_path(), what="budget")
    if data.get("enabled") is False:
        return Budget(enabled=False, confirm_above_requests=_confirm(data.get("confirm_above_requests")))
    return Budget(
        max_requests=_positive_int(data.get("max_requests"), "max_requests"),
        max_output_tokens=_positive_int(data.get("max_output_tokens"), "max_output_tokens"),
        estimated_usd=_optional_money(data.get("estimated_usd"), "estimated_usd"),
        confirm_above_requests=_confirm(data.get("confirm_above_requests")),
        enabled=True,
    )


def save_budget(budget: Budget) -> None:
    save_feature(
        budget_path(),
        {
            "enabled": budget.enabled,
            "max_requests": budget.max_requests,
            "max_output_tokens": budget.max_output_tokens,
            "estimated_usd": None if budget.estimated_usd is None else format(budget.estimated_usd, "f"),
            "confirm_above_requests": budget.confirm_above_requests,
        },
    )


def budget_off() -> Budget:
    current = load_budget()
    cleared = Budget(enabled=False, confirm_above_requests=current.confirm_above_requests)
    save_budget(cleared)
    return cleared


def _positive_int(value, name: str) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"Invalid {name} {value!r}.", hint="Use a positive whole number.") from None
    if number <= 0:
        raise ConfigError(f"Invalid {name} {value!r}.", hint="Use a positive whole number.")
    return number


def _optional_money(value, name: str) -> Decimal | None:
    if value is None or value == "":
        return None
    return _money(value, name)


def _confirm(value) -> int:
    if value is None:
        return 4
    number = _positive_int(value, "confirm_above_requests")
    return 4 if number is None else number


def lookup_price(prices: dict[str, Price], *, alias: str | None, provider: str, model: str) -> Price | None:
    if alias and alias in prices:
        return prices[alias]
    composite = f"{provider}:{model}"
    if composite in prices:
        return prices[composite]
    return None


def _quantize(number: Decimal) -> Decimal:
    return number.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def estimate_cost(price: Price, *, input_tokens: int, output_tokens: int) -> Decimal:
    cost = (
        Decimal(input_tokens) * price.input_per_million
        + Decimal(output_tokens) * price.output_per_million
    ) / Decimal(1_000_000)
    return _quantize(cost)


def preflight(
    calls: list[PlannedCall],
    prices: dict[str, Price],
    budget: Budget,
    *,
    dollar_gate: Decimal | None = None,
) -> Preflight:
    """Decide whether a fan-out, including synthesis, may start.

    ``dollar_gate`` is an extra per-invocation ceiling (``--budget-usd``). The
    tighter of that value and the saved budget is used. Neither is a bill.
    """

    request_count = len(calls)
    reasons: list[str] = []
    unknown: list[str] = []
    caps: dict[str, int | None] = {}
    currencies: set[str] = set()
    total: Decimal | None = Decimal(0)
    gate = dollar_gate
    saved_gate = budget.estimated_usd if budget.enabled else None
    if gate is not None and saved_gate is not None:
        gate = min(gate, saved_gate)
    elif gate is None:
        gate = saved_gate

    max_requests = budget.max_requests if budget.enabled else None
    max_output = budget.max_output_tokens if budget.enabled else None
    if max_requests is not None and request_count > max_requests:
        reasons.append(
            f"This would send {request_count} requests, above the budget of {max_requests}."
        )

    for call in calls:
        cap = call.output_token_cap
        if max_output is not None:
            if cap is None:
                cap = max_output
            elif cap > max_output:
                reasons.append(
                    f"{call.label} asks for {cap} output tokens, above the budget of {max_output}."
                )
        caps[call.label] = cap
        price = prices.get(call.key)
        if price is None:
            unknown.append(call.label)
            total = None
            continue
        currencies.add(price.currency)
        if cap is None:
            unknown.append(call.label)
            total = None
            continue
        if total is not None:
            total += estimate_cost(price, input_tokens=call.input_tokens_estimate, output_tokens=cap)

    if len(currencies) > 1:
        total = None
        reasons.append(
            "Saved prices use more than one currency (" + ", ".join(sorted(currencies)) + ")."
        )
    currency = next(iter(currencies)) if len(currencies) == 1 else None

    if gate is not None:
        if unknown or total is None:
            names = ", ".join(unknown) or "one or more targets"
            reasons.append(
                f"A dollar gate is set ({gate}), but the estimate does not cover {names}. "
                "Refusing rather than treating an unknown price as zero."
            )
        elif total > gate:
            reasons.append(
                f"Heuristic estimate {_format_money(total, currency)} is above the dollar gate "
                f"{_format_money(gate, currency or 'USD')}."
            )

    if total is not None and currency:
        money = _format_money(_quantize(total), currency)
        estimate_line = f"Heuristic estimate {money}. {DOLLAR_NOTE}"
    elif unknown:
        estimate_line = (
            "Estimate unknown for: " + ", ".join(unknown) + ". Unknown is not zero. " + DOLLAR_NOTE
        )
    else:
        estimate_line = "Estimate unknown. " + DOLLAR_NOTE

    disclosure = f"{request_count} request(s). {HEURISTIC_NOTE} {estimate_line}"
    allowed = not reasons
    return Preflight(
        allowed=allowed,
        reason="; ".join(reasons) if reasons else None,
        currency=currency,
        estimate=None if total is None else _quantize(total),
        unknown=tuple(unknown),
        request_count=request_count,
        output_caps=caps,
        disclosure=disclosure,
    )


def receipt(usage, price: Price | None) -> dict:
    """Separate reported provider tokens from what was only estimated.

    Reasoning and cached tokens are shown on their own. They are not added on
    top of input and output, because providers usually already include them.
    """

    def reported(value):
        return value if isinstance(value, int) else None

    if usage is None:
        reported_input = reported_output = reported_reasoning = reported_cached = None
    else:
        reported_input = reported(usage.input_tokens)
        reported_output = reported(usage.output_tokens)
        reported_reasoning = reported(usage.reasoning_tokens)
        reported_cached = reported(usage.cached_tokens)

    cost = None
    currency = None
    if (
        price is not None
        and reported_input is not None
        and reported_output is not None
    ):
        cost = format(
            estimate_cost(price, input_tokens=reported_input, output_tokens=reported_output),
            "f",
        )
        currency = price.currency
    return {
        "input_tokens": "unknown" if reported_input is None else reported_input,
        "output_tokens": "unknown" if reported_output is None else reported_output,
        "reasoning_tokens": "unknown" if reported_reasoning is None else reported_reasoning,
        "cached_tokens": "unknown" if reported_cached is None else reported_cached,
        "estimated_cost": "unknown" if cost is None else cost,
        "currency": currency,
        "cost_basis": "reported" if cost is not None else "unknown",
        "note": DOLLAR_NOTE if cost is not None else "Cost unknown. Missing usage or price is not shown as zero.",
    }


def _format_money(amount: Decimal, currency: str | None) -> str:
    return f"{format(amount, 'f')} {currency or ''}".strip()


def tighter_dollar_gate(text: str | None, budget: Budget) -> Decimal | None:
    if text is None or str(text).strip() == "":
        return None
    return _money(text, "budget")


def register(register_command=register_repl) -> None:
    register_command("budget", repl_budget, ("/budget ...", "request and estimate gates"))


def repl_budget(repl, rest: str) -> None:
    try:
        tokens = shlex.split(rest or "")
    except ValueError as exc:
        raise UsageError(f"Could not parse /budget: {exc}") from None
    if not tokens or tokens[0] in {"show", "status"}:
        _print_budget(repl, load_budget())
        return
    if tokens[0] == "off":
        budget_off()
        repl.say(repl.style.dim("budget gates off (partial-merge confirmation still applies)"))
        return
    if tokens[0] != "set":
        raise UsageError(
            "Usage: /budget set --max-requests N --max-output-tokens N --estimated-usd N, or /budget off."
        )
    current = load_budget()
    values = {
        "max_requests": current.max_requests,
        "max_output_tokens": current.max_output_tokens,
        "estimated_usd": current.estimated_usd,
        "confirm_above_requests": current.confirm_above_requests,
    }
    index = 1
    flags = {
        "--max-requests": "max_requests",
        "--max-output-tokens": "max_output_tokens",
        "--estimated-usd": "estimated_usd",
        "--confirm-above-requests": "confirm_above_requests",
    }
    while index < len(tokens):
        flag = tokens[index]
        if flag not in flags:
            raise UsageError(f"Unknown /budget flag {flag!r}.")
        if index + 1 >= len(tokens):
            raise UsageError(f"{flag} needs a value.")
        raw = tokens[index + 1]
        key = flags[flag]
        if key == "estimated_usd":
            values[key] = _money(raw, "estimated_usd")
        else:
            values[key] = _positive_int(raw, key)
        index += 2
    updated = Budget(
        max_requests=values["max_requests"],
        max_output_tokens=values["max_output_tokens"],
        estimated_usd=values["estimated_usd"],
        confirm_above_requests=values["confirm_above_requests"] or 4,
        enabled=True,
    )
    save_budget(updated)
    _print_budget(repl, updated)


def _print_budget(repl, budget: Budget) -> None:
    if not budget.enabled:
        repl.say("budget: off")
        return
    repl.say("budget:")
    repl.say(f"  max requests: {budget.max_requests if budget.max_requests is not None else 'unset'}")
    repl.say(
        "  max output tokens: "
        + (str(budget.max_output_tokens) if budget.max_output_tokens is not None else "unset")
    )
    money = "unset" if budget.estimated_usd is None else format(budget.estimated_usd, "f")
    repl.say(f"  estimated usd gate: {money}")
    repl.say(f"  confirm above requests: {budget.confirm_above_requests}")
    repl.say(repl.style.dim("  " + DOLLAR_NOTE))


def repl_usage_cost(repl) -> None:
    prices = load_prices()
    session = repl.session
    alias = getattr(session, "target_alias", None)
    price = lookup_price(
        prices,
        alias=alias,
        provider=session.provider.name,
        model=session.model,
    )
    info = receipt(session.usage if session.turns else None, price)
    repl.say(repl.style.bold("Usage cost"))
    repl.say(f"  input:     {info['input_tokens']}")
    repl.say(f"  output:    {info['output_tokens']}")
    repl.say(f"  reasoning: {info['reasoning_tokens']} (not added on top of output)")
    repl.say(f"  cached:    {info['cached_tokens']} (not added on top of input)")
    repl.say(f"  estimate:  {info['estimated_cost']} {info['currency'] or ''}".rstrip())
    repl.say(repl.style.dim("  " + info["note"]))
