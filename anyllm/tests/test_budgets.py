"""Price notes and local gates. No network and no invented provider prices."""

from __future__ import annotations

import os
import sys
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyllm.adapters import Usage
from anyllm.budgets import (
    DOLLAR_NOTE,
    HEURISTIC_NOTE,
    Budget,
    PlannedCall,
    estimate_cost,
    estimate_tokens,
    load_budget,
    load_prices,
    preflight,
    receipt,
    save_budget,
    set_price,
)
from anyllm.errors import ConfigError


def _home(case: unittest.TestCase) -> Path:
    path = ROOT / ".test-run" / f"budgets-{os.getpid()}-{id(case)}"
    path.mkdir(parents=True, exist_ok=True)
    case.addCleanup(lambda: __import__("shutil").rmtree(path, True))
    return path


class BudgetTests(unittest.TestCase):
    def test_decimal_rates_roundtrip_and_do_not_use_binary_float(self):
        home = _home(self)
        with _env(home):
            price = set_price("fast", input_rate="0.20", output_rate="0.40", currency="usd")
            self.assertEqual(price.currency, "USD")
            self.assertEqual(price.input_per_million, Decimal("0.20"))
            stored = load_prices()["fast"]
            self.assertEqual(
                estimate_cost(stored, input_tokens=1_000_000, output_tokens=2_000_000),
                Decimal("1.000000"),
            )
            raw = (home / "prices.json").read_text(encoding="utf-8")
            self.assertIn('"0.20"', raw)
            self.assertIn(stored.updated, raw)

    def test_unknown_usage_is_not_zero(self):
        home = _home(self)
        with _env(home):
            price = set_price("fast", input_rate="1", output_rate="1")
            missing = receipt(None, price)
            self.assertEqual(missing["input_tokens"], "unknown")
            self.assertEqual(missing["estimated_cost"], "unknown")
            self.assertNotEqual(missing["estimated_cost"], 0)
            self.assertNotIn("0.00", str(missing["estimated_cost"]))
            partial = receipt(Usage(input_tokens=10, output_tokens=None), price)
            self.assertEqual(partial["estimated_cost"], "unknown")
            self.assertEqual(partial["output_tokens"], "unknown")

    def test_receipt_does_not_add_reasoning_or_cached_on_top(self):
        home = _home(self)
        with _env(home):
            price = set_price("fast", input_rate="0.20", output_rate="0.40")
            info = receipt(
                Usage(input_tokens=1_000_000, output_tokens=1_000_000, reasoning_tokens=5_000_000, cached_tokens=9),
                price,
            )
            # 0.20 + 0.40, not also the reasoning tokens.
            self.assertEqual(Decimal(info["estimated_cost"]), Decimal("0.600000"))
            self.assertEqual(info["reasoning_tokens"], 5_000_000)
            self.assertEqual(info["cached_tokens"], 9)
            self.assertEqual(info["cost_basis"], "reported")

    def test_dollar_gate_refuses_unknown_and_oversized_estimates(self):
        calls = [
            PlannedCall("fast", "fast", 1_000_000, 1_000_000),
            PlannedCall("slow", "slow", 1_000_000, 1_000_000),
        ]
        prices = {
            "fast": set_price_memory("fast", "0.20", "0.40"),
        }
        unknown = preflight(calls, prices, Budget(estimated_usd=Decimal("10")))
        self.assertFalse(unknown.allowed)
        self.assertIn("unknown", unknown.reason.lower())
        self.assertIn("slow", unknown.reason)
        self.assertIsNone(unknown.estimate)

        both = {
            "fast": prices["fast"],
            "slow": set_price_memory("slow", "0.20", "0.40"),
        }
        over = preflight(calls, both, Budget(estimated_usd=Decimal("0.50")))
        self.assertFalse(over.allowed)
        self.assertGreater(over.estimate, Decimal("0.50"))
        under = preflight(calls, both, Budget(estimated_usd=Decimal("2")))
        self.assertTrue(under.allowed)
        self.assertIn("not a hard spending cap", under.disclosure)
        self.assertIn(HEURISTIC_NOTE.split()[0], under.disclosure)

    def test_request_and_output_caps_are_enforced(self):
        calls = [
            PlannedCall("a", "a", 10, 1000),
            PlannedCall("b", "b", 10, None),
            PlannedCall("c", "c", 10, 50),
        ]
        prices = {name: set_price_memory(name, "0", "0") for name in ("a", "b", "c")}
        too_many = preflight(calls, prices, Budget(max_requests=2))
        self.assertFalse(too_many.allowed)
        self.assertIn("3 requests", too_many.reason)
        too_long = preflight(calls[:1], prices, Budget(max_output_tokens=100))
        self.assertFalse(too_long.allowed)
        filled = preflight(calls[1:2], prices, Budget(max_output_tokens=100))
        self.assertTrue(filled.allowed)
        self.assertEqual(filled.output_caps["b"], 100)

    def test_mixed_currency_is_not_summed(self):
        calls = [
            PlannedCall("a", "a", 1000, 1000),
            PlannedCall("b", "b", 1000, 1000),
        ]
        prices = {
            "a": set_price_memory("a", "1", "1", "USD"),
            "b": set_price_memory("b", "1", "1", "EUR"),
        }
        result = preflight(calls, prices, Budget(estimated_usd=Decimal("1")))
        self.assertFalse(result.allowed)
        self.assertIn("currency", result.reason.lower())
        self.assertIsNone(result.estimate)

    def test_bad_rates_are_rejected(self):
        home = _home(self)
        with _env(home):
            with self.assertRaises(ConfigError):
                set_price("fast", input_rate="-1", output_rate="1")
            with self.assertRaises(ConfigError):
                set_price("fast", input_rate="nope", output_rate="1")
            budget = Budget(max_requests=4, estimated_usd=Decimal("0.02"), max_output_tokens=512)
            save_budget(budget)
            loaded = load_budget()
            self.assertEqual(loaded.max_requests, 4)
            self.assertEqual(loaded.estimated_usd, Decimal("0.02"))

    def test_heuristic_is_labeled_and_stable(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("abcde"), 2)
        self.assertIn("not a provider tokenizer", HEURISTIC_NOTE)
        self.assertIn("not a hard spending cap", DOLLAR_NOTE)


def set_price_memory(name, input_rate, output_rate, currency="USD"):
    from anyllm.budgets import Price

    return Price(
        name=name,
        input_per_million=Decimal(input_rate),
        output_per_million=Decimal(output_rate),
        currency=currency,
        updated="2026-01-01T00:00:00+00:00",
    )


def _env(home: Path):
    return mock.patch.dict(os.environ, {"ANYLLM_HOME": str(home)}, clear=False)


if __name__ == "__main__":
    unittest.main()
