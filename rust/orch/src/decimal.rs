//! A small unsigned fixed-point decimal.
//!
//! The Python code uses `decimal.Decimal` for price notes: exact base-10
//! arithmetic, `ROUND_HALF_UP` quantization to six places, and `format(x, "f")`
//! which keeps the parsed scale. This module reproduces exactly the subset of
//! that behavior the budgets code relies on, without pulling in an external
//! decimal crate.

use std::cmp::Ordering;
use std::fmt;

const MAX_SCALE: u32 = 24;

/// Non-negative fixed-point decimal: `mantissa * 10^-scale`.
#[derive(Clone, Copy, Debug)]
pub struct Decimal {
    mantissa: u128,
    scale: u32,
}

#[derive(Debug)]
pub struct DecimalError;

impl Decimal {
    pub const ZERO: Decimal = Decimal { mantissa: 0, scale: 0 };
    pub const ONE: Decimal = Decimal { mantissa: 1, scale: 0 };

    /// Parse a plain or scientific decimal literal, as `Decimal(str)` does.
    pub fn parse(text: &str) -> Result<Decimal, DecimalError> {
        let raw = text.trim();
        if raw.is_empty() {
            return Err(DecimalError);
        }
        let bytes = raw.as_bytes();
        let mut index = 0;
        if bytes[0] == b'+' {
            index = 1;
        } else if bytes[0] == b'-' {
            // The callers never accept negative values, but the parser must
            // recognize them so validation can reject them with a clear error.
            return Err(DecimalError);
        }
        let mut digits_before: Vec<u8> = Vec::new();
        let mut digits_after: Vec<u8> = Vec::new();
        let mut seen_digit = false;
        let mut seen_dot = false;
        let mut exponent: i64 = 0;
        while index < bytes.len() {
            let byte = bytes[index];
            match byte {
                b'0'..=b'9' => {
                    seen_digit = true;
                    if seen_dot {
                        digits_after.push(byte);
                    } else {
                        digits_before.push(byte);
                    }
                    index += 1;
                }
                b'.' if !seen_dot => {
                    seen_dot = true;
                    index += 1;
                }
                b'e' | b'E' if seen_digit => {
                    let rest = &raw[index + 1..];
                    exponent = rest
                        .parse::<i64>()
                        .map_err(|_| DecimalError)?;
                    index = bytes.len();
                }
                _ => return Err(DecimalError),
            }
        }
        if !seen_digit {
            return Err(DecimalError);
        }
        let mut all = digits_before;
        all.extend_from_slice(&digits_after);
        let mut scale = digits_after.len() as i64 - exponent;
        if scale < 0 {
            for _ in 0..(-scale) {
                all.push(b'0');
            }
            scale = 0;
        }
        if scale > MAX_SCALE as i64 {
            return Err(DecimalError);
        }
        // The parsed scale is kept, like Python's Decimal, so `format_f`
        // round-trips "0.20" as "0.20".
        let mantissa = parse_digits(&all).ok_or(DecimalError)?;
        Ok(Decimal { mantissa, scale: scale as u32 })
    }

    fn rescaled(&self, scale: u32) -> u128 {
        if scale >= self.scale {
            let factor = pow10(scale - self.scale);
            self.mantissa * factor
        } else {
            let factor = pow10(self.scale - scale);
            self.mantissa / factor
        }
    }

    /// Exact addition. Scales are aligned without rounding.
    pub fn add(&self, other: Decimal) -> Decimal {
        let scale = self.scale.max(other.scale);
        let mantissa = self.rescaled(scale) + other.rescaled(scale);
        Decimal { mantissa, scale }
    }

    /// Multiply by a whole number.
    pub fn mul_int(&self, factor: u128) -> Decimal {
        Decimal { mantissa: self.mantissa.saturating_mul(factor), scale: self.scale }
    }

    /// Divide by 10^6, exactly.
    pub fn div_million(&self) -> Decimal {
        Decimal { mantissa: self.mantissa, scale: self.scale + 6 }
    }

    /// Quantize to six decimal places with `ROUND_HALF_UP`.
    pub fn quantize6(&self) -> Decimal {
        let target: u32 = 6;
        if self.scale <= target {
            let factor = pow10(target - self.scale);
            Decimal { mantissa: self.mantissa * factor, scale: target }
        } else {
            let factor = pow10(self.scale - target);
            let quotient = self.mantissa / factor;
            let remainder = self.mantissa % factor;
            let mut rounded = quotient;
            if remainder * 2 >= factor {
                rounded += 1;
            }
            Decimal { mantissa: rounded, scale: target }
        }
    }

    pub fn cmp_decimal(&self, other: &Decimal) -> Ordering {
        let scale = self.scale.max(other.scale);
        self.rescaled(scale).cmp(&other.rescaled(scale))
    }

    pub fn is_zero(&self) -> bool {
        self.mantissa == 0
    }

    pub fn from_int(value: u128) -> Decimal {
        Decimal { mantissa: value, scale: 0 }
    }

    /// `format(x, "f")`: fixed notation, no exponent.
    pub fn format_f(&self) -> String {
        let text = self.mantissa.to_string();
        let scale = self.scale as usize;
        if scale == 0 {
            return text;
        }
        if text.len() <= scale {
            let mut out = String::from("0.");
            for _ in 0..(scale - text.len()) {
                out.push('0');
            }
            out.push_str(&text);
            out
        } else {
            let split = text.len() - scale;
            let mut out = String::new();
            out.push_str(&text[..split]);
            out.push('.');
            out.push_str(&text[split..]);
            out
        }
    }
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_f())
    }
}

impl PartialOrd for Decimal {
    fn partial_cmp(&self, other: &Decimal) -> Option<Ordering> {
        Some(self.cmp_decimal(other))
    }
}

impl PartialEq for Decimal {
    fn eq(&self, other: &Decimal) -> bool {
        self.cmp_decimal(other) == Ordering::Equal
    }
}

impl Eq for Decimal {}

fn pow10(power: u32) -> u128 {
    let mut value: u128 = 1;
    for _ in 0..power {
        value = match value.checked_mul(10) {
            Some(next) => next,
            None => return u128::MAX,
        };
    }
    value
}

fn parse_digits(digits: &[u8]) -> Option<u128> {
    let mut value: u128 = 0;
    for byte in digits {
        value = value.checked_mul(10)?.checked_add((*byte - b'0') as u128)?;
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_and_keeps_scale() {
        let value = Decimal::parse("0.20").unwrap();
        assert_eq!(value.format_f(), "0.20");
        assert_eq!(Decimal::parse("1").unwrap().format_f(), "1");
        assert_eq!(Decimal::parse("1.500").unwrap().format_f(), "1.500");
    }

    #[test]
    fn parses_scientific_and_rejects_garbage() {
        assert_eq!(Decimal::parse("1e3").unwrap().format_f(), "1000");
        assert_eq!(Decimal::parse("2.5E-1").unwrap().format_f(), "0.25");
        assert!(Decimal::parse("nope").is_err());
        assert!(Decimal::parse("").is_err());
        assert!(Decimal::parse("-1").is_err());
        assert!(Decimal::parse("1.2.3").is_err());
    }

    #[test]
    fn exact_base10_arithmetic() {
        let price = Decimal::parse("0.20").unwrap();
        let cost = price
            .mul_int(1_000_000)
            .add(Decimal::parse("0.40").unwrap().mul_int(2_000_000))
            .div_million();
        assert_eq!(cost.quantize6().format_f(), "1.000000");
    }

    #[test]
    fn quantize_rounds_half_up() {
        let value = Decimal::parse("0.0000005").unwrap();
        assert_eq!(value.quantize6().format_f(), "0.000001");
        let value = Decimal::parse("0.00000049").unwrap();
        assert_eq!(value.quantize6().format_f(), "0.000000");
    }

    #[test]
    fn compares_across_scales() {
        assert!(Decimal::parse("0.20").unwrap() > Decimal::parse("0.199999").unwrap());
        assert!(Decimal::parse("10").unwrap() == Decimal::parse("10.00").unwrap());
        assert!(Decimal::parse("1000000").unwrap() > Decimal::parse("999999.9").unwrap());
    }
}
