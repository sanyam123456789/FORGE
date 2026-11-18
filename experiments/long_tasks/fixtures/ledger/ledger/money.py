"""Money helpers: parsing, rounding, formatting and currency conversion.

All amounts are ``decimal.Decimal``. Expenses are negative, income is
positive. Exchange rates are expressed as "units of a currency per 1 USD".
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

CENT = Decimal("0.01")
ONE = Decimal("1")

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "INR": "₹",
    "JPY": "¥",
}

#: Currencies that have no minor unit (no cents).
ZERO_DECIMAL_CURRENCIES = frozenset({"JPY"})


class MoneyError(ValueError):
    """Raised for amounts that cannot be parsed, split or converted."""


def minor_unit(currency: str = "USD") -> Decimal:
    """Smallest amount of *currency*: 0.01, or 1 for zero-decimal currencies."""
    return ONE if currency in ZERO_DECIMAL_CURRENCIES else CENT


def round_money(amount: Decimal, currency: str = "USD") -> Decimal:
    """Round half-up to the currency's minor unit."""
    return Decimal(amount).quantize(minor_unit(currency), rounding=ROUND_HALF_UP)


def parse_amount(text: str | None) -> Decimal:
    """Parse an amount as written in bank exports.

    Accepted forms: ``1234.5``, ``1,234.50``, ``-12``, ``+12``, ``$5.00``,
    ``5.00 USD``, ``1_000``, and accounting negatives ``(12.50)``, which mean
    -12.50. Raises ``MoneyError`` for anything else.
    """
    if text is None:
        raise MoneyError("amount is missing")
    raw = str(text).strip()
    if not raw:
        raise MoneyError("amount is empty")

    negative = False
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
        raw = raw[1:-1].strip()
    if raw.startswith("-"):
        negative = not negative
        raw = raw[1:].strip()
    elif raw.startswith("+"):
        raw = raw[1:].strip()

    for symbol in CURRENCY_SYMBOLS.values():
        if raw.startswith(symbol):
            raw = raw[len(symbol):].strip()
            break

    parts = raw.split()
    if len(parts) == 2 and parts[1].upper() in CURRENCY_SYMBOLS:
        raw = parts[0]

    raw = raw.replace(",", "").replace("_", "")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise MoneyError(f"not an amount: {text!r}") from exc
    if not value.is_finite():
        raise MoneyError(f"not a finite amount: {text!r}")
    return -value if negative else value


def format_amount(amount: Decimal, currency: str = "USD") -> str:
    """Format like ``-$1,234.50``; unknown currencies get the code as a suffix."""
    value = round_money(Decimal(amount), currency)
    sign = "-" if value < 0 else ""
    places = 0 if currency in ZERO_DECIMAL_CURRENCIES else 2
    body = f"{abs(value):,.{places}f}"
    symbol = CURRENCY_SYMBOLS.get(currency)
    if symbol:
        return f"{sign}{symbol}{body}"
    return f"{sign}{body} {currency}"


def convert(
    amount: Decimal,
    from_currency: str,
    to_currency: str,
    rates: dict[str, Decimal],
) -> Decimal:
    """Convert *amount* between currencies using per-USD *rates*.

    ``rates["EUR"] == Decimal("0.9")`` means 1 USD buys 0.9 EUR. USD itself
    never needs a rate. The result is rounded to the target currency.
    """
    if from_currency == to_currency:
        return Decimal(amount)
    try:
        in_usd = (
            Decimal(amount)
            if from_currency == "USD"
            else Decimal(amount) / Decimal(rates[from_currency])
        )
        result = in_usd if to_currency == "USD" else in_usd * Decimal(rates[to_currency])
    except KeyError as exc:
        raise MoneyError(f"no exchange rate for {exc.args[0]}") from exc
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise MoneyError(f"bad exchange rate: {exc}") from exc
    return round_money(result, to_currency)


def sum_amounts(amounts: Iterable[Decimal]) -> Decimal:
    """Exact sum of amounts (an empty iterable sums to 0)."""
    total = Decimal("0")
    for amount in amounts:
        total += Decimal(amount)
    return total


def split_amount(amount: Decimal, parts: int, currency: str = "USD") -> list[Decimal]:
    """Split *amount* into *parts* shares that add up exactly.

    The first shares get the leftover minor units, so ``split_amount(10, 3)``
    is ``[3.34, 3.33, 3.33]``.
    """
    if parts < 1:
        raise MoneyError("parts must be at least 1")
    unit = minor_unit(currency)
    total_units = int((round_money(amount, currency) / unit).to_integral_value())
    sign = -1 if total_units < 0 else 1
    base, extra = divmod(abs(total_units), parts)
    return [sign * (base + (1 if i < extra else 0)) * unit for i in range(parts)]


def percent(part: Decimal, whole: Decimal) -> Decimal | None:
    """``part / whole`` as a percentage rounded to 0.1, or None if whole is 0."""
    if whole == 0:
        return None
    return (Decimal(part) / Decimal(whole) * 100).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )
