"""Price calculations with some duplicated rounding logic."""


def price_with_tax(amount, rate):
    total = amount * (1 + rate)
    cents = round(total * 100)
    return cents / 100


def discounted_price(amount, discount):
    total = amount * (1 - discount)
    cents = round(total * 100)
    return cents / 100
