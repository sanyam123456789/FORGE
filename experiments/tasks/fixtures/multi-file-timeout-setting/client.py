"""A tiny client that currently hardcodes its timeout."""


def make_request(url):
    timeout = 10
    return {"url": url, "timeout": timeout}
