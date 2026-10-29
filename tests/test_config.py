"""Guard against YAML quirks: PyYAML reads 2.82e6 (no sign after e) as a STRING."""
import re

from src.config import load_config

NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _walk(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, node


def test_no_number_was_read_as_text():
    bad = [p for p, v in _walk(load_config()) if isinstance(v, str) and NUMERIC.match(v.strip())]
    assert not bad, f"these config values look numeric but loaded as text: {bad}"
