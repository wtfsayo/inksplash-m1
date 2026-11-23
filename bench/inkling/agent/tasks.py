"""Eight small tool-use tasks, each a sandbox of files, a prompt and a checker.

A task's checker runs after the agent stops and decides on the files it left
(hidden tests it never saw) or on its final answer. `reference` applies a known
good solution, which self_test() checks passes, while the untouched files fail:
so a checker that accepts everything or nothing is caught before a model runs.
"""

import csv
import io
import json
import pathlib
import random
import subprocess
import sys
import tempfile
import textwrap

PYTHON = sys.executable


def run(sandbox, *command, timeout=30):
    result = subprocess.run(command, cwd=sandbox, capture_output=True, text=True, timeout=timeout)
    return result.returncode, (result.stdout + result.stderr)


def unittest_passes(sandbox, module):
    try:
        return run(sandbox, PYTHON, "-m", "unittest", module)[0] == 0
    except subprocess.TimeoutExpired:
        return False


def dedent(text):
    return textwrap.dedent(text).lstrip("\n")


class Task:
    def __init__(self, name, prompt, files, check, reference):
        self.name, self.prompt, self.files, self.check, self.reference = name, prompt, files, check, reference

    def build(self, sandbox):
        for name, content in self.files.items():
            path = pathlib.Path(sandbox) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


def log_lines():
    rng = random.Random(11)
    files = {}
    target = "req-7f3a91"
    for index in range(5):
        lines = []
        for n in range(60):
            rid = f"req-{rng.randrange(16**6):06x}"
            status = rng.choice([200, 200, 200, 201, 404, 500])
            latency = round(rng.uniform(0.02, 1.9), 3)
            lines.append(f"2026-11-0{index + 1}T10:{n:02d}:00Z {rid} status={status} latency={latency}s")
        if index == 3:
            lines[41] = f"2026-11-04T10:41:00Z {target} status=500 latency=2.847s"
        files[f"logs/app-{index + 1}.log"] = "\n".join(lines) + "\n"
    return files, target


def csv_data():
    rng = random.Random(5)
    rows = [("id", "category", "value")]
    for n in range(240):
        rows.append((n, rng.choice("ABC"), round(rng.uniform(1, 100), 2)))
    selected = [row[2] for row in rows[1:] if row[1] == "B"]
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue(), f"{sum(selected) / len(selected):.2f}"


LOGS, LOG_ANSWER = log_lines()
CSV, CSV_ANSWER = csv_data()


def paginate_check(sandbox, answer):
    hidden = dedent("""
        import unittest
        from paginate import paginate
        class Hidden(unittest.TestCase):
            def test_pages(self):
                items = list(range(10))
                self.assertEqual(paginate(items, 1, 4), [0, 1, 2, 3])
                self.assertEqual(paginate(items, 3, 4), [8, 9])
                self.assertEqual(paginate(items, 4, 4), [])
            def test_bad(self):
                with self.assertRaises(ValueError):
                    paginate([1], 0, 4)
                with self.assertRaises(ValueError):
                    paginate([1], 1, 0)
        """)
    (pathlib.Path(sandbox) / "hidden_paginate.py").write_text(hidden)
    return unittest_passes(sandbox, "hidden_paginate") and unittest_passes(sandbox, "test_paginate")


def durations_check(sandbox, answer):
    hidden = dedent("""
        import unittest
        from durations import parse_duration
        class Hidden(unittest.TestCase):
            def test_cases(self):
                for text, seconds in [("90s", 90), ("2m", 120), ("1h30m", 5400), ("1d", 86400), ("1d2h3m4s", 93784), ("0s", 0), (" 5m ", 300)]:
                    self.assertEqual(parse_duration(text), seconds, text)
            def test_errors(self):
                for text in ["", "5", "m5", "5x", "5m5", "-3s", "1h 30m"]:
                    with self.assertRaises(ValueError, msg=text):
                        parse_duration(text)
        """)
    (pathlib.Path(sandbox) / "hidden_durations.py").write_text(hidden)
    return unittest_passes(sandbox, "hidden_durations")


def logs_check(sandbox, answer):
    return LOG_ANSWER in answer


def rename_check(sandbox, answer):
    root = pathlib.Path(sandbox)
    sources = [p for p in root.rglob("*.py")]
    text = "\n".join(p.read_text() for p in sources if not p.name.startswith("hidden"))
    return "calc_total" not in text and text.count("compute_total") >= 6 and unittest_passes(sandbox, "test_orders")


def merge_check(sandbox, answer):
    path = pathlib.Path(sandbox) / "merged.json"
    if not path.exists():
        return False
    try:
        got = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    want = {"name": "svc", "port": 9090, "debug": True, "db": {"host": "db2", "port": 5432, "pool": 8}, "tags": ["a", "b", "c"]}
    return got == want


def csv_check(sandbox, answer):
    return CSV_ANSWER in answer


def import_check(sandbox, answer):
    return unittest_passes(sandbox, "test_shop") and unittest_passes(sandbox, "test_cart")


def cli_check(sandbox, answer):
    root = str(pathlib.Path(sandbox))
    out = run(root, PYTHON, "cli.py", "--limit", "2", "a", "b", "c", "d")
    verbose = run(root, PYTHON, "cli.py", "--verbose", "x")
    plain = run(root, PYTHON, "cli.py", "x", "y")
    return (out[0] == 0 and out[1].split() == ["a", "b"] and verbose[0] == 0 and "verbose" in verbose[1].lower()
            and verbose[1].split()[-1] == "x" and plain[1].split() == ["x", "y"] and unittest_passes(sandbox, "test_cli"))


def write(sandbox, files):
    for name, content in files.items():
        path = pathlib.Path(sandbox) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


TASKS = [
    Task(
        "bugfix_paginate",
        "The tests in test_paginate.py fail. Find and fix the bug in paginate.py without changing the tests, then run them.",
        {
            "paginate.py": dedent('''
                def paginate(items, page, size):
                    """Return the items on a 1-based page of `size` items."""
                    if page < 1 or size < 1:
                        raise ValueError("page and size must be positive")
                    start = page * size
                    return items[start : start + size]
                '''),
            "test_paginate.py": dedent("""
                import unittest
                from paginate import paginate
                class T(unittest.TestCase):
                    def test_first(self):
                        self.assertEqual(paginate(list(range(10)), 1, 3), [0, 1, 2])
                    def test_last(self):
                        self.assertEqual(paginate(list(range(10)), 4, 3), [9])
                if __name__ == "__main__":
                    unittest.main()
                """),
        },
        paginate_check,
        {"paginate.py": dedent('''
            def paginate(items, page, size):
                """Return the items on a 1-based page of `size` items."""
                if page < 1 or size < 1:
                    raise ValueError("page and size must be positive")
                start = (page - 1) * size
                return items[start : start + size]
            ''')},
    ),
    Task(
        "implement_duration_parser",
        "Implement parse_duration in durations.py exactly as its docstring says, so that test_durations.py passes. Run the tests when done.",
        {
            "durations.py": dedent('''
                def parse_duration(text):
                    """Parse a duration like "1h30m" or "90s" into whole seconds.

                    Units: d (86400 s), h (3600), m (60), s (1), each at most once, in that order,
                    each preceded by a non-negative integer, with no spaces inside. Surrounding
                    whitespace is ignored. Anything else raises ValueError (including an empty
                    string, a number with no unit, a unit with no number, an unknown unit, or a
                    negative number).
                    """
                    raise NotImplementedError
                '''),
            "test_durations.py": dedent("""
                import unittest
                from durations import parse_duration
                class T(unittest.TestCase):
                    def test_basic(self):
                        self.assertEqual(parse_duration("90s"), 90)
                        self.assertEqual(parse_duration("1h30m"), 5400)
                    def test_error(self):
                        with self.assertRaises(ValueError):
                            parse_duration("5x")
                """),
        },
        durations_check,
        {"durations.py": dedent('''
            import re
            def parse_duration(text):
                match = re.fullmatch(r"(?:(\\d+)d)?(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", text.strip())
                if not match or not any(group is not None for group in match.groups()):
                    raise ValueError(text)
                d, h, m, s = (int(group or 0) for group in match.groups())
                return d * 86400 + h * 3600 + m * 60 + s
            ''')},
    ),
    Task(
        "log_forensics",
        "The files in logs/ are five days of request logs. Exactly one request has status=500 and a latency above 2.5s. "
        "Find it and reply with its request id (req-xxxxxx) in your final answer.",
        dict(LOGS),
        logs_check,
        {},
    ),
    Task(
        "rename_refactor",
        "Rename the function calc_total to compute_total everywhere in this project (definitions, imports, calls, tests) "
        "and make sure the tests still pass. Do not leave any occurrence of the old name.",
        {
            "orders.py": dedent("""
                def calc_total(items):
                    return sum(price * qty for price, qty in items)
                def calc_total_with_tax(items, rate):
                    return round(calc_total(items) * (1 + rate), 2)
                """),
            "report.py": dedent("""
                from orders import calc_total
                def summary(items):
                    return f"total={calc_total(items)}"
                """),
            "invoice.py": dedent("""
                import orders
                def invoice(items):
                    return {"total": orders.calc_total(items), "lines": len(items)}
                """),
            "test_orders.py": dedent("""
                import unittest
                import orders, report, invoice
                class T(unittest.TestCase):
                    def test_total(self):
                        self.assertEqual(orders.calc_total([(2, 3), (1, 5)]), 11)
                    def test_tax(self):
                        self.assertEqual(orders.calc_total_with_tax([(10, 1)], 0.5), 15.0)
                    def test_report(self):
                        self.assertEqual(report.summary([(2, 2)]), "total=4")
                    def test_invoice(self):
                        self.assertEqual(invoice.invoice([(1, 1)]), {"total": 1, "lines": 1})
                """),
        },
        rename_check,
        {
            "orders.py": dedent("""
                def compute_total(items):
                    return sum(price * qty for price, qty in items)
                def compute_total_with_tax(items, rate):
                    return round(compute_total(items) * (1 + rate), 2)
                """),
            "report.py": dedent("""
                from orders import compute_total
                def summary(items):
                    return f"total={compute_total(items)}"
                """),
            "invoice.py": dedent("""
                import orders
                def invoice(items):
                    return {"total": orders.compute_total(items), "lines": len(items)}
                """),
            "test_orders.py": dedent("""
                import unittest
                import orders, report, invoice
                class T(unittest.TestCase):
                    def test_total(self):
                        self.assertEqual(orders.compute_total([(2, 3), (1, 5)]), 11)
                    def test_tax(self):
                        self.assertEqual(orders.compute_total_with_tax([(10, 1)], 0.5), 15.0)
                    def test_report(self):
                        self.assertEqual(report.summary([(2, 2)]), "total=4")
                    def test_invoice(self):
                        self.assertEqual(invoice.invoice([(1, 1)]), {"total": 1, "lines": 1})
                """),
        },
    ),
    Task(
        "config_merge",
        "Merge base.json and override.json into a new file merged.json. Rules: override values win; objects merge "
        "recursively; lists are concatenated with duplicates removed, keeping first-seen order. Write merged.json.",
        {
            "base.json": json.dumps({"name": "svc", "port": 8080, "debug": False, "db": {"host": "db1", "port": 5432}, "tags": ["a", "b"]}, indent=2),
            "override.json": json.dumps({"port": 9090, "debug": True, "db": {"host": "db2", "pool": 8}, "tags": ["b", "c"]}, indent=2),
        },
        merge_check,
        {"merged.json": json.dumps({"name": "svc", "port": 9090, "debug": True, "db": {"host": "db2", "port": 5432, "pool": 8}, "tags": ["a", "b", "c"]})},
    ),
    Task(
        "csv_stats",
        "data.csv has columns id, category, value. Compute the mean of value over the rows whose category is B, "
        "rounded to 2 decimals, and give that number in your final answer. Use the tools to compute it; do not estimate.",
        {"data.csv": CSV},
        csv_check,
        {},
    ),
    Task(
        "fix_import_error",
        "test_shop.py and test_cart.py both fail to even import. Find the cause and fix the code (not the tests) so that both pass.",
        {
            "cart.py": dedent("""
                from shop import price_of
                class Cart:
                    def __init__(self):
                        self.items = []
                    def add(self, sku, qty=1):
                        self.items.append((sku, qty))
                    def total(self):
                        return sum(price_of(sku) * qty for sku, qty in self.items)
                """),
            "shop.py": dedent("""
                from cart import Cart
                PRICES = {"apple": 3, "pear": 4}
                def price_of(sku):
                    return PRICES[sku]
                def new_cart():
                    return Cart()
                """),
            "test_shop.py": dedent("""
                import unittest
                import shop
                class T(unittest.TestCase):
                    def test_cart(self):
                        c = shop.new_cart(); c.add("apple", 2)
                        self.assertEqual(c.total(), 6)
                    def test_price(self):
                        self.assertEqual(shop.price_of("pear"), 4)
                """),
            "test_cart.py": dedent("""
                import unittest
                from cart import Cart
                class T(unittest.TestCase):
                    def test_total(self):
                        c = Cart(); c.add("pear"); c.add("apple", 3)
                        self.assertEqual(c.total(), 13)
                """),
        },
        import_check,
        {"shop.py": dedent("""
            PRICES = {"apple": 3, "pear": 4}
            def price_of(sku):
                return PRICES[sku]
            def new_cart():
                from cart import Cart
                return Cart()
            """)},
    ),
    Task(
        "add_cli_flags",
        "cli.py prints its positional arguments, one per line. Add two flags with argparse: --limit N (print only the first N "
        "arguments) and --verbose (print a first line 'verbose' before the arguments). Keep test_cli.py passing.",
        {
            "cli.py": dedent("""
                import argparse

                def main(argv=None):
                    parser = argparse.ArgumentParser()
                    parser.add_argument("words", nargs="*")
                    args = parser.parse_args(argv)
                    for word in args.words:
                        print(word)

                if __name__ == "__main__":
                    main()
                """),
            "test_cli.py": dedent("""
                import io, unittest
                from contextlib import redirect_stdout
                import cli
                class T(unittest.TestCase):
                    def test_plain(self):
                        out = io.StringIO()
                        with redirect_stdout(out):
                            cli.main(["x", "y"])
                        self.assertEqual(out.getvalue().split(), ["x", "y"])
                """),
        },
        cli_check,
        {"cli.py": dedent("""
            import argparse

            def main(argv=None):
                parser = argparse.ArgumentParser()
                parser.add_argument("words", nargs="*")
                parser.add_argument("--limit", type=int)
                parser.add_argument("--verbose", action="store_true")
                args = parser.parse_args(argv)
                words = args.words if args.limit is None else args.words[: args.limit]
                if args.verbose:
                    print("verbose")
                for word in words:
                    print(word)

            if __name__ == "__main__":
                main()
            """)},
    ),
]


def self_test():
    for task in TASKS:
        for solved in (False, True):
            if not solved and not task.reference and task.name not in ("log_forensics", "csv_stats"):
                continue
            with tempfile.TemporaryDirectory() as sandbox:
                task.build(sandbox)
                answer = ""
                if solved:
                    write(sandbox, task.reference)
                    answer = {"log_forensics": LOG_ANSWER, "csv_stats": CSV_ANSWER}.get(task.name, "")
                else:
                    answer = "I could not find it."
                ok = task.check(sandbox, answer)
                status = "ok" if ok == solved else "WRONG"
                print(f"{task.name:28s} {'solved   ' if solved else 'untouched'} -> checker says {ok}  {status}")
                if ok != solved:
                    sys.exit(1)


if __name__ == "__main__":
    self_test()
