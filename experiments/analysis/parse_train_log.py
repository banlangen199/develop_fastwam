#!/usr/bin/env python3
"""Parse FastWAM/RoutedWAM training logs into a tidy per-step metric table.

The trainer prints metrics as a *wrapped* block under each
``>> epoch=.. step=N/TOTAL`` header, one or two ``key=value`` pairs per physical
line, padded to a fixed column width.  That means no single line contains a
whole step, so the usual ``grep 'step=' | ...`` one-liner silently captures only
whichever metrics happened to land on the header line.

This module instead treats the header as a record separator and accumulates every
``key=value`` seen until the next header.  Values that are not plain floats
(``lr=1.00e-04``, ``speed=0.09 step/s``, ``eta=01:14:26``) are kept as strings.

Usage:
    python experiments/analysis/parse_train_log.py <log> [...] --csv out.csv
    python experiments/analysis/parse_train_log.py <log> --keys router_gate_mean
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

# `>> epoch=0 step=30/416 loss=0.5952` -- the `>>` marker is what distinguishes
# a metric header from the many other lines that mention a step.
_HEADER = re.compile(r">>\s+epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)/(?P<total>\d+)")
# Keys include `@`, `/` and digits: `router_keep_dino@t1/wrist`.
_KV = re.compile(r"(?P<key>[A-Za-z_][\w@/.]*)=(?P<val>-?[\d.]+(?:e[-+]?\d+)?|-?[\d.]+)")


def iter_steps(text: str) -> Iterator[Dict[str, Any]]:
    """Yield one dict per logged step, in file order."""
    record: Dict[str, Any] | None = None
    for line in text.splitlines():
        header = _HEADER.search(line)
        if header:
            if record is not None:
                yield record
            record = {
                "epoch": int(header.group("epoch")),
                "step": int(header.group("step")),
                "total": int(header.group("total")),
            }
            # The header line itself carries `loss=...`; fall through so the
            # generic scraper below picks it up.
        if record is None:
            continue
        # `step=30/416` would otherwise be re-scraped as step=30 -- harmless,
        # but `total` would be lost, so the header span is excluded.
        body = line[header.end() :] if header else line
        for match in _KV.finditer(body):
            key, raw = match.group("key"), match.group("val")
            try:
                record[key] = float(raw)
            except ValueError:
                record[key] = raw
    if record is not None:
        yield record


def parse_file(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(errors="replace")
    steps = list(iter_steps(text))
    # A resumed or multi-rank log can repeat a step; last write wins, matching
    # what a reader scrolling the log would conclude.
    deduped: Dict[int, Dict[str, Any]] = {}
    for rec in steps:
        deduped[rec["step"]] = rec
    return [deduped[k] for k in sorted(deduped)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, help="write the merged table here")
    ap.add_argument("--keys", nargs="*", help="only print these keys")
    args = ap.parse_args()

    tables: Dict[str, List[Dict[str, Any]]] = {}
    for log in args.logs:
        if not log.exists():
            print(f"missing: {log}", file=sys.stderr)
            continue
        rows = parse_file(log)
        tables[log.parent.name if log.name == "job_log.txt" else log.stem] = rows
        print(f"{log}: {len(rows)} steps", file=sys.stderr)

    if args.csv:
        fields: List[str] = ["run"]
        for rows in tables.values():
            for rec in rows:
                for key in rec:
                    if key not in fields:
                        fields.append(key)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for run, rows in tables.items():
                for rec in rows:
                    writer.writerow({"run": run, **rec})
        print(f"-> {args.csv}", file=sys.stderr)

    if args.keys:
        for run, rows in tables.items():
            print(f"\n== {run}")
            for rec in rows:
                vals = " ".join(
                    f"{k}={rec[k]}" for k in args.keys if k in rec
                )
                print(f"  step={rec['step']:>4} {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
