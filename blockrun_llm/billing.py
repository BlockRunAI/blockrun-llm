"""Local billing / cost-tracking CLI for the BlockRun Python SDK.

Reads ``~/.blockrun/cost_log.jsonl`` and prints / exports filtered, grouped
summaries. No HTTP requests, no payment, no wallet signing — operates purely
on the local append-only ledger written by ``save_to_cache`` after every paid
API call.

Usage:
    python -m blockrun_llm.billing summary [--from] [--to]
                                            [--wallet] [--network]
                                            [--group-by FIELD]
    python -m blockrun_llm.billing export {csv|json}
                                          [--from] [--to]
                                          [--wallet] [--network]
                                          [--output PATH]

Examples:
    python -m blockrun_llm.billing summary
    python -m blockrun_llm.billing summary --group-by model
    python -m blockrun_llm.billing summary --from 2026-04-01 --group-by month
    python -m blockrun_llm.billing export csv --from 2026-05-01 --output may.csv
    python -m blockrun_llm.billing export json --wallet 0x...
"""

from __future__ import annotations

import argparse
import sys

from .cache import (
    COST_LOG_PATH,
    export_cost_log_csv,
    export_cost_log_json,
    get_cost_log_summary,
)


def _fmt_usd(value: float) -> str:
    return f"${value:.4f}"


def _print_summary(args: argparse.Namespace) -> int:
    summary = get_cost_log_summary(
        from_date=args.from_date,
        to_date=args.to_date,
        wallet=args.wallet,
        network=args.network,
        group_by=args.group_by,
    )

    print("=" * 64)
    print("BLOCKRUN — LOCAL COST LOG SUMMARY")
    print("=" * 64)
    print(f"  log file       : {COST_LOG_PATH}")
    if args.from_date:
        print(f"  from           : {args.from_date}")
    if args.to_date:
        print(f"  to             : {args.to_date}")
    if args.wallet:
        print(f"  wallet         : {args.wallet}")
    if args.network:
        print(f"  network        : {args.network}")
    print(f"  group_by       : {args.group_by}")
    print(f"  total          : {_fmt_usd(summary['total_usd'])} ({summary['calls']} calls)")
    print()

    groups = summary.get("groups") or {
        # legacy no-arg shape — adapt
        k: {"calls": 0, "cost_usd": v}
        for k, v in (summary.get("by_endpoint") or {}).items()
    }
    if not groups:
        print("  (no entries match the filter)")
        return 0

    rows = sorted(groups.items(), key=lambda kv: kv[1]["cost_usd"], reverse=True)
    width = max((len(str(k)) for k, _ in rows), default=10)
    width = min(max(width, 12), 60)
    print(f"  {'KEY':<{width}}  {'CALLS':>7}  {'COST':>10}")
    print(f"  {'-' * width}  {'-' * 7}  {'-' * 10}")
    for key, slot in rows:
        calls = slot.get("calls") or 0
        cost = slot.get("cost_usd") or 0.0
        display = (str(key) or "(none)")[:width]
        print(f"  {display:<{width}}  {calls:>7}  {_fmt_usd(cost):>10}")
    print()
    return 0


def _print_export(args: argparse.Namespace) -> int:
    fmt = args.format
    if fmt == "csv":
        text = export_cost_log_csv(
            args.output,
            from_date=args.from_date,
            to_date=args.to_date,
            wallet=args.wallet,
            network=args.network,
        )
    elif fmt == "json":
        text = export_cost_log_json(
            args.output,
            from_date=args.from_date,
            to_date=args.to_date,
            wallet=args.wallet,
            network=args.network,
        )
    else:
        sys.stderr.write(f"unknown format: {fmt}\n")
        return 2

    if args.output:
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--from",
        dest="from_date",
        type=str,
        default=None,
        help="lower bound (inclusive) — YYYY-MM-DD or ISO datetime",
    )
    p.add_argument(
        "--to",
        dest="to_date",
        type=str,
        default=None,
        help="upper bound (inclusive) — YYYY-MM-DD or ISO datetime",
    )
    p.add_argument("--wallet", type=str, default=None, help="filter to one wallet")
    p.add_argument(
        "--network",
        type=str,
        default=None,
        help="filter to one network (base-mainnet / base-sepolia / solana-mainnet)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m blockrun_llm.billing",
        description="Local BlockRun cost-log reader / exporter.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    s_summary = sub.add_parser("summary", help="Print an aggregated summary.")
    _add_filter_args(s_summary)
    s_summary.add_argument(
        "--group-by",
        choices=("endpoint", "model", "wallet", "network", "client_kind", "day", "month"),
        default="endpoint",
        help="bucket dimension (default: endpoint)",
    )
    s_summary.set_defaults(handler=_print_summary)

    s_export = sub.add_parser("export", help="Export raw entries.")
    s_export.add_argument("format", choices=("csv", "json"), help="output format")
    _add_filter_args(s_export)
    s_export.add_argument(
        "--output",
        type=str,
        default=None,
        help="write to this path (default: stdout)",
    )
    s_export.set_defaults(handler=_print_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
