"""Command-line entry point: raw GRIN-Global export(s) -> CSV views + HTML report.

    seedbank-survival --inventory PATH [--accessions PATH]
                       (--genus NAME [--genus NAME ...] | --list-genera)
                       [--as-of-year YEAR] [--out-dir PATH]
                       [--model {quadratic,weibull,breakpoint}]

Raw-GRIN-only for now -- the reformatted CSV/XLSX path stays available
programmatically (data_prep.load_accessions, etc.) but isn't exposed here.

run() holds all the real logic and takes an already-parsed Namespace, so it's
callable directly in tests without spawning a subprocess; main() is a thin
argparse + exit-code wrapper. --as-of-year is the one place this package
reads a live clock -- everywhere else (the library layer) takes it as an
explicit parameter.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from . import data_prep
from .deterioration import ModelKind, fit_global_model, fit_group_models
from .grin_import import adapt_raw_export, assemble_model_dataset, list_genera
from .hierarchical import predict_hierarchical
from .priority import build_priority_table
from .report import ModelReportData, build_report

_ALL_MODEL_KINDS: tuple[ModelKind, ...] = ("quadratic", "weibull", "breakpoint")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seedbank-survival",
        description="Build regeneration-priority and inventory-status views from a raw GRIN-Global export.",
    )
    parser.add_argument(
        "--inventory", required=True, type=Path,
        help="Raw GRIN-Global inventory export (.xlsx or .csv)",
    )
    parser.add_argument(
        "--accessions", type=Path, default=None,
        help="Optional raw GRIN-Global accession-level export, for the Received Date SeedAge fallback",
    )
    parser.add_argument(
        "--genus", "-g", action="append", dest="genera", metavar="NAME",
        help="Genus to include (repeatable, e.g. -g Astragalus -g Homalobus). Required unless --list-genera.",
    )
    parser.add_argument(
        "--list-genera", action="store_true",
        help="Print the Taxon-derived genera found in --inventory and exit, without writing anything.",
    )
    parser.add_argument(
        "--as-of-year", type=int, default=None,
        help="Year to compute SeedAge as of (default: the current year)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."),
        help="Directory to write accession_view.csv / inventory_view.csv / report.html into (default: current directory)",
    )
    parser.add_argument(
        "--model", choices=list(_ALL_MODEL_KINDS), default="quadratic",
        help="Deterioration curve form for the CSVs and the report's initial view "
             "(default: quadratic). All three are fit regardless and switchable live "
             "in the report itself -- weibull and breakpoint are guaranteed "
             "non-increasing with age; quadratic isn't but is sometimes the better "
             "fit -- see README.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    df_raw = data_prep.load_accessions(args.inventory)

    if args.list_genera:
        print(list_genera(df_raw).to_string(index=False))
        return 0

    if not args.genera:
        print(
            "error: --genus is required (use --list-genera to see what's in this file)",
            file=sys.stderr,
        )
        return 1

    df_accessions = data_prep.load_accessions(args.accessions) if args.accessions else None
    as_of_year = args.as_of_year if args.as_of_year is not None else date.today().year

    adapted = adapt_raw_export(
        df_raw, as_of_year=as_of_year, genera=args.genera, df_accessions=df_accessions
    )
    if len(adapted.df_primary) == 0:
        known = list_genera(df_raw)
        print(
            f"error: no rows matched genus/genera {args.genera!r}. "
            f"Known genera in this file:\n{known.to_string(index=False)}",
            file=sys.stderr,
        )
        return 1

    df_primary_clean = data_prep.clean_ages(adapted.df_primary)
    df_model = assemble_model_dataset(df_primary_clean, adapted.df_borrowed)
    if len(df_model) == 0:
        print(
            "error: no rows with both a resolvable age and a measured viability -- can't fit any model",
            file=sys.stderr,
        )
        return 1

    df_ranking = data_prep.build_ranking_dataset(df_primary_clean)
    df_inventory = data_prep.build_inventory_view(df_primary_clean)

    # Fit all three curve kinds, not just --model: the report embeds every
    # one that succeeds and lets a curator switch between them live in the
    # browser, no server round-trip. --model only picks which kind's table
    # goes to the plain CSVs (which can't be interactive) and which one the
    # report shows first. A kind that fails to converge (weibull only --
    # quadratic and breakpoint can't fail this way) is skipped with a
    # warning rather than aborting the whole run, as long as at least one
    # other kind succeeds.
    models: dict[ModelKind, ModelReportData] = {}
    for kind in _ALL_MODEL_KINDS:
        try:
            global_model = fit_global_model(df_model, kind)
        except RuntimeError as exc:
            print(f"warning: skipping --model {kind}: {exc}", file=sys.stderr)
            continue
        species_models = fit_group_models(df_model, "SpeciesGroup", kind)
        origin_models = fit_group_models(df_model, "Origin", kind)

        kind_ranking = predict_hierarchical(df_ranking, species_models, origin_models, global_model)
        kind_inventory = predict_hierarchical(df_inventory, species_models, origin_models, global_model)

        # Both views export every qualifying row -- never an artificial
        # top-N cap here, since the whole point is not to silently drop
        # real inventory. The HTML report's checkbox/filters, not a CLI
        # flag, are how a curator narrows what they're looking at.
        models[kind] = ModelReportData(
            accession_table=build_priority_table(
                kind_ranking, species_models, origin_models, global_model, top_n=len(kind_ranking)
            ),
            inventory_table=build_priority_table(
                kind_inventory, species_models, origin_models, global_model, top_n=len(kind_inventory)
            ),
            df_model=df_model,
            species_models=species_models,
            origin_models=origin_models,
            global_model=global_model,
        )

    if not models:
        print("error: every deterioration curve kind failed to fit -- can't build a report", file=sys.stderr)
        return 1

    csv_kind = args.model if args.model in models else next(iter(models))
    accession_table = models[csv_kind].accession_table
    inventory_table = models[csv_kind].inventory_table

    args.out_dir.mkdir(parents=True, exist_ok=True)
    accession_csv = args.out_dir / "accession_view.csv"
    inventory_csv = args.out_dir / "inventory_view.csv"
    report_path = args.out_dir / "report.html"

    accession_table.to_csv(accession_csv, index=False)
    inventory_table.to_csv(inventory_csv, index=False)

    html = build_report(
        models,
        genera=args.genera,
        as_of_year=as_of_year,
        default_model=args.model,
    )
    report_path.write_text(html, encoding="utf-8")

    print(f"Models fit: {', '.join(models)} (CSVs use {csv_kind})")
    print(f"Accession view: {len(accession_table)} rows -> {accession_csv}")
    print(f"Inventory view: {len(inventory_table)} rows -> {inventory_csv}")
    print(f"Report: {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
