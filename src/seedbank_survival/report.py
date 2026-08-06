"""Builds a single self-contained HTML report -- no server, no external
resources, opens in any browser. This is the project's "dashboard" for now:
a curator runs the CLI and gets one file back with both views and charts.

build_report() is IO-free (returns a string; the CLI does the file write),
matching the rest of this package's pure-function style.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .deterioration import BreakpointCurve, Curve, QuadraticCurve, WeibullCurve

_MODEL_LABELS = {"quadratic": "Quadratic", "weibull": "Weibull", "breakpoint": "Breakpoint"}

_TABLE_COLUMNS = [
    ("Accession", "Accession"),
    ("Suffix", "Suffix"),
    ("Status", "Status"),
    ("Species", "Species"),
    ("Origin", "Origin"),
    ("SeedAge", "Seed age"),
    ("PrimaryReason", "Reason"),
    ("EstimatedViability_2026", "Est. viability %"),
    ("YearsRemainingTo0%", "Years to 0%"),
    ("ModelUsed", "Model"),
    ("ModelConfidence", "Confidence"),
    ("Location", "Location"),
]

# Carried into each row's JSON payload for the on-site filter checkbox, but
# deliberately excluded from _TABLE_COLUMNS -- MaintenanceSite (e.g. "W6") is
# a filter key, not something a curator needs its own visible column for.
_FILTER_ONLY_COLUMNS = ["MaintenanceSite"]

# Every per-species/per-global chart uses this SAME color pair -- each chart
# is its own small multiple (one series + one dashed reference), never two
# categorical hues sharing a legend, so there's no "tell these colors apart"
# problem the dataviz skill's all-pairs cap exists for.
_SERIES_COLOR = "var(--series-1)"
_REFERENCE_COLOR = "var(--series-ref)"


def _select_charted_species(
    accession_table: pd.DataFrame,
    species_models: dict[str, Curve],
    top_n_charts: int,
) -> list[str]:
    """Species that actually won the confidence comparison for >=1 row, ranked
    by how many accession-view rows use that tier, capped at top_n_charts.

    A species with its own fitted Curve whose points all got overridden
    back to Global by the R^2/significance comparison does NOT get its own
    chart -- the charts would otherwise visually contradict
    ModelUsed/ModelConfidence.
    """
    won = accession_table[accession_table["ModelUsed"] == "Species"]
    counts = won["Species"].value_counts()
    return [name for name in counts.index if name in species_models][:top_n_charts]


def _curve_coeffs(model: Curve) -> dict:
    """Serialize a fitted curve's parameters for the JS chart renderer.

    Each curve kind carries different parameters -- "kind" tells the JS
    side (buildChartSvg's evalCurve) which formula to evaluate; see there
    for the matching math.
    """
    base = {"r2": model.r2, "n": model.n}
    if isinstance(model, QuadraticCurve):
        return {**base, "kind": "quadratic", "intercept": model.intercept,
                "linearCoef": model.linear_coef, "quadCoef": model.quad_coef}
    if isinstance(model, WeibullCurve):
        return {**base, "kind": "weibull", "v0": model.v0, "lam": model.lam, "k": model.k}
    if isinstance(model, BreakpointCurve):
        return {**base, "kind": "breakpoint", "t0": model.t0,
                "plateau": model.plateau, "slope": model.slope}
    raise TypeError(f"unknown curve type: {type(model)!r}")


def _build_charts_payload(
    df_model: pd.DataFrame,
    species_models: dict[str, Curve],
    global_model: Curve,
    charted_species: list[str],
    species_group_col: str,
) -> list[dict]:
    x_max = float(df_model["AgeAtTest"].max()) if len(df_model) else 1.0
    x_max = max(x_max, 1.0)

    charts = [
        {
            "title": f"Genus-wide (all {len(df_model)} test points)",
            "points": df_model[["AgeAtTest", "Viability"]].round(1).to_numpy().tolist(),
            "model": _curve_coeffs(global_model),
            "reference": None,
            "xMax": x_max,
        }
    ]
    for name in charted_species:
        model = species_models[name]
        points = df_model.loc[df_model[species_group_col] == name, ["AgeAtTest", "Viability"]]
        charts.append(
            {
                "title": name,
                "points": points.round(1).to_numpy().tolist(),
                "model": _curve_coeffs(model),
                "reference": _curve_coeffs(global_model),
                "xMax": x_max,
            }
        )
    return charts


@dataclass(frozen=True)
class ModelReportData:
    """One fitted model kind's worth of inputs to build_report.

    Every curve kind (typically quadratic/weibull/breakpoint, but only
    whichever the caller successfully fit -- see cli.py) gets its own
    ModelReportData, and build_report embeds ALL of them -- a curator
    switches models live in the browser with no server round-trip, so this
    carries everything needed to rebuild both tables and the charts for
    that one kind.
    """

    accession_table: pd.DataFrame
    inventory_table: pd.DataFrame
    df_model: pd.DataFrame
    species_models: dict[str, Curve]
    origin_models: dict[str, Curve]
    global_model: Curve


def _build_model_payload(
    data: ModelReportData, top_n_charts: int, species_group_col: str, row_columns: list[str]
) -> dict:
    charted_species = _select_charted_species(data.accession_table, data.species_models, top_n_charts)
    charts = _build_charts_payload(
        data.df_model, data.species_models, data.global_model, charted_species, species_group_col
    )
    return {
        "accessionRows": data.accession_table[row_columns].to_dict("records"),
        "inventoryRows": data.inventory_table[row_columns].to_dict("records"),
        "charts": charts,
        "globalR2": round(data.global_model.r2, 2),
        "globalN": data.global_model.n,
    }


def build_report(
    models: dict[str, ModelReportData],
    genera: list[str],
    as_of_year: int,
    default_model: str = "quadratic",
    top_n_charts: int = 8,
    species_group_col: str = "SpeciesGroup",
) -> str:
    """Build the self-contained HTML report. Returns HTML; writes nothing.

    `models` carries one ModelReportData per fitted curve kind -- every
    kind's full table + chart data is embedded in the page, and a "Model"
    selector (present on both views, kept in sync between them -- the
    underlying fitted curves don't depend on which view you're looking at,
    only which rows each view selects) switches which one is displayed,
    entirely client-side.

    default_model picks which kind is shown on first load; falls back to
    whichever key happens to be present if it's missing from `models` (e.g.
    a weibull fit that failed to converge -- see cli.py).

    Row counts (accession/inventory) don't vary by model kind -- every
    qualifying row is exported regardless of which curve fit it, only the
    PREDICTED values differ -- so they're hoisted to one shared, model-
    independent field rather than duplicated per model.
    """
    if default_model not in models:
        default_model = next(iter(models))

    row_columns = [c for c, _ in _TABLE_COLUMNS] + _FILTER_ONLY_COLUMNS
    model_payloads = {
        kind: _build_model_payload(data, top_n_charts, species_group_col, row_columns)
        for kind, data in models.items()
    }

    any_data = next(iter(models.values()))
    payload = {
        "genera": genera,
        "asOfYear": as_of_year,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "columns": _TABLE_COLUMNS,
        "models": model_payloads,
        "modelLabels": {kind: _MODEL_LABELS[kind] for kind in models},
        "defaultModel": default_model,
        "counts": {
            "accession": len(any_data.accession_table),
            "inventory": len(any_data.inventory_table),
        },
    }

    return _PAGE_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, default=_json_default))


def _json_default(value):
    if pd.isna(value):
        return None
    raise TypeError(f"not JSON serializable: {value!r}")


_PAGE_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>SeedbankSurvival report</title>
<style>
.viz-root {
  color-scheme: light;
  --page:           #f7f8f3;
  --surface-1:      #fdfdfb;
  --surface-2:      #f2f3ec;
  --text-primary:   #14150f;
  --text-secondary: #54564a;
  --text-muted:     #8b8d80;
  --grid:           #e3e4da;
  --axis:           #c4c6ba;
  --accent:         #5c7a52;
  --border:         rgba(20,21,15,0.09);
  --series-1:       #2a78d6;
  --series-ref:     #9a9c8f;
  --tooltip-bg:     #14150f;
  --tooltip-text:   #fdfdfb;
  --row-hover:      #f2f3ec;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --page:           #121309; --surface-1: #1b1c14; --surface-2: #23241a;
    --text-primary:   #f4f5ee; --text-secondary: #bcbeaf; --text-muted: #83857a;
    --grid: #2b2c22; --axis: #3a3c30; --accent: #8fae82;
    --border: rgba(244,245,238,0.10);
    --series-1: #3987e5; --series-ref: #6c6e60;
    --tooltip-bg: #f4f5ee; --tooltip-text: #14150f; --row-hover: #23241a;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --page:           #121309; --surface-1: #1b1c14; --surface-2: #23241a;
  --text-primary:   #f4f5ee; --text-secondary: #bcbeaf; --text-muted: #83857a;
  --grid: #2b2c22; --axis: #3a3c30; --accent: #8fae82;
  --border: rgba(244,245,238,0.10);
  --series-1: #3987e5; --series-ref: #6c6e60;
  --tooltip-bg: #f4f5ee; --tooltip-text: #14150f; --row-hover: #23241a;
}

* { box-sizing: border-box; }
body { margin: 0; }
.viz-root {
  background: var(--page);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--text-primary);
  padding: 32px 20px 60px;
}
.wrap { max-width: 1080px; margin: 0 auto; }
header.page-head { margin-bottom: 28px; }
.eyebrow {
  font-size: 11px; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--accent); margin: 0 0 6px;
}
h1 { font-size: 24px; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 6px; text-wrap: balance; }
.meta { font-size: 13px; color: var(--text-secondary); }
.meta span { font-variant-numeric: tabular-nums; }

.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px 26px;
  margin-bottom: 22px;
}
.card h2 { font-size: 17px; font-weight: 600; margin: 0 0 4px; }
.card .sub { font-size: 12.5px; color: var(--text-secondary); margin: 0 0 16px; }

.view-tabs {
  display: flex; gap: 4px; padding: 4px; margin-bottom: 18px;
  background: var(--surface-2); border-radius: 10px; width: fit-content;
}
.view-tab {
  font: inherit; font-size: 13px; font-weight: 600; color: var(--text-secondary);
  background: transparent; border: none; border-radius: 7px; padding: 8px 16px;
  cursor: pointer;
}
.view-tab.active { background: var(--surface-1); color: var(--text-primary); box-shadow: 0 1px 2px var(--border); }
.view-tab:hover:not(.active) { color: var(--text-primary); }

.controls { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-bottom: 6px; }
.filter-input {
  flex: 1; min-width: 200px; padding: 7px 11px; font-size: 13px;
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface-2); color: var(--text-primary);
}
.radio-group {
  display: flex; flex-wrap: wrap; gap: 14px; margin: 0; padding: 0; border: none;
}
.radio-group label {
  display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-secondary);
  white-space: nowrap; cursor: pointer;
}
.checkbox-label {
  display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text-secondary);
  white-space: nowrap; cursor: pointer;
}
.control-label {
  font-size: 13px; font-weight: 600; color: var(--text-secondary); white-space: nowrap;
}
.col-toggle { position: relative; }
.col-toggle-btn {
  font: inherit; font-size: 13px; font-weight: 600; color: var(--text-secondary);
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 7px 12px; cursor: pointer; white-space: nowrap;
}
.col-toggle-btn:hover { color: var(--text-primary); }
.col-menu {
  position: absolute; top: calc(100% + 6px); right: 0; z-index: 5;
  display: flex; flex-direction: column; gap: 5px; min-width: 180px;
  max-height: 280px; overflow-y: auto;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}
/* `display: flex` above otherwise beats the UA [hidden] rule at equal
   specificity, since this stylesheet loads after the UA one -- without this,
   the menu would render open by default instead of only on click. */
.col-menu[hidden] { display: none; }
.col-menu label {
  display: flex; align-items: center; gap: 7px; font-size: 12.5px; color: var(--text-secondary);
  white-space: nowrap; cursor: pointer;
}
.sort-note {
  font-size: 12px; color: var(--accent); margin: 0 0 14px; font-weight: 600;
}
.row-count { font-size: 12px; color: var(--text-muted); margin-top: 8px; font-variant-numeric: tabular-nums; }

.table-scroll {
  overflow: auto;
  max-height: 62vh; /* fallback until JS sizes this to fill the remaining viewport */
  border: 1px solid var(--border);
  border-radius: 8px;
}
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th, td {
  text-align: left; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
th {
  font-weight: 600; color: var(--text-secondary);
  border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--surface-1);
  padding: 0;
}
.th-label {
  display: block; cursor: pointer; user-select: none;
  padding: 7px 16px 7px 10px; overflow: hidden; text-overflow: ellipsis;
}
.th-label:hover { color: var(--text-primary); }
th .arrow { opacity: 0.4; margin-left: 3px; }
td { padding: 7px 10px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
tbody tr:hover { background: var(--row-hover); }

.col-resize-handle {
  position: absolute; top: 0; right: 0; width: 7px; height: 100%; cursor: col-resize;
}
.col-resize-handle:hover, .col-resize-handle.resizing { background: var(--accent); opacity: 0.35; }
table.resizing, table.resizing * { cursor: col-resize !important; user-select: none !important; }

.chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }
.chart-card { background: var(--surface-2); border-radius: 10px; padding: 14px 16px 10px; }
.chart-card h3 { font-size: 13px; font-weight: 600; margin: 0 0 8px; }
svg.chart { display: block; width: 100%; height: auto; overflow: visible; }
.axis-label { font-size: 9.5px; fill: var(--text-muted); }
.axis-line { stroke: var(--axis); stroke-width: 1; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.tick-label { font-size: 9px; fill: var(--text-muted); font-variant-numeric: tabular-nums; }
.dot { opacity: 0.55; }
.trend { fill: none; stroke-width: 2; }
.trend.reference { stroke: var(--series-ref); stroke-width: 1.5; stroke-dasharray: 5 4; }
.chart-stat { font-size: 10.5px; color: var(--text-muted); font-variant-numeric: tabular-nums; margin-top: 2px; }

.tooltip {
  position: fixed; pointer-events: none; background: var(--tooltip-bg); color: var(--tooltip-text);
  font-size: 11px; padding: 5px 8px; border-radius: 6px; line-height: 1.4; white-space: nowrap;
  opacity: 0; transform: translate(-50%, -100%); transition: opacity 0.1s; z-index: 10;
  font-variant-numeric: tabular-nums;
}
.tooltip.visible { opacity: 1; }
</style>

<div class="viz-root">
  <div class="wrap">
    <header class="page-head">
      <p class="eyebrow">SeedbankSurvival report</p>
      <h1 id="reportTitle">Regeneration priority &amp; inventory status</h1>
      <p class="meta" id="reportMeta"></p>
    </header>

    <div class="view-tabs" role="tablist">
      <button class="view-tab active" type="button" role="tab" aria-selected="true" data-view="accession">Accession view</button>
      <button class="view-tab" type="button" role="tab" aria-selected="false" data-view="inventory">Inventory view</button>
    </div>

    <section class="card" id="accessionCard">
      <h2>Accession view</h2>
      <p class="sub">One row per accession -- the best-representative packet, for deciding what to regenerate.</p>
      <p class="sort-note">Sorted by regeneration urgency: highest need (lowest predicted viability) first.</p>
      <div class="controls">
        <input class="filter-input" type="text" placeholder="Filter by accession, status, species, origin, reason..." data-filter-for="accessionTable">
      </div>
      <div class="controls">
        <fieldset class="radio-group" aria-label="Viability filter">
          <label><input type="radio" name="viabilityFilterAccession" value="all" checked> Show All</label>
          <label><input type="radio" name="viabilityFilterAccession" value="exclude0"> Exclude 0% Expected Viability</label>
          <label><input type="radio" name="viabilityFilterAccession" value="only0"> Show Only 0% Expected Viability</label>
        </fieldset>
        <label class="checkbox-label"><input type="checkbox" id="onSiteAccession"> On-site only (W6, Pullman)</label>
        <div class="col-toggle">
          <button type="button" class="col-toggle-btn" data-menu="colMenuAccession">Columns &#9662;</button>
          <div class="col-menu" id="colMenuAccession" hidden></div>
        </div>
      </div>
      <div class="controls">
        <span class="control-label">Model:</span>
        <fieldset class="radio-group" aria-label="Deterioration model" id="modelSelectAccession"></fieldset>
      </div>
      <div class="table-scroll"><table id="accessionTable"></table></div>
      <p class="row-count" id="accessionCount"></p>
    </section>

    <section class="card" id="inventoryCard" hidden>
      <h2>Inventory view</h2>
      <p class="sub">Every physical packet with seed on hand, dead or alive -- for deciding what to test or discard.</p>
      <p class="sort-note">Sorted by urgency: lowest predicted viability packets first.</p>
      <div class="controls">
        <input class="filter-input" type="text" placeholder="Filter by accession, status, species, origin, reason..." data-filter-for="inventoryTable">
      </div>
      <div class="controls">
        <fieldset class="radio-group" aria-label="Viability filter">
          <label><input type="radio" name="viabilityFilterInventory" value="all" checked> Show All</label>
          <label><input type="radio" name="viabilityFilterInventory" value="exclude0"> Exclude 0% Expected Viability</label>
          <label><input type="radio" name="viabilityFilterInventory" value="only0"> Show Only 0% Expected Viability</label>
        </fieldset>
        <label class="checkbox-label"><input type="checkbox" id="onSiteInventory"> On-site only (W6, Pullman)</label>
        <div class="col-toggle">
          <button type="button" class="col-toggle-btn" data-menu="colMenuInventory">Columns &#9662;</button>
          <div class="col-menu" id="colMenuInventory" hidden></div>
        </div>
      </div>
      <div class="controls">
        <span class="control-label">Model:</span>
        <fieldset class="radio-group" aria-label="Deterioration model" id="modelSelectInventory"></fieldset>
      </div>
      <div class="table-scroll"><table id="inventoryTable"></table></div>
      <p class="row-count" id="inventoryCount"></p>
    </section>

    <section class="card" id="chartsCard">
      <h2>Deterioration curves</h2>
      <p class="sub" id="chartsSub">Real fitted data. Species get their own chart only when they actually won the confidence comparison against the genus-wide model.</p>
      <div class="chart-grid" id="chartGrid"></div>
    </section>
  </div>
  <div class="tooltip" id="tooltip"></div>
</div>

<script>
const DATA = __PAYLOAD__;
let currentModel = DATA.defaultModel;

// ---------- model switching ----------
// The fitted curves (and therefore the charts) don't depend on which view
// you're looking at -- only which rows each view selects does -- so model
// selection is ONE shared piece of state, with a selector duplicated on
// both views (kept in sync) for convenience regardless of which tab is
// active, rather than two independently-selectable models.
function updateMeta() {
  const m = DATA.models[currentModel];
  document.getElementById("reportMeta").textContent =
    `Genera: ${DATA.genera.join(", ")} · as of ${DATA.asOfYear} · generated ${DATA.generatedAt} · ` +
    `${DATA.counts.accession} accessions · ${DATA.counts.inventory} packets · ` +
    `${DATA.modelLabels[currentModel]} model, genus-wide R²=${m.globalR2} (n=${m.globalN})`;
  document.getElementById("chartsSub").textContent =
    `${DATA.modelLabels[currentModel]} curves, real fitted data. Species get their own chart only when ` +
    `they actually won the confidence comparison against the genus-wide model.`;
}
updateMeta();

// ---------- tables ----------
// Columns start auto-sized (matching the old plain-table look). Right after
// the first paint, natural widths are measured and locked into a <colgroup>
// under table-layout:fixed -- from then on, resizing (drag the handle at a
// header's right edge) and hide/show (the Columns menu) both just rebuild
// that colgroup, so neither one fights the browser reflowing column widths
// from whatever happens to be on screen after a sort/filter.
function renderTable(tableId, rows, columns) {
  const table = document.getElementById(tableId);
  let sortCol = null, sortDir = 1;
  const colWidths = new Map();
  const hiddenCols = new Set();
  let measured = false;

  function visibleColumns() {
    return columns.filter(([key]) => !hiddenCols.has(key));
  }

  function startResize(e, key, th) {
    e.preventDefault();
    e.stopPropagation();
    const startX = e.clientX;
    const startWidth = th.getBoundingClientRect().width;
    const handle = e.target;
    table.classList.add("resizing");
    handle.classList.add("resizing");

    function onMove(ev) {
      const width = Math.max(48, startWidth + (ev.clientX - startX));
      colWidths.set(key, width);
      const col = table.querySelector(`col[data-key="${CSS.escape(key)}"]`);
      if (col) col.style.width = width + "px";
    }
    function onUp() {
      table.classList.remove("resizing");
      handle.classList.remove("resizing");
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    }
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  function draw(data) {
    const cols = visibleColumns();
    let colgroup = "<colgroup>" + cols.map(([key]) => {
      const w = colWidths.get(key);
      return `<col data-key="${key}"${w ? ` style="width:${w}px"` : ""}>`;
    }).join("") + "</colgroup>";
    let head = "<thead><tr>" + cols.map(([key, label]) =>
      `<th data-key="${key}"><span class="th-label" data-key="${key}">${label}` +
      `<span class="arrow">${sortCol === key ? (sortDir > 0 ? "↑" : "↓") : ""}</span></span>` +
      `<span class="col-resize-handle" data-key="${key}"></span></th>`
    ).join("") + "</tr></thead>";
    let body = "<tbody>" + data.map(row =>
      "<tr>" + cols.map(([key]) => {
        const v = row[key] === null || row[key] === undefined ? "" : row[key];
        return `<td title="${String(v).replace(/"/g, "&quot;")}">${v}</td>`;
      }).join("") + "</tr>"
    ).join("") + "</tbody>";
    table.innerHTML = colgroup + head + body;

    table.querySelectorAll(".th-label").forEach(label => {
      label.addEventListener("click", () => {
        const key = label.dataset.key;
        sortDir = (sortCol === key) ? -sortDir : 1;
        sortCol = key;
        state.rows = state.rows.slice().sort((a, b) => {
          const av = a[key], bv = b[key];
          if (av === null || av === undefined) return 1;
          if (bv === null || bv === undefined) return -1;
          if (typeof av === "number" && typeof bv === "number") return (av - bv) * sortDir;
          return String(av).localeCompare(String(bv)) * sortDir;
        });
        draw(state.rows);
      });
    });
    table.querySelectorAll(".col-resize-handle").forEach(handle => {
      handle.addEventListener("mousedown", (e) => startResize(e, handle.dataset.key, handle.closest("th")));
    });

    if (!measured) {
      measured = true;
      table.querySelectorAll("thead th").forEach(th => {
        colWidths.set(th.dataset.key, th.getBoundingClientRect().width);
      });
      table.style.tableLayout = "fixed";
      draw(data);
    }
  }

  const state = { rows: rows };
  draw(state.rows);
  return {
    setRows(newRows) { state.rows = newRows; draw(state.rows); },
    buildColumnMenu(menuEl) {
      menuEl.innerHTML = columns.map(([key, label]) =>
        `<label><input type="checkbox" data-key="${key}" ${hiddenCols.has(key) ? "" : "checked"}> ${label}</label>`
      ).join("");
      menuEl.querySelectorAll("input").forEach(cb => {
        cb.addEventListener("change", () => {
          if (cb.checked) hiddenCols.delete(cb.dataset.key); else hiddenCols.add(cb.dataset.key);
          draw(state.rows);
        });
      });
    },
  };
}

const accessionTableCtl = renderTable("accessionTable", DATA.models[currentModel].accessionRows, DATA.columns);
const inventoryTableCtl = renderTable("inventoryTable", DATA.models[currentModel].inventoryRows, DATA.columns);
document.getElementById("accessionCount").textContent = DATA.models[currentModel].accessionRows.length + " rows";
document.getElementById("inventoryCount").textContent = DATA.models[currentModel].inventoryRows.length + " rows";

accessionTableCtl.buildColumnMenu(document.getElementById("colMenuAccession"));
inventoryTableCtl.buildColumnMenu(document.getElementById("colMenuInventory"));
document.querySelectorAll(".col-toggle-btn").forEach(btn => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const menu = document.getElementById(btn.dataset.menu);
    const wasHidden = menu.hidden;
    document.querySelectorAll(".col-menu").forEach(m => { m.hidden = true; });
    menu.hidden = !wasHidden;
  });
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".col-toggle")) {
    document.querySelectorAll(".col-menu").forEach(m => { m.hidden = true; });
  }
});

function applyFilters() {
  document.querySelectorAll(".filter-input").forEach(input => {
    const targetId = input.dataset.filterFor;
    const query = input.value.trim().toLowerCase();
    const modelData = DATA.models[currentModel];
    const source = targetId === "accessionTable" ? modelData.accessionRows : modelData.inventoryRows;
    const ctl = targetId === "accessionTable" ? accessionTableCtl : inventoryTableCtl;

    let filtered = source;
    if (query) {
      filtered = filtered.filter(row =>
        ["Accession", "Status", "Species", "Origin", "PrimaryReason"].some(k =>
          String(row[k] || "").toLowerCase().includes(query)
        )
      );
    }
    const radioName = targetId === "accessionTable" ? "viabilityFilterAccession" : "viabilityFilterInventory";
    const viabilityFilter = document.querySelector(`input[name="${radioName}"]:checked`).value;
    if (viabilityFilter === "exclude0") {
      filtered = filtered.filter(row => Math.round(row["EstimatedViability_2026"] || 0) !== 0);
    } else if (viabilityFilter === "only0") {
      filtered = filtered.filter(row => Math.round(row["EstimatedViability_2026"] || 0) === 0);
    }
    const onSiteId = targetId === "accessionTable" ? "onSiteAccession" : "onSiteInventory";
    if (document.getElementById(onSiteId).checked) {
      filtered = filtered.filter(row => row["MaintenanceSite"] === "W6");
    }
    ctl.setRows(filtered);
    document.getElementById(targetId === "accessionTable" ? "accessionCount" : "inventoryCount").textContent = filtered.length + " rows";
  });
}
document.querySelectorAll(".filter-input").forEach(el => el.addEventListener("input", applyFilters));
// Model-select radios are excluded here -- their own handler (setModel,
// below) needs to run first to swap currentModel before applyFilters reads
// DATA.models[currentModel]; setModel calls applyFilters itself afterward.
document.querySelectorAll('.radio-group input[type="radio"]:not([name^="modelSelect"])').forEach(el => el.addEventListener("change", applyFilters));
document.querySelectorAll('.checkbox-label input[type="checkbox"]').forEach(el => el.addEventListener("change", applyFilters));

function buildModelSelector(container, groupName) {
  container.innerHTML = Object.keys(DATA.models).map(kind =>
    `<label><input type="radio" name="${groupName}" value="${kind}" ${kind === currentModel ? "checked" : ""}> ${DATA.modelLabels[kind]}</label>`
  ).join("");
}
buildModelSelector(document.getElementById("modelSelectAccession"), "modelSelectAccession");
buildModelSelector(document.getElementById("modelSelectInventory"), "modelSelectInventory");

function setModel(kind) {
  currentModel = kind;
  document.querySelectorAll('input[name="modelSelectAccession"], input[name="modelSelectInventory"]').forEach(el => {
    el.checked = el.value === kind;
  });
  updateMeta();
  renderCharts(DATA.models[currentModel].charts);
  applyFilters();
}
document.querySelectorAll('input[name="modelSelectAccession"], input[name="modelSelectInventory"]').forEach(el => {
  el.addEventListener("change", (e) => setModel(e.target.value));
});

// ---------- view toggle ----------
document.querySelectorAll(".view-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".view-tab").forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");

    const view = tab.dataset.view;
    document.getElementById("accessionCard").hidden = view !== "accession";
    document.getElementById("inventoryCard").hidden = view !== "inventory";
    fitTablePanes();
  });
});

// ---------- fill-the-window table sizing ----------
// A fixed vh percentage wastes/overflows space depending on how tall the
// header/controls above it happen to render -- size each visible table pane
// to use exactly the viewport height remaining below it instead.
function fitTablePanes() {
  document.querySelectorAll(".table-scroll").forEach(el => {
    if (el.offsetParent === null) return; // hidden (inactive view)
    const top = el.getBoundingClientRect().top;
    const available = window.innerHeight - top - 36; // room for the row-count line below
    el.style.maxHeight = Math.max(240, available) + "px";
  });
}
window.addEventListener("resize", fitTablePanes);
window.addEventListener("load", fitTablePanes);
fitTablePanes();
requestAnimationFrame(fitTablePanes); // catch a viewport/layout not yet settled on first paint

// ---------- charts ----------
const chartGrid = document.getElementById("chartGrid");
const tooltip = document.getElementById("tooltip");

function evalCurve(model, age) {
  if (model.kind === "weibull") {
    const a = Math.max(age, 0);
    return model.v0 * Math.exp(-Math.pow(a / model.lam, model.k));
  }
  if (model.kind === "breakpoint") {
    return age <= model.t0 ? model.plateau : model.plateau + model.slope * (age - model.t0);
  }
  return model.intercept + model.linearCoef * age + model.quadCoef * age * age;
}

function buildChartSvg(chart) {
  const W = 320, H = 210;
  const M = { top: 8, right: 10, bottom: 26, left: 30 };
  const plotW = W - M.left - M.right, plotH = H - M.top - M.bottom;
  const xMax = chart.xMax, yMax = 100;
  const sx = age => M.left + (age / xMax) * plotW;
  const sy = v => M.top + plotH - (v / yMax) * plotH;

  let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${chart.title}: viability vs seed age at test">`;
  for (let v = 0; v <= 100; v += 50) {
    const y = sy(v);
    svg += `<line class="gridline" x1="${M.left}" y1="${y}" x2="${W - M.right}" y2="${y}"/>`;
    svg += `<text class="tick-label" x="${M.left - 5}" y="${y}" text-anchor="end" dominant-baseline="central">${v}</text>`;
  }
  svg += `<line class="axis-line" x1="${M.left}" y1="${H - M.bottom}" x2="${W - M.right}" y2="${H - M.bottom}"/>`;
  svg += `<text class="axis-label" x="${M.left + plotW / 2}" y="${H - 4}" text-anchor="middle">age (yrs)</text>`;

  // Quadratic, not a straight line -- sample it at many points rather than
  // just two endpoints, so the drawn path actually follows the curve
  // (including any plateau-then-drop shape) instead of cutting a chord
  // across it. A linear fit (quadCoef 0) still renders correctly this way,
  // just as a visually straight sampled path.
  function trendPath(model) {
    const STEPS = 40;
    let d = "";
    for (let i = 0; i <= STEPS; i++) {
      const age = (i / STEPS) * xMax;
      const v = Math.max(0, Math.min(100, evalCurve(model, age)));
      d += (i === 0 ? "M " : "L ") + sx(age) + " " + sy(v) + " ";
    }
    return d.trim();
  }

  if (chart.reference) {
    svg += `<path class="trend reference" d="${trendPath(chart.reference)}"/>`;
  }
  chart.points.forEach(([age, v]) => {
    svg += `<circle class="dot" cx="${sx(age)}" cy="${sy(v)}" r="2.6" fill="var(--series-1)" data-age="${age}" data-v="${v}"/>`;
  });
  svg += `<path class="trend" style="stroke:var(--series-1)" d="${trendPath(chart.model)}"/>`;
  svg += `</svg>`;
  return svg;
}

function renderCharts(charts) {
  chartGrid.innerHTML = "";
  charts.forEach(chart => {
    const div = document.createElement("div");
    div.className = "chart-card";
    // A curve has no single constant slope -- show the average rate of
    // change over the plotted age range instead of one instantaneous value.
    // Clipped the same way the drawn curve is: an unclipped quadratic can
    // extrapolate to wildly implausible values (e.g. -300%) at the edge of a
    // wide age range even when the curve is visibly flat at 0 for most of
    // it, which would otherwise contradict what the chart actually shows.
    const clip = v => Math.max(0, Math.min(100, v));
    const avgSlope = (clip(evalCurve(chart.model, chart.xMax)) - clip(evalCurve(chart.model, 0))) / chart.xMax;
    div.innerHTML = `<h3>${chart.title}</h3>` + buildChartSvg(chart) +
      `<div class="chart-stat">n=${chart.model.n} · R²=${chart.model.r2.toFixed(2)} · avg ${avgSlope.toFixed(2)}%/yr</div>`;
    chartGrid.appendChild(div);
  });
}
renderCharts(DATA.models[currentModel].charts);

chartGrid.addEventListener("mousemove", (e) => {
  const target = e.target.closest(".dot");
  if (!target) { tooltip.classList.remove("visible"); return; }
  tooltip.textContent = `age ${target.dataset.age}y, viability ${target.dataset.v}%`;
  tooltip.style.left = e.clientX + "px";
  tooltip.style.top = (e.clientY - 10) + "px";
  tooltip.classList.add("visible");
});
chartGrid.addEventListener("mouseleave", () => tooltip.classList.remove("visible"));
</script>
"""
