"""Builds a single self-contained HTML report -- no server, no external
resources, opens in any browser. This is the project's "dashboard" for now:
a curator runs the CLI and gets one file back with both views and charts.

build_report() is IO-free (returns a string; the CLI does the file write),
matching the rest of this package's pure-function style.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from .deterioration import SlopeModel

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
    species_models: dict[str, SlopeModel],
    top_n_charts: int,
) -> list[str]:
    """Species that actually won the confidence comparison for >=1 row, ranked
    by how many accession-view rows use that tier, capped at top_n_charts.

    A species with its own fitted SlopeModel whose points all got overridden
    back to Global by the R^2 comparison does NOT get its own chart -- the
    charts would otherwise visually contradict ModelUsed/ModelConfidence.
    """
    won = accession_table[accession_table["ModelUsed"] == "Species"]
    counts = won["Species"].value_counts()
    return [name for name in counts.index if name in species_models][:top_n_charts]


def _trend_endpoints(model: SlopeModel, x_max: float) -> tuple[float, float]:
    y0 = max(0.0, min(100.0, model.intercept))
    y1 = max(0.0, min(100.0, model.intercept + model.slope * x_max))
    return y0, y1


def _build_charts_payload(
    df_model: pd.DataFrame,
    species_models: dict[str, SlopeModel],
    global_model: SlopeModel,
    charted_species: list[str],
    species_group_col: str,
) -> list[dict]:
    x_max = float(df_model["AgeAtTest"].max()) if len(df_model) else 1.0
    x_max = max(x_max, 1.0)

    charts = [
        {
            "title": f"Genus-wide (all {len(df_model)} test points)",
            "points": df_model[["AgeAtTest", "Viability"]].round(1).to_numpy().tolist(),
            "model": {"intercept": global_model.intercept, "slope": global_model.slope,
                      "r2": global_model.r2, "n": global_model.n},
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
                "model": {"intercept": model.intercept, "slope": model.slope,
                          "r2": model.r2, "n": model.n},
                "reference": {"intercept": global_model.intercept, "slope": global_model.slope},
                "xMax": x_max,
            }
        )
    return charts


def build_report(
    accession_table: pd.DataFrame,
    inventory_table: pd.DataFrame,
    df_model: pd.DataFrame,
    species_models: dict[str, SlopeModel],
    origin_models: dict[str, SlopeModel],
    global_model: SlopeModel,
    genera: list[str],
    as_of_year: int,
    top_n_charts: int = 8,
    species_group_col: str = "SpeciesGroup",
) -> str:
    """Build the self-contained HTML report. Returns HTML; writes nothing."""
    charted_species = _select_charted_species(accession_table, species_models, top_n_charts)
    charts = _build_charts_payload(
        df_model, species_models, global_model, charted_species, species_group_col
    )
    _row_columns = [c for c, _ in _TABLE_COLUMNS] + _FILTER_ONLY_COLUMNS

    payload = {
        "genera": genera,
        "asOfYear": as_of_year,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "columns": _TABLE_COLUMNS,
        "accessionRows": accession_table[_row_columns].to_dict("records"),
        "inventoryRows": inventory_table[_row_columns].to_dict("records"),
        "charts": charts,
        "counts": {
            "accession": len(accession_table),
            "inventory": len(inventory_table),
            "globalR2": round(global_model.r2, 2),
            "globalN": global_model.n,
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
th, td { text-align: left; padding: 7px 10px; white-space: nowrap; }
th {
  font-weight: 600; color: var(--text-secondary); cursor: pointer; user-select: none;
  border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--surface-1);
}
th:hover { color: var(--text-primary); }
th .arrow { opacity: 0.4; margin-left: 3px; }
td { border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
tbody tr:hover { background: var(--row-hover); }

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
      </div>
      <div class="table-scroll"><table id="inventoryTable"></table></div>
      <p class="row-count" id="inventoryCount"></p>
    </section>

    <section class="card" id="chartsCard">
      <h2>Deterioration curves</h2>
      <p class="sub">Real fitted data. Species get their own chart only when they actually won the confidence comparison against the genus-wide model.</p>
      <div class="chart-grid" id="chartGrid"></div>
    </section>
  </div>
  <div class="tooltip" id="tooltip"></div>
</div>

<script>
const DATA = __PAYLOAD__;

document.getElementById("reportMeta").textContent =
  `Genera: ${DATA.genera.join(", ")} · as of ${DATA.asOfYear} · generated ${DATA.generatedAt} · ` +
  `${DATA.counts.accession} accessions · ${DATA.counts.inventory} packets · genus-wide R²=${DATA.counts.globalR2} (n=${DATA.counts.globalN})`;

// ---------- tables ----------
function renderTable(tableId, rows, columns) {
  const table = document.getElementById(tableId);
  let sortCol = null, sortDir = 1;

  function draw(data) {
    let head = "<thead><tr>" + columns.map(([key, label]) =>
      `<th data-key="${key}">${label}<span class="arrow">${sortCol === key ? (sortDir > 0 ? "↑" : "↓") : ""}</span></th>`
    ).join("") + "</tr></thead>";
    let body = "<tbody>" + data.map(row =>
      "<tr>" + columns.map(([key]) => `<td>${row[key] === null || row[key] === undefined ? "" : row[key]}</td>`).join("") + "</tr>"
    ).join("") + "</tbody>";
    table.innerHTML = head + body;
    table.querySelectorAll("th").forEach(th => {
      th.addEventListener("click", () => {
        const key = th.dataset.key;
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
  }

  const state = { rows: rows };
  draw(state.rows);
  return {
    setRows(newRows) { state.rows = newRows; draw(state.rows); },
  };
}

const accessionTableCtl = renderTable("accessionTable", DATA.accessionRows, DATA.columns);
const inventoryTableCtl = renderTable("inventoryTable", DATA.inventoryRows, DATA.columns);
document.getElementById("accessionCount").textContent = DATA.accessionRows.length + " rows";
document.getElementById("inventoryCount").textContent = DATA.inventoryRows.length + " rows";

function applyFilters() {
  document.querySelectorAll(".filter-input").forEach(input => {
    const targetId = input.dataset.filterFor;
    const query = input.value.trim().toLowerCase();
    const source = targetId === "accessionTable" ? DATA.accessionRows : DATA.inventoryRows;
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
document.querySelectorAll('.radio-group input[type="radio"]').forEach(el => el.addEventListener("change", applyFilters));
document.querySelectorAll('.checkbox-label input[type="checkbox"]').forEach(el => el.addEventListener("change", applyFilters));

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

  function trendPath(model) {
    const y0 = Math.max(0, Math.min(100, model.intercept));
    const y1 = Math.max(0, Math.min(100, model.intercept + model.slope * xMax));
    return `M ${sx(0)} ${sy(y0)} L ${sx(xMax)} ${sy(y1)}`;
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

DATA.charts.forEach(chart => {
  const div = document.createElement("div");
  div.className = "chart-card";
  div.innerHTML = `<h3>${chart.title}</h3>` + buildChartSvg(chart) +
    `<div class="chart-stat">n=${chart.model.n} · R²=${chart.model.r2.toFixed(2)} · ${chart.model.slope.toFixed(2)}%/yr</div>`;
  chartGrid.appendChild(div);
});

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
