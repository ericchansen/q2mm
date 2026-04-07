const BENCHMARK_TABLES = [
  {
    id: "small-molecules",
    filterColumns: [0, 1, 2, 3],
    sortTypes: {
      4: "number",
      5: "number",
      6: "duration",
      7: "number",
    },
  },
  {
    id: "rh-enamide",
    filterColumns: [0, 1, 2, 3],
    sortTypes: {
      3: "status",
      4: "result",
      5: "number",
      6: "duration",
    },
  },
  {
    id: "gpu-comparisons",
    filterColumns: [0, 1, 2],
    sortTypes: {
      3: "number",
      4: "number",
      5: "duration",
      6: "relative-speed",
    },
  },
];

function cleanText(value) {
  return value.replace(/\s+/g, " ").trim();
}

function getCellText(cell) {
  return cleanText(cell.textContent || "");
}

function getHeaderText(header) {
  return cleanText(header.textContent || "");
}

function parseNumber(value) {
  const text = cleanText(value).replace(/,/g, "");
  if (!text || text === "-") {
    return null;
  }

  const matches = text.match(/-?\d+(?:\.\d+)?/g);
  if (!matches || matches.length === 0) {
    return null;
  }

  return Number(matches[0]);
}

function parseDuration(value) {
  return parseNumber(value);
}

function parseResult(value) {
  const text = cleanText(value).replace(/,/g, "");
  if (!text || text === "-") {
    return null;
  }

  const matches = text.match(/-?\d+(?:\.\d+)?/g);
  if (!matches || matches.length === 0) {
    return null;
  }

  return Number(matches[matches.length - 1]);
}

function parseRelativeSpeed(value) {
  const text = cleanText(value).toLowerCase();
  if (!text) {
    return null;
  }
  if (text === "baseline") {
    return 1;
  }
  return parseNumber(text);
}

function parseStatus(value) {
  const text = cleanText(value).toLowerCase();
  const order = {
    success: 0,
    failed: 1,
  };
  return Object.prototype.hasOwnProperty.call(order, text) ? order[text] : 99;
}

function parseSortValue(value, type) {
  if (type === "duration") {
    return parseDuration(value);
  }
  if (type === "result") {
    return parseResult(value);
  }
  if (type === "relative-speed") {
    return parseRelativeSpeed(value);
  }
  if (type === "status") {
    return parseStatus(value);
  }
  if (type === "number") {
    return parseNumber(value);
  }
  return cleanText(value).toLowerCase();
}

function compareRows(leftRow, rightRow, state, config) {
  if (state.sortColumn === null) {
    return Number(leftRow.dataset.benchmarkOriginalIndex) - Number(rightRow.dataset.benchmarkOriginalIndex);
  }

  const columnIndex = state.sortColumn;
  const sortType = config.sortTypes[columnIndex] || "text";
  const leftValue = parseSortValue(getCellText(leftRow.cells[columnIndex]), sortType);
  const rightValue = parseSortValue(getCellText(rightRow.cells[columnIndex]), sortType);
  const leftMissing = leftValue === null || leftValue === undefined || Number.isNaN(leftValue);
  const rightMissing = rightValue === null || rightValue === undefined || Number.isNaN(rightValue);

  if (leftMissing && rightMissing) {
    return Number(leftRow.dataset.benchmarkOriginalIndex) - Number(rightRow.dataset.benchmarkOriginalIndex);
  }
  if (leftMissing) {
    return 1;
  }
  if (rightMissing) {
    return -1;
  }

  let comparison = 0;
  if (typeof leftValue === "number" && typeof rightValue === "number") {
    comparison = leftValue - rightValue;
  } else {
    comparison = String(leftValue).localeCompare(String(rightValue), undefined, {
      numeric: true,
      sensitivity: "base",
    });
  }

  if (comparison === 0) {
    comparison = Number(leftRow.dataset.benchmarkOriginalIndex) - Number(rightRow.dataset.benchmarkOriginalIndex);
  }

  return state.sortDirection === "desc" ? -comparison : comparison;
}

function updateSortIndicators(headers, state) {
  headers.forEach((header, index) => {
    const indicator = header.querySelector(".benchmark-sort-indicator");
    const button = header.querySelector(".benchmark-sort-button");
    let symbol = "↕";
    let ariaSort = "none";

    if (state.sortColumn === index && state.sortDirection === "asc") {
      symbol = "↑";
      ariaSort = "ascending";
    } else if (state.sortColumn === index && state.sortDirection === "desc") {
      symbol = "↓";
      ariaSort = "descending";
    }

    if (indicator) {
      indicator.textContent = symbol;
    }
    if (button) {
      button.setAttribute("aria-label", `${header.dataset.benchmarkHeader} (${ariaSort})`);
    }
    header.setAttribute("aria-sort", ariaSort);
  });
}

function renderTable(table, state, config, originalRows, emptyState, headers) {
  const tbody = table.tBodies[0];
  const rows = originalRows
    .filter((row) => {
      for (const [columnIndex, select] of state.filters.entries()) {
        if (select.value && getCellText(row.cells[columnIndex]) !== select.value) {
          return false;
        }
      }
      return true;
    })
    .sort((leftRow, rightRow) => compareRows(leftRow, rightRow, state, config));

  tbody.replaceChildren(...rows);
  emptyState.classList.toggle("is-visible", rows.length === 0);
  updateSortIndicators(headers, state);
}

function buildFilterControl(columnIndex, headerText, rows, state, rerender) {
  const label = document.createElement("label");
  label.className = "benchmark-filter";

  const title = document.createElement("span");
  title.className = "benchmark-filter__label";
  title.textContent = headerText;

  const select = document.createElement("select");
  select.setAttribute("aria-label", `Filter ${headerText}`);

  const allOption = document.createElement("option");
  allOption.value = "";
  allOption.textContent = `All ${headerText}`;
  select.append(allOption);

  const seen = new Set();
  rows.forEach((row) => {
    const value = getCellText(row.cells[columnIndex]);
    if (seen.has(value)) {
      return;
    }
    seen.add(value);

    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  });

  select.addEventListener("change", rerender);
  state.filters.set(columnIndex, select);
  label.append(title, select);
  return label;
}

function makeHeadersSortable(headers, state, rerender) {
  headers.forEach((header, columnIndex) => {
    const label = getHeaderText(header);
    header.dataset.benchmarkHeader = label;
    header.scope = "col";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "benchmark-sort-button";

    const text = document.createElement("span");
    text.textContent = label;

    const indicator = document.createElement("span");
    indicator.className = "benchmark-sort-indicator";
    indicator.setAttribute("aria-hidden", "true");
    indicator.textContent = "↕";

    button.append(text, indicator);
    button.addEventListener("click", () => {
      if (state.sortColumn !== columnIndex) {
        state.sortColumn = columnIndex;
        state.sortDirection = "asc";
      } else if (state.sortDirection === "asc") {
        state.sortDirection = "desc";
      } else {
        state.sortColumn = null;
        state.sortDirection = null;
      }

      rerender();
    });

    header.textContent = "";
    header.append(button);
  });
}

function findNextTable(anchor) {
  let current = anchor.nextElementSibling;
  while (current) {
    if (current.tagName === "TABLE") {
      return current;
    }
    // Material wraps tables: scrollwrap > md-typeset__table > table
    const nested = current.querySelector("table");
    if (nested) {
      return nested;
    }
    current = current.nextElementSibling;
  }
  return null;
}

function enhanceBenchmarkTable(config) {
  const anchor = document.querySelector(`.benchmark-table-anchor[data-benchmark-table="${config.id}"]`);
  if (!anchor) {
    return;
  }

  const table = findNextTable(anchor);
  if (!table || table.dataset.benchmarkEnhanced === "true" || !table.tHead || !table.tBodies[0]) {
    return;
  }

  table.dataset.benchmarkEnhanced = "true";
  table.classList.add("benchmark-table");

  const headers = Array.from(table.tHead.querySelectorAll("th"));
  const originalRows = Array.from(table.tBodies[0].rows);
  originalRows.forEach((row, index) => {
    row.dataset.benchmarkOriginalIndex = String(index);
  });

  const shell = document.createElement("section");
  shell.className = "benchmark-table-shell";
  shell.dataset.benchmarkTable = config.id;

  const toolbar = document.createElement("div");
  toolbar.className = "benchmark-table-toolbar";

  const scroll = document.createElement("div");
  scroll.className = "benchmark-table-scroll";

  const emptyState = document.createElement("p");
  emptyState.className = "benchmark-table-empty";
  emptyState.textContent = "No rows match the current filters.";

  const state = {
    filters: new Map(),
    sortColumn: null,
    sortDirection: null,
  };

  const rerender = () => renderTable(table, state, config, originalRows, emptyState, headers);

  config.filterColumns.forEach((columnIndex) => {
    toolbar.append(buildFilterControl(columnIndex, headers[columnIndex].textContent, originalRows, state, rerender));
  });

  makeHeadersSortable(headers, state, rerender);

  anchor.replaceWith(shell);
  shell.append(toolbar, scroll, emptyState);
  scroll.append(table);
  rerender();
}

function initBenchmarkTables() {
  BENCHMARK_TABLES.forEach(enhanceBenchmarkTable);
}

if (typeof document$ !== "undefined" && typeof document$.subscribe === "function") {
  document$.subscribe(initBenchmarkTables);
} else if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initBenchmarkTables);
} else {
  initBenchmarkTables();
}
