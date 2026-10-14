// Run with: node test\test_benchmark_tables.js
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

let ready;
const source = fs.readFileSync(path.join(__dirname, "..", "docs", "javascripts", "benchmarks.js"), "utf8");
const tableCode = vm.runInNewContext(
  `${source}\n({ BENCHMARK_TABLES, getCellText, parseSortValue, compareRows, renderTable })`,
  {
    document: {
      readyState: "loading",
      addEventListener(event, callback) {
        assert.equal(event, "DOMContentLoaded");
        ready = callback;
      },
    },
  },
);
assert.equal(typeof ready, "function");
assert.deepEqual(Array.from(tableCode.BENCHMARK_TABLES, (config) => config.id), ["small-molecules", "gpu-comparisons"]);
assert.equal(tableCode.getCellText({ textContent: "  Wall\n time  " }), "Wall time");
assert.equal(tableCode.parseSortValue("6,009 s", "duration"), 6009);
assert.equal(tableCode.parseSortValue("baseline", "relative-speed"), 1);
assert.equal(tableCode.parseSortValue("5.61x", "relative-speed"), 5.61);
assert.equal(tableCode.parseSortValue("-", "number"), null);

const row = (value, index) => ({
  cells: [{ textContent: value }],
  dataset: { benchmarkOriginalIndex: String(index) },
});
const config = { sortTypes: { 0: "number" } };
const state = { sortColumn: 0, sortDirection: "asc", filters: new Map() };
const low = row("2", 0), high = row("10", 1), missing = row("-", 2), tied = row("2", 3);
assert.ok(tableCode.compareRows(low, high, state, config) < 0);
assert.ok(tableCode.compareRows(low, tied, state, config) < 0);
state.sortDirection = "desc";
assert.ok(tableCode.compareRows(low, high, state, config) > 0);
assert.ok(tableCode.compareRows(missing, high, state, config) > 0);

let displayed, empty, ariaSort, label;
const indicator = {};
const header = {
  dataset: { benchmarkHeader: "Value" },
  querySelector(selector) {
    return selector === ".benchmark-sort-indicator" ? indicator : {
      setAttribute(name, value) { label = value; },
    };
  },
  setAttribute(name, value) { ariaSort = value; },
};
const table = { tBodies: [{ replaceChildren(...rows) { displayed = rows; } }] };
const emptyState = { classList: { toggle(name, value) { empty = value; } } };
const rows = [low, high, missing];
tableCode.renderTable(table, state, config, rows, emptyState, [header]);
assert.deepEqual(displayed, [high, low, missing]);
assert.equal(ariaSort, "descending");
assert.equal(label, "Value (descending)");
assert.equal(empty, false);
state.filters.set(0, { value: "absent" });
tableCode.renderTable(table, state, config, rows, emptyState, [header]);
assert.deepEqual(displayed, []);
assert.equal(empty, true);
state.filters.clear();
state.sortColumn = state.sortDirection = null;
tableCode.renderTable(table, state, config, rows, emptyState, [header]);
assert.deepEqual(displayed, rows);
assert.equal(ariaSort, "none");
console.log("Benchmark table parsing, sorting, filtering, reset and ARIA checks passed.");
