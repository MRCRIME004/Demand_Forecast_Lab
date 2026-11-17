# Start here: Demand Forecasting and Inventory Planning source

The website has two ways to run the **same Python forecasting engine**. On the hosted site, `dist/worker.mjs` loads Python through Pyodide. On a local computer, `server.py` calls `dist/engine.py` directly. The JavaScript files below handle the interface; they do not calculate the forecasts or inventory values.

## Find the code by task

| What you want to inspect | Where the working code is |
|---|---|
| Input form and tab layout | `dist/index.html` (`settings-sidebar`, `data`, `overview`, `graphs`, `purchase`, `tables`) |
| File upload, CSV parsing, column selection, Data-tab editing | `dist/js/data/02-data-input-and-editing.js` (`parseCSV`, `loadCSV`, `renderInput`, `syncRowToRaw`) |
| Setup controls, tab navigation, shared display formatting | `dist/js/interface/01-setup-and-navigation.js` |
| Graph cards, chart search, picker, image viewer | `dist/js/charts/03-charts-and-gallery.js` |
| Forecast report and purchase-plan screen | `dist/js/reports/04-forecast-report-and-inventory.js` |
| Table selection, sorting, CSV download | `dist/js/tables/05-tables.js` |
| Create analysis payload, call local server or browser worker, display result | `dist/js/runtime/06-run-engine.js` (`payload`, `browserRun`, `runAnalysis`, `showResult`) |
| Final presentation and accessibility controls | `dist/js/interface/07-presentation-and-accessibility.js` and `dist/v2.js` |
| Python input checks and data preparation | `dist/engine.py`: `validate` |
| Model fitting and tuning | `dist/engine.py`: `fit_ses`, `fit_holt`, `fit_holt_winters_state`, `tune_ridge_and_rf`, `search_sarima_order_by_aic` |
| Forecast comparison and prediction intervals | `dist/engine.py`: `rolling_origin_cv`, `compute_intervals_and_coverage`, `analyze` |
| EOQ, safety stock, reorder point, purchase dates | `dist/engine.py`: `build_purchase_plan`; `dist/analysis_core.py`: `inventory`, `simulate_policy` |
| Chart drawing and four-part reading guides | `dist/visualization.py`; figures and guide records are assembled in `dist/engine.py` |
| Hosted Python startup | `dist/worker.mjs` |
| Local Python server | `server.py`; Windows launcher: `START_WEBSITE.cmd` |
| Tests | `tests/test_engine.py`, `tests/check_boundaries.py` |

## Run it

For the hosted version, open `https://yourfuturedemand.netlify.app/`. For local Windows, install Python 3.13, extract the project and double-click `START_WEBSITE.cmd`; it installs only `requirements.txt` packages if missing and opens `http://127.0.0.1:8767/`. Keep the terminal window open. To inspect code, open the files directly; no download command is needed to read them.

The seven numbered JavaScript files are loaded in order by `dist/index.html` as ordinary browser scripts. The split only moves existing interface code into named files. The numerical engine files remain intact.
