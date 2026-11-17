# Demand Forecast Lab

Demand Forecast Lab is a browser-based tool for daily demand forecasting and inventory planning. It compares forecasting methods on chronological train, validation, and test splits, then uses the forecast for EOQ, safety stock, reorder-point, and purchase-plan calculations.

## Live website

The deployed website is served from the `dist/` directory through Netlify. Python forecasting runs in the browser with Pyodide, so uploaded data remains in the browser session.

## Run locally

From the repository root, start any static file server in `dist/`. For example:

```powershell
cd dist
python -m http.server 8000
```

Then open `http://localhost:8000`.

The first analysis needs an internet connection so Pyodide can load the scientific Python packages from its CDN.

## Find the working code

Open [CODE_MAP.md](CODE_MAP.md) for a task-by-task map of input, Data tab, charts, tables, forecasting, EOQ, and the purchase schedule. The browser interface now lives in clearly named folders under `dist/js/`. These are the files loaded by the website itself.

## Repository structure

- `dist/index.html` — application page
- `dist/js/data/` — CSV input and Data-tab editing
- `dist/js/charts/` — graph gallery and image viewer
- `dist/js/reports/` — forecast and inventory report screens
- `dist/js/tables/` — result table viewer and downloads
- `dist/js/runtime/` — run control and engine connection
- `dist/js/interface/` and `dist/v2.js` — setup, navigation, presentation, accessibility
- `dist/worker.mjs` — browser Python worker
- `dist/engine.py`, `dist/analysis_core.py`, and `dist/visualization.py` — forecasting, evaluation, inventory calculations, and charts
- `dist/*.css` — application styles
- `dist/sample.csv` and `dist/template.csv` — sample input and reusable template
- `netlify.toml` — deployment configuration

## Project

P. Nagendra Reddy  
Data Science in Manufacturing, NIT Warangal
