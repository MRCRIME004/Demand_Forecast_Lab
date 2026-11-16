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

## Repository structure

- `dist/index.html` — application page
- `dist/app.js` and `dist/v2.js` — interface and workflow
- `dist/worker.mjs` — browser Python worker
- `dist/engine.py`, `dist/analysis_core.py`, and `dist/visualization.py` — forecasting, evaluation, inventory calculations, and charts
- `dist/*.css` — application styles
- `dist/sample.csv` and `dist/template.csv` — sample input and reusable template
- `netlify.toml` — deployment configuration

## Project

P. Nagendra Reddy  
Data Science in Manufacturing, NIT Warangal
