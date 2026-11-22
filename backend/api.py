"""WellMind web API (FastAPI).

Run (from the project root, hackathon env):
    python -m uvicorn backend.api:app --port 8000
Then open http://localhost:8000  (the built React app in frontend/dist is served from here).
During frontend development, `npm run dev` in frontend/ proxies /api calls to this server.
"""
import math
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import data, pages, report



@asynccontextmanager
async def lifespan(app):
    """On start-up: load the provided dataset and pre-build the report in the background (instant pages)."""
    threading.Thread(target=report.warm_up, daemon=True).start()
    yield


app = FastAPI(title='WellMind API', lifespan=lifespan)
DIST = Path(__file__).resolve().parent.parent / 'frontend' / 'dist'
XLSX = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def to_json(obj):
    """Make numpy / pandas values JSON-safe: numpy numbers -> Python numbers, NaN / inf -> null."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, np.ndarray):
        return to_json(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.get('/api/overview')
def overview():
    return JSONResponse(to_json(data.overview()))


@app.get('/api/pta/events')
def pta_events():
    return JSONResponse(to_json(data.pta_events()))


@app.get('/api/pta/event/{event_id}')
def pta_event(event_id: int, start: float | None = None, end: float | None = None):
    """Plot data + results for one shut-in. Optional start / end [h] override the radial-flow window."""
    if not 0 <= event_id < len(data.pta_results()):
        raise HTTPException(404, 'unknown event')
    window = None
    if start is not None and end is not None:
        if not 0 < start < end:
            raise HTTPException(400, 'window start must be > 0 and < end')
        window = (start, end)
    return JSONResponse(to_json(data.pta_event_detail(event_id, window)))


def excel_response(kind, filename):
    return Response(data.excel_bytes(kind), media_type=XLSX,
                    headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@app.get('/api/pta/export.xlsx')
def pta_export():
    return excel_response('PTA', f"PTA_results_{data.ACTIVE['id']}.xlsx")


@app.get('/api/vfm/export.xlsx')
def vfm_export():
    return excel_response('VFM', f"VFM_predictions_{data.ACTIVE['id']}.xlsx")


# ---------------------------------------------------------------------------------------------
# VFM page, Insights page, Report page
# ---------------------------------------------------------------------------------------------
@app.get('/api/vfm/overview')
def vfm_overview():
    return JSONResponse(to_json(pages.vfm_overview()))


@app.get('/api/vfm/well/{well}')
def vfm_well(well: str):
    if well not in set(data.daily_rates().well_name):
        raise HTTPException(404, 'unknown well')
    return JSONResponse(to_json(pages.vfm_well(well)))


@app.get('/api/insights')
def insights(oil_price: float = 70.0):
    if not 0 < oil_price < 1000:
        raise HTTPException(400, 'oil price must be between 0 and 1000 $/bbl')
    return JSONResponse(to_json(pages.insights(oil_price)))


@app.get('/api/report.html')
def report_page(oil_price: float = 70.0, download: bool = False):
    """Self-contained engineering report (open in the browser and print to PDF, or download)."""
    headers = {'Content-Disposition': f'attachment; filename="WellMind_report_{data.ACTIVE["id"]}.html"'} if download else {}
    return HTMLResponse(report.report_html(data.ACTIVE['id'], round(oil_price, 2)), headers=headers)


@app.get('/api/model.pkl')
def model_file():
    """The trained VFM model used for every prediction (deliverable: model in .pkl)."""
    return FileResponse(data.ROOT / 'models' / 'vfm_model.pkl', media_type='application/octet-stream', filename='vfm_model.pkl')


@app.get('/api/pta/plots.zip')
def pta_plots():
    return Response(report.pta_plots_zip(data.ACTIVE['id']), media_type='application/zip',
                    headers={'Content-Disposition': f'attachment; filename="PTA_plots_{data.ACTIVE["id"]}.zip"'})


# ---------------------------------------------------------------------------------------------
# Datasets: upload raw files -> pipeline (background job) -> the new dataset becomes active
# ---------------------------------------------------------------------------------------------
@app.get('/api/dataset')
def dataset():
    return JSONResponse(to_json(data.dataset_info()))


@app.get('/api/datasets')
def datasets():
    return JSONResponse(to_json(data.list_datasets()))


@app.post('/api/datasets/{bundle_id}/activate')
def activate(bundle_id: str):
    try:
        data.activate(bundle_id)
    except KeyError:
        raise HTTPException(404, 'unknown dataset')
    return JSONResponse(to_json(data.dataset_info()))


@app.post('/api/upload')
def upload(files: list[UploadFile] = File(...)):
    """Save the uploaded CSV files and start the pipeline in the background. Returns the job to poll."""
    bundle_id = 'upload-' + uuid.uuid4().hex[:6]
    folder = data.UPLOADS / bundle_id
    folder.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = Path(f.filename).name
        if not name.lower().endswith('.csv'):
            raise HTTPException(400, f'{name}: only .csv files are accepted')
        target = folder / name
        with open(target, 'wb') as out:
            shutil.copyfileobj(f.file, out)
        saved.append(target)
    if not saved:
        raise HTTPException(400, 'no files received')
    return JSONResponse(to_json(data.start_upload_job(saved, bundle_id, on_done=report.warm_up)))


@app.get('/api/jobs/{job_id}')
def job(job_id: str):
    if job_id not in data.JOBS:
        raise HTTPException(404, 'unknown job')
    return JSONResponse(to_json(data.JOBS[job_id]))


@app.get('/api/data/signals/{well}')
def signals(well: str, resample: str = '1h'):
    if well not in set(data.daily_rates().well_name):
        raise HTTPException(404, 'unknown well')
    return JSONResponse(to_json(data.signals(well, resample)))


# ---------------------------------------------------------------------------------------------
# Serve the built frontend (single-page app: unknown paths return index.html so client routing works)
# ---------------------------------------------------------------------------------------------
if DIST.exists():
    app.mount('/assets', StaticFiles(directory=DIST / 'assets'), name='assets')

    @app.get('/{path:path}')
    def spa(path: str):
        file = DIST / path
        if path and file.is_file():
            return FileResponse(file)
        return FileResponse(DIST / 'index.html')
