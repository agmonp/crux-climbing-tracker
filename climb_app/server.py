"""Loopback-only web application, persistent jobs and a single CPU worker."""
from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Config, ROOT
from .pipeline import AnalysisCancelled, VideoProcessor, video_metadata

DATA = ROOT / 'data' / 'jobs'
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MAX_ACTIVE_JOBS = 3
SAMPLE_FILES = ('body-trajectory-input.mp4', 'warp-dynamic-input.mp4', 'warp-fixed-input.mp4')
ARTIFACTS = {'video': 'annotated.mp4', 'preview': 'preview.mp4', 'report': 'report.json',
             'csv': 'metrics.csv', 'poster': 'poster.jpg'}
pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='climb-analysis')
lock = threading.RLock()
jobs: dict[str, dict] = {}
cancellations: dict[str, threading.Event] = {}


def persist(job: dict) -> None:
    """Atomically update metadata; callers hold the registry lock."""
    path = DATA / job['id'] / 'job.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(job, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA.mkdir(parents=True, exist_ok=True)
    with lock:
        for path in DATA.glob('*/job.json'):
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                if job['status'] in ('queued', 'running'):
                    job.update(status='failed', error='Processing was interrupted by a server restart. Upload the clip again.')
                    persist(job)
                jobs[job['id']] = job
            except (OSError, ValueError, KeyError):
                logging.exception('Cannot restore job %s', path)
    yield
    for event in cancellations.values():
        event.set()
    pool.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title='CRUX • Local climbing analysis', lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])


@app.middleware('http')
async def same_origin_mutations(request: Request, call_next):
    origin = request.headers.get('origin')
    if request.method in ('POST', 'DELETE', 'PUT', 'PATCH') and origin:
        if urlparse(origin).netloc != request.headers.get('host'):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail': 'Only same-origin local requests are accepted.'}, status_code=403)
    return await call_next(request)


def require_job(job_id: str) -> dict:
    with lock:
        if job_id not in jobs:
            raise HTTPException(404, 'Analysis not found.')
        return dict(jobs[job_id])


def register(job_id: str, filename: str, source: Path) -> dict:
    try:
        metadata = video_metadata(source, Config())
    except ValueError as exc:
        source.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from exc
    capture = cv2.VideoCapture(str(source))
    try:
        ok, frame = capture.read()
        if not ok:
            raise HTTPException(422, 'The first frame cannot be decoded.')
        scale = min(1.0, 960/max(frame.shape[:2]))
        cv2.imwrite(str(source.parent/'thumbnail.jpg'), cv2.resize(frame, None, fx=scale, fy=scale))
    finally:
        capture.release()
    job = {'id': job_id, 'name': filename, 'source_file': source.name,
           'created': datetime.now(timezone.utc).isoformat(), 'status': 'ready',
           'stage': 'ready', 'progress': 0.0, 'metadata': metadata, 'error': None}
    with lock:
        jobs[job_id] = job
        persist(job)
    return dict(job)


@app.get('/api/health')
def health() -> dict:
    return {'status': 'ok', 'model_ready': Config().model_path.is_file()}


@app.get('/api/jobs')
def list_jobs() -> list[dict]:
    with lock:
        return sorted((dict(j) for j in jobs.values()), key=lambda j: j['created'], reverse=True)


@app.post('/api/upload', status_code=201)
async def upload(file: UploadFile) -> dict:
    extension = Path(file.filename or '').suffix.lower()
    if extension not in ('.mp4', '.mov', '.webm', '.m4v'):
        raise HTTPException(415, 'Choose an MP4, MOV, M4V, or WebM video.')
    job_id = uuid.uuid4().hex
    directory = DATA/job_id
    directory.mkdir(parents=True)
    source = directory/f'source{extension}'
    total = 0
    try:
        with source.open('wb') as target:
            while chunk := await file.read(CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, 'Video must be smaller than 500 MB.')
                target.write(chunk)
        if total == 0:
            raise HTTPException(422, 'The uploaded file is empty.')
        return register(job_id, Path(file.filename).name, source)
    except Exception:
        source.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


class AnalysisOptions(BaseModel):
    crop: tuple[float, float, float, float] | None = None
    static_speed: float = Field(default=0.012, gt=0, le=0.5)
    smoothing_seconds: float = Field(default=0.55, ge=0.15, le=2)
    visibility_threshold: float = Field(default=0.5, ge=0.1, le=1)
    roi_size: int = Field(default=40, ge=20, le=100)
    contact_radius_px: float = Field(default=8, ge=1, le=50)
    mask_skin: bool = True


def worker(job_id: str, config: Config, crop) -> None:
    directory = DATA/job_id
    def update(stage: str, progress: float) -> None:
        with lock:
            jobs[job_id].update(stage=stage, progress=progress)
            persist(jobs[job_id])
    with lock:
        job = jobs[job_id]
        job['status'] = 'running'
        persist(job)
    try:
        result = VideoProcessor(config).process(directory/job['source_file'], directory/'result',
                                                crop, update, cancellations[job_id])
        with lock:
            job.update(status='complete', summary=result.summary, warnings=result.warnings, progress=1.0)
    except AnalysisCancelled:
        with lock:
            job.update(status='cancelled', stage='cancelled')
    except Exception as exc:
        logging.exception('Analysis failed for %s', job_id)
        with lock:
            job.update(status='failed', error=str(exc))
    finally:
        with lock:
            persist(job)
            cancellations.pop(job_id, None)


@app.post('/api/jobs/{job_id}/analyze', status_code=202)
def analyze(job_id: str, options: AnalysisOptions) -> dict:
    require_job(job_id)
    if not Config().model_path.is_file():
        raise HTTPException(503, 'Pose model missing. Run: python setup_model.py')
    crop = options.crop
    if crop and (not all(0 <= n <= 1 for n in crop) or crop[2]-crop[0] < 0.05 or crop[3]-crop[1] < 0.05):
        raise HTTPException(422, 'Select a crop at least 5% wide and tall inside the image.')
    cfg = Config(**options.model_dump(exclude={'crop'}))
    with lock:
        job = jobs[job_id]
        if job['status'] != 'ready':
            raise HTTPException(409, 'This clip has already been submitted. Upload it again for another run.')
        if sum(j['status'] in ('running', 'queued') for j in jobs.values()) >= MAX_ACTIVE_JOBS:
            raise HTTPException(429, 'The processing queue is full. Wait for an analysis to finish.')
        job.update(status='queued', stage='queued', options=options.model_dump())
        cancellations[job_id] = threading.Event()
        persist(job)
        pool.submit(worker, job_id, cfg, crop)
        return dict(job)


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str) -> dict:
    return require_job(job_id)


@app.post('/api/jobs/{job_id}/cancel')
def cancel(job_id: str) -> dict:
    require_job(job_id)
    with lock:
        if job_id in cancellations:
            cancellations[job_id].set()
            return {'status': 'cancelling'}
        return {'status': jobs[job_id]['status']}


@app.get('/api/jobs/{job_id}/files/{kind}')
def artifact(job_id: str, kind: str) -> FileResponse:
    job = require_job(job_id)
    if kind in ('source', 'thumbnail'):
        path = DATA/job_id/(job['source_file'] if kind == 'source' else 'thumbnail.jpg')
    elif kind in ARTIFACTS and job['status'] == 'complete':
        path = DATA/job_id/'result'/ARTIFACTS[kind]
    else:
        raise HTTPException(404, 'Artifact is not available.')
    if not path.is_file():
        raise HTTPException(404, 'Artifact is missing.')
    if kind in ('video', 'csv'):
        return FileResponse(path, filename=f"crux-{job_id[:8]}{path.suffix}")
    return FileResponse(path)


@app.get('/api/samples')
def samples() -> list[dict]:
    return [{'name': name, 'source': 'tommyjtl / climbing-analysis-toolbox',
             'url': 'https://github.com/tommyjtl/climbing-analysis-toolbox/tree/main/examples/videos'}
            for name in SAMPLE_FILES if (ROOT/'samples'/name).is_file()]


@app.post('/api/samples/{name}', status_code=201)
def use_sample(name: str) -> dict:
    if name not in SAMPLE_FILES or not (ROOT/'samples'/name).is_file():
        raise HTTPException(404, 'Sample not found.')
    job_id = uuid.uuid4().hex
    directory = DATA/job_id
    directory.mkdir(parents=True)
    source = directory/'source.mp4'
    shutil.copyfile(ROOT/'samples'/name, source)
    return register(job_id, name, source)


@app.get('/')
def index() -> FileResponse:
    return FileResponse(ROOT/'static'/'index.html')


app.mount('/static', StaticFiles(directory=ROOT/'static'), name='static')
