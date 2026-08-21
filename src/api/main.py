"""Abuse-Ring Sentinel v2 — FastAPI application.

Features:
- API key authentication (set SENTINEL_API_KEY env var)
- CORS middleware
- Request logging with timing
- Global exception handler (never leak stack traces)
- Health check with feature store staleness monitoring
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from src.api.schemas import ScoringRequest, ScoringResponse
from src.api.decision_gate import evaluate_risk_pipeline
from src.monitoring.feature_store_health import check_feature_store_health

logger = logging.getLogger('sentinel')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)

# ---------------------------------------------------------------------------
# API Key auth (optional — if SENTINEL_API_KEY is set, enforce it)
# ---------------------------------------------------------------------------
_API_KEY = os.environ.get('SENTINEL_API_KEY')
_api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)


async def verify_api_key(key: str = Security(_api_key_header)):
    """Enforce API key if SENTINEL_API_KEY is configured."""
    if _API_KEY is None:
        # No key configured → open access (local dev / hackathon demo)
        return
    if not key or key != _API_KEY:
        raise HTTPException(status_code=403, detail='Invalid or missing API key')


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    auth_mode = 'API key required' if _API_KEY else 'open (set SENTINEL_API_KEY to secure)'
    logger.info('Abuse-Ring Sentinel v2 starting up — auth: %s', auth_mode)
    yield
    logger.info('Abuse-Ring Sentinel v2 shutting down')


app = FastAPI(
    title='Abuse-Ring Sentinel API',
    version='2.0',
    lifespan=lifespan,
)

# CORS — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['POST', 'GET'],
    allow_headers=['*', 'X-API-Key'],
)


@app.middleware('http')
async def request_logging_middleware(request: Request, call_next):
    """Log every request with timing for observability."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        'method=%s path=%s status=%s elapsed_ms=%.2f',
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: never leak stack traces to the client."""
    logger.exception('Unhandled exception on %s %s', request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={'detail': 'Internal server error'},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post('/score', response_model=ScoringResponse)
async def score_transaction(
    req: ScoringRequest,
    _: None = Depends(verify_api_key),
):
    return evaluate_risk_pipeline(req)


@app.get('/health')
async def health():
    fs_healthy = check_feature_store_health(quiet=True)
    return {
        'status': 'ok' if fs_healthy else 'degraded',
        'feature_store': 'healthy' if fs_healthy else 'stale_or_unreachable',
    }
