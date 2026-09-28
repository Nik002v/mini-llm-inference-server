import asyncio
import json
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from app.api.schemas import GenerateRequest, GenerateResponse
from app.inference.request import InferenceError

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    scheduler = request.app.state.scheduler
    return JSONResponse(
        {
            "status": "ok" if scheduler.healthy else "unavailable",
            "model_loaded": scheduler.worker.loaded,
        },
        status_code=200 if scheduler.healthy else 503,
    )


@router.get("/metrics")
async def metrics(request: Request):
    return Response(
        request.app.state.scheduler.metrics.render(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@router.get("/stats")
async def stats(request: Request):
    return request.app.state.scheduler.stats()


@router.post("/generate", response_model=GenerateResponse)
async def generate(params: GenerateRequest, request: Request):
    scheduler = request.app.state.scheduler
    job = scheduler.submit(params)
    if params.stream:

        async def events():
            try:
                while True:
                    try:
                        kind, payload = await asyncio.wait_for(job.events.get(), 5)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if kind in ("done", "error"):
                        break
            finally:
                if not job.future.done():
                    scheduler.cancel(job)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": job.id,
            },
        )
    try:
        while not job.future.done():
            if await request.is_disconnected():
                scheduler.cancel(job)
                raise InferenceError("client_cancelled", 499)
            await asyncio.wait({job.future}, timeout=0.05)
        return job.future.result()
    except asyncio.CancelledError:
        scheduler.cancel(job)
        raise
