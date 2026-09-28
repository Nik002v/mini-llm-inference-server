from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.config import Settings
from app.inference.worker import ModelWorker
from app.inference.scheduler import Scheduler
from app.inference.request import InferenceError
from app.api.routes import router


class BodyLimit:
    """Bound actual body bytes, including chunked transfer, before JSON parsing."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        parts, size = [], 0
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            size += len(msg.get("body", b""))
            if size > self.limit:
                return await JSONResponse(
                    {"error": "request_body_too_large"}, status_code=413
                )(scope, receive, send)
            parts.append(msg.get("body", b""))
            if not msg.get("more_body", False):
                break
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {
                    "type": "http.request",
                    "body": b"".join(parts),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, worker=None):
    settings = settings or Settings()
    scheduler = Scheduler(settings, worker or ModelWorker(settings))

    @asynccontextmanager
    async def lifespan(app):
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.close()

    app = FastAPI(title="Mini LLM Inference Server", version="1.0.0", lifespan=lifespan)
    app.state.scheduler = scheduler
    app.add_middleware(BodyLimit, limit=settings.max_body_bytes)

    @app.exception_handler(InferenceError)
    async def inference_error(request: Request, exc: InferenceError):
        return JSONResponse(
            {"error": exc.code},
            status_code=exc.status,
            headers={"Retry-After": "1"} if exc.status == 503 else None,
        )

    app.include_router(router)
    return app


app = create_app()
