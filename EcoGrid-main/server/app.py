import json
import os

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError("openenv-core>=0.2.0 is required for the server.") from e

from fastapi import Request
from fastapi.responses import JSONResponse

from models.schemas import GridAction
from server.ecogrid_environment import ServerEcoGridEnv, ServerObservation

app = create_app(
    ServerEcoGridEnv,
    GridAction,
    ServerObservation,
    env_name="eco-grid-openenv",
    max_concurrent_envs=10,
)


@app.middleware("http")
async def normalize_step_payload(request: Request, call_next):
    """Allow /step payloads with either wrapped or direct action JSON."""
    if request.method == "POST" and request.url.path == "/step":
        body = await request.body()
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return JSONResponse(status_code=422, content={"detail": "Invalid JSON body"})

            if isinstance(payload, dict) and "action" not in payload:
                wrapped = json.dumps({"action": payload}).encode("utf-8")
                request._body = wrapped

                async def _receive():
                    return {"type": "http.request", "body": wrapped, "more_body": False}

                request._receive = _receive

    return await call_next(request)


@app.get("/", include_in_schema=False)
def root():
    """Landing route for judges/operators."""
    return JSONResponse(
        {
            "name": "eco-grid-openenv",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
            "schema": "/schema",
            "version": "/version",
        }
    )


@app.get("/version", include_in_schema=False)
def version():
    return {"version": "1.1.0-stabilized"}


def main():
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
