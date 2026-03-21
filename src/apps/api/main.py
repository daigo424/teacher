from fastapi import FastAPI
from packages.core.config import settings

from apps.api.routers.ask import router as ask_router
from apps.api.routers.health import router as health_router

if settings.env == "local":
    import debugpy

    debugpy.listen(("0.0.0.0", 5678))
    print("🐛 waiting for debugger...")
    debugpy.wait_for_client()

app = FastAPI(title="Teacher API")
app.include_router(health_router)
app.include_router(ask_router)
