# api/main.py
from fastapi import FastAPI
from api.routes import health as health_routes
from fastapi.middleware.cors import CORSMiddleware
from api.routes import anemia, news

app = FastAPI()
app.include_router(health_routes.router)

# tighten for prod: list your real origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # replace with ["https://your.app"] in prod
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

app.include_router(anemia.router)
app.include_router(news.router) 

@app.get("/health")
def health():
    return {"ok": True, "version": "eyes_defy_anemia_v0_1"}

@app.get("/ready")
def ready():
    return {"ready": bool(getattr(app.state, "model_ready", False))}

@app.on_event("startup")
async def _startup():
    try:
        anemia.preload_model()           # strict load; raises on failure
        app.state.model_ready = True
        print("[ISHI] Startup: model ready")
    except Exception as e:
        app.state.model_ready = False
        print(f"[ISHI] Startup: model NOT ready: {e}")
