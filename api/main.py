# /workspaces/ishi/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import anemia  # make sure api/routes/__init__.py exists
from api.utils.config import get_anemia_threshold

app = FastAPI(title="ISHI Anemia API", version="v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Routers
app.include_router(anemia.router)

# Health
@app.get("/health")
def health():
    return {
        "ok": True,
        "version": "eyes_defy_anemia_v0_1",
        "threshold": get_anemia_threshold(),   # ← report live value
    }
