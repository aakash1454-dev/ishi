from fastapi import APIRouter
router = APIRouter(tags=["meta"])

@router.get("/healthz")
def healthz():
    return {"ok": True}
