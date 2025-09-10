# scripts/run_api_prod.sh
export PYTHONPATH=/workspaces/ishi
export ANEMIA_CKPT=/workspaces/ishi/api/models/anemia_cnn.pth
export ANEMIA_ARCH=simple_cnn
export ANEMIA_THRESHOLD=0.50
# optional while testing to skip mask stage:
# export ISHI_DISABLE_CROPPER=1
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers=1
