import os

def get_anemia_threshold() -> float:
    # Reads the env var each time; falls back to 0.303 if missing/invalid
    v = os.getenv("ANEMIA_THRESHOLD", "0.303")
    try:
        return float(v)
    except ValueError:
        return 0.303
