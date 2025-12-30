import os

def get_anemia_threshold() -> float:
    """
    Get the anemia classification threshold.
    
    NOTE: Model outputs are INVERTED in api/routes/anemia.py
    After inversion: higher score = more likely anemic
    
    With inverted scores:
    - Anemic patients avg ~0.50
    - Healthy patients avg ~0.46
    
    Set ANEMIA_THRESHOLD env var to override.
    """
    v = os.getenv("ANEMIA_THRESHOLD", "0.55")
    try:
        return float(v)
    except ValueError:
        return 0.55
