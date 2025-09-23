from pathlib import Path

# === Config for GEE Processing ===

LAKES = [
    {"name": "Bodensee (Lake Constance)", "lat": 47.600, "lon": 9.300,  "buffer_km": 50},
    {"name": "Chiemsee",                  "lat": 47.880, "lon": 12.410, "buffer_km": 15},
    {"name": "Starnberger See",           "lat": 47.900, "lon": 11.340, "buffer_km": 10},
    {"name": "Müggelsee",                 "lat": 52.430, "lon": 13.650, "buffer_km": 10},
    {"name": "Steinhuder Meer",           "lat": 52.450, "lon": 9.350,  "buffer_km": 10},
]

SELECT_LAKE = "ALL"

DATE_START = "2025-06-01"
DATE_END   = "2025-08-31"
SCALE_M    = 10

ERODE_PIXELS  = 2
BUFFER_PIXELS = 3
MIN_PATCH_PX  = 8

IQR_K = 1.5
Z_THRESHOLD = 3.0
ROBUST_K = 3.0

NDCI_COEFFS = dict(a0=-0.40, a1=1.10, a2=0.00)

DOG_RED = dict(lambda_nm=665, A=230.0,  C=0.170)
DOG_NIR = dict(lambda_nm=865, A=1300.0, C=0.212)
DOG_SWITCH_RRS = 0.03

EE_CLOUD_PROJECT = "big-query-demo-460609"

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)