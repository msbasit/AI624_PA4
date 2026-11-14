"""Download official NATS-Bench metadata, without pretrained weights."""

import re
from pathlib import Path
import requests

root = Path(__file__).resolve().parents[1]
url = "https://drive.usercontent.google.com/download"
params = {"id": "1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul", "export": "download"}
r = requests.get(url, params=params, timeout=90)
if "text/html" in r.headers.get("Content-Type", ""):
    params.update(
        dict(re.findall(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', r.text))
    )
    r = requests.get(url, params=params, stream=True, timeout=90)
r.raise_for_status()
assert "text/html" not in r.headers.get(
    "Content-Type", ""
), "Download returned an HTML page"
path = root / "data" / "NATS-tss-v1_0-3ffb9.pickle.pbz2"
with path.open("wb") as f:
    for chunk in r.iter_content(2**20):
        f.write(chunk)
print(path, path.stat().st_size)
