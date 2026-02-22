"""Check Zenodo depositions status."""
from dotenv import load_dotenv
import os, requests, json

load_dotenv(".env")
token = os.getenv("ZENODO_ACCESS_TOKEN")
headers = {"Authorization": f"Bearer {token}"}

resp = requests.get("https://zenodo.org/api/deposit/depositions", headers=headers, params={"size": 10})
resp.raise_for_status()
deps = resp.json()

print(f"Found {len(deps)} depositions:")
for d in deps:
    title = d.get("title", "No title")[:80]
    state = d.get("state", "unknown")
    dep_id = d.get("id", "?")
    print(f"  ID: {dep_id} | State: {state} | Title: {title}")
